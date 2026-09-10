from __future__ import annotations

import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from choto.log import get_logger

_log = get_logger(__name__)


SANDBOX_DIR_NAME = "choto-bench-sandbox"

GLOBAL_DOMAIN = "NSGlobalDomain"
DARK_MODE_KEY = "AppleInterfaceStyle"
DOCK_DOMAIN = "com.apple.dock"
DOCK_AUTOHIDE_KEY = "autohide"
DOCK_TILESIZE_KEY = "tilesize"

SETTINGS_WHITELIST: dict[tuple[str, str], str] = {
    (GLOBAL_DOMAIN, DARK_MODE_KEY): "string",
    (DOCK_DOMAIN, DOCK_AUTOHIDE_KEY): "bool",
    (DOCK_DOMAIN, DOCK_TILESIZE_KEY): "int",
}

QUITTABLE_APPS = frozenset({"Calculator", "System Settings"})

COMMAND_TIMEOUT_S = 30.0

RESTORE_POLL_INTERVAL_S = 0.2
RESTORE_TIMEOUT_S = 3.0

APPLESCRIPT_NOT_AUTHORIZED = -1743
_APPLESCRIPT_ABSENT_CODES = frozenset({-600, -1719, -1728})

_NOT_RUNNING = "!not-running"
_NO_WINDOW = "!no-window"


_DIRECTIONALITY_MARKS = frozenset("\u200e\u200f\u2066\u2067\u2068\u2069\u061c")

_GROUPING_CHARACTERS = frozenset(" \u00a0\u202f\u2009\u2007'\u2019")

_MINUS_SIGNS = str.maketrans({"\u2212": "-", "\ufe63": "-", "\uff0d": "-"})

_GROUP_SIZE = 3

_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

_MAX_AX_DEPTH = 12

_SEEN_LIMIT = 12


class OpError(RuntimeError): ...


class PreconditionError(OpError): ...


class AppleScriptError(OpError):
    def __init__(self, message: str, code: int | None) -> None:
        super().__init__(message)
        self.code = code


class SystemBridge(ABC):
    @abstractmethod
    def read_default(self, domain: str, key: str) -> str | None: ...

    @abstractmethod
    def write_default(self, domain: str, key: str, value: str, kind: str) -> None: ...

    @abstractmethod
    def delete_default(self, domain: str, key: str) -> None: ...

    @abstractmethod
    def set_dark_mode(self, enabled: bool) -> None: ...

    @abstractmethod
    def restart_dock(self) -> None: ...

    @abstractmethod
    def run_applescript(self, script: str) -> str: ...


class MacSystemBridge(SystemBridge):
    def read_default(self, domain: str, key: str) -> str | None:
        completed = self._run(["defaults", "read", domain, key], check=False)
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def write_default(self, domain: str, key: str, value: str, kind: str) -> None:
        flag = {"string": "-string", "bool": "-bool", "int": "-int"}.get(kind)
        if flag is None:
            raise OpError(f"unknown preference kind {kind!r} for {domain} {key}")
        self._run(["defaults", "write", domain, key, flag, value], check=True)

    def delete_default(self, domain: str, key: str) -> None:
        self._run(["defaults", "delete", domain, key], check=False)

    def set_dark_mode(self, enabled: bool) -> None:
        value = "true" if enabled else "false"
        self.run_applescript(
            'tell application "System Events" to tell appearance preferences '
            f"to set dark mode to {value}"
        )

    def restart_dock(self) -> None:
        self._run(["killall", "Dock"], check=False)

    def run_applescript(self, script: str) -> str:
        completed = self._run(["osascript", "-e", script], check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise AppleScriptError(
                f"osascript failed: {stderr or 'no diagnostic'}", _applescript_code(stderr)
            )
        return completed.stdout.strip()

    def _run(self, command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_S,
                check=False,
            )
        except FileNotFoundError as exc:
            raise OpError(f"{command[0]} is not available: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise OpError(
                f"{command[0]} did not finish within {COMMAND_TIMEOUT_S:.0f}s: {' '.join(command)}"
            ) from exc
        if check and completed.returncode != 0:
            raise OpError(
                f"{' '.join(command)} failed with code {completed.returncode}: "
                f"{completed.stderr.strip() or 'no diagnostic'}"
            )
        return completed


def _applescript_code(stderr: str) -> int | None:
    start = stderr.rfind("(-")
    if start == -1:
        return None
    end = stderr.find(")", start)
    if end == -1:
        return None
    try:
        return int(stderr[start + 1 : end])
    except ValueError:
        return None


def sandbox_root() -> Path:
    return Path.home() / SANDBOX_DIR_NAME


@dataclass
class TaskContext:
    sandbox: Path
    system: SystemBridge
    snapshots: dict[str, Any] = field(default_factory=dict)


def resolve_in_sandbox(sandbox: Path, relative: str) -> Path:
    if not relative or relative.startswith("/"):
        raise OpError(f"sandbox path must be relative and non-empty, got {relative!r}")
    candidate = (sandbox / relative).resolve()
    root = sandbox.resolve()
    if candidate != root and root not in candidate.parents:
        raise OpError(f"sandbox path {relative!r} escapes {root}")
    return candidate


def _require_whitelisted(domain: str, key: str) -> str:
    kind = SETTINGS_WHITELIST.get((domain, key))
    if kind is None:
        allowed = ", ".join(f"{d} {k}" for d, k in sorted(SETTINGS_WHITELIST))
        raise OpError(f"preference {domain} {key} is not on the stand's whitelist ({allowed})")
    return kind


def _snapshot_key(domain: str, key: str) -> str:
    return f"{domain}:{key}"


class SetupOp(ABC):
    @abstractmethod
    def apply(self, ctx: TaskContext) -> None: ...

    @abstractmethod
    def describe(self) -> str: ...


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str


class CheckOp(ABC):
    @abstractmethod
    def evaluate(self, ctx: TaskContext) -> CheckResult: ...

    @abstractmethod
    def describe(self) -> str: ...


class TeardownOp(ABC):
    @abstractmethod
    def revert(self, ctx: TaskContext) -> None: ...

    @abstractmethod
    def describe(self) -> str: ...

    def residue(self, ctx: TaskContext) -> list[str]:
        return []


@dataclass(frozen=True)
class ResetSandbox(SetupOp):
    def apply(self, ctx: TaskContext) -> None:
        _guard_sandbox(ctx.sandbox)
        if ctx.sandbox.is_symlink():
            raise OpError(f"{ctx.sandbox} is a symlink; refusing to touch it")
        if ctx.sandbox.exists():
            shutil.rmtree(ctx.sandbox)
        ctx.sandbox.mkdir(parents=True)

    def describe(self) -> str:
        return f"reset the sandbox ({SANDBOX_DIR_NAME}/)"


def _guard_sandbox(sandbox: Path) -> None:
    if sandbox.name != SANDBOX_DIR_NAME:
        raise OpError(f"refusing to use {sandbox} as the sandbox: name must be {SANDBOX_DIR_NAME}")
    if sandbox.parent == sandbox:
        raise OpError(f"refusing to use {sandbox} as the sandbox: it has no parent")


@dataclass(frozen=True)
class WriteFile(SetupOp):
    path: str
    content: str

    def apply(self, ctx: TaskContext) -> None:
        target = resolve_in_sandbox(ctx.sandbox, self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.content, encoding="utf-8")

    def describe(self) -> str:
        return f"write {self.path} ({len(self.content)} chars)"


@dataclass(frozen=True)
class MakeDir(SetupOp):
    path: str

    def apply(self, ctx: TaskContext) -> None:
        resolve_in_sandbox(ctx.sandbox, self.path).mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return f"create directory {self.path}/"


@dataclass(frozen=True)
class SnapshotDefault(SetupOp):
    domain: str
    key: str

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def apply(self, ctx: TaskContext) -> None:
        value = ctx.system.read_default(self.domain, self.key)
        ctx.snapshots[_snapshot_key(self.domain, self.key)] = value
        _log.info("bench.task.snapshot", domain=self.domain, key=self.key, value=value)

    def describe(self) -> str:
        return f"remember {self.domain} {self.key}"


@dataclass(frozen=True)
class RequireDefaultIn(SetupOp):
    domain: str
    key: str
    accepted: tuple[str | None, ...]
    hint: str

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def apply(self, ctx: TaskContext) -> None:
        value = ctx.system.read_default(self.domain, self.key)
        if value not in self.accepted:
            raise PreconditionError(
                f"{self.domain} {self.key} is {value!r}, task needs one of "
                f"{self.accepted!r}. {self.hint}"
            )

    def describe(self) -> str:
        return f"require {self.domain} {self.key} in {self.accepted!r}"


@dataclass(frozen=True)
class RequireDefaultBelow(SetupOp):
    domain: str
    key: str
    limit: int
    hint: str

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def apply(self, ctx: TaskContext) -> None:
        raw = ctx.system.read_default(self.domain, self.key)
        if raw is None:
            return
        try:
            value = float(raw)
        except ValueError as exc:
            raise OpError(f"{self.domain} {self.key} is not numeric: {raw!r}") from exc
        if value >= self.limit:
            raise PreconditionError(
                f"{self.domain} {self.key} is {raw}, task needs it below {self.limit}. {self.hint}"
            )

    def describe(self) -> str:
        return f"require {self.domain} {self.key} < {self.limit}"


@dataclass(frozen=True)
class QuitApp(SetupOp, TeardownOp):
    app: str

    def __post_init__(self) -> None:
        if self.app not in QUITTABLE_APPS:
            allowed = ", ".join(sorted(QUITTABLE_APPS))
            raise OpError(f"refusing to quit {self.app!r}; the stand may only quit: {allowed}")

    def apply(self, ctx: TaskContext) -> None:
        self._quit(ctx)

    def revert(self, ctx: TaskContext) -> None:
        self._quit(ctx)

    def _quit(self, ctx: TaskContext) -> None:
        script = (
            f'tell application "System Events" to if not (exists process "{self.app}") '
            'then return "absent"\n'
            f'tell application "{self.app}" to quit\n'
            'return "quit"'
        )
        try:
            ctx.system.run_applescript(script)
        except AppleScriptError as exc:
            if exc.code in _APPLESCRIPT_ABSENT_CODES:
                return
            raise

    def describe(self) -> str:
        return f"quit {self.app}"


@dataclass(frozen=True)
class SnapshotTextEditDocuments(SetupOp):
    snapshot_name = "textedit:untitled"

    def apply(self, ctx: TaskContext) -> None:
        ctx.snapshots[self.snapshot_name] = _textedit_untitled_documents(ctx)

    def describe(self) -> str:
        return "remember TextEdit's unsaved documents"


def _textedit_untitled_documents(ctx: TaskContext) -> list[str]:
    script = "\n".join(
        [
            'tell application "System Events" to if not (exists process "TextEdit") then return ""',
            'set collected to ""',
            'tell application "TextEdit"',
            "    repeat with doc in (every document)",
            '        set docPath to ""',
            "        try",
            "            set docPath to (path of doc) as text",
            "        end try",
            '        if docPath is "" then set collected to collected & (name of doc) & linefeed',
            "    end repeat",
            "end tell",
            "return collected",
        ]
    )
    try:
        output = ctx.system.run_applescript(script)
    except AppleScriptError as exc:
        if exc.code in _APPLESCRIPT_ABSENT_CODES:
            return []
        raise
    return [line.strip() for line in output.splitlines() if line.strip()]


@dataclass(frozen=True)
class FileContentEquals(CheckOp):
    path: str
    expected: str
    exact: bool = True

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        target = resolve_in_sandbox(ctx.sandbox, self.path)
        if not target.is_file():
            return CheckResult(False, f"{self.path}: no such file")
        try:
            actual = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return CheckResult(False, f"{self.path}: unreadable as utf-8 ({exc})")
        wanted = self.expected if self.exact else self.expected.strip()
        found = actual if self.exact else actual.strip()
        if found == wanted:
            return CheckResult(True, f"{self.path} holds {_clip(found)!r}")
        return CheckResult(False, f"{self.path} holds {_clip(found)!r}, expected {_clip(wanted)!r}")

    def describe(self) -> str:
        mode = "byte-exact" if self.exact else "trimmed"
        return f"{self.path} == {_clip(self.expected)!r} ({mode})"


@dataclass(frozen=True)
class FileContains(CheckOp):
    path: str
    needle: str
    present: bool = True

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        target = resolve_in_sandbox(ctx.sandbox, self.path)
        if not target.is_file():
            return CheckResult(False, f"{self.path}: no such file")
        try:
            actual = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return CheckResult(False, f"{self.path}: unreadable ({exc})")
        found = self.needle in actual
        if found == self.present:
            verb = "contains" if found else "does not contain"
            return CheckResult(True, f"{self.path} {verb} {self.needle!r}")
        verb = "contains" if found else "does not contain"
        return CheckResult(False, f"{self.path} {verb} {self.needle!r}, expected otherwise")

    def describe(self) -> str:
        verb = "contains" if self.present else "does not contain"
        return f"{self.path} {verb} {self.needle!r}"


@dataclass(frozen=True)
class PathIs(CheckOp):
    path: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("file", "dir", "absent"):
            raise OpError(f"unknown path kind {self.kind!r}")

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        target = resolve_in_sandbox(ctx.sandbox, self.path)
        actual = "file" if target.is_file() else "dir" if target.is_dir() else "absent"
        if actual == self.kind:
            return CheckResult(True, f"{self.path} is {actual}")
        return CheckResult(False, f"{self.path} is {actual}, expected {self.kind}")

    def describe(self) -> str:
        return f"{self.path} is {self.kind}"


@dataclass(frozen=True)
class DefaultEquals(CheckOp):
    domain: str
    key: str
    expected: str | None

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        value = ctx.system.read_default(self.domain, self.key)
        label = f"{self.domain} {self.key}"
        if value == self.expected:
            return CheckResult(True, f"{label} is {value!r}")
        return CheckResult(False, f"{label} is {value!r}, expected {self.expected!r}")

    def describe(self) -> str:
        return f"defaults read {self.domain} {self.key} == {self.expected!r}"


@dataclass(frozen=True)
class DefaultAtLeast(CheckOp):
    domain: str
    key: str
    minimum: int

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        raw = ctx.system.read_default(self.domain, self.key)
        label = f"{self.domain} {self.key}"
        if raw is None:
            return CheckResult(False, f"{label} is unset, expected >= {self.minimum}")
        try:
            value = float(raw)
        except ValueError:
            return CheckResult(False, f"{label} is {raw!r}, which is not a number")
        if value >= self.minimum:
            return CheckResult(True, f"{label} is {raw}")
        return CheckResult(False, f"{label} is {raw}, expected >= {self.minimum}")

    def describe(self) -> str:
        return f"defaults read {self.domain} {self.key} >= {self.minimum}"


@dataclass(frozen=True)
class WindowTitleIsOneOf(CheckOp):
    app: str
    titles: tuple[str, ...]

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        script = (
            'tell application "System Events"\n'
            f'    if not (exists process "{self.app}") then return "{_NOT_RUNNING}"\n'
            f'    tell process "{self.app}"\n'
            f'        if (count of windows) is 0 then return "{_NO_WINDOW}"\n'
            "        return title of window 1\n"
            "    end tell\n"
            "end tell"
        )
        title = _inspect(ctx, script)
        if title == _NOT_RUNNING:
            return CheckResult(False, f"{self.app} is not running")
        if title == _NO_WINDOW:
            return CheckResult(False, f"{self.app} has no window open")
        wanted = {name.casefold() for name in self.titles}
        if title.casefold() in wanted:
            return CheckResult(True, f"{self.app} front window is {title!r}")
        return CheckResult(
            False, f"{self.app} front window is {title!r}, expected one of {list(self.titles)}"
        )

    def describe(self) -> str:
        return f"{self.app} front window title in {list(self.titles)}"


_CALCULATOR_DISPLAY_SCRIPT = "\n".join(
    [
        'tell application "System Events"',
        f'    if not (exists process "Calculator") then return "{_NOT_RUNNING}"',
        '    tell process "Calculator"',
        f'        if (count of windows) is 0 then return "{_NO_WINDOW}"',
        "        set displayRoot to window 1",
        "    end tell",
        "end tell",
        "set shownValues to my collectDisplayText(displayRoot, 0)",
        'set report to ""',
        "repeat with shownValue in shownValues",
        "    set report to report & shownValue & linefeed",
        "end repeat",
        "return report",
        "",
        "on collectDisplayText(uiElement, depth)",
        "    set found to {}",
        f"    if depth > {_MAX_AX_DEPTH} then return found",
        '    tell application "System Events"',
        "        try",
        "            set kids to UI elements of uiElement",
        "        on error",
        "            return found",
        "        end try",
        "        repeat with kid in kids",
        "            try",
        "                set elementClass to class of kid",
        "                if elementClass is static text or elementClass is text field then",
        "                    set end of found to (value of kid as text)",
        "                end if",
        "            end try",
        "            set found to found & (my collectDisplayText(kid, depth + 1))",
        "        end repeat",
        "    end tell",
        "    return found",
        "end collectDisplayText",
    ]
)


@dataclass(frozen=True)
class CalculatorDisplayEquals(CheckOp):
    expected: str
    wanted: Decimal = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        parsed = parse_display_number(self.expected)
        if parsed is None:
            raise ValueError(
                f"CalculatorDisplayEquals needs a number to compare against, got {self.expected!r}"
            )
        object.__setattr__(self, "wanted", parsed)

    def evaluate(self, ctx: TaskContext) -> CheckResult:
        output = _inspect(ctx, _CALCULATOR_DISPLAY_SCRIPT)
        if output == _NOT_RUNNING:
            return CheckResult(False, "Calculator is not running")
        if output == _NO_WINDOW:
            return CheckResult(False, "Calculator has no window open")
        seen = [line.strip() for line in output.splitlines() if line.strip()]
        for value in seen:
            if parse_display_number(value) == self.wanted:
                return CheckResult(True, f"Calculator display reads {value!r}")
        shown = [_clip(value) for value in seen[:_SEEN_LIMIT]] or ["nothing readable"]
        if len(seen) > _SEEN_LIMIT:
            shown.append(f"… and {len(seen) - _SEEN_LIMIT} more")
        return CheckResult(False, f"Calculator shows {shown}, expected {self.expected!r}")

    def describe(self) -> str:
        return f"Calculator display == {self.expected}"


def _inspect(ctx: TaskContext, script: str) -> str:
    try:
        return ctx.system.run_applescript(script)
    except AppleScriptError as exc:
        if exc.code == APPLESCRIPT_NOT_AUTHORIZED:
            raise OpError(
                "the stand is not allowed to ask System Events about other applications; "
                "grant Automation (and Accessibility) permission to the terminal running it: "
                f"{exc}"
            ) from exc
        if exc.code in _APPLESCRIPT_ABSENT_CODES:
            return _NOT_RUNNING
        raise


def parse_display_number(text: str) -> Decimal | None:
    cleaned = "".join(char for char in text if char not in _DIRECTIONALITY_MARKS)
    cleaned = cleaned.strip().translate(_MINUS_SIGNS)
    cleaned = "".join(char for char in cleaned if char not in _GROUPING_CHARACTERS)
    decimal_point = _decimal_separator(cleaned)
    for separator in (".", ","):
        if separator != decimal_point:
            cleaned = cleaned.replace(separator, "")
    if decimal_point is not None:
        cleaned = cleaned.replace(decimal_point, ".")
    if not _NUMBER_PATTERN.match(cleaned):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _decimal_separator(text: str) -> str | None:
    dots, commas = text.count("."), text.count(",")
    if dots and commas:
        return "." if text.rindex(".") > text.rindex(",") else ","
    if dots > 1 or commas > 1:
        return None
    if dots:
        return "."
    if commas:
        fraction = text.rsplit(",", 1)[1]
        return None if len(fraction) == _GROUP_SIZE and fraction.isdigit() else ","
    return None


def _clip(text: str, limit: int = 60) -> str:
    collapsed = text.replace("\n", "\\n")
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


@dataclass(frozen=True)
class RemoveSandbox(TeardownOp):
    def revert(self, ctx: TaskContext) -> None:
        _guard_sandbox(ctx.sandbox)
        if ctx.sandbox.is_symlink():
            raise OpError(f"{ctx.sandbox} is a symlink; refusing to remove it")
        if ctx.sandbox.exists():
            shutil.rmtree(ctx.sandbox)

    def residue(self, ctx: TaskContext) -> list[str]:
        if ctx.sandbox.exists():
            return [f"sandbox still exists: {ctx.sandbox}"]
        return []

    def describe(self) -> str:
        return f"delete the sandbox ({SANDBOX_DIR_NAME}/)"


@dataclass(frozen=True)
class RestoreDefault(TeardownOp):
    domain: str
    key: str

    def __post_init__(self) -> None:
        _require_whitelisted(self.domain, self.key)

    def revert(self, ctx: TaskContext) -> None:
        name = _snapshot_key(self.domain, self.key)
        if name not in ctx.snapshots:
            raise OpError(
                f"no snapshot of {self.domain} {self.key}: the task must snapshot a "
                "setting during setup before it may be restored"
            )
        previous = ctx.snapshots[name]
        if ctx.system.read_default(self.domain, self.key) == previous:
            return

        _log.info("bench.task.restore", domain=self.domain, key=self.key, value=previous)
        if (self.domain, self.key) == (GLOBAL_DOMAIN, DARK_MODE_KEY):
            ctx.system.set_dark_mode(previous == "Dark")
        else:
            if previous is None:
                ctx.system.delete_default(self.domain, self.key)
            else:
                ctx.system.write_default(
                    self.domain, self.key, previous, SETTINGS_WHITELIST[(self.domain, self.key)]
                )
            ctx.system.restart_dock()

        current = self._await_value(ctx, previous)
        if current != previous:
            raise OpError(
                f"could not restore {self.domain} {self.key}: it reads {current!r}, "
                f"expected {previous!r} — revert it by hand"
            )

    def _await_value(self, ctx: TaskContext, wanted: str | None) -> str | None:
        deadline = time.monotonic() + RESTORE_TIMEOUT_S
        current = ctx.system.read_default(self.domain, self.key)
        while current != wanted and time.monotonic() < deadline:
            time.sleep(RESTORE_POLL_INTERVAL_S)
            current = ctx.system.read_default(self.domain, self.key)
        return current

    def residue(self, ctx: TaskContext) -> list[str]:
        name = _snapshot_key(self.domain, self.key)
        if name not in ctx.snapshots:
            return []
        previous = ctx.snapshots[name]
        current = ctx.system.read_default(self.domain, self.key)
        if current != previous:
            return [f"{self.domain} {self.key} is {current!r}, was {previous!r}"]
        return []

    def describe(self) -> str:
        return f"restore {self.domain} {self.key}"


@dataclass(frozen=True)
class CloseSandboxTextEditDocuments(TeardownOp):
    def revert(self, ctx: TaskContext) -> None:
        prefix = _applescript_string(f"{ctx.sandbox.resolve()}/")
        script = "\n".join(
            [
                'tell application "System Events" to if not (exists process "TextEdit") '
                'then return "0"',
                "set closedCount to 0",
                'tell application "TextEdit"',
                "    repeat with doc in (every document)",
                '        set docPath to ""',
                "        try",
                "            set docPath to (path of doc) as text",
                "        end try",
                f"        if docPath starts with {prefix} then",
                "            close doc saving no",
                "            set closedCount to closedCount + 1",
                "        end if",
                "    end repeat",
                "end tell",
                "return closedCount as text",
            ]
        )
        try:
            ctx.system.run_applescript(script)
        except AppleScriptError as exc:
            if exc.code in _APPLESCRIPT_ABSENT_CODES:
                return
            raise

    def residue(self, ctx: TaskContext) -> list[str]:
        before = Counter(ctx.snapshots.get(SnapshotTextEditDocuments.snapshot_name, []))
        after = Counter(_textedit_untitled_documents(ctx))
        new = after - before
        if not new:
            return []
        names = ", ".join(sorted(new.elements()))
        return [f"TextEdit has unsaved document(s) this run left behind: {names}"]

    def describe(self) -> str:
        return "close TextEdit documents opened from the sandbox"


def _applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
