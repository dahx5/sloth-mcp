from __future__ import annotations

import threading
import time
from typing import Any, Protocol

from AppKit import (
    NSApplicationActivateAllWindows,
    NSApplicationActivateIgnoringOtherApps,
    NSApplicationActivationPolicyProhibited,
    NSRunningApplication,
    NSWorkspace,
    NSWorkspaceOpenConfiguration,
)
from CoreFoundation import (
    CFRunLoopRunInMode,
    kCFRunLoopDefaultMode,
    kCFRunLoopRunFinished,
)
from Foundation import NSURL, NSBundle

from choto.executor.applauncher import AppActivationError, AppIdentity
from choto.log import get_logger
from choto.overlay.identity import is_overlay_helper
from choto.vision.parser import FrontmostApp, frontmost_app_owner

_log = get_logger(__name__)

__all__ = [
    "AppActivationError",
    "AppIdentity",
    "MacAppLauncher",
    "activate_app",
    "app_version",
    "resolve_running_app_name",
    "same_app",
]

_ACTIVATE_OPTIONS = NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps

_OPEN_TIMEOUT_S = 10.0
_ACTIVATION_RETRY_INTERVAL_S = 0.4
_MAX_ACTIVATION_ATTEMPTS = 6

_FRONTMOST_TIMEOUT_WARM_S = 4.0
_FRONTMOST_TIMEOUT_COLD_S = 15.0
_FRONTMOST_POLL_INTERVAL_S = 0.05


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


_WORKSPACE_REFRESH_SPIN_S = 0.02

_VERSION_KEYS = ("CFBundleShortVersionString", "CFBundleVersion")


def activate_app(app_name: str, clock: _Clock = time) -> str:
    wanted = app_name.strip()
    if not wanted:
        raise ValueError("app_name must be a non-empty string.")

    workspace = _shared_workspace()
    target = _select_running(wanted, _activatable_apps(workspace))

    was_running = target is not None

    if target is not None:
        name = _display_name(target)
        _reopen_running(workspace, target, name)
    else:
        target = _launch(workspace, wanted)
        name = _display_name(target) or wanted

    attempts, waited_s = _confirm_frontmost(
        workspace, target, name, _confirmation_timeout(was_running), clock
    )
    _log.info(
        "app.activated",
        requested=wanted,
        resolved=name,
        cold_launch=not was_running,
        attempts=attempts,
        waited_s=round(waited_s, 3),
    )
    return name


def resolve_running_app_name(app_name: str) -> str | None:
    wanted = app_name.strip()
    if not wanted:
        return None
    app = _select_running(wanted, _activatable_apps(_shared_workspace()))
    return _display_name(app) or None


def same_app(observed: str, wanted: str) -> bool:
    left = observed.strip().casefold()
    right = wanted.strip().casefold()
    if not left or not right:
        return False
    if left == right:
        return True
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    return longer.startswith(f"{shorter} ")


def app_version(app_name: str) -> str:
    wanted = app_name.strip()
    if not wanted:
        return ""

    try:
        workspace = _shared_workspace()
        running = _select_running(wanted, _activatable_apps(workspace))
        url = running.bundleURL() if running is not None else None
        if url is None:
            path = workspace.fullPathForApplication_(wanted)
            if not path:
                _log.debug("app.version_unknown", app_name=wanted, reason="no such application")
                return ""
            url = _file_url(str(path))
        info = _bundle_info(url)
    except AppActivationError as exc:
        _log.warning("app.version_ambiguous", app_name=wanted, error=str(exc))
        return ""
    except Exception:  # noqa: BLE001 - an unreadable version is an unknown one
        _log.exception("app.version_read_failed", app_name=wanted)
        return ""

    if info is None:
        _log.debug("app.version_unknown", app_name=wanted, reason="no bundle information")
        return ""
    return _version_from_info(info)


def _version_from_info(info: Any) -> str:
    for key in _VERSION_KEYS:
        value = info.get(key)
        if value is None:
            continue
        version = str(value).strip()
        if version:
            return version
    return ""


def _shared_workspace() -> Any:
    return NSWorkspace.sharedWorkspace()


def _spin_run_loop(timeout_s: float) -> int:
    return int(CFRunLoopRunInMode(kCFRunLoopDefaultMode, timeout_s, False))


def _refresh_workspace_state() -> bool:
    if _spin_run_loop(_WORKSPACE_REFRESH_SPIN_S) == int(kCFRunLoopRunFinished):
        _log.warning(
            "app.workspace_refresh_ineffective",
            thread=threading.current_thread().name,
            reason="this thread's run loop has no workspace sources; app list may be stale",
        )
        return False
    return True


def _open_configuration() -> Any:
    configuration = NSWorkspaceOpenConfiguration.configuration()
    configuration.setActivates_(True)
    return configuration


def _file_url(path: str) -> Any:
    return NSURL.fileURLWithPath_(path)


def _running_application(pid: int) -> Any | None:
    return NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)


def _bundle_path(url: Any) -> str:
    if url is None:
        return ""
    path = url.path()
    return str(path) if path else ""


def _bundle_info(url: Any) -> Any | None:
    bundle = NSBundle.bundleWithURL_(url)
    if bundle is None:
        return None
    return bundle.infoDictionary()


def _activatable_apps(workspace: Any) -> list[tuple[str, Any]]:
    _refresh_workspace_state()
    apps: list[tuple[str, Any]] = []
    for app in workspace.runningApplications():
        if bool(app.isTerminated()):
            continue
        if int(app.activationPolicy()) == int(NSApplicationActivationPolicyProhibited):
            continue
        if is_overlay_helper(int(app.processIdentifier())):
            continue
        name = app.localizedName()
        if name:
            apps.append((str(name), app))
    return apps


def _select_running(wanted: str, apps: list[tuple[str, Any]]) -> Any | None:
    folded = wanted.casefold()

    exact = [entry for entry in apps if entry[0].casefold() == folded]
    if exact:
        return exact[0][1]

    kin = [entry for entry in apps if same_app(entry[0], wanted)]
    if kin:
        return _unique(wanted, kin, "name extension")

    return None


def _unique(wanted: str, candidates: list[tuple[str, Any]], tier: str) -> Any:
    if len(candidates) == 1:
        return candidates[0][1]
    names = ", ".join(f'"{name}"' for name, _ in sorted(candidates, key=lambda e: e[0]))
    raise AppActivationError(
        f'app name "{wanted}" is ambiguous: {len(candidates)} running apps match it by '
        f"{tier} ({names}); use the full name"
    )


def _display_name(app: Any | None) -> str:
    if app is None:
        return ""
    name = app.localizedName()
    return str(name) if name else ""


def _reopen_running(workspace: Any, app: Any, name: str) -> None:
    url = app.bundleURL()
    if url is not None:
        error = _open_application(workspace, url)[1]
        if not error:
            return
        _log.warning("app.reopen_failed", app_name=name, error=error)

    if not bool(app.activateWithOptions_(_ACTIVATE_OPTIONS)):
        raise AppActivationError(f'could not activate running app "{name}"')


def _launch(workspace: Any, wanted: str) -> Any:
    path = workspace.fullPathForApplication_(wanted)
    if not path:
        raise AppActivationError(
            f'app "{wanted}" is not running and no installed application has that name'
        )

    app, error = _open_application(workspace, _file_url(str(path)))
    if error:
        raise AppActivationError(f'could not launch app "{wanted}" from {path}: {error}')
    if app is None:
        raise AppActivationError(
            f'could not launch app "{wanted}" from {path}: the system reported neither an '
            "application nor an error"
        )
    return app


def _open_application(
    workspace: Any, url: Any, timeout_s: float | None = None
) -> tuple[Any | None, str]:
    timeout = _OPEN_TIMEOUT_S if timeout_s is None else max(0.0, min(timeout_s, _OPEN_TIMEOUT_S))
    finished = threading.Event()
    outcome: dict[str, Any] = {"app": None, "error": ""}

    def completion_handler(app: Any, error: Any) -> None:
        outcome["app"] = app
        outcome["error"] = _error_text(error)
        finished.set()

    workspace.openApplicationAtURL_configuration_completionHandler_(
        url, _open_configuration(), completion_handler
    )
    if not finished.wait(timeout):
        return None, f"the open request did not complete within {timeout:.1f}s"
    error = str(outcome["error"])
    if error:
        return None, error
    return outcome["app"], ""


def _error_text(error: Any) -> str:
    if error is None:
        return ""
    description = error.localizedDescription()
    return str(description) if description else str(error)


def _confirmation_timeout(was_running: bool) -> float:
    return _FRONTMOST_TIMEOUT_WARM_S if was_running else _FRONTMOST_TIMEOUT_COLD_S


def _confirm_frontmost(
    workspace: Any, app: Any, name: str, timeout_s: float, clock: _Clock
) -> tuple[int, float]:
    pid = int(app.processIdentifier())
    started = clock.monotonic()
    deadline = started + timeout_s
    attempts = 1
    next_request_at = started + _ACTIVATION_RETRY_INTERVAL_S

    while True:
        front = frontmost_app_owner()
        rivals: list[str] = []
        if front.pid is not None:
            if _is_same_application(app, pid, front.pid):
                return attempts, clock.monotonic() - started
        elif same_app(front.name, name):
            rivals = _same_named_rivals(workspace, front.name, pid)
            if not rivals:
                return attempts, clock.monotonic() - started

        active, refreshed = _is_active(app)
        if active:
            return attempts, clock.monotonic() - started

        now = clock.monotonic()
        if now >= deadline:
            raise AppActivationError(
                _confirmation_failure(
                    name, pid, front, rivals, active, refreshed, timeout_s, attempts, now - started
                )
            )

        if now >= next_request_at and attempts < _MAX_ACTIVATION_ATTEMPTS:
            _request_activation_again(workspace, app, name, attempts, deadline, clock)
            attempts += 1
            next_request_at = clock.monotonic() + _ACTIVATION_RETRY_INTERVAL_S

        clock.sleep(_FRONTMOST_POLL_INTERVAL_S)


def _request_activation_again(
    workspace: Any, app: Any, name: str, attempts_so_far: int, deadline: float, clock: _Clock
) -> None:
    retry_number = attempts_so_far
    url = app.bundleURL()
    reopen = retry_number % 2 == 0 and url is not None

    if reopen:
        error = _open_application(workspace, url, timeout_s=deadline - clock.monotonic())[1]
        outcome = error or "ok"
    else:
        outcome = "ok" if bool(app.activateWithOptions_(_ACTIVATE_OPTIONS)) else "refused"

    _log.warning(
        "app.activation_retried",
        app_name=name,
        attempt=retry_number + 1,
        call="reopen" if reopen else "activate",
        outcome=outcome,
    )


def _same_named_rivals(workspace: Any, front: str, pid: int) -> list[str]:
    return sorted(
        name
        for name, other in _activatable_apps(workspace)
        if int(other.processIdentifier()) != pid and same_app(name, front)
    )


def _is_same_application(app: Any, wanted_pid: int, front_pid: int) -> bool:
    if front_pid == wanted_pid:
        return True
    return _shares_bundle(app, front_pid)


def _shares_bundle(app: Any, front_pid: int) -> bool:
    ours = _bundle_path(app.bundleURL()).rstrip("/")
    theirs = _bundle_path_of_pid(front_pid).rstrip("/")
    if not ours or not theirs:
        return False
    return ours == theirs or theirs.startswith(f"{ours}/") or ours.startswith(f"{theirs}/")


def _is_active(app: Any) -> tuple[bool, bool]:
    refreshed = _refresh_workspace_state()
    return bool(app.isActive()), refreshed


def _owner_clause(pid: int | None) -> str:
    if pid is None:
        return ""
    bundle = _bundle_path_of_pid(pid)
    located = f' out of "{bundle}"' if bundle else " (no bundle path)"
    return f", owned by pid {pid}{located}, which is not this application"


def _confirmation_failure(
    name: str,
    pid: int,
    front: FrontmostApp,
    rivals: list[str],
    active: bool,
    refreshed: bool,
    timeout_s: float,
    attempts: int,
    waited_s: float,
) -> str:
    who = front.name or "none"
    where = f'the window server reports "{who}" in front{_owner_clause(front.pid)}'
    message = (
        f'app "{name}" (pid {pid}) did not become frontmost within {timeout_s:.1f}s: '
        f"{where} and the app itself reports isActive={str(active).lower()}"
    )
    if not refreshed:
        message += (
            " (this thread cannot refresh NSWorkspace, so that is the value cached when the "
            "workspace was first read, not a live answer)"
        )
    if rivals:
        others = ", ".join(f'"{rival}"' for rival in rivals)
        message += (
            f'; the front window\'s name matches "{name}" but also {len(rivals)} other '
            f"running app(s) ({others}), so it does not identify ours"
        )
    plural = "" if attempts == 1 else "s"
    message += f"; {attempts} activation request{plural} issued over {waited_s:.2f}s"
    return message


def _bundle_path_of_pid(pid: int) -> str:
    app = _running_application(pid)
    return _bundle_path(app.bundleURL()) if app is not None else ""


class MacAppLauncher:
    def activate(self, app_name: str) -> str:
        return activate_app(app_name)

    def resolve(self, app_name: str) -> AppIdentity | None:
        name = resolve_running_app_name(app_name)
        if name is None:
            return None
        return AppIdentity(display_name=name)

    def same_app(self, observed: str, wanted: str) -> bool:
        return same_app(observed, wanted)

    def app_version(self, app_name: str) -> str:
        return app_version(app_name)

    def close(self) -> None: ...
