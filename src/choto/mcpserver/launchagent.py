from __future__ import annotations

import os
import plistlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from choto import userpaths
from choto.log import get_logger
from choto.mcpserver import appbundle
from choto.mcpserver.appbundle import BundleBuild, InputPermission
from choto.mcpserver.daemonprobe import DaemonHealth, probe_daemon
from choto.mcpserver.service import PROJECT_ROOT, ServiceError
from choto.mcpserver.socket_path import (
    LAUNCHD_SOCKET_NAME,
    LAUNCHD_SOCKET_OPTION,
    SOCKET_MODE,
)

__all__ = [
    "LABEL",
    "DaemonHealth",
    "InstallResult",
    "LaunchctlError",
    "LaunchdInfo",
    "ServiceError",
    "ServiceStatus",
    "build_plist",
    "dry_run_fields",
    "install",
    "install_guidance",
    "log_dir",
    "plist_path",
    "probe_daemon",
    "read_launchd_info",
    "render_definition",
    "render_plist",
    "restart",
    "starts_on_demand",
    "status",
    "status_fields",
    "supervisor_description",
    "uninstall",
]

_log = get_logger(__name__)


LABEL = "com.choto.daemon"
PLIST_FILENAME = f"{LABEL}.plist"

SUPERVISOR_NAME = "launchd"

LAUNCH_AGENTS_DIR = Path("~/Library/LaunchAgents")


LAUNCHCTL = "/bin/launchctl"

LAUNCHCTL_TIMEOUT_SECONDS = 30.0

LAUNCHCTL_NO_SUCH_SERVICE = 113
LAUNCHCTL_NO_SUCH_PROCESS = 3


def _report_field(key: str, value: str = r".+?") -> re.Pattern[str]:
    return re.compile(rf"^\s*{key} = ({value})\s*$", re.MULTILINE)


_PID_RE = _report_field("pid", r"\d+")
_STATE_RE = _report_field("state")
_LAST_EXIT_CODE_RE = _report_field("last exit code", r"-?\d+")
_LAST_EXIT_REASON_RE = _report_field("last exit reason")


class LaunchctlError(ServiceError):
    def __init__(self, argv: list[str], returncode: int | None, stderr: str) -> None:
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr.strip()
        rendered = " ".join(argv)
        status = "timed out" if returncode is None else f"exited with {returncode}"
        detail = f": {self.stderr}" if self.stderr else ""
        super().__init__(f"`{rendered}` {status}{detail}")


@dataclass(frozen=True)
class LaunchdInfo:
    loaded: bool
    pid: int | None = None
    state: str | None = None
    last_exit_code: int | None = None
    last_exit_reason: str | None = None


@dataclass(frozen=True)
class ServiceStatus:
    plist_path: Path
    plist_installed: bool
    socket_path: Path
    launchd: LaunchdInfo
    health: DaemonHealth
    bundle_path: Path
    bundle_signed: bool
    input_permission: InputPermission
    health_detail: str | None = None
    bundle_detail: str | None = None
    input_permission_detail: str | None = None


@dataclass(frozen=True)
class InstallResult:
    plist_path: Path
    bundle: BundleBuild


def uid() -> int:
    return os.getuid()


def domain_target() -> str:
    return f"gui/{uid()}"


def service_target() -> str:
    return f"{domain_target()}/{LABEL}"


def plist_path() -> Path:
    return (LAUNCH_AGENTS_DIR / PLIST_FILENAME).expanduser()


def log_dir() -> Path:
    return userpaths.log_dir()


def build_plist(
    socket_path: Path,
    *,
    bundle: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    app = appbundle.bundle_path() if bundle is None else bundle
    logs = log_dir()

    for name, path in (
        ("bundle path", app),
        ("project root", project_root),
        ("socket path", socket_path),
    ):
        if not path.is_absolute():
            raise ServiceError(f"{name} must be absolute, got {path}")

    return {
        "Label": LABEL,
        "ProgramArguments": [
            *appbundle.daemon_argv(app, socket_path),
            LAUNCHD_SOCKET_OPTION,
            LAUNCHD_SOCKET_NAME,
        ],
        "Sockets": {
            LAUNCHD_SOCKET_NAME: {
                "SockPathName": str(socket_path),
                "SockPathMode": SOCKET_MODE,
                "SockFamily": "Unix",
                "SockType": "stream",
            }
        },
        "ProcessType": "Interactive",
        "WorkingDirectory": str(project_root),
        "StandardOutPath": str(logs / userpaths.DAEMON_LOG_FILENAME),
        "StandardErrorPath": str(logs / userpaths.DAEMON_ERROR_LOG_FILENAME),
    }


def render_plist(
    socket_path: Path,
    *,
    bundle: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> bytes:
    return plistlib.dumps(
        build_plist(socket_path, bundle=bundle, project_root=project_root),
        fmt=plistlib.FMT_XML,
    )


def render_definition(socket_path: Path) -> str:
    return render_plist(socket_path).decode("utf-8")


def dry_run_fields(socket_path: Path) -> dict[str, object]:
    return {
        "plist": str(plist_path()),
        "socket": str(socket_path),
        "bundle": str(appbundle.bundle_path()),
    }


def _ask_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    argv = [LAUNCHCTL, *args]
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=LAUNCHCTL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LaunchctlError(argv, None, f"no answer within {LAUNCHCTL_TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise LaunchctlError(argv, None, str(exc)) from exc


def _run_launchctl(
    args: list[str], *, allowed_returncodes: tuple[int, ...] = ()
) -> subprocess.CompletedProcess[str]:
    completed = _ask_launchctl(args)
    if completed.returncode != 0 and completed.returncode not in allowed_returncodes:
        raise LaunchctlError([LAUNCHCTL, *args], completed.returncode, completed.stderr)
    return completed


def _field(report: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(report)
    return match.group(1) if match else None


def _int_field(report: str, pattern: re.Pattern[str]) -> int | None:
    value = _field(report, pattern)
    return None if value is None else int(value)


def _parse_launchd_report(report: str) -> LaunchdInfo:
    return LaunchdInfo(
        loaded=True,
        pid=_int_field(report, _PID_RE),
        state=_field(report, _STATE_RE),
        last_exit_code=_int_field(report, _LAST_EXIT_CODE_RE),
        last_exit_reason=_field(report, _LAST_EXIT_REASON_RE),
    )


def read_launchd_info() -> LaunchdInfo:
    completed = _run_launchctl(
        ["print", service_target()],
        allowed_returncodes=(LAUNCHCTL_NO_SUCH_SERVICE,),
    )
    if completed.returncode == LAUNCHCTL_NO_SUCH_SERVICE:
        return LaunchdInfo(loaded=False)
    return _parse_launchd_report(completed.stdout)


def _bootout() -> bool:
    completed = _run_launchctl(
        ["bootout", service_target()],
        allowed_returncodes=(LAUNCHCTL_NO_SUCH_SERVICE, LAUNCHCTL_NO_SUCH_PROCESS),
    )
    if completed.returncode == 0:
        _log.info("service.bootout", target=service_target())
        return True
    _log.info("service.bootout.absent", target=service_target())
    return False


def _bootstrap(path: Path) -> None:
    _run_launchctl(["bootstrap", domain_target(), str(path)])
    _log.info("service.bootstrap", domain=domain_target(), plist=str(path))


def _write_plist(target: Path, document: bytes) -> None:
    try:
        log_dir().mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(document)
    except OSError as exc:
        raise ServiceError(f"cannot write the launch agent at {target}: {exc}") from exc


def install(
    socket_path: Path,
    *,
    bundle_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> InstallResult:
    info = read_launchd_info()
    health, detail = probe_daemon(socket_path)

    if health in (DaemonHealth.RESPONSIVE, DaemonHealth.BUSY) and not info.loaded:
        raise ServiceError(
            f"a Choto daemon that launchd does not manage is already listening on "
            f"{socket_path} (probe: {health.value}). Installing the agent now would "
            f"leave two daemons fighting for the socket. Stop that daemon first "
            f"(it was started by hand, e.g. `choto --socket`), then rerun "
            f"`choto service install`."
        )

    build = appbundle.build(bundle_dir=bundle_dir)
    document = render_plist(socket_path, bundle=build.path, project_root=project_root)

    target = plist_path()
    _write_plist(target, document)
    _log.info("service.plist_written", plist=str(target), socket=str(socket_path))

    if info.loaded:
        _bootout()
    _bootstrap(target)
    _log.info(
        "service.installed",
        label=LABEL,
        plist=str(target),
        socket=str(socket_path),
        bundle=str(build.path),
        bundle_rebuilt=build.changed,
        previous_health=health.value,
        previous_health_detail=detail,
    )
    return InstallResult(plist_path=target, bundle=build)


def uninstall() -> bool:
    unloaded = _bootout()

    target = plist_path()
    if not target.exists():
        _log.info("service.uninstall.absent", plist=str(target), unloaded=unloaded)
        return unloaded

    try:
        target.unlink()
    except OSError as exc:
        raise ServiceError(f"cannot remove the launch agent at {target}: {exc}") from exc
    _log.info("service.uninstalled", label=LABEL, plist=str(target))
    return True


def restart() -> None:
    completed = _run_launchctl(
        ["kickstart", "-k", service_target()],
        allowed_returncodes=(LAUNCHCTL_NO_SUCH_SERVICE,),
    )
    if completed.returncode == LAUNCHCTL_NO_SUCH_SERVICE:
        raise ServiceError(
            f"{LABEL} is not loaded in {domain_target()}; run `choto service install` first"
        )
    _log.info("service.restarted", target=service_target())


def status(socket_path: Path, *, bundle_dir: Path | None = None) -> ServiceStatus:
    target = plist_path()
    info = read_launchd_info()
    health, detail = probe_daemon(socket_path)
    app = appbundle.bundle_path(bundle_dir)
    signed, signature_detail = appbundle.signature_status(app)
    permission, permission_detail = appbundle.check_input_permission(app)
    return ServiceStatus(
        plist_path=target,
        plist_installed=target.is_file(),
        socket_path=socket_path,
        launchd=info,
        health=health,
        health_detail=detail,
        bundle_path=app,
        bundle_signed=signed,
        bundle_detail=signature_detail,
        input_permission=permission,
        input_permission_detail=permission_detail,
    )


def status_fields(report: ServiceStatus) -> dict[str, object]:
    return {
        "label": LABEL,
        "plist": str(report.plist_path),
        "plist_installed": report.plist_installed,
        "loaded": report.launchd.loaded,
        "pid": report.launchd.pid,
        "launchd_state": report.launchd.state,
        "last_exit_code": report.launchd.last_exit_code,
        "last_exit_reason": report.launchd.last_exit_reason,
        "socket": str(report.socket_path),
        "daemon_health": report.health.value,
        "daemon_detail": report.health_detail,
        "bundle": str(report.bundle_path),
        "bundle_signed": report.bundle_signed,
        "bundle_detail": report.bundle_detail,
        "input_permission": report.input_permission.value,
        "input_permission_detail": report.input_permission_detail,
    }


def install_guidance(result: InstallResult) -> str:
    identity = (
        "The bundle was rebuilt, so its code identity changed: macOS will ask "
        "again even if you approved an earlier build."
        if result.bundle.changed
        else "The bundle was already up to date, so an existing approval still stands."
    )
    return (
        f"\nInstalled {LABEL}.\n"
        f"  app bundle: {result.bundle.path}\n"
        f"  launch agent: {result.plist_path}\n\n"
        f"{identity}\n\n"
        "The first time the daemon acts on the screen macOS shows\n"
        '  "Choto wants to control your computer" -> click Allow.\n\n'
        "If that prompt was dismissed before, macOS remembers the refusal and will\n"
        "not ask again. Clear it yourself with:\n"
        f"  tccutil reset Accessibility {appbundle.BUNDLE_IDENTIFIER}\n"
        "  choto service restart\n\n"
        "Verify at any time with `choto service status` (input_permission=granted).\n"
    )


def _installed_plist() -> dict[str, object] | None:
    target = plist_path()
    try:
        return plistlib.loads(target.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        _log.warning("service.plist_unreadable", plist=str(target), error=str(exc))
        return None


def _declared_sockets(document: dict[str, object]) -> list[str]:
    sockets = document.get("Sockets")
    if not isinstance(sockets, dict):
        return []
    return [
        str(entry.get("SockPathName", "")) for entry in sockets.values() if isinstance(entry, dict)
    ]


def starts_on_demand(socket_path: Path) -> bool:
    if not read_launchd_info().loaded:
        return False
    document = _installed_plist()
    if document is None:
        return False
    wanted = os.path.realpath(socket_path)
    return any(os.path.realpath(declared) == wanted for declared in _declared_sockets(document))


def supervisor_description() -> str | None:
    info = read_launchd_info()
    if info.loaded and info.pid is not None:
        return f"{SUPERVISOR_NAME} is running it as pid {info.pid}"
    return None
