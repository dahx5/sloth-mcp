from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from choto import userpaths

__all__ = [
    "PROJECT_ROOT",
    "SERVICE_MANAGERS",
    "ServiceError",
    "ServiceManager",
    "resolve_service_manager",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ServiceError(RuntimeError): ...


class ServiceManager(Protocol):
    SUPERVISOR_NAME: str

    def render_definition(self, socket_path: Path) -> str: ...

    def dry_run_fields(self, socket_path: Path) -> dict[str, object]: ...

    def install(self, socket_path: Path) -> Any: ...

    def install_guidance(self, result: Any) -> str: ...

    def uninstall(self) -> bool: ...

    def status(self, socket_path: Path) -> Any: ...

    def status_fields(self, report: Any) -> dict[str, object]: ...

    def restart(self) -> None: ...

    def supervisor_description(self) -> str | None: ...

    def starts_on_demand(self, socket_path: Path) -> bool: ...

    def probe_daemon(self, socket_path: Path) -> tuple[Any, str | None]: ...


def _launchd_manager() -> ServiceManager:
    from choto.mcpserver import launchagent

    return cast(ServiceManager, launchagent)


SERVICE_MANAGERS = {
    userpaths.OS_MACOS: _launchd_manager,
}


def resolve_service_manager() -> ServiceManager:
    try:
        os_name = userpaths.current_os()
    except userpaths.UnsupportedPlatformError as exc:
        raise ServiceError(str(exc)) from exc

    builder = SERVICE_MANAGERS.get(os_name)
    if builder is None:
        raise ServiceError(
            f"Choto has no service manager for {os_name!r}; "
            f"supported systems are {sorted(SERVICE_MANAGERS)}."
        )
    return builder()
