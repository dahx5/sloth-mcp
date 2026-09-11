from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "APP_DIR_NAME",
    "DAEMON_ERROR_LOG_FILENAME",
    "DAEMON_LOG_FILENAME",
    "DB_FILENAME",
    "MODEL_DIR_NAME",
    "OS_LINUX",
    "OS_MACOS",
    "UnsupportedPlatformError",
    "current_os",
    "default_db_path",
    "log_dir",
    "model_dir",
    "socket_dir",
]

APP_DIR_NAME = "choto"

DB_FILENAME = "choto.db"

MODEL_DIR_NAME = "models"

DAEMON_LOG_FILENAME = "daemon.log"
DAEMON_ERROR_LOG_FILENAME = "daemon.err.log"

OS_MACOS = "macos"
OS_LINUX = "linux"

_SYS_PLATFORM_PREFIXES = {
    "darwin": OS_MACOS,
    "linux": OS_LINUX,
}


MACOS_APPLICATION_SUPPORT = Path("~/Library/Application Support")
MACOS_LOGS = Path("~/Library/Logs")

_DATA_ROOTS = {OS_MACOS: MACOS_APPLICATION_SUPPORT, OS_LINUX: Path("~/.local/share")}
_LOG_ROOTS = {OS_MACOS: MACOS_LOGS, OS_LINUX: Path("~/.local/state")}

PROJECT_DB_PATH = Path("data") / DB_FILENAME


class UnsupportedPlatformError(RuntimeError): ...


def current_os() -> str:
    for prefix, os_name in _SYS_PLATFORM_PREFIXES.items():
        if sys.platform.startswith(prefix):
            return os_name
    raise UnsupportedPlatformError(
        f"Choto has no per-user directory layout for sys.platform {sys.platform!r}; "
        f"supported systems are {sorted(_SYS_PLATFORM_PREFIXES.values())}."
    )


def socket_dir() -> Path:
    return _app_dir_under(_DATA_ROOTS)


def log_dir() -> Path:
    return _app_dir_under(_LOG_ROOTS)


def model_dir() -> Path:
    return _app_dir_under(_DATA_ROOTS) / MODEL_DIR_NAME


def default_db_path() -> Path:
    current_os()
    return PROJECT_DB_PATH


def _app_dir_under(roots: dict[str, Path]) -> Path:
    return (roots[current_os()] / APP_DIR_NAME).expanduser()
