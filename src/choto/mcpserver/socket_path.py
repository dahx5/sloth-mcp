from __future__ import annotations

import os
from pathlib import Path

from choto import userpaths

SOCKET_FILENAME = "choto.sock"

SOCKET_OPTION = "--socket"

LAUNCHD_SOCKET_OPTION = "--launchd-socket"
LAUNCHD_SOCKET_NAME = "Listener"

SOCKET_DIR_MODE = 0o700

SOCKET_MODE = 0o600

MACOS_MAX_SOCKET_PATH_BYTES = 103


def max_socket_path_bytes() -> int:
    return MACOS_MAX_SOCKET_PATH_BYTES


def default_socket_path() -> str:
    return str(userpaths.socket_dir() / SOCKET_FILENAME)


def resolve_socket_path(raw: str | os.PathLike[str] | None) -> Path:
    if raw is not None and not str(raw).strip():
        raise ValueError("socket path must not be empty")

    path = Path(raw if raw is not None else default_socket_path()).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    limit = max_socket_path_bytes()
    encoded = len(os.fsencode(str(path)))
    if encoded > limit:
        raise ValueError(
            f"socket path is {encoded} bytes, exceeding this platform's AF_UNIX limit "
            f"of {limit}: {path}"
        )
    return path


def ensure_socket_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=SOCKET_DIR_MODE)
