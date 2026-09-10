from __future__ import annotations

import json
import socket
from enum import Enum
from pathlib import Path

PROBE_PROTOCOL_VERSION = "2024-11-05"
PROBE_CLIENT_NAME = "choto-service-status"
PROBE_CLIENT_VERSION = "1"

PROBE_CONNECT_TIMEOUT_SECONDS = 2.0
PROBE_REPLY_TIMEOUT_SECONDS = 5.0
PROBE_BUFFER_BYTES = 65536
PROBE_MAX_REPLY_BYTES = 1024 * 1024
_PROBE_REQUEST_ID = 1

SESSION_CAP_ERROR_CODE = -32000

_HUNG_UP_DETAIL = "the daemon closed the connection without replying"


class DaemonHealth(str, Enum):
    RESPONSIVE = "responsive"
    BUSY = "busy"
    NOT_LISTENING = "not_listening"
    ABSENT = "absent"
    UNREACHABLE = "unreachable"


_INITIALIZE_REQUEST = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": _PROBE_REQUEST_ID,
            "method": "initialize",
            "params": {
                "protocolVersion": PROBE_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": PROBE_CLIENT_NAME,
                    "version": PROBE_CLIENT_VERSION,
                },
            },
        }
    ).encode("utf-8")
    + b"\n"
)


class _NoAnswer(Exception):
    def __init__(self, health: DaemonHealth, detail: str) -> None:
        super().__init__(detail)
        self.health = health
        self.detail = detail


def probe_daemon(socket_path: Path) -> tuple[DaemonHealth, str | None]:
    if not socket_path.exists():
        return DaemonHealth.ABSENT, f"no socket file at {socket_path}"
    try:
        reply = _initialize(socket_path)
    except _NoAnswer as silence:
        return silence.health, silence.detail
    return _read_health(reply)


def _initialize(socket_path: Path) -> str:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(PROBE_CONNECT_TIMEOUT_SECONDS)
        try:
            probe.connect(str(socket_path))
        except (ConnectionRefusedError, FileNotFoundError) as exc:
            raise _NoAnswer(
                DaemonHealth.NOT_LISTENING, f"nothing is bound to {socket_path}: {exc}"
            ) from exc
        except OSError as exc:
            raise _NoAnswer(
                DaemonHealth.UNREACHABLE, f"cannot connect to {socket_path}: {exc}"
            ) from exc

        probe.settimeout(PROBE_REPLY_TIMEOUT_SECONDS)
        try:
            probe.sendall(_INITIALIZE_REQUEST)
            line = _receive_line(probe)
        except TimeoutError as exc:
            raise _NoAnswer(
                DaemonHealth.BUSY, "connected, but no MCP reply arrived before the timeout"
            ) from exc
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise _NoAnswer(DaemonHealth.BUSY, _HUNG_UP_DETAIL) from exc
        except OSError as exc:
            raise _NoAnswer(DaemonHealth.UNREACHABLE, f"MCP handshake failed: {exc}") from exc
    finally:
        probe.close()

    if line is None:
        raise _NoAnswer(DaemonHealth.BUSY, _HUNG_UP_DETAIL)
    return line


def _read_health(line: str) -> tuple[DaemonHealth, str | None]:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        return DaemonHealth.UNREACHABLE, f"reply is not JSON: {exc}"

    if not isinstance(message, dict):
        return DaemonHealth.UNREACHABLE, f"unexpected MCP reply: {line[:200]!r}"
    if "result" in message:
        return DaemonHealth.RESPONSIVE, None

    error = message.get("error")
    if (
        isinstance(error, dict)
        and error.get("code") == SESSION_CAP_ERROR_CODE
        and isinstance(error.get("message"), str)
    ):
        return DaemonHealth.BUSY, error["message"]
    return DaemonHealth.UNREACHABLE, f"unexpected MCP reply: {line[:200]!r}"


def _receive_line(sock: socket.socket) -> str | None:
    buffer = bytearray()
    while b"\n" not in buffer:
        chunk = sock.recv(PROBE_BUFFER_BYTES)
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > PROBE_MAX_REPLY_BYTES:
            raise OSError(f"MCP reply exceeded {PROBE_MAX_REPLY_BYTES} bytes without a newline")
    return buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace")
