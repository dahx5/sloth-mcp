from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

from choto.log import get_logger, setup_logging
from choto.mcpserver.socket_path import default_socket_path, resolve_socket_path

_log = get_logger(__name__)

_CONNECT_TIMEOUT_SECONDS = 2.0

_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0)

_BUFFER_BYTES = 65536

_SHUTDOWN_JOIN_SECONDS = 0.2

_STDIN_FD = 0
_STDOUT_FD = 1

_EXIT_OK = 0
_EXIT_FAILURE = 1


class BridgeConnectError(RuntimeError): ...


def _connect(path: Path) -> socket.socket:
    attempts = len(_RETRY_DELAYS_SECONDS) + 1
    budget = sum(_RETRY_DELAYS_SECONDS)
    last_error: OSError | None = None

    for attempt in range(attempts):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
            sock.connect(str(path))
        except OSError as exc:
            sock.close()
            last_error = exc
            if attempt < len(_RETRY_DELAYS_SECONDS):
                delay = _RETRY_DELAYS_SECONDS[attempt]
                _log.warning(
                    "bridge.connect.retry",
                    socket=str(path),
                    attempt=attempt + 1,
                    of=attempts,
                    error=str(exc),
                    retry_in_s=delay,
                )
                time.sleep(delay)
            continue
        sock.settimeout(None)
        _log.info("bridge.connected", socket=str(path), attempt=attempt + 1)
        return sock

    raise BridgeConnectError(
        f"could not reach the Choto daemon at {path} after {attempts} attempts "
        f"over ~{budget:.1f}s (last error: {last_error}). Start the daemon from a "
        f"terminal that has Accessibility permission: choto --socket {path}"
    )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


class _Traffic:
    __slots__ = ("received", "sent")

    def __init__(self) -> None:
        self.sent = 0
        self.received = 0


def _pump_stdin_to_socket(sock: socket.socket, traffic: _Traffic) -> None:
    try:
        while True:
            chunk = os.read(_STDIN_FD, _BUFFER_BYTES)
            if not chunk:
                _log.info("bridge.stdin.eof")
                break
            traffic.sent += len(chunk)
            sock.sendall(chunk)
    except OSError as exc:
        _log.warning("bridge.stdin.pump_stopped", error=str(exc))
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            _log.debug("bridge.stdin.shutdown_skipped")


def _pump_socket_to_stdout(sock: socket.socket, traffic: _Traffic) -> None:
    while True:
        chunk = sock.recv(_BUFFER_BYTES)
        if not chunk:
            _log.info("bridge.socket.eof", received_bytes=traffic.received)
            return
        _write_all(_STDOUT_FD, chunk)
        traffic.received += len(chunk)


def _explain_silence(traffic: _Traffic, path: Path) -> bool:
    if traffic.received or not traffic.sent:
        return False
    _log.error(
        "bridge.session.refused",
        socket=str(path),
        sent_bytes=traffic.sent,
        error=(
            f"the Choto daemon at {path} accepted the connection, took {traffic.sent} byte(s) "
            "of requests and closed it without answering any of them — not even to refuse it. "
            "A daemon at its session limit answers with a JSON-RPC error saying so, so this "
            "is something else: a daemon that died mid-handshake, or one older than that "
            "refusal, which closed a client over its limit in silence. Check the daemon's own "
            "log for session.rejected, and `choto service status` for whether it is still up."
        ),
    )
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="choto-bridge",
        description=(
            "Bridge Claude Desktop's stdio MCP transport to a running Choto "
            "daemon listening on a unix socket."
        ),
    )
    parser.add_argument(
        "--socket",
        dest="socket_path",
        metavar="PATH",
        default=None,
        help=f"Path of the daemon's unix socket (default: {default_socket_path()}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _parse_args(argv)

    try:
        path = resolve_socket_path(args.socket_path)
    except ValueError as exc:
        _log.error("bridge.socket_path.invalid", error=str(exc))
        return _EXIT_FAILURE

    try:
        sock = _connect(path)
    except BridgeConnectError as exc:
        _log.error("bridge.connect.failed", socket=str(path), error=str(exc))
        return _EXIT_FAILURE

    traffic = _Traffic()
    uplink = threading.Thread(
        target=_pump_stdin_to_socket,
        args=(sock, traffic),
        name="choto-bridge-uplink",
        daemon=True,
    )
    uplink.start()

    exit_code = _EXIT_OK
    try:
        _pump_socket_to_stdout(sock, traffic)
    except OSError as exc:
        _log.error("bridge.socket.pump_failed", socket=str(path), error=str(exc))
        exit_code = _EXIT_FAILURE
    else:
        if _explain_silence(traffic, path):
            exit_code = _EXIT_FAILURE
    finally:
        try:
            sock.close()
        except OSError as exc:
            _log.warning("bridge.socket.close_failed", error=str(exc))
        uplink.join(timeout=_SHUTDOWN_JOIN_SECONDS)

    _log.info("bridge.stopped", exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
