from __future__ import annotations

import json
import os
import signal
import socket
import stat
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import anyio.lowlevel
import mcp.types as types
from anyio._core._eventloop import get_async_backend
from anyio.abc import SocketListener, SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.fastmcp import FastMCP
from mcp.shared.message import SessionMessage

from choto.log import get_logger
from choto.mcpserver.daemonprobe import SESSION_CAP_ERROR_CODE
from choto.mcpserver.socket_path import SOCKET_MODE, ensure_socket_dir

_log = get_logger(__name__)

MAX_LINE_BYTES = 8 * 1024 * 1024

LIVENESS_PROBE_TIMEOUT_SECONDS = 1.0

DEFAULT_MAX_SESSIONS = 8

SESSION_HANDOVER_GRACE_SECONDS = 0.5

REFUSAL_REQUEST_TIMEOUT_SECONDS = 2.0

_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class SocketUnavailableError(RuntimeError): ...


class LineTooLongError(ValueError):
    def __init__(self) -> None:
        super().__init__(
            f"JSON-RPC message exceeded {MAX_LINE_BYTES} bytes without a newline terminator"
        )


class JsonLines:
    __slots__ = ("_reader", "_stream")

    def __init__(self, stream: SocketStream) -> None:
        self._stream = stream
        self._reader = BufferedByteReceiveStream(stream)

    async def receive(self) -> str | None:
        try:
            line = await self._reader.receive_until(b"\n", MAX_LINE_BYTES)
        except anyio.DelimiterNotFound as exc:
            raise LineTooLongError from exc
        except (anyio.IncompleteRead, anyio.BrokenResourceError, anyio.ClosedResourceError):
            return None
        return line.decode("utf-8", errors="replace")

    async def send(self, text: str) -> None:
        await self._stream.send(text.encode("utf-8") + b"\n")


@asynccontextmanager
async def socket_streams(
    stream: SocketStream,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    lines = JsonLines(stream)

    async def socket_reader() -> None:
        try:
            async with read_stream_writer:
                while True:
                    try:
                        line = await lines.receive()
                    except LineTooLongError as exc:
                        await read_stream_writer.send(exc)
                        break
                    if line is None:
                        break

                    text = line.strip()
                    if not text:
                        continue

                    try:
                        message = types.JSONRPCMessage.model_validate_json(text)
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue

                    await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def socket_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    await lines.send(
                        session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    )
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(socket_reader)
        tg.start_soon(socket_writer)
        try:
            yield read_stream, write_stream
        finally:
            tg.cancel_scope.cancel()


def _probe_is_alive(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(LIVENESS_PROBE_TIMEOUT_SECONDS)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except TimeoutError:
        return True
    except OSError as exc:
        raise SocketUnavailableError(f"cannot probe the existing socket at {path}: {exc}") from exc
    else:
        return True
    finally:
        probe.close()


def prepare_socket_path(path: Path) -> None:
    ensure_socket_dir(path)

    if not path.exists() and not path.is_symlink():
        return

    if not path.is_socket():
        raise SocketUnavailableError(
            f"{path} exists and is not a unix socket; refusing to replace it"
        )

    if _probe_is_alive(path):
        raise SocketUnavailableError(f"another Choto daemon is already listening on {path}")

    path.unlink()
    _log.warning("socket.stale_removed", socket=str(path))


async def _serve_connection(mcp: FastMCP, stream: SocketStream) -> None:
    lowlevel = mcp._mcp_server
    async with socket_streams(stream) as (read_stream, write_stream):
        await lowlevel.run(read_stream, write_stream, lowlevel.create_initialization_options())


async def _watch_signals(signals: AsyncIterator[signal.Signals], scope: anyio.CancelScope) -> None:
    async for signum in signals:
        _log.info("daemon.signal", signal=signal.Signals(signum).name)
        scope.cancel()
        return


class _SessionSlots:
    __slots__ = ("_capacity", "_freed", "_idle_since", "_live")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"a daemon must be allowed at least one session, not {capacity}")
        self._capacity = capacity
        self._live = 0
        self._freed = anyio.Event()
        self._idle_since: float | None = time.monotonic()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def live(self) -> int:
        return self._live

    async def claim(self, grace: float) -> bool:
        if self._live < self._capacity:
            self._live += 1
            self._idle_since = None
            return True
        with anyio.move_on_after(grace):
            await self._freed.wait()
        if self._live >= self._capacity:
            return False
        self._live += 1
        self._idle_since = None
        return True

    def release(self) -> None:
        self._live -= 1
        if self._live == 0:
            self._idle_since = time.monotonic()
        freed, self._freed = self._freed, anyio.Event()
        freed.set()

    def seconds_until_idle(self, timeout: float) -> float | None:
        if self._idle_since is None:
            return None
        return timeout - (time.monotonic() - self._idle_since)


async def _first_request_id(lines: JsonLines) -> int | str | None:
    with anyio.move_on_after(REFUSAL_REQUEST_TIMEOUT_SECONDS):
        try:
            line = await lines.receive()
        except LineTooLongError:
            return None
        if line is None:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(message, dict):
            identifier = message.get("id")
            if isinstance(identifier, int | str) and not isinstance(identifier, bool):
                return identifier
    return None


def _session_cap_message(capacity: int) -> str:
    return (
        f"the Choto daemon is already serving {capacity} MCP session(s), which is all it "
        "allows (CHOTO_MAX_MCP_SESSIONS). Every connected client holds one for as long as "
        "it stays connected — each Claude Desktop surface spawns its own choto-bridge — so "
        "the usual cause is a bridge still running for a client that has gone away. Quit "
        "the clients you are not using (or kill the stray choto-bridge processes); raising "
        "the limit needs the daemon restarted."
    )


async def _refuse_session(stream: SocketStream, path: Path, capacity: int, session: int) -> None:
    _log.warning(
        "session.rejected",
        socket=str(path),
        session=session,
        capacity=capacity,
        reason="session limit reached",
    )
    try:
        async with stream:
            lines = JsonLines(stream)
            identifier = await _first_request_id(lines)
            if identifier is None:
                _log.warning(
                    "session.rejection_unaddressed",
                    socket=str(path),
                    session=session,
                    capacity=capacity,
                )
            payload = {
                "jsonrpc": "2.0",
                "id": identifier,
                "error": {
                    "code": SESSION_CAP_ERROR_CODE,
                    "message": _session_cap_message(capacity),
                },
            }
            await lines.send(json.dumps(payload))
    except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError) as exc:
        _log.warning(
            "session.rejection_undelivered", socket=str(path), session=session, error=str(exc)
        )


async def _accept_sessions(
    mcp: FastMCP,
    listener: SocketListener,
    path: Path,
    slots: _SessionSlots,
    idle_exit_seconds: float | None,
) -> None:
    accept_scope: anyio.CancelScope | None = None

    def wake_accept() -> None:
        if accept_scope is not None:
            accept_scope.cancel()

    async def serve(stream: SocketStream, session: int) -> None:
        try:
            async with stream:
                await _serve_connection(mcp, stream)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError) as exc:
            _log.warning("session.disconnected", socket=str(path), session=session, error=str(exc))
        except Exception:
            _log.exception("session.failed", socket=str(path), session=session)
        finally:
            slots.release()
            _log.info("session.closed", socket=str(path), session=session, live=slots.live)
            if idle_exit_seconds is not None and slots.live == 0:
                wake_accept()

    async with anyio.create_task_group() as tg:
        session = 0
        while True:
            remaining = (
                None if idle_exit_seconds is None else slots.seconds_until_idle(idle_exit_seconds)
            )
            if remaining is not None and remaining <= 0:
                _log.info(
                    "daemon.idle_exit",
                    socket=str(path),
                    idle_exit_seconds=idle_exit_seconds,
                    sessions_served=session,
                )
                return

            stream: SocketStream | None = None
            with anyio.move_on_after(remaining) as accept_scope:
                stream = await listener.accept()
            accept_scope = None
            if stream is None:
                continue
            session += 1

            if not await slots.claim(SESSION_HANDOVER_GRACE_SECONDS):
                tg.start_soon(_refuse_session, stream, path, slots.capacity, session)
                continue

            _log.info(
                "session.opened",
                socket=str(path),
                session=session,
                live=slots.live,
                capacity=slots.capacity,
            )
            tg.start_soon(serve, stream, session)


@dataclass(frozen=True)
class _Endpoint:
    open_listener: Callable[[], Awaitable[SocketListener]]
    path: Path
    unlink_on_close: bool
    idle_exit_seconds: float | None


async def _run_daemon(mcp: FastMCP, endpoint: _Endpoint, max_sessions: int) -> None:
    slots = _SessionSlots(max_sessions)
    with anyio.open_signal_receiver(*_SIGNALS) as signals:
        listener = await endpoint.open_listener()
        _log.info(
            "socket.listening",
            socket=str(endpoint.path),
            activated=not endpoint.unlink_on_close,
            idle_exit_seconds=endpoint.idle_exit_seconds,
        )

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_watch_signals, signals, tg.cancel_scope)
                await _accept_sessions(
                    mcp, listener, endpoint.path, slots, endpoint.idle_exit_seconds
                )
                tg.cancel_scope.cancel()
        finally:
            await listener.aclose()
            if endpoint.unlink_on_close:
                _unlink_endpoint(endpoint.path)


def _unlink_endpoint(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _log.warning("socket.unlink_failed", socket=str(path), error=str(exc))
    else:
        _log.info("socket.closed", socket=str(path))


def _adopt_listening_socket(fd: int) -> tuple[socket.socket, Path]:
    try:
        if not stat.S_ISSOCK(os.fstat(fd).st_mode):
            raise SocketUnavailableError(f"inherited descriptor {fd} is not a socket")
        inherited = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0, fileno=fd)
    except OSError as exc:
        raise SocketUnavailableError(f"inherited descriptor {fd} is unusable: {exc}") from exc

    try:
        kind = inherited.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        bound = inherited.getsockname()
    except OSError as exc:
        inherited.close()
        raise SocketUnavailableError(f"cannot inspect inherited descriptor {fd}: {exc}") from exc

    if kind != socket.SOCK_STREAM:
        inherited.close()
        raise SocketUnavailableError(
            f"inherited descriptor {fd} is a socket of type {kind}; the daemon speaks "
            f"JSON-RPC over a stream ({int(socket.SOCK_STREAM)})"
        )
    if not isinstance(bound, str) or not bound:
        inherited.close()
        raise SocketUnavailableError(
            f"inherited descriptor {fd} is not a unix socket bound to a path "
            f"(its address is {bound!r})"
        )

    inherited.setblocking(False)
    return inherited, Path(bound)


def _bound_endpoint(socket_path: Path) -> _Endpoint:

    async def open_listener() -> SocketListener:
        prepare_socket_path(socket_path)
        return await anyio.create_unix_listener(socket_path, mode=SOCKET_MODE)

    return _Endpoint(
        open_listener=open_listener,
        path=socket_path,
        unlink_on_close=True,
        idle_exit_seconds=None,
    )


def _activated_endpoint(
    inherited: socket.socket, path: Path, expected_path: Path, idle_exit_seconds: float
) -> _Endpoint:
    if os.path.realpath(path) != os.path.realpath(expected_path):
        _log.warning(
            "socket.activated_path_mismatch",
            activated=str(path),
            expected=str(expected_path),
        )

    async def open_listener() -> SocketListener:
        return get_async_backend().create_unix_listener(inherited)

    return _Endpoint(
        open_listener=open_listener,
        path=path,
        unlink_on_close=False,
        idle_exit_seconds=idle_exit_seconds,
    )


def serve_socket(
    mcp: FastMCP,
    socket_path: Path,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
) -> None:
    anyio.run(_run_daemon, mcp, _bound_endpoint(socket_path), max_sessions)


def serve_activated_socket(
    mcp: FastMCP,
    fd: int,
    socket_path: Path,
    max_sessions: int,
    idle_exit_seconds: float,
) -> None:
    if idle_exit_seconds <= 0:
        raise ValueError(f"the idle timeout must be positive, not {idle_exit_seconds}")
    inherited, bound_path = _adopt_listening_socket(fd)
    anyio.run(
        _run_daemon,
        mcp,
        _activated_endpoint(inherited, bound_path, socket_path, idle_exit_seconds),
        max_sessions,
    )
