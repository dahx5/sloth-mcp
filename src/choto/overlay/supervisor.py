from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass

from choto.log import get_logger
from choto.overlay.backend import OverlayCapabilities
from choto.overlay.geometry import (
    FRAME_THICKNESS_PT,
    STOP_BUTTON_WIDTH_PT,
    Rect,
    frame_bands,
    stop_button_rect,
)
from choto.overlay.identity import forget_helper, register_helper
from choto.overlay.protocol import (
    FrameEvent,
    HideCommand,
    OverlayProtocolError,
    ShowCommand,
    StopEvent,
    encode,
    parse_event,
)

_log = get_logger(__name__)

_HELPER_BOOTSTRAP = (
    "import site, sys\n"
    "for path in sys.argv[1:]:\n"
    "    site.addsitedir(path)\n"
    "from choto.overlay.__main__ import main\n"
    "sys.exit(main())\n"
)

_LOGGED_LINE_CHARS = 200

_HELPER_EXIT_TIMEOUT_S = 5.0

Spawner = Callable[[], "subprocess.Popen[str]"]


@dataclass(frozen=True, slots=True)
class _FrameSeen:
    bands: bool
    button: bool


MAC_OVERLAY_CAPABILITIES = OverlayCapabilities(
    draws_frame=True,
    stop_button=True,
    excluded_from_capture=True,
)
"""What the macOS backend can do while the frame it draws is reaching the screen.

The starting position, not a promise: two of the three are claims about pixels
on the user's display, and this platform has been observed to swallow them
(:func:`choto.overlay.app._application`). They stand until the helper's own
measurement contradicts them — see :attr:`MacOverlayBackend.capabilities` and
:data:`UNSEEN_OVERLAY_CAPABILITIES`.
"""

UNSEEN_OVERLAY_CAPABILITIES = OverlayCapabilities(
    draws_frame=False,
    stop_button=False,
    excluded_from_capture=True,
)
"""What is left when the window server says the frame never reached the screen.

Not "the helper is unhealthy" — the helper may be perfectly alive. This is the
state where Choto has counted its own panels and found none of them on the
user's display, which means a run has no visible frame and no visible stop
button, and is obliged to say so.
"""


def helper_argv() -> list[str]:
    roots = [path for path in sys.path if path]
    return [sys.executable, "-c", _HELPER_BOOTSTRAP, *roots]


def _spawn_helper() -> subprocess.Popen[str]:
    return subprocess.Popen(
        helper_argv(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        close_fds=True,
    )


class MacOverlayBackend:
    def __init__(self, spawn: Spawner = _spawn_helper) -> None:
        self._spawn = spawn
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._sent = 0
        self._stop = threading.Event()
        self._measured: _FrameSeen | None = None
        self._shown: Rect | None = None

    @property
    def capabilities(self) -> OverlayCapabilities:
        measured = self._measured
        if measured is None:
            return MAC_OVERLAY_CAPABILITIES
        if measured.bands and measured.button:
            return MAC_OVERLAY_CAPABILITIES
        if not measured.bands:
            return UNSEEN_OVERLAY_CAPABILITIES
        return OverlayCapabilities(
            draws_frame=True,
            stop_button=False,
            excluded_from_capture=MAC_OVERLAY_CAPABILITIES.excluded_from_capture,
        )

    def begin(self) -> None:
        with self._lock:
            self._stop.clear()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def show(self, rect: Rect) -> None:
        try:
            with self._lock:
                if rect == self._shown and self._alive():
                    return
                process = self._ensure_process()
                if process is None:
                    return
                command = ShowCommand(x=rect.x, y=rect.y, w=rect.w, h=rect.h)
                if self._send(process, command):
                    self._shown = rect
        except Exception:  # noqa: BLE001 - an indicator must never fail a run
            _log.exception("overlay.show_failed")

    def hide(self) -> None:
        try:
            with self._lock:
                process = self._process
                if process is None or not self._alive():
                    self._shown = None
                    return
                self._send(process, HideCommand())
                self._shown = None
        except Exception:  # noqa: BLE001 - see show()
            _log.exception("overlay.hide_failed")

    def drawn_rects(self, screen: Rect) -> tuple[Rect, ...] | None:
        target = self._shown
        if target is None:
            return None
        bands = frame_bands(target, screen, FRAME_THICKNESS_PT)
        if not bands:
            return None
        button = stop_button_rect(target, screen, FRAME_THICKNESS_PT, STOP_BUTTON_WIDTH_PT)
        if button.empty:
            return bands
        return (*bands, button)

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _ensure_process(self) -> subprocess.Popen[str] | None:
        if self._process is not None and not self._alive():
            _log.warning("overlay.helper_died", returncode=self._process.returncode)
            forget_helper(self._process.pid)
            self._process = None
            self._shown = None

        if self._process is not None:
            return self._process

        try:
            process = self._spawn()
        except OSError as exc:
            _log.warning("overlay.spawn_failed", error=str(exc))
            return None
        register_helper(process.pid)
        if process.stdin is None or process.stdout is None:
            _log.warning("overlay.spawn_without_pipes", pid=process.pid)
            self._retire(process, "started without the pipes to talk to it")
            return None

        self._process = process
        self._sent = 0
        self._measured = None
        reader = threading.Thread(
            target=self._read_events,
            args=(process,),
            name="choto-overlay-events",
            daemon=True,
        )
        reader.start()
        _log.info("overlay.helper_started", pid=process.pid)
        return process

    def _send(self, process: subprocess.Popen[str], message: ShowCommand | HideCommand) -> bool:
        stream = process.stdin
        if stream is None:
            return False
        try:
            stream.write(encode(message))
            stream.flush()
        except (OSError, ValueError) as exc:
            _log.warning("overlay.send_failed", command=message.cmd, error=str(exc))
            self._process = None
            self._shown = None
            self._retire(process, f"writing {message.cmd} failed: {exc}")
            return False
        self._sent += 1
        return True

    def _retire(self, process: subprocess.Popen[str], reason: str) -> None:
        _log.warning("overlay.helper_retired", pid=process.pid, reason=reason)
        threading.Thread(
            target=self._reap,
            args=(process,),
            name="choto-overlay-reaper",
            daemon=True,
        ).start()

    def _reap(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdin
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                _log.debug("overlay.helper_stdin_close_failed", pid=process.pid, error=str(exc))
        try:
            process.wait(timeout=_HELPER_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _log.warning(
                "overlay.helper_would_not_exit", pid=process.pid, timeout_s=_HELPER_EXIT_TIMEOUT_S
            )
            process.kill()
            try:
                process.wait(timeout=_HELPER_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _log.error("overlay.helper_unkillable", pid=process.pid)
                return
        forget_helper(process.pid)
        _log.info("overlay.helper_reaped", pid=process.pid, returncode=process.returncode)

    def _stop_pressed(self, pid: int) -> None:
        _log.warning("overlay.stop_pressed", pid=pid)
        self._stop.set()

    def _frame_measured(self, event: FrameEvent, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._process is not process:
                return
            if event.applied != self._sent:
                _log.debug(
                    "overlay.stale_frame_report",
                    pid=process.pid,
                    applied=event.applied,
                    sent=self._sent,
                )
                return
            if event.expected == 0:
                self._shown = None
                return
            self._measured = _FrameSeen(
                bands=event.onscreen > (1 if event.button_onscreen else 0),
                button=event.button_onscreen,
            )
            if event.onscreen == 0:
                self._shown = None
        if event.onscreen < event.expected:
            _log.error(
                "overlay.frame_not_on_screen",
                pid=process.pid,
                expected=event.expected,
                onscreen=event.onscreen,
                button=event.button_onscreen,
            )

    def _read_events(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                try:
                    event = parse_event(line)
                except OverlayProtocolError as exc:
                    _log.warning(
                        "overlay.unreadable_event",
                        line=line.strip()[:_LOGGED_LINE_CHARS],
                        error=str(exc),
                    )
                    continue
                if isinstance(event, StopEvent):
                    self._stop_pressed(process.pid)
                elif isinstance(event, FrameEvent):
                    self._frame_measured(event, process)
                else:
                    _log.info("overlay.helper_ready", pid=process.pid)
        except (OSError, ValueError) as exc:
            _log.warning("overlay.event_stream_failed", pid=process.pid, error=str(exc))
        finally:
            forget_helper(process.pid)
            with self._lock:
                if self._process is process:
                    self._shown = None
            _log.debug("overlay.event_stream_closed", pid=process.pid)
