from __future__ import annotations

import threading
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

import numpy as np

from choto.capture.grabber import CaptureError
from choto.capture.monitor import MonitorInfo
from choto.capture.sckgrabber import SckScreenGrabber
from choto.capture.source import Frame
from choto.log import get_logger
from choto.models import BBox

_log = get_logger(__name__)

__all__ = [
    "LINGER_S",
    "SharedSckSource",
    "SharedStream",
    "open_shared_sck_source",
]

LINGER_S = 30.0

_T = TypeVar("_T")


class SharedStream:
    def __init__(
        self,
        build: Callable[[], SckScreenGrabber] = SckScreenGrabber,
        linger_s: float = LINGER_S,
    ) -> None:
        self._build = build
        self._linger_s = linger_s
        self._lock = threading.Lock()
        self._grabber: SckScreenGrabber | None = None
        self._leases = 0
        self._timer: threading.Timer | None = None

    def acquire(self) -> SckScreenGrabber:
        with self._lock:
            self._cancel_linger()
            if self._grabber is None:
                self._grabber = self._build()
                _log.info("sckshared.stream_opened")
            self._leases += 1
            return self._grabber

    def release(self) -> None:
        with self._lock:
            if self._leases == 0:
                return
            self._leases -= 1
            if self._leases > 0 or self._grabber is None:
                return
            self._timer = threading.Timer(self._linger_s, self._expire)
            self._timer.daemon = True
            self._timer.start()
            _log.debug("sckshared.lingering", seconds=self._linger_s)

    def retire_if_broken(self, grabber: SckScreenGrabber) -> None:
        reason = grabber.broken
        if not reason:
            return
        with self._lock:
            if self._grabber is not grabber:
                return
            self._grabber = None
            self._cancel_linger()
        _log.warning("sckshared.stream_retired", reason=reason)
        grabber.close()

    def close(self) -> None:
        with self._lock:
            self._cancel_linger()
            grabber, self._grabber = self._grabber, None
            self._leases = 0
        if grabber is None:
            return
        _log.info("sckshared.stream_closed", after_idle_s=0.0)
        grabber.close()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._grabber is not None

    def _cancel_linger(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _expire(self) -> None:
        with self._lock:
            self._timer = None
            if self._leases > 0 or self._grabber is None:
                return
            grabber, self._grabber = self._grabber, None
        _log.info("sckshared.stream_closed", after_idle_s=self._linger_s)
        grabber.close()


class SharedSckSource:
    def __init__(self, stream: SharedStream) -> None:
        self._stream = stream
        self._grabber: SckScreenGrabber | None = self._stream.acquire()

    def __enter__(self) -> SharedSckSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._grabber is None:
            return
        self._grabber = None
        self._stream.release()

    def latest(self) -> Frame:
        return self._through(lambda grabber: grabber.latest())

    def region(self, box: BBox) -> Frame:
        return self._through(lambda grabber: grabber.region(box))

    def geometry(self) -> MonitorInfo:
        return self._through(lambda grabber: grabber.geometry())

    def grab(self) -> np.ndarray:
        return self._through(lambda grabber: grabber.grab())

    def grab_region(self, box: BBox) -> np.ndarray:
        return self._through(lambda grabber: grabber.grab_region(box))

    def display_points(self) -> tuple[int, int]:
        return self._through(lambda grabber: grabber.display_points())

    def monitor_info(self) -> MonitorInfo:
        return self._through(lambda grabber: grabber.monitor_info())

    def _through(self, call: Callable[[SckScreenGrabber], _T]) -> _T:
        grabber = self._grabber
        if grabber is None:
            raise CaptureError(
                "This capture handle has been closed; its lease on the shared "
                "capture stream was given up."
            )
        try:
            return call(grabber)
        except CaptureError:
            self._stream.retire_if_broken(grabber)
            raise


def open_shared_sck_source(stream: SharedStream) -> SharedSckSource:
    return SharedSckSource(stream)
