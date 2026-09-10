from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from choto.capture.monitor import MonitorInfo
from choto.models import BBox


class CaptureError(RuntimeError): ...


def has_screen_recording_permission() -> bool:
    import Quartz

    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001 - preflight is best-effort; assume denied.
        return False


@runtime_checkable
class PullGrabber(Protocol):
    def grab(self) -> np.ndarray: ...

    def grab_region(self, box: BBox) -> np.ndarray: ...

    def display_points(self) -> tuple[int, int]: ...

    def monitor_info(self) -> MonitorInfo: ...

    def close(self) -> None: ...
