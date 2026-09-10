from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from choto.capture.monitor import MonitorInfo
from choto.models import BBox
from choto.overlay.geometry import Rect


def _now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass(frozen=True)
class OverlayInk:
    rects: tuple[Rect, ...] = ()
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                f"Overlay ink scale must be a positive finite number, got {self.scale!r}."
            )


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    box: BBox
    captured_at_ms: float
    age_ms: float
    overlay: OverlayInk | None = None

    def __post_init__(self) -> None:
        image = self.image
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Frame image must be a numpy array, got {type(image).__name__}.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Frame image must be (H, W, 3) RGB, got shape {image.shape}.")
        if image.dtype != np.uint8:
            raise ValueError(f"Frame image must be uint8, got dtype {image.dtype}.")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Frame image is empty (shape {image.shape}).")
        if not image.flags["C_CONTIGUOUS"]:
            raise ValueError("Frame image must be C-contiguous; a numpy view is not enough.")
        if self.box.w != int(image.shape[1]) or self.box.h != int(image.shape[0]):
            raise ValueError(
                f"Frame box {self.box.w}x{self.box.h} does not match the "
                f"{int(image.shape[1])}x{int(image.shape[0])} image it describes."
            )
        if self.age_ms < 0.0:
            raise ValueError(f"Frame age must not be negative, got {self.age_ms}.")


@runtime_checkable
class ScreenSource(Protocol):
    def latest(self) -> Frame: ...

    def region(self, box: BBox) -> Frame: ...

    def geometry(self) -> MonitorInfo: ...

    def close(self) -> None: ...
