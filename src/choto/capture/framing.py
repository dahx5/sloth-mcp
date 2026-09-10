from __future__ import annotations

import numpy as np

from choto.models import BBox


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    if width <= 0 or height <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {width}x{height}.")
    x0 = min(max(box.x, 0), width)
    y0 = min(max(box.y, 0), height)
    x1 = min(max(box.x + box.w, 0), width)
    y1 = min(max(box.y + box.h, 0), height)
    if x1 <= x0 or y1 <= y0:
        return BBox(x=x0, y=y0, w=0, h=0)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def crop_frame(frame: np.ndarray, box: BBox | None) -> np.ndarray:
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"Cannot crop an empty frame, got shape {frame.shape}.")
    if box is None:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    clamped = clamp_box(box, width, height)
    if clamped.w == 0 or clamped.h == 0:
        raise ValueError(
            f"Crop box {box.x},{box.y},{box.w},{box.h} lies outside the {width}x{height} frame."
        )
    return np.ascontiguousarray(
        frame[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w]
    )
