from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from choto.overlay.geometry import Rect

MASK_FILL_VALUE = 0

MASK_MARGIN_PX = 1


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Frame to mask must be a numpy array, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Frame to mask must be (H, W, 3) RGB, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Frame to mask must be uint8, got dtype {image.dtype}.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Frame to mask is empty (shape {image.shape}).")


def _span(
    low_pt: float, high_pt: float, scale: float, margin_px: int, limit: int
) -> tuple[int, int]:
    low = int(math.floor(low_pt * scale)) - margin_px
    high = int(math.ceil(high_pt * scale)) + margin_px
    return max(low, 0), min(high, limit)


def mask_rects(
    image: np.ndarray,
    rects: Sequence[Rect],
    scale: float,
    *,
    margin_px: int = MASK_MARGIN_PX,
) -> np.ndarray:
    _validate_image(image)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Pixels per point must be a positive finite number, got {scale!r}.")
    if margin_px < 0:
        raise ValueError(f"Mask margin must not be negative, got {margin_px}.")

    height, width = image.shape[0], image.shape[1]
    spans: list[tuple[int, int, int, int]] = []
    for rect in rects:
        if not all(math.isfinite(value) for value in (rect.x, rect.y, rect.w, rect.h)):
            raise ValueError(f"Rectangle to mask has a non-finite coordinate: {rect!r}.")
        if rect.empty:
            continue
        x0, x1 = _span(rect.x, rect.right, scale, margin_px, width)
        y0, y1 = _span(rect.y, rect.bottom, scale, margin_px, height)
        if x0 >= x1 or y0 >= y1:
            continue
        spans.append((y0, y1, x0, x1))

    if not spans:
        return image

    masked = image.copy()
    for y0, y1, x0, x1 in spans:
        masked[y0:y1, x0:x1] = MASK_FILL_VALUE
    return masked
