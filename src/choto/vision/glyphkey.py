from __future__ import annotations

import math

import numpy as np

from choto.graph import GLYPH_COLOUR_CLASSES
from choto.vision.imaging import phash_image

_SILHOUETTE_LEVEL = 0.5

_MIN_CHROMA = 24.0

_MIN_CHROMATIC_SHARE = 0.5

_ACHROMATIC_CLASS = 0

_HUE_BUCKETS = GLYPH_COLOUR_CLASSES - 1
_HUE_BUCKET_WIDTH_DEGREES = 360.0 / _HUE_BUCKETS
_HUE_ORIGIN_DEGREES = 25.0

_SEXTANT_DEGREES = 60.0


def background_colour(patch: np.ndarray) -> np.ndarray:
    ring = np.concatenate(
        [
            patch[0].reshape(-1, 3),
            patch[-1].reshape(-1, 3),
            patch[:, 0].reshape(-1, 3),
            patch[:, -1].reshape(-1, 3),
        ]
    )
    return np.median(ring, axis=0)


def ink_mask(crop: np.ndarray) -> np.ndarray:
    patch = crop.astype(np.float32)
    ink = np.linalg.norm(patch - background_colour(patch), axis=2)
    peak = float(ink.max())
    if peak <= 0.0:
        return np.zeros(crop.shape[:2], dtype=bool)
    return ink > peak * _SILHOUETTE_LEVEL


def glyph_parts(crop: np.ndarray) -> tuple[str, int]:
    mask = ink_mask(crop)
    return phash_image(_silhouette(mask)), _colour_class(crop, mask)


def _silhouette(mask: np.ndarray) -> np.ndarray:
    shape = np.where(mask, 255, 0).astype(np.uint8)
    return np.repeat(shape[:, :, None], 3, axis=2)


def _colour_class(crop: np.ndarray, mask: np.ndarray) -> int:
    pixels = crop[mask].astype(np.float32)
    if pixels.shape[0] == 0:
        return _ACHROMATIC_CLASS
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    coloured = chroma >= _MIN_CHROMA
    if float(coloured.mean()) < _MIN_CHROMATIC_SHARE:
        return _ACHROMATIC_CLASS
    hue = _dominant_hue(pixels[coloured], chroma[coloured])
    bucket = int((hue - _HUE_ORIGIN_DEGREES) % 360.0 // _HUE_BUCKET_WIDTH_DEGREES)
    return _ACHROMATIC_CLASS + 1 + bucket


def _dominant_hue(pixels: np.ndarray, chroma: np.ndarray) -> float:
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    high = pixels.max(axis=1)
    sextant = np.where(
        high == red,
        ((green - blue) / chroma) % 6.0,
        np.where(high == green, (blue - red) / chroma + 2.0, (red - green) / chroma + 4.0),
    )
    angle = np.radians(sextant * _SEXTANT_DEGREES)
    x = float(np.sum(chroma * np.cos(angle)))
    y = float(np.sum(chroma * np.sin(angle)))
    return math.degrees(math.atan2(y, x)) % 360.0
