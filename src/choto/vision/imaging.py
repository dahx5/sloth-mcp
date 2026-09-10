from __future__ import annotations

import io

import imagehash
import numpy as np
from PIL import Image


def _to_pil(img: np.ndarray) -> Image.Image:
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a numpy array, got {type(img).__name__}.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {img.shape}.")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype {img.dtype}.")
    return Image.fromarray(img, mode="RGB")


def phash_image(img: np.ndarray) -> str:
    return str(imagehash.phash(_to_pil(img)))


def screenshot_png(img: np.ndarray, *, max_width: int = 1280) -> bytes:
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}.")

    pil = _to_pil(img)
    if pil.width > max_width:
        new_height = max(1, round(pil.height * max_width / pil.width))
        pil = pil.resize((max_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()
