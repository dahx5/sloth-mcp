from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from choto.config import OCR_ENGINE_APPLE_VISION, Settings
from choto.models import BBox, OcrLine

if TYPE_CHECKING:
    import numpy as np


class OcrError(RuntimeError): ...


MIN_RECOGNIZABLE_SIDE_PX = 3
"""Shortest side, in pixels, an image may have and still be recognizable.

Not a tuning knob and not a guess: Apple Vision refuses a smaller image itself,
and says so in the refusal — ``com.apple.Vision`` code 13, *"The image is too
small in at least one dimension 2 x 2 (each dimension has to be more than 2
pixels)"*. Three is that sentence read literally. It is stated here, on the
contract, rather than beside the one engine that quotes it, because the *reason*
a rectangle gets this small has nothing to do with recognition: it is a fact
about a window, and every engine will meet the same windows.

Windows really do get this small. Measured live on 2026-07-29: the plaque Chrome
draws in the bottom-left corner while the pointer rests on a link was listed by
the window server at 447x22 pt, then at 2x2 as it withdrew — and that rectangle
went to the recognizer as a region of interest, which is how one hovered link
took a whole run down.
"""


def too_small_to_recognize(width: int, height: int) -> str | None:
    if width >= MIN_RECOGNIZABLE_SIDE_PX and height >= MIN_RECOGNIZABLE_SIDE_PX:
        return None
    return (
        f"{width}x{height} px is too small to recognize text in; a recognizer needs at "
        f"least {MIN_RECOGNIZABLE_SIDE_PX} px on each side"
    )


def bbox_overlap_fraction(a: BBox, b: BBox) -> float:
    inter_w = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    inter_h = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    area_a, area_b = a.w * a.h, b.w * b.h
    smaller = min(area_a, area_b)
    if smaller == 0:
        inner, outer = (a, b) if area_a <= area_b else (b, a)
        cx, cy = inner.center
        inside = outer.x <= cx <= outer.x + outer.w and outer.y <= cy <= outer.y + outer.h
        return 1.0 if inside else 0.0
    return (inter_w * inter_h) / smaller


@dataclass(frozen=True)
class OcrPasses:
    lines: list[OcrLine]
    text_lines: list[OcrLine]


@runtime_checkable
class OcrEngine(Protocol):
    def recognize(
        self,
        image: np.ndarray,
        *,
        glyph_regions: Sequence[BBox] | None = None,
    ) -> list[OcrLine]: ...

    def recognize_passes(
        self,
        image: np.ndarray,
        *,
        glyph_regions: Sequence[BBox] | None = None,
    ) -> OcrPasses: ...


@runtime_checkable
class OcrRequestCounter(Protocol):
    def request_count(
        self,
        width: int,
        height: int,
        glyph_regions: Sequence[BBox] | None = None,
    ) -> int: ...


OcrEngineFactory = Callable[[Settings], OcrEngine]


def _build_apple_vision(settings: Settings) -> OcrEngine:
    from choto.ocr.vision import AppleVisionOcr

    return AppleVisionOcr(settings)


OCR_ENGINES: dict[str, OcrEngineFactory] = {
    OCR_ENGINE_APPLE_VISION: _build_apple_vision,
}
"""The name → builder table ``ocr_engine`` selects from.

This is the one place an engine is wired in: adding an implementation means
adding a row here, and every consumer picks it up because none of them names an
engine directly.
"""


def resolve_ocr_factory(name: str) -> OcrEngineFactory:
    factory = OCR_ENGINES.get(name)
    if factory is None:
        raise OcrError(
            f"Unknown OCR engine {name!r} (setting ocr_engine); "
            f"available engines are {sorted(OCR_ENGINES)}."
        )
    return factory


def create_ocr_engine(settings: Settings) -> OcrEngine:
    return resolve_ocr_factory(settings.ocr_engine)(settings)
