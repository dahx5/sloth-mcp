from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

import numpy as np

from choto.capture.monitor import MonitorInfo
from choto.capture.source import Frame, OverlayInk, ScreenSource
from choto.log import get_logger
from choto.models import BBox
from choto.overlay.geometry import Rect
from choto.overlay.masking import mask_rects

_log = get_logger(__name__)


@runtime_checkable
class OverlayRects(Protocol):
    def drawn_rects(self, screen: Rect) -> tuple[Rect, ...] | None: ...


def to_image_space(rects: Sequence[Rect], box: BBox, scale: float) -> tuple[Rect, ...]:
    if not rects:
        return ()
    if box.x == 0 and box.y == 0:
        return tuple(rects)
    dx = box.x / scale
    dy = box.y / scale
    return tuple(Rect(x=rect.x - dx, y=rect.y - dy, w=rect.w, h=rect.h) for rect in rects)


class MaskedScreenSource:
    def __init__(self, source: ScreenSource, overlay: OverlayRects) -> None:
        self._source = source
        self._overlay = overlay
        self._geometry: MonitorInfo | None = None
        self._failure: str | None = None
        self._suppressed = 0

    def __enter__(self) -> MaskedScreenSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def latest(self) -> Frame:
        return self._masked(self._source.latest(), whole_display=True)

    def region(self, box: BBox) -> Frame:
        return self._masked(self._source.region(box), whole_display=False)

    def geometry(self) -> MonitorInfo:
        geometry = self._source.geometry()
        self._geometry = geometry
        return geometry

    def close(self) -> None:
        self._source.close()

    def _masked(self, frame: Frame, *, whole_display: bool) -> Frame:
        try:
            geometry = self._geometry_for(frame, whole_display)
            screen = Rect(x=0.0, y=0.0, w=float(geometry.width_pt), h=float(geometry.height_pt))
            rects = self._overlay.drawn_rects(screen) or ()
            scale = geometry.scale
            image = mask_rects(frame.image, to_image_space(rects, frame.box, scale), scale)
        except Exception as exc:  # noqa: BLE001 - the overlay may never fail a capture.
            self._note_failure(exc)
            return frame
        self._note_success()
        return Frame(
            image=image,
            box=frame.box,
            captured_at_ms=frame.captured_at_ms,
            age_ms=frame.age_ms,
            overlay=OverlayInk(rects=tuple(rects), scale=scale),
        )

    def _geometry_for(self, frame: Frame, whole_display: bool) -> MonitorInfo:
        cached = self._geometry
        if cached is not None and not (
            whole_display and (cached.width_px != frame.box.w or cached.height_px != frame.box.h)
        ):
            return cached
        return self.geometry()

    def _note_failure(self, exc: BaseException) -> None:
        signature = f"{type(exc).__name__}: {exc}"
        if signature == self._failure:
            self._suppressed += 1
            return
        self._failure = signature
        self._suppressed = 0
        _log.warning("overlay.mask_failed", error=signature, exc_info=True)

    def _note_success(self) -> None:
        if self._failure is None:
            return
        _log.info(
            "overlay.mask_recovered", after=self._failure, suppressed_repeats=self._suppressed
        )
        self._failure = None
        self._suppressed = 0


def mask_overlay(source: ScreenSource, overlay: OverlayRects | None) -> ScreenSource:
    if overlay is None:
        return source
    return MaskedScreenSource(source, overlay)


def _union_masked(frame: Frame, other_rects: tuple[Rect, ...], scale: float) -> np.ndarray:
    if not other_rects:
        return frame.image
    return mask_rects(frame.image, to_image_space(other_rects, frame.box, scale), scale)


def aligned_images(a: Frame, b: Frame) -> tuple[np.ndarray, np.ndarray]:
    ink_a, ink_b = a.overlay, b.overlay
    if ink_a is None and ink_b is None:
        return a.image, b.image
    if ink_a is not None and ink_b is not None and ink_a == ink_b:
        return a.image, b.image
    scale_a = ink_a.scale if ink_a is not None else ink_b.scale  # type: ignore[union-attr]
    scale_b = ink_b.scale if ink_b is not None else ink_a.scale  # type: ignore[union-attr]
    rects_a = ink_a.rects if ink_a is not None else ()
    rects_b = ink_b.rects if ink_b is not None else ()
    return (
        _union_masked(a, rects_b, scale_a),
        _union_masked(b, rects_a, scale_b),
    )
