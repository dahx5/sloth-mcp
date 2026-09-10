from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol

import numpy as np

from choto.capture.framing import clamp_box
from choto.capture.monitor import MonitorInfo
from choto.log import get_logger
from choto.models import BBox, OcrLine, Scope
from choto.vision.titlebar import Rect, Refusal, refine_window_rect
from choto.vision.windowsource import (
    Obstructions,
    Window,
    WindowList,
    WindowRect,
    rects_overlap,
    window_rect_to_pixels,
)

_log = get_logger(__name__)

_UNTITLED = "<untitled>"

CHROME_BAND_COVERAGE_MIN = 0.9
CHROME_BAND_DEPTH_MAX = 0.25

FrameOrSource = np.ndarray | Callable[[], np.ndarray]


class WindowSelectionError(RuntimeError): ...


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _titled(title: str) -> str:
    return f'"{title}"' if title else _UNTITLED


def _suffix(note: str | None) -> str:
    return f"; {note}" if note else ""


def _empty(box: BBox) -> bool:
    return min(box.w, box.h) < 1


def window_titles(windows: Sequence[Window]) -> str:
    if not windows:
        return "none"
    return ", ".join(_titled(window.title) for window in windows)


class ObstructionSource(Protocol):
    def __call__(self, rect: WindowRect, *, owner: str) -> Obstructions: ...


@dataclass(frozen=True)
class Covered:
    rects: tuple[BBox, ...] = ()
    owners: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def any(self) -> bool:
        return bool(self.rects)

    def hides(self, x: int, y: int) -> bool:
        return _inside(self.rects, x, y)

    @property
    def named(self) -> str:
        return ", ".join(f'"{owner}"' for owner in self.owners) or "another window"

    def note(self) -> str | None:
        if self.reason is not None:
            return f"could not check what is drawn over it: {self.reason}"
        if not self.rects:
            return None
        return f"partly covered by {self.named}"


@dataclass(frozen=True)
class ScopedArea:
    app_name: str
    title: str
    window: BBox
    chrome: tuple[BBox, ...] = ()
    refinement: str | None = None
    covered: Covered = field(default_factory=Covered)

    scoped: ClassVar[Literal[True]] = True

    @property
    def rects(self) -> tuple[BBox, ...]:
        return (self.window, *self.chrome)

    def select(self, lines: Sequence[OcrLine], scope: Scope) -> list[OcrLine]:
        if scope is Scope.CHROME:
            return [line for line in lines if _inside(self.chrome, *line.bbox.center)]
        return self.visible(self.in_window(lines))

    def visible(self, lines: Sequence[OcrLine]) -> list[OcrLine]:
        if not self.covered.any:
            return list(lines)
        return [line for line in lines if not self.covered.hides(*line.bbox.center)]

    def in_window(self, lines: Sequence[OcrLine]) -> list[OcrLine]:
        return [line for line in lines if _inside((self.window,), *line.bbox.center)]

    def label(self) -> str:
        return f'"{self.app_name}" window {_titled(self.title)}'

    def window_line(self) -> str:
        refined = f" ({self.refinement})" if self.refinement else ""
        return f"{self.label()}{refined}"

    def scan_note(self) -> str:
        return f"{self.label()} + {len(self.chrome)} chrome window(s)"

    def describe(self, kept: int, total: int, scope: Scope) -> str:
        refined = _suffix(self.refinement)
        if scope is Scope.CHROME:
            return (
                f"searched chrome of {self.label()} "
                f"({_plural(len(self.chrome), 'chrome window')}; {kept} of {total} elements"
                f"{refined})"
            )
        obstructed = _suffix(self.covered.note())
        return f"searched {self.label()} ({kept} of {total} elements{refined}{obstructed})"


@dataclass(frozen=True)
class BlindArea:
    fallback_reason: str
    app_name: str = ""
    blind_region: BBox | None = None

    scoped: ClassVar[Literal[False]] = False
    rects: ClassVar[tuple[BBox, ...]] = ()

    def select(self, lines: Sequence[OcrLine], scope: Scope) -> list[OcrLine]:
        if scope is Scope.CHROME:
            return list(lines)
        return self.in_window(lines)

    def in_window(self, lines: Sequence[OcrLine]) -> list[OcrLine]:
        if self.blind_region is None:
            return list(lines)
        return [line for line in lines if _inside((self.blind_region,), *line.bbox.center)]

    def label(self) -> str:
        return "whole screen minus system chrome" if self.blind_region else "whole screen"

    def window_line(self) -> str:
        return (
            f"whole screen (window scope unavailable: {self.fallback_reason}; "
            "the elements below may belong to several applications and are NOT remembered "
            "— recall will not have this later)"
        )

    def scan_note(self) -> str:
        return "whole screen"

    def describe(self, kept: int, total: int, scope: Scope) -> str:
        counted = (
            _plural(total, "element")
            if self.blind_region is None
            else f"{kept} of {total} elements"
        )
        return (
            f"searched {self.label()} ({counted}; window scope unavailable: {self.fallback_reason})"
        )


WorkArea = ScopedArea | BlindArea


def _inside(rects: Sequence[BBox], x: int, y: int) -> bool:
    return any(rect.x <= x < rect.x + rect.w and rect.y <= y < rect.y + rect.h for rect in rects)


def select_window(listing: WindowList, query: str | None) -> Window:
    candidates = listing.app_windows
    if not candidates:
        raise WindowSelectionError(f'"{listing.app_name}" has no window to work in')

    wanted = query.strip().casefold() if query is not None else ""
    if not wanted:
        return candidates[0]

    matches = [window for window in candidates if wanted in window.title.casefold()]
    if not matches:
        raise WindowSelectionError(
            f'no window of "{listing.app_name}" has a title containing "{query}"; '
            f"open windows: {window_titles(candidates)}"
        )
    if len(matches) > 1:
        raise WindowSelectionError(
            f'"{query}" matches {len(matches)} open windows of "{listing.app_name}" '
            f"({window_titles(matches)}); name the one you mean more precisely"
        )
    return matches[0]


def covers_frame(window: Window, *, scale: float, frame_width: int, frame_height: int) -> bool:
    box = clamp_box(window_rect_to_pixels(window.rect, scale), frame_width, frame_height)
    return not _empty(box)


def windows_on_frame(
    windows: Sequence[Window], *, scale: float, frame_width: int, frame_height: int
) -> tuple[Window, ...]:
    return tuple(
        window
        for window in windows
        if covers_frame(window, scale=scale, frame_width=frame_width, frame_height=frame_height)
    )


def _trim_edges(near: int, far: int, *, start: int, size: int, extent: int) -> tuple[int, int]:
    if start <= 0:
        return max(near, start + size), far
    if start + size >= extent:
        return near, min(far, start)
    return near, far


def blind_read_region(
    chrome: Sequence[BBox], *, frame_width: int, frame_height: int
) -> BBox | None:
    if frame_width < 1 or frame_height < 1:
        raise ValueError(f"Frame must have a positive size, got {frame_width}x{frame_height}.")
    left, top = 0, 0
    right, bottom = frame_width, frame_height
    for box in chrome:
        band = clamp_box(box, frame_width, frame_height)
        if _empty(band):
            continue
        spans_width = band.w >= CHROME_BAND_COVERAGE_MIN * frame_width
        spans_height = band.h >= CHROME_BAND_COVERAGE_MIN * frame_height
        if spans_width and band.h <= CHROME_BAND_DEPTH_MAX * frame_height:
            top, bottom = _trim_edges(top, bottom, start=band.y, size=band.h, extent=frame_height)
        if spans_height and band.w <= CHROME_BAND_DEPTH_MAX * frame_width:
            left, right = _trim_edges(left, right, start=band.x, size=band.w, extent=frame_width)
    if (left, top, right, bottom) == (0, 0, frame_width, frame_height):
        return None
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def covering_rects(
    listing: WindowList,
    window: Window,
    *,
    scale: float,
    frame_width: int,
    frame_height: int,
    obstructions: ObstructionSource | None,
) -> Covered:
    rects: list[BBox] = []
    owners: list[str] = []

    def add(covering: Window) -> None:
        box = clamp_box(window_rect_to_pixels(covering.rect, scale), frame_width, frame_height)
        if _empty(box):
            return
        rects.append(box)
        name = covering.app_name or "(unnamed)"
        if name not in owners:
            owners.append(name)

    reason: str | None = None
    if obstructions is not None:
        answer = obstructions(window.rect, owner=listing.app_name)
        reason = answer.reason
        for covering in answer.windows:
            add(covering)
    for sibling in listing.app_windows:
        if sibling is window:
            break
        if rects_overlap(sibling.rect, window.rect):
            add(sibling)
    return Covered(rects=tuple(rects), owners=tuple(owners), reason=reason)


def _on_frame(
    listing: WindowList, *, scale: float, frame_width: int, frame_height: int
) -> WindowList:
    app_windows = windows_on_frame(
        listing.app_windows, scale=scale, frame_width=frame_width, frame_height=frame_height
    )
    chrome = windows_on_frame(
        listing.chrome, scale=scale, frame_width=frame_width, frame_height=frame_height
    )
    dropped = len(listing.app_windows) - len(app_windows)
    if not app_windows:
        return WindowList(
            app_name=listing.app_name,
            windows=(),
            reason=(
                f'no window of "{listing.app_name}" lies on the captured display '
                f"({_plural(dropped, 'window')} entirely outside it)"
            ),
        )
    if dropped or len(listing.chrome) != len(chrome):
        _log.info(
            "workarea.off_frame_windows_dropped",
            app_name=listing.app_name,
            app_windows_dropped=dropped,
            chrome_dropped=len(listing.chrome) - len(chrome),
        )
    return WindowList(app_name=listing.app_name, windows=app_windows + chrome)


@dataclass(frozen=True)
class RefinedWindow:
    box: BBox
    note: str | None


_RefineKey = tuple[str, str, tuple[int, int, int, int], float]


class WindowRectRefiner:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._remembered: dict[_RefineKey, RefinedWindow] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def refine(
        self, window: Window, box: BBox, *, frame: FrameOrSource, scale: float
    ) -> RefinedWindow:
        if not self._enabled:
            return RefinedWindow(box=box, note=None)
        key: _RefineKey = (window.app_name, window.title, (box.x, box.y, box.w, box.h), scale)
        remembered = self._remembered.get(key)
        if remembered is not None:
            if remembered.note is not None:
                _log.debug(
                    "workarea.refinement_reused",
                    app_name=window.app_name,
                    window_title=window.title,
                    note=remembered.note,
                )
            return remembered
        pixels = frame() if callable(frame) else frame
        answer = self._measure(window, box, frame=pixels, scale=scale)
        self._remembered = {key: answer}
        return answer

    def _measure(
        self, window: Window, box: BBox, *, frame: np.ndarray, scale: float
    ) -> RefinedWindow:
        unchanged = RefinedWindow(box=box, note=None)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        visible = clamp_box(box, width, height)
        if _empty(box) or _empty(visible):
            _log.debug(
                "workarea.refinement_skipped",
                app_name=window.app_name,
                window_title=window.title,
                box=(box.x, box.y, box.w, box.h),
                frame=(width, height),
            )
            return unchanged
        outcome = refine_window_rect(
            frame,
            scale,
            Rect(left=box.x, top=box.y, right=box.x + box.w - 1, bottom=box.y + box.h - 1),
        )
        if isinstance(outcome, Refusal):
            _log.debug(
                "workarea.refinement_refused",
                app_name=window.app_name,
                window_title=window.title,
                code=outcome.code.value,
                detail=outcome.detail,
            )
            return unchanged
        if not any(outcome.shifts):
            return unchanged
        refined = BBox(
            x=outcome.rect.left,
            y=outcome.rect.top,
            w=outcome.rect.width,
            h=outcome.rect.height,
        )
        _log.info(
            "workarea.rect_refined",
            app_name=window.app_name,
            window_title=window.title,
            given=(box.x, box.y, box.w, box.h),
            refined=(refined.x, refined.y, refined.w, refined.h),
            shifts=outcome.shifts,
        )
        left, top, right, bottom = outcome.shifts
        return RefinedWindow(
            box=refined,
            note=(
                "rectangle refined against the title bar, sides moved by "
                f"{left}, {top}, {right}, {bottom} px"
            ),
        )


def resolve_work_area(
    listing: WindowList,
    *,
    monitor: MonitorInfo,
    target_app: str | None,
    window_query: str | None,
    same_app: Callable[[str, str], bool],
    frame: FrameOrSource | None = None,
    refiner: WindowRectRefiner | None = None,
    obstructions: ObstructionSource | None = None,
) -> WorkArea:
    candidates, reason = _windows_to_work_in(
        listing, monitor=monitor, target_app=target_app, same_app=same_app
    )
    if reason is None:
        return _scoped_area(
            candidates,
            window_query,
            monitor=monitor,
            frame=frame,
            refiner=refiner,
            obstructions=obstructions,
        )
    if window_query is not None:
        raise WindowSelectionError(f'cannot work in window "{window_query}": {reason}')
    return _blind_area(listing, reason, monitor=monitor, target_app=target_app)


def _windows_to_work_in(
    listing: WindowList,
    *,
    monitor: MonitorInfo,
    target_app: str | None,
    same_app: Callable[[str, str], bool],
) -> tuple[WindowList, str | None]:
    if listing.reason is not None:
        return listing, listing.reason
    if target_app is not None and not same_app(listing.app_name, target_app):
        return listing, (
            f'the frontmost window belongs to "{listing.app_name or "unknown"}", '
            f'not to "{target_app}"'
        )
    on_frame = _on_frame(
        listing,
        scale=monitor.scale or 1.0,
        frame_width=monitor.width_px,
        frame_height=monitor.height_px,
    )
    return on_frame, on_frame.reason


def _blind_area(
    listing: WindowList, reason: str, *, monitor: MonitorInfo, target_app: str | None
) -> BlindArea:
    scale = monitor.scale or 1.0
    region = blind_read_region(
        window_rects_to_pixels_of(
            windows_on_frame(
                listing.chrome,
                scale=scale,
                frame_width=monitor.width_px,
                frame_height=monitor.height_px,
            ),
            scale,
        ),
        frame_width=monitor.width_px,
        frame_height=monitor.height_px,
    )
    _log.warning(
        "workarea.whole_screen",
        target_app=target_app,
        reason=reason,
        blind_region=None if region is None else (region.x, region.y, region.w, region.h),
    )
    return BlindArea(fallback_reason=reason, app_name=listing.app_name, blind_region=region)


def _scoped_area(
    listing: WindowList,
    window_query: str | None,
    *,
    monitor: MonitorInfo,
    frame: FrameOrSource | None,
    refiner: WindowRectRefiner | None,
    obstructions: ObstructionSource | None,
) -> ScopedArea:
    scale = monitor.scale or 1.0
    window = select_window(listing, window_query)
    located = window_rect_to_pixels(window.rect, scale)
    refinement: str | None = None
    if frame is not None and refiner is not None:
        sharpened = refiner.refine(window, located, frame=frame, scale=scale)
        located, refinement = sharpened.box, sharpened.note
    covered = covering_rects(
        listing,
        window,
        scale=scale,
        frame_width=monitor.width_px,
        frame_height=monitor.height_px,
        obstructions=obstructions,
    )
    area = ScopedArea(
        app_name=listing.app_name,
        title=window.title,
        window=located,
        chrome=window_rects_to_pixels_of(listing.chrome, scale),
        refinement=refinement,
        covered=covered,
    )
    if covered.any or covered.reason is not None:
        _log.info(
            "workarea.covered",
            app_name=area.app_name,
            window_title=area.title,
            covering=len(covered.rects),
            owners=list(covered.owners),
            reason=covered.reason,
        )
    _log.debug(
        "workarea.resolved",
        app_name=area.app_name,
        window_title=area.title,
        chrome=len(area.chrome),
        refinement=refinement,
    )
    return area


def window_rects_to_pixels_of(windows: Sequence[Window], scale: float) -> tuple[BBox, ...]:
    return tuple(window_rect_to_pixels(window.rect, scale) for window in windows)
