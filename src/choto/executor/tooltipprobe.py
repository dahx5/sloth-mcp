from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from choto.capture.framing import clamp_box, crop_frame
from choto.capture.grabber import PullGrabber
from choto.capture.monitor import MonitorInfo
from choto.capture.stability import frame_diff
from choto.config import Settings
from choto.executor.applauncher import AppLauncher
from choto.executor.screen_reader import ScreenReader
from choto.executor.workarea import (
    ScopedArea,
    WindowRectRefiner,
    WindowSelectionError,
    resolve_work_area,
)
from choto.graph.db_models import Element
from choto.graph.icons import GlyphLabel
from choto.graph.repository import GraphRepository, normalize_text
from choto.inputctl.errors import InputControlError
from choto.inputctl.gestures import in_failsafe_corner
from choto.inputctl.protocol import InputBackend
from choto.log import get_logger
from choto.models import BBox, ElementKind, IconLabelSource, OcrLine
from choto.ocr.engine import OcrEngine
from choto.overlay.backend import NullOverlayBackend, OverlayBackend
from choto.overlay.geometry import Rect
from choto.vision.windowsource import Window, WindowList, WindowSource

_log = get_logger(__name__)

DEFAULT_TOOLTIP_PROBE_LIMIT = 12

MAX_TOOLTIP_PROBE_LIMIT = 40

TOOLTIP_PROBE_MARGIN_X_PX = 360
TOOLTIP_PROBE_MARGIN_Y_PX = 120

TOOLTIP_APPEARED_PIXEL_FRACTION = 0.01

MIN_TOOLTIP_LABEL_CHARS = 2

_MIN_TOOLTIP_CONFIDENCE = 0.5

_CORNER_REASON = "kill-switch: mouse moved to the failsafe corner; the pass stopped there"
_STOP_BUTTON_REASON = "kill-switch: stop button pressed; the pass stopped there"

_PERMISSION_REASON = (
    "input permission missing — hovering needs Accessibility. Enable this process under "
    "System Settings -> Privacy & Security -> Accessibility and call probe_tooltips again"
)


@dataclass(frozen=True)
class TooltipTiming:
    appear_timeout_ms: int = 2500
    settle_timeout_ms: int = 400
    vanish_timeout_ms: int = 800


@dataclass(frozen=True)
class TooltipProbeReport:
    app_name: str = ""
    window_title: str = ""
    refused: str | None = None
    candidates: int = 0
    hovered: int = 0
    named: int = 0
    labels: tuple[str, ...] = ()
    without_tooltip: int = 0
    unreadable: int = 0
    stopped: str | None = None
    write_error: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.candidates - self.named)

    def render(self) -> str:
        if self.refused is not None:
            return f"probe_tooltips did nothing: {self.refused}"
        where = _where(self.app_name, self.window_title)
        if self.candidates == 0:
            return (
                f"nothing to probe on {where}: it has no icons without a name. Read a window "
                "with a toolbar (observe, or a plan that walks through it) and call this again."
            )

        lines = [
            f"probed {where}: hovered {self.hovered} of {self.candidates} unnamed drawing(s)",
            f"named {self.named} from tooltips, {self.without_tooltip} showed none, "
            f"{self.unreadable} showed nothing readable",
        ]
        lines.extend(note for worth_saying, note in self._notes() if worth_saying)
        return "\n".join(lines)

    def _notes(self) -> tuple[tuple[bool, str], ...]:
        return (
            (
                bool(self.labels),
                "names written (click them like any other text): "
                + ", ".join(f'"{label}"' for label in self.labels),
            ),
            (self.stopped is not None, f"stopped early: {self.stopped}"),
            (self.write_error is not None, f"nothing was written: {self.write_error}"),
            (
                bool(self.remaining),
                f"{self.remaining} drawing(s) here still have no name — call probe_tooltips "
                "again for the ones not reached, or annotate_icons for the ones with no tooltip",
            ),
        )


def _where(app_name: str, window_title: str) -> str:
    titled = f'window "{window_title}"' if window_title else "window <untitled>"
    return f'"{app_name}" {titled}'


def validate_probe_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"limit must be at least 1 icon, got {limit}")
    if limit > MAX_TOOLTIP_PROBE_LIMIT:
        raise ValueError(
            f"limit must be at most {MAX_TOOLTIP_PROBE_LIMIT} icons per call, got {limit}; "
            "each hover waits out the application's tooltip delay"
        )
    return limit


def prose_ocr_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"ocr_glyph_pass_enabled": False})


_NO_OVERLAY = NullOverlayBackend()

_DEFAULT_TIMING = TooltipTiming()


@dataclass(frozen=True)
class _Stored:
    named: int = 0
    labels: tuple[str, ...] = ()
    error: str | None = None


class TooltipProbe:
    def __init__(
        self,
        reader: ScreenReader,
        grabber: PullGrabber,
        ocr: OcrEngine,
        input_controller: InputBackend,
        windows: WindowSource,
        apps: AppLauncher,
        repo: GraphRepository,
        settings: Settings,
        overlay: OverlayBackend = _NO_OVERLAY,
        timing: TooltipTiming = _DEFAULT_TIMING,
    ) -> None:
        self._reader = reader
        self._grabber = grabber
        self._ocr = ocr
        self._input = input_controller
        self._windows = windows
        self._apps = apps
        self._repo = repo
        self._settings = settings
        self._overlay = overlay
        self._timing = timing
        self._rect_refiner = WindowRectRefiner(enabled=settings.window_rect_refinement_enabled)

    def probe(self, limit: int = DEFAULT_TOOLTIP_PROBE_LIMIT) -> TooltipProbeReport:
        validate_probe_limit(limit)
        if not self._ensure_input_permission():
            return _refused(_PERMISSION_REASON)

        listing = self._windows.frontmost_windows()
        monitor = self._reader.monitor_info()
        try:
            area = resolve_work_area(
                listing,
                monitor=monitor,
                target_app=None,
                window_query=None,
                same_app=self._apps.same_app,
                frame=lambda: self._reader.grab_region(
                    BBox(x=0, y=0, w=monitor.width_px, h=monitor.height_px)
                ),
                refiner=self._rect_refiner,
            )
        except WindowSelectionError as exc:  # pragma: no cover - needs a window query
            return _refused(str(exc))
        if not area.scoped:
            return _refused(
                f"the frontmost window could not be located ({area.fallback_reason}), and a "
                "hover pass has nothing to aim at without one"
            )

        result = self._reader.read(area)
        screen_id = result.screen_db_id
        if screen_id is None:
            return _refused(
                "the read of the frontmost window was not remembered, so it has no icons to name"
            )

        app_name = result.state.app_name
        candidates = self._unnamed_glyphs(screen_id)
        _log.info(
            "tooltips.probe_started",
            app_name=app_name,
            window_title=area.title,
            candidates=len(candidates),
            limit=limit,
        )
        if not candidates:
            return _nothing_to_probe(app_name, area.title)

        return self._run(area, monitor, listing, app_name, candidates[:limit], len(candidates))

    def _run(
        self,
        area: ScopedArea,
        monitor: MonitorInfo,
        listing: WindowList,
        app_name: str,
        targets: Sequence[Element],
        candidates: int,
    ) -> TooltipProbeReport:
        origin = self._input.mouse_position()
        front = _front_window_of(listing)
        labels: list[GlyphLabel] = []
        hovered = 0
        without_tooltip = 0
        unreadable = 0
        stopped: str | None = None

        self._begin_overlay(area, monitor)
        try:
            for element in targets:
                stopped = self._abort_requested()
                if stopped is not None:
                    break
                stopped = self._front_changed(front)
                if stopped is not None:
                    break
                hovered += 1
                try:
                    text = self._probe_one(element, origin, monitor)
                except InputControlError as exc:
                    _log.warning("tooltips.input_failed", app_name=app_name, error=str(exc))
                    stopped = f"the pointer could not be moved: {exc}"
                    break
                if text is None:
                    without_tooltip += 1
                    continue
                if len(text) < MIN_TOOLTIP_LABEL_CHARS:
                    _log.info("tooltips.unreadable", app_name=app_name, text=text)
                    unreadable += 1
                    continue
                labels.append(
                    GlyphLabel(
                        app_name=app_name,
                        phash=str(element.icon_phash),
                        label=text,
                        source=IconLabelSource.TOOLTIP,
                    )
                )
        finally:
            self._restore_pointer(origin)
            self._end_overlay()

        stored = self._store(labels)
        _log.info(
            "tooltips.probe_finished",
            app_name=app_name,
            hovered=hovered,
            named=stored.named,
            without_tooltip=without_tooltip,
            unreadable=unreadable,
            stopped=stopped,
        )
        return TooltipProbeReport(
            app_name=app_name,
            window_title=area.title,
            candidates=candidates,
            hovered=hovered,
            named=stored.named,
            labels=stored.labels,
            without_tooltip=without_tooltip,
            unreadable=unreadable,
            stopped=stopped,
            write_error=stored.error,
        )

    def _probe_one(
        self, element: Element, park: tuple[float, float], monitor: MonitorInfo
    ) -> str | None:
        icon = BBox(x=element.x, y=element.y, w=element.w, h=element.h)
        region = self._probe_region(icon, monitor)
        baseline = self._grabber.grab()

        self._input.move(*_to_points(monitor, *icon.center))
        shown = self._wait_for_tooltip(baseline, region, icon)
        text = None if shown is None else self._read_tooltip(baseline, shown, region, icon)

        self._input.move(*park)
        if shown is not None and not self._wait_for_vanish(baseline, region, icon):
            _log.info(
                "tooltips.still_shown",
                icon=(icon.x, icon.y, icon.w, icon.h),
                timeout_ms=self._timing.vanish_timeout_ms,
            )
        return text

    def _wait_for_tooltip(
        self, baseline: np.ndarray, region: BBox, icon: BBox
    ) -> np.ndarray | None:
        deadline = time.monotonic() + self._timing.appear_timeout_ms / 1000.0
        for frame in self._frames(deadline):
            if self._masked_diff(baseline, frame, region, icon) > TOOLTIP_APPEARED_PIXEL_FRACTION:
                return self._settled(frame, region, icon)
        return None

    def _settled(self, frame: np.ndarray, region: BBox, icon: BBox) -> np.ndarray:
        deadline = time.monotonic() + self._timing.settle_timeout_ms / 1000.0
        previous = frame
        for current in self._frames(deadline):
            if (
                self._masked_diff(previous, current, region, icon)
                < self._settings.stable_diff_threshold
            ):
                return current
            previous = current
        return previous

    def _wait_for_vanish(self, baseline: np.ndarray, region: BBox, icon: BBox) -> bool:
        deadline = time.monotonic() + self._timing.vanish_timeout_ms / 1000.0
        for frame in self._frames(deadline):
            if self._masked_diff(baseline, frame, region, icon) <= TOOLTIP_APPEARED_PIXEL_FRACTION:
                return True
        return False

    def _frames(self, deadline: float) -> Iterator[np.ndarray]:
        interval = self._settings.stable_poll_interval_ms / 1000.0
        next_at = time.monotonic() + interval
        while True:
            now = time.monotonic()
            if now < next_at:
                if next_at >= deadline:
                    return
                time.sleep(next_at - now)
            if time.monotonic() >= deadline:
                return
            next_at = time.monotonic() + interval
            yield self._grabber.grab()

    def _masked_diff(
        self, before: np.ndarray, after: np.ndarray, region: BBox, icon: BBox
    ) -> float:
        left = crop_frame(before, region).copy()
        right = crop_frame(after, region).copy()
        hole = clamp_box(
            BBox(x=icon.x - region.x, y=icon.y - region.y, w=icon.w, h=icon.h),
            int(left.shape[1]),
            int(left.shape[0]),
        )
        left[hole.y : hole.y + hole.h, hole.x : hole.x + hole.w] = 0
        right[hole.y : hole.y + hole.h, hole.x : hole.x + hole.w] = 0
        return frame_diff(left, right)

    def _probe_region(self, icon: BBox, monitor: MonitorInfo) -> BBox:
        return clamp_box(
            BBox(
                x=icon.x - TOOLTIP_PROBE_MARGIN_X_PX,
                y=icon.y - TOOLTIP_PROBE_MARGIN_Y_PX,
                w=icon.w + 2 * TOOLTIP_PROBE_MARGIN_X_PX,
                h=icon.h + 2 * TOOLTIP_PROBE_MARGIN_Y_PX,
            ),
            monitor.width_px,
            monitor.height_px,
        )

    def _read_tooltip(
        self, baseline: np.ndarray, shown: np.ndarray, region: BBox, icon: BBox
    ) -> str:
        before = {normalize_text(line.text) for line in self._lines(baseline, region, icon)}
        added = [
            line
            for line in self._lines(shown, region, icon)
            if normalize_text(line.text) not in before
        ]
        added.sort(key=lambda line: (line.bbox.y, line.bbox.x))
        return " ".join(" ".join(line.text.split()) for line in added).strip()

    def _lines(self, frame: np.ndarray, region: BBox, icon: BBox) -> list[OcrLine]:
        crop = crop_frame(frame, region)
        hole = BBox(x=icon.x - region.x, y=icon.y - region.y, w=icon.w, h=icon.h)
        kept = []
        for line in self._ocr.recognize(crop):
            if line.confidence < _MIN_TOOLTIP_CONFIDENCE:
                continue
            cx, cy = line.bbox.center
            if hole.x <= cx < hole.x + hole.w and hole.y <= cy < hole.y + hole.h:
                continue
            kept.append(line)
        return kept

    def _unnamed_glyphs(self, screen_id: int) -> list[Element]:
        icons = [
            element
            for element in self._repo.get_elements(screen_id)
            if element.kind is ElementKind.ICON and element.icon_phash and not element.text.strip()
        ]
        icons.sort(key=lambda element: (element.y, element.x))
        unique: dict[str, Element] = {}
        for element in icons:
            unique.setdefault(str(element.icon_phash), element)
        return list(unique.values())

    def _store(self, labels: Sequence[GlyphLabel]) -> _Stored:
        if not labels:
            return _Stored()
        try:
            result = self._repo.apply_glyph_labels(list(labels))
        except KeyError as exc:
            _log.warning("tooltips.write_failed", error=str(exc.args[0]))
            return _Stored(error=f"{exc.args[0]} The names read this pass were not stored.")
        _log.info(
            "tooltips.labels_written",
            glyphs=result.glyphs_labeled,
            elements=result.elements_updated,
        )
        return _Stored(
            named=result.glyphs_labeled,
            labels=tuple(label.text for label in labels),
        )

    def _ensure_input_permission(self) -> bool:
        if self._input.ensure_permission():
            return True
        return self._input.request_permission()

    def _abort_requested(self) -> str | None:
        if in_failsafe_corner(self._input.mouse_position()):
            return _CORNER_REASON
        if self._overlay.stop_requested():
            return _STOP_BUTTON_REASON
        return None

    def _front_changed(self, front: Window | None) -> str | None:
        listing = self._windows.frontmost_windows()
        if listing.reason is not None:
            return f"the window server stopped reporting the front window ({listing.reason})"
        current = _front_window_of(listing)
        if front is None or current is None:
            return "the front window could no longer be identified"
        changes = (
            (
                not self._apps.same_app(listing.app_name, front.app_name),
                f'the front application changed to "{listing.app_name or "unknown"}" '
                f'from "{front.app_name}"',
            ),
            (
                current.title != front.title,
                f'the front window changed to "{current.title or "<untitled>"}" '
                f'from "{front.title or "<untitled>"}"',
            ),
            (
                current.rect != front.rect,
                "the window moved or was resized, so the icon positions no longer hold",
            ),
        )
        return next((reason for changed, reason in changes if changed), None)

    def _restore_pointer(self, origin: tuple[float, float]) -> None:
        try:
            self._input.move(*origin)
        except Exception:  # noqa: BLE001 - restoring the pointer must not mask the outcome
            _log.exception("tooltips.pointer_not_restored", origin=origin)

    def _begin_overlay(self, area: ScopedArea, monitor: MonitorInfo) -> None:
        self._overlay.begin()
        scale = _pixel_scale(monitor)
        window = area.window
        self._overlay.show(
            Rect(x=window.x / scale, y=window.y / scale, w=window.w / scale, h=window.h / scale)
        )

    def _end_overlay(self) -> None:
        self._overlay.hide()


def _pixel_scale(monitor: MonitorInfo) -> float:
    return monitor.scale or 1.0


def _to_points(monitor: MonitorInfo, px: int, py: int) -> tuple[float, float]:
    scale = _pixel_scale(monitor)
    return (px / scale, py / scale)


def _front_window_of(listing: WindowList) -> Window | None:
    windows = listing.app_windows
    return windows[0] if windows else None


def _refused(reason: str) -> TooltipProbeReport:
    return TooltipProbeReport(refused=reason)


def _nothing_to_probe(app_name: str, window_title: str) -> TooltipProbeReport:
    return TooltipProbeReport(app_name=app_name, window_title=window_title)
