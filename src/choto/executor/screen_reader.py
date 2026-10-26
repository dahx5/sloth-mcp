from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from statistics import median

import numpy as np
from rapidfuzz import fuzz

from choto.capture.framing import clamp_box, crop_frame
from choto.capture.monitor import MonitorInfo
from choto.capture.overlaymask import to_image_space
from choto.capture.source import Frame, ScreenSource
from choto.capture.stability import Settled, wait_until_stable
from choto.config import Settings
from choto.executor.workarea import BlindArea, ScopedArea, WorkArea
from choto.graph import normalize_text
from choto.graph.db_models import Element, Screen
from choto.graph.icons import IconSaveResult, icon_record
from choto.graph.repository import GraphRepository, hamming_hex
from choto.log import get_logger
from choto.models import BBox, ElementKind, OcrLine, ScreenState
from choto.ocr.engine import OcrEngine, create_ocr_engine, too_small_to_recognize
from choto.overlay.geometry import Rect, frame_bands, stop_button_rect
from choto.overlay.masking import mask_rects
from choto.vision.anchors import select_anchors
from choto.vision.icondetect import IconDetectorParams, detect_icon_boxes, icon_candidates
from choto.vision.imaging import phash_image
from choto.vision.navigation import (
    NO_BAND,
    BrowserBand,
    Navigation,
    address_line,
    address_typed,
    band_still_holds,
    browser_band,
    derive_navigation,
)
from choto.vision.windowsource import WindowSource, window_rect_to_pixels

_log = get_logger(__name__)

GLYPH_SCOPE_WHOLE_FRAME = "whole-frame"

TITLE_MATCH_MIN_OVERLAP = 0.9

# PR-004
CACHE_EXACT_HAMMING = 0

# PR-040
ANCHOR_PROBE_PAD_LINES = 1.0


def box_overlap(left: BBox, right: BBox) -> float:
    x0, y0 = max(left.x, right.x), max(left.y, right.y)
    x1 = min(left.x + left.w, right.x + right.w)
    y1 = min(left.y + left.h, right.y + right.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union > 0 else 0.0


def own_ink_rects(window: BBox, monitor: MonitorInfo) -> tuple[Rect, ...]:
    scale = monitor.scale or 1.0
    screen = Rect(x=0.0, y=0.0, w=float(monitor.width_pt), h=float(monitor.height_pt))
    target = Rect(x=window.x / scale, y=window.y / scale, w=window.w / scale, h=window.h / scale)
    return (*frame_bands(target, screen), stop_button_rect(target, screen))


# PR-024
def nav_rects(areas: Sequence[BBox], box: BBox, scale: float) -> tuple[Rect, ...]:
    return tuple(
        Rect(
            x=(area.x - box.x) / scale,
            y=(area.y - box.y) / scale,
            w=area.w / scale,
            h=area.h / scale,
        )
        for area in areas
    )


_MIN_CONTENT_SIDE_PX = 64


# PR-003, PR-024
def _content_slice(image: np.ndarray, navigation: Sequence[Rect]) -> np.ndarray:
    h, w = image.shape[0], image.shape[1]
    left = top = 0
    right, bottom = w, h
    for area in navigation:
        x0, y0 = int(round(area.x)), int(round(area.y))
        x1, y1 = int(round(area.x + area.w)), int(round(area.y + area.h))
        if x0 <= left and y0 <= top and y1 >= bottom and x1 < right:
            left = max(left, x1)
        elif x1 >= right and y0 <= top and y1 >= bottom and x0 > left:
            right = min(right, x0)
        elif y0 <= top and x0 <= left and x1 >= right and y1 < bottom:
            top = max(top, y1)
        elif y1 >= bottom and x0 <= left and x1 >= right and y0 > top:
            bottom = min(bottom, y0)
    if right - left < _MIN_CONTENT_SIDE_PX or bottom - top < _MIN_CONTENT_SIDE_PX:
        return image
    return image[top:bottom, left:right]


def identity_phash(
    crop: Frame,
    window: BBox,
    monitor: MonitorInfo,
    navigation: Sequence[BBox] = (),
    content_top: int = 0,
) -> str:
    scale = monitor.scale or 1.0
    ink = to_image_space(own_ink_rects(window, monitor), crop.box, scale)
    masked = mask_rects(crop.image, ink, scale, margin_px=0)
    # PR-038
    if content_top and masked.shape[0] - content_top >= _MIN_CONTENT_SIDE_PX:
        masked = masked[content_top:]
    else:
        content_top = 0
    body = BBox(x=crop.box.x, y=crop.box.y + content_top, w=crop.box.w, h=crop.box.h)
    return phash_image(_content_slice(masked, nav_rects(navigation, body, scale)))


# PR-046
def glyph_scope_signature(regions: Sequence[BBox] | None, origin: tuple[int, int] = (0, 0)) -> str:
    if not regions:
        return GLYPH_SCOPE_WHOLE_FRAME
    dx, dy = origin
    return ";".join(sorted(f"{box.x - dx},{box.y - dy},{box.w},{box.h}" for box in regions))


def _element_lines(elements: Sequence[Element]) -> list[OcrLine]:
    return [
        OcrLine(
            text=element.text,
            bbox=BBox(x=element.x, y=element.y, w=element.w, h=element.h),
            confidence=element.ocr_confidence,
        )
        for element in elements
        if element.text.strip()
    ]


def _center_inside(rects: Sequence[BBox], line: OcrLine) -> bool:
    cx, cy = line.bbox.center
    return any(rect.x <= cx < rect.x + rect.w and rect.y <= cy < rect.y + rect.h for rect in rects)


@dataclass(frozen=True)
class SurfaceSplit:
    window: list[OcrLine]
    chrome: list[OcrLine]
    elsewhere: list[OcrLine]


def split_surfaces(area: WorkArea, lines: Sequence[OcrLine]) -> SurfaceSplit:
    if not area.scoped:
        return SurfaceSplit(window=list(lines), chrome=[], elsewhere=[])

    window: list[OcrLine] = []
    chrome: list[OcrLine] = []
    elsewhere: list[OcrLine] = []
    for line in lines:
        in_window = _center_inside((area.window,), line)
        in_chrome = _center_inside(area.chrome, line)
        if in_window:
            window.append(line)
        if in_chrome:
            chrome.append(line)
        if not in_window and not in_chrome:
            elsewhere.append(line)
    return SurfaceSplit(window=window, chrome=chrome, elsewhere=elsewhere)


_ROW_TOLERANCE_RATIO = 0.5

_FALLBACK_CHAR_WIDTH_PX = 8.0

_COLUMN_SNAP_RATIO = 0.5


def _content(line: OcrLine) -> str:
    return line.text.strip()


def _normalized(text: str) -> str:
    return " ".join(text.split())


def group_rows(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    usable = [line for line in lines if _content(line)]
    if not usable:
        return []
    heights = [line.bbox.h for line in usable if line.bbox.h > 0]
    tolerance = max(1, round(median(heights) * _ROW_TOLERANCE_RATIO)) if heights else 1

    rows: list[list[OcrLine]] = []
    current: list[OcrLine] = []
    row_top = 0
    for line in sorted(usable, key=lambda item: (item.bbox.center[1], item.bbox.x)):
        centre_y = line.bbox.center[1]
        if current and centre_y - row_top > tolerance:
            rows.append(sorted(current, key=lambda item: item.bbox.x))
            current = []
        if not current:
            row_top = centre_y
        current.append(line)
    rows.append(sorted(current, key=lambda item: item.bbox.x))
    return rows


def _char_width(lines: Sequence[OcrLine]) -> float:
    widths = [
        line.bbox.w / len(_content(line)) for line in lines if _content(line) and line.bbox.w > 0
    ]
    if not widths:
        return _FALLBACK_CHAR_WIDTH_PX
    return median(widths)


def _column_starts(lines: Sequence[OcrLine], tolerance: int) -> dict[int, float]:
    edges = sorted({line.bbox.x for line in lines})
    snapped: dict[int, float] = {}
    cluster: list[int] = []

    def flush() -> None:
        if not cluster:
            return
        shared = sum(cluster) / len(cluster)
        for edge in cluster:
            snapped[edge] = shared

    for edge in edges:
        if cluster and edge - cluster[0] > tolerance:
            flush()
            cluster = []
        cluster.append(edge)
    flush()
    return snapped


def layout_lines(lines: Sequence[OcrLine]) -> list[str]:
    rows = group_rows(lines)
    if not rows:
        return []
    placed = [line for row in rows for line in row]
    char_width = _char_width(placed)
    starts = _column_starts(placed, max(1, round(char_width * _COLUMN_SNAP_RATIO)))
    origin = min(starts.values())

    rendered: list[str] = []
    for row in rows:
        text = ""
        for line in row:
            column = round((starts[line.bbox.x] - origin) / char_width)
            if column > len(text):
                text += " " * (column - len(text))
            elif text:
                text += " "
            text += _content(line)
        rendered.append(text.rstrip())
    return rendered


def _row_index_of(rows: Sequence[Sequence[OcrLine]], line: OcrLine | None, fallback: int) -> int:
    if line is None:
        return fallback
    for index, row in enumerate(rows):
        if any(item is line for item in row):
            return index
    return fallback


def lines_between(
    lines: Sequence[OcrLine], first: OcrLine | None = None, last: OcrLine | None = None
) -> list[OcrLine]:
    rows = group_rows(lines)
    if not rows:
        return []
    start = _row_index_of(rows, first, 0)
    end = _row_index_of(rows, last, len(rows) - 1)
    if start > end:
        start, end = end, start
    return [line for row in rows[start : end + 1] for line in row]


def stitch_lines(collected: Sequence[str], fresh: Sequence[str]) -> list[str]:
    tail = [_normalized(line) for line in collected]
    head = [_normalized(line) for line in fresh]
    for size in range(min(len(tail), len(head)), 0, -1):
        if tail[-size:] == head[:size]:
            return [*collected, *fresh[size:]]
    return [*collected, *fresh]


# PR-027
ScreenPin = Callable[[Sequence[OcrLine]], int | None]


@dataclass(frozen=True)
class ReadResult:
    state: ScreenState
    screen_db_id: int | None
    window_title: str
    from_cache: bool
    frame: Frame
    crop: Frame
    phash: str
    # PR-029
    navigation: Navigation = Navigation()
    # PR-059
    page_top: int = 0
    # PR-058
    unmapped: str = ""


ADDRESS_TYPED = "the browser's address bar is being typed into: not a page, not remembered"


def _page_top(look: _Look, band: BrowserBand) -> int:
    return look.crop.box.y + band.content_top if band else 0


@dataclass(frozen=True)
class _Identity:
    app_name: str
    window_title: str
    phash: str
    glyph_scope: str
    navigation: Navigation
    origin: tuple[int, int]
    site: str = ""
    band: BrowserBand = NO_BAND

    def own(self, line: OcrLine, top: int) -> bool:
        if self.band.holds(line, top) or self.navigation.holds(line):
            return False
        return normalize_text(line.text) not in self.navigation.shared


@dataclass(frozen=True)
class _Look:
    frame: Frame
    crop: Frame
    regions: tuple[BBox, ...]
    width: int
    height: int
    monitor: MonitorInfo

    @property
    def image(self) -> np.ndarray:
        return self.frame.image

    @property
    def offset(self) -> tuple[int, int]:
        return self.frame.box.x, self.frame.box.y

    @property
    def scale(self) -> float:
        return self.monitor.scale or 1.0

    def to_display(self, box: BBox) -> BBox:
        dx, dy = self.offset
        return BBox(x=box.x + dx, y=box.y + dy, w=box.w, h=box.h)

    def to_picture(self, box: BBox) -> BBox:
        dx, dy = self.offset
        return BBox(x=box.x - dx, y=box.y - dy, w=box.w, h=box.h)

    def to_crop(self, line: OcrLine) -> OcrLine:
        box = line.bbox
        moved = BBox(x=box.x - self.crop.box.x, y=box.y - self.crop.box.y, w=box.w, h=box.h)
        return line.model_copy(update={"bbox": moved})

    def glyph_regions(self) -> list[BBox] | None:
        if not self.regions:
            return None
        return [self.to_picture(box) for box in self.regions]

    def place(self, lines: Sequence[OcrLine]) -> list[OcrLine]:
        dx, dy = self.offset
        if not dx and not dy:
            return list(lines)
        return [line.model_copy(update={"bbox": self.to_display(line.bbox)}) for line in lines]

    def pixels(self, box: BBox) -> np.ndarray | None:
        placed = clamp_box(self.to_picture(box), int(self.image.shape[1]), int(self.image.shape[0]))
        if too_small_to_recognize(placed.w, placed.h) is not None:
            return None
        return crop_frame(self.image, placed)


class ScreenReader:
    def __init__(
        self,
        source: ScreenSource,
        ocr: OcrEngine,
        repo: GraphRepository,
        settings: Settings,
        windows: WindowSource,
    ) -> None:
        self._source = source
        self._ocr = ocr
        self._repo = repo
        self._settings = settings
        self._windows = windows
        self._prose: OcrEngine | None = None
        self._bands: dict[tuple[str, int, int], BrowserBand] = {}

    def read(
        self,
        area: WorkArea,
        force_ocr: bool = False,
        *,
        persist: bool = True,
        pin: ScreenPin | None = None,
        pin_keeps_node: bool = False,
    ) -> ReadResult:
        look = self._look(area)
        app_name = area.app_name or self._windows.frontmost_app_name()
        if not area.scoped:
            return self._read_unscoped(area, look, app_name)
        return self._read_window(
            area,
            look,
            app_name,
            force_ocr=force_ocr,
            persist=persist,
            pin=pin,
            pin_keeps_node=pin_keeps_node,
        )

    # PR-037
    def _browser(self, app_name: str) -> bool:
        return app_name in self._settings.browser_apps

    @staticmethod
    def _band_key(app_name: str, area: ScopedArea) -> tuple[str, int, int]:
        return app_name, area.window.w, area.window.h

    def _identity(
        self,
        area: ScopedArea,
        look: _Look,
        app_name: str,
        window_title: str,
        site: str,
        band: BrowserBand,
    ) -> _Identity:
        origin = (area.window.x, area.window.y)
        # PR-023, PR-024
        navigation = derive_navigation(
            self._repo.app_label_places(app_name, site=site, origin=origin),
            self._repo.app_anchor_sets(app_name, site=site),
        )
        return _Identity(
            app_name=app_name,
            window_title=window_title,
            phash=identity_phash(
                look.crop, area.window, look.monitor, navigation.areas, band.content_top
            ),
            glyph_scope=glyph_scope_signature(look.regions, origin),
            navigation=navigation,
            origin=origin,
            site=site,
            band=band,
        )

    def _read_window(
        self,
        area: ScopedArea,
        look: _Look,
        app_name: str,
        *,
        force_ocr: bool,
        persist: bool,
        pin: ScreenPin | None = None,
        pin_keeps_node: bool = False,
    ) -> ReadResult:
        window_title = self._window_title(area, look.monitor)
        # PR-027
        must_ocr = force_ocr or pin is not None
        # PR-058
        if self._browser(app_name) and not window_title:
            must_ocr = True
        band = NO_BAND
        site = ""
        if self._browser(app_name) and not must_ocr:
            band = self._bands.get(self._band_key(app_name, area), NO_BAND)
            # PR-038
            if not band or not band_still_holds(look.crop.image, band, look.scale):
                _log.info("window.band_unknown", app_name=app_name, known=bool(band))
                must_ocr = True
            else:
                site = self._band_site(look, band)
                must_ocr = not site
        if not must_ocr:
            ident = self._identity(area, look, app_name, window_title, site, band)
            cached = self._repo.find_screen_by_phash(
                ident.phash, CACHE_EXACT_HAMMING, app_name=app_name, site=site
            )
            if cached is not None:
                elements = self._repo.get_elements(cached.id, origin=ident.origin)
                if self._cache_serves(cached, elements, look, ident):
                    return self._serve_cached(cached, elements, look, ident, persist=persist)
                must_ocr = True
        return self._parse_window(
            area,
            look,
            app_name,
            window_title,
            persist=persist,
            forced=must_ocr,
            pin=pin,
            pin_keeps_node=pin_keeps_node,
        )

    # PR-037
    def _band_site(self, look: _Look, band: BrowserBand) -> str:
        strip = BBox(x=look.crop.box.x, y=look.crop.box.y, w=look.crop.box.w, h=band.content_top)
        pixels = look.pixels(strip)
        if pixels is None:
            return ""
        lines = self._prose_ocr().recognize(pixels, glyph_regions=None)
        found = address_line(lines, band.content_top)
        return "" if found is None else found.site

    def _cache_serves(
        self, cached: Screen, elements: Sequence[Element], look: _Look, ident: _Identity
    ) -> bool:
        refusal = None
        if cached.is_stale:
            refusal = "stale"
        elif cached.glyph_scope != ident.glyph_scope:
            refusal = "glyph_scope_changed"
        # PR-046
        elif cached.window_x is None or cached.window_y is None:
            refusal = "origin_unknown"
        # PR-037
        elif cached.site != ident.site:
            refusal = "site_unknown" if not cached.site else "site_changed"
        # PR-040
        elif not self._anchor_holds(cached, elements, look, ident):
            refusal = "anchor_not_in_place"
        if refusal is None:
            return True
        _log.info(
            "window.cache_refused",
            screen_id=cached.id,
            reason=refusal,
            stored_scope=cached.glyph_scope,
            requested_scope=ident.glyph_scope,
            stored_site=cached.site,
            site=ident.site,
        )
        return False

    # PR-040
    def _anchor_holds(
        self, cached: Screen, elements: Sequence[Element], look: _Look, ident: _Identity
    ) -> bool:
        placed: dict[str, Element] = {}
        for element in elements:
            if element.kind is ElementKind.TEXT and element.x is not None:
                placed.setdefault(element.norm_text, element)
        tried = 0
        for text in cached.anchors:
            key = normalize_text(str(text))
            element = placed.get(key)
            if element is None or key in ident.navigation.ignored:
                continue
            pad = round(element.h * ANCHOR_PROBE_PAD_LINES)
            pixels = look.pixels(
                BBox(
                    x=int(element.x or 0) - pad,
                    y=int(element.y or 0) - pad,
                    w=element.w + 2 * pad,
                    h=element.h + 2 * pad,
                )
            )
            if pixels is None:
                continue
            tried += 1
            read = self._prose_ocr().recognize(pixels, glyph_regions=None)
            seen = normalize_text(" ".join(line.text for line in read))
            if fuzz.partial_ratio(key, seen) >= self._settings.fuzzy_match_threshold:
                _log.info("window.anchor_in_place", screen_id=cached.id, anchor=text, tried=tried)
                return True
        _log.info("window.anchor_missing", screen_id=cached.id, tried=tried)
        return False

    def _serve_cached(
        self,
        cached: Screen,
        elements: Sequence[Element],
        look: _Look,
        ident: _Identity,
        *,
        persist: bool,
    ) -> ReadResult:
        lines = _element_lines(elements)
        state = ScreenState(
            app_name=ident.app_name,
            width=cached.width,
            height=cached.height,
            phash=cached.phash,
            lines=lines,
        )
        if persist:
            # PR-024
            ignored = ident.navigation.ignored
            self._repo.upsert_screen(
                state,
                [text for text in cached.anchors if normalize_text(str(text)) not in ignored],
                glyph_scope=ident.glyph_scope,
                window_title=ident.window_title,
                site=ident.site,
                window_origin=ident.origin,
                screen_id=cached.id,
            )
        _log.info(
            "window.cache_hit" if persist else "window.probed",
            screen_id=cached.id,
            phash=cached.phash,
            frame_phash=ident.phash,
            window_title=ident.window_title,
            site=ident.site,
            line_count=len(lines),
            glyph_scope=ident.glyph_scope,
        )
        return ReadResult(
            state=state,
            screen_db_id=cached.id,
            window_title=ident.window_title,
            from_cache=True,
            frame=look.frame,
            crop=look.crop,
            phash=ident.phash,
            navigation=ident.navigation,
            page_top=_page_top(look, ident.band),
        )

    # PR-038
    def _learn_band(
        self, area: ScopedArea, look: _Look, app_name: str, lines: Sequence[OcrLine]
    ) -> BrowserBand:
        key = self._band_key(app_name, area)
        band = browser_band(look.crop.image, [look.to_crop(line) for line in lines], look.scale)
        if band:
            self._bands[key] = band
            # PR-037
            band = replace(band, site=self._band_site(look, band))
        else:
            self._bands.pop(key, None)
        _log.info(
            "window.band_read",
            app_name=app_name,
            content_top=band.content_top,
            site=band.site,
        )
        return band

    # PR-036
    def _merge_twins(self, node: Screen, ident: _Identity, anchors: Sequence[str]) -> Screen:
        twins = self._repo.screens_agreeing(
            app_name=ident.app_name,
            site=ident.site or node.site,
            anchors=anchors,
            ignored=ident.navigation.ignored,
        )
        group = sorted({*twins, node.id})
        if len(group) < 2:
            return node
        return self._repo.merge_screens(group)

    def _parse_window(
        self,
        area: ScopedArea,
        look: _Look,
        app_name: str,
        window_title: str,
        *,
        persist: bool,
        forced: bool,
        pin: ScreenPin | None = None,
        pin_keeps_node: bool = False,
    ) -> ReadResult:
        passes = self._ocr.recognize_passes(look.image, glyph_regions=look.glyph_regions())
        lines = look.place(passes.lines)
        window_lines = area.in_window(lines)
        band = NO_BAND
        if self._browser(app_name):
            band = self._learn_band(area, look, app_name, window_lines)
        ident = self._identity(area, look, app_name, window_title, band.site, band)
        # PR-058
        if self._browser(app_name) and address_typed(
            [look.to_crop(line) for line in window_lines],
            look.scale,
            titled=bool(window_title),
            band=band,
        ):
            return self._unmapped(look, ident, lines, ADDRESS_TYPED)
        # PR-024, PR-038
        anchors = select_anchors(
            [line for line in window_lines if ident.own(line, look.crop.box.y)]
        )
        node = None
        pinned = False
        # PR-027
        held = pin(window_lines) if pin is not None else None
        if held is not None:
            node = self._repo.get_screen(held)
            pinned = node is not None
            if node is not None:
                _log.info(
                    "window.pinned_to_pass",
                    screen_id=node.id,
                    phash=node.phash,
                    frame_phash=ident.phash,
                    drift=hamming_hex(ident.phash, node.phash),
                )
        if node is None:
            # PR-004
            node = self._repo.find_screen_by_phash(
                ident.phash,
                self._settings.phash_max_hamming,
                app_name=ident.app_name,
                site=ident.site,
                anchors=anchors,
                ignored=ident.navigation.ignored,
            )
            if node is not None:
                _log.info(
                    "window.anchors_matched",
                    screen_id=node.id,
                    phash=ident.phash,
                    distance=hamming_hex(ident.phash, node.phash),
                    window_title=ident.window_title,
                    anchors=anchors,
                )
                if persist:
                    node = self._merge_twins(node, ident, anchors)
        node_drift = 0 if node is None else hamming_hex(ident.phash, node.phash)
        # PR-038
        kept_phash = node.phash if pinned and node is not None else ident.phash
        state = ScreenState(
            app_name=ident.app_name,
            width=look.width,
            height=look.height,
            phash=kept_phash,
            lines=lines,
        )
        # PR-039
        if persist and not (pinned and pin_keeps_node):
            screen = self._repo.upsert_screen(
                state,
                anchors,
                glyph_scope=ident.glyph_scope,
                window_title=ident.window_title,
                site=ident.site,
                window_origin=ident.origin,
                screen_id=None if node is None else node.id,
            )
            screen_id: int | None = screen.id
            state = state.model_copy(update={"phash": screen.phash})
            self._repo.replace_elements(screen.id, window_lines)
            icon_lines = self._read_icons(
                look, area, ident, screen.id, look.place(passes.text_lines)
            )
            if icon_lines:
                state = state.model_copy(update={"lines": [*lines, *icon_lines]})
            _log.info(
                "window.parsed",
                screen_id=screen.id,
                phash=screen.phash,
                frame_phash=ident.phash,
                drift=node_drift,
                window_title=ident.window_title,
                site=ident.site,
                content_top=band.content_top,
                line_count=len(lines),
                stored_count=len(window_lines),
                named_icons=len(icon_lines),
                forced=forced,
                was_stale=bool(node is not None and node.is_stale),
                glyph_scope=ident.glyph_scope,
                nav_slots=len(ident.navigation.slots),
                nav_areas=len(ident.navigation.areas),
                shared_labels=len(ident.navigation.shared),
            )
        else:
            screen_id = node.id if node is not None else None
            if node is not None:
                state = state.model_copy(update={"phash": node.phash})
            _log.info(
                "window.probed",
                screen_id=screen_id,
                phash=state.phash,
                frame_phash=ident.phash,
                drift=node_drift,
                window_title=ident.window_title,
                site=ident.site,
                line_count=len(lines),
                glyph_scope=ident.glyph_scope,
            )
        return ReadResult(
            state=state,
            screen_db_id=screen_id,
            window_title=ident.window_title,
            from_cache=False,
            frame=look.frame,
            crop=look.crop,
            phash=ident.phash,
            navigation=ident.navigation,
            page_top=_page_top(look, band),
        )

    # PR-058
    def _unmapped(
        self, look: _Look, ident: _Identity, lines: list[OcrLine], why: str
    ) -> ReadResult:
        _log.info(
            "window.unmapped",
            why=why,
            frame_phash=ident.phash,
            window_title=ident.window_title,
            content_top=ident.band.content_top,
            line_count=len(lines),
        )
        state = ScreenState(
            app_name=ident.app_name,
            width=look.width,
            height=look.height,
            phash=ident.phash,
            lines=lines,
        )
        return ReadResult(
            state=state,
            screen_db_id=None,
            window_title=ident.window_title,
            from_cache=False,
            frame=look.frame,
            crop=look.crop,
            phash=ident.phash,
            navigation=ident.navigation,
            unmapped=why,
        )

    # PR-007
    def _window_title(self, area: ScopedArea, monitor: MonitorInfo) -> str:
        window = area.window
        listing = self._windows.frontmost_windows()
        if listing.app_name != area.app_name:
            _log.info(
                "window.title_foreign_app",
                work_area_app=area.app_name,
                frontmost_app=listing.app_name,
                window_title=area.title,
            )
            return area.title
        scale = monitor.scale or 1.0
        best = area.title
        best_overlap = 0.0
        for candidate in listing.app_windows:
            overlap = box_overlap(window, window_rect_to_pixels(candidate.rect, scale))
            if overlap > best_overlap:
                best, best_overlap = candidate.title, overlap
        if best_overlap < TITLE_MATCH_MIN_OVERLAP:
            _log.info(
                "window.title_unmatched",
                app_name=area.app_name,
                window=(window.x, window.y, window.w, window.h),
                best_overlap=round(best_overlap, 3),
                window_title=area.title,
            )
            return area.title
        if best != area.title:
            _log.info(
                "window.title_moved_on",
                app_name=area.app_name,
                resolved_title=area.title,
                fresh_title=best,
                overlap=round(best_overlap, 3),
            )
        return best

    def _look(self, area: WorkArea) -> _Look:
        monitor = self._source.geometry()
        frame = self._source.latest()
        return _Look(
            frame=frame,
            crop=self._window_frame(frame, area.window) if area.scoped else frame,
            regions=area.rects,
            width=int(frame.image.shape[1]),
            height=int(frame.image.shape[0]),
            monitor=monitor,
        )

    @staticmethod
    def _window_frame(frame: Frame, window: BBox) -> Frame:
        image = crop_frame(frame.image, window)
        placed = clamp_box(window, int(frame.image.shape[1]), int(frame.image.shape[0]))
        return Frame(
            image=image,
            box=BBox(
                x=frame.box.x + placed.x,
                y=frame.box.y + placed.y,
                w=placed.w,
                h=placed.h,
            ),
            captured_at_ms=frame.captured_at_ms,
            age_ms=frame.age_ms,
            overlay=frame.overlay,
        )

    def _read_icons(
        self,
        look: _Look,
        area: ScopedArea,
        ident: _Identity,
        screen_id: int,
        text_lines: Sequence[OcrLine],
    ) -> list[OcrLine]:
        if not self._settings.icon_detection_enabled:
            return []

        monitor = look.monitor
        params = IconDetectorParams(
            scale=max(1, round(monitor.scale)), confidence=self._settings.icon_confidence
        )
        boxes = detect_icon_boxes(look.image, look.to_picture(area.window), params)
        candidates = icon_candidates(
            look.image,
            boxes,
            [look.to_picture(line.bbox) for line in text_lines],
            params,
        )
        records = [
            icon_record(look.to_display(item.bbox), item.phash, item.crop) for item in candidates
        ]
        saved = self._repo.save_icon_candidates(screen_id, ident.app_name, records)
        return self._named_icon_lines(screen_id, saved, ident.origin)

    def _named_icon_lines(
        self, screen_id: int, saved: IconSaveResult, origin: tuple[int, int]
    ) -> list[OcrLine]:
        if not saved.elements_named:
            return []
        return _element_lines(
            [
                element
                for element in self._repo.get_elements(screen_id, origin=origin)
                if element.kind is ElementKind.ICON
            ]
        )

    def _read_unscoped(self, area: BlindArea, look: _Look, app_name: str) -> ReadResult:
        phash = phash_image(look.crop.image)
        lines = self._ocr.recognize(look.frame.image, glyph_regions=None)
        state = ScreenState(
            app_name=app_name,
            width=look.width,
            height=look.height,
            phash=phash,
            lines=lines,
        )
        _log.info(
            "window.unscoped_parse",
            app_name=app_name,
            phash=phash,
            line_count=len(lines),
            reason=area.fallback_reason,
        )
        return ReadResult(
            state=state,
            screen_db_id=None,
            window_title="",
            from_cache=False,
            frame=look.frame,
            crop=look.crop,
            phash=phash,
        )

    def monitor_info(self) -> MonitorInfo:
        return self._source.geometry()

    def grab_frame(self, box: BBox) -> Frame:
        return self._source.region(box)

    def grab_region(self, box: BBox) -> np.ndarray:
        return self.grab_frame(box).image

    def read_region_text(self, box: BBox) -> list[OcrLine]:
        monitor = self._source.geometry()
        clamped = clamp_box(box, monitor.width_px, monitor.height_px)
        pixels = self._source.region(clamped).image
        lines = self._prose_ocr().recognize(pixels, glyph_regions=None)
        return [
            line.model_copy(
                update={
                    "bbox": BBox(
                        x=line.bbox.x + clamped.x,
                        y=line.bbox.y + clamped.y,
                        w=line.bbox.w,
                        h=line.bbox.h,
                    )
                }
            )
            for line in lines
        ]

    def _prose_ocr(self) -> OcrEngine:
        if self._prose is None:
            from choto.executor.tooltipprobe import prose_ocr_settings

            self._prose = create_ocr_engine(prose_ocr_settings(self._settings))
        return self._prose

    def latest_frame(self) -> Frame:
        return self._source.latest()

    def wait_until_stable(
        self,
        timeout_ms: int,
        *,
        quiet_grace_ms: int,
        region: BBox | None = None,
        acted_at: tuple[int, int] | None = None,
        baseline: Frame | None = None,
        since_ms: int = 0,
    ) -> Settled:
        return wait_until_stable(
            self._source,
            timeout_ms=timeout_ms,
            settings=self._settings,
            quiet_grace_ms=quiet_grace_ms,
            region=region,
            acted_at=acted_at,
            baseline=baseline,
            since_ms=since_ms,
        )

    def close(self) -> None:
        self._source.close()
