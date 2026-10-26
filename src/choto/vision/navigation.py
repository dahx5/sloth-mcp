from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from choto.graph import normalize_text
from choto.models import BBox, OcrLine
from choto.vision.anchors import anchors_agree

# PR-023
NAV_MIN_WINDOWS = 3

# PR-023
NAV_MIN_SHARE = 0.35

# PR-023
NAV_PLACE_TOLERANCE_PX = 4


@dataclass(frozen=True)
class LabelPlace:
    screen_id: int
    text: str
    bbox: BBox


@dataclass(frozen=True)
class NavSlot:
    text: str
    bbox: BBox


@dataclass(frozen=True)
class Navigation:
    slots: tuple[NavSlot, ...] = ()
    areas: tuple[BBox, ...] = ()
    windows: int = 0
    quorum: int = 0
    shared: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.slots)

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(slot.text for slot in self.slots)

    # PR-024
    @property
    def ignored(self) -> frozenset[str]:
        return self.labels | self.shared

    # PR-024
    def holds(self, line: OcrLine) -> bool:
        text = normalize_text(line.text)
        if not text:
            return False
        cx, cy = line.bbox.center
        return any(slot.text == text and _near(slot.bbox, cx, cy) for slot in self.slots)


def _near(box: BBox, cx: int, cy: int) -> bool:
    bx, by = box.center
    return abs(cx - bx) <= NAV_PLACE_TOLERANCE_PX and abs(cy - by) <= NAV_PLACE_TOLERANCE_PX


def place_clusters(places: Sequence[LabelPlace]) -> list[list[LabelPlace]]:
    grouped: list[list[LabelPlace]] = []
    for place in sorted(places, key=lambda item: (item.bbox.y, item.bbox.x)):
        cx, cy = place.bbox.center
        for cluster in grouped:
            if _near(cluster[0].bbox, cx, cy):
                cluster.append(place)
                break
        else:
            grouped.append([place])
    return grouped


def _median_box(places: Sequence[LabelPlace]) -> BBox:
    return BBox(
        x=round(median(place.bbox.x for place in places)),
        y=round(median(place.bbox.y for place in places)),
        w=round(median(place.bbox.w for place in places)),
        h=round(median(place.bbox.h for place in places)),
    )


def _grow(box: BBox, pad: int) -> BBox:
    return BBox(x=box.x - pad, y=box.y - pad, w=box.w + 2 * pad, h=box.h + 2 * pad)


def _touching(left: BBox, right: BBox) -> bool:
    return (
        left.x <= right.x + right.w
        and right.x <= left.x + left.w
        and left.y <= right.y + right.h
        and right.y <= left.y + left.h
    )


def _hull(boxes: Sequence[BBox]) -> BBox:
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    right = max(box.x + box.w for box in boxes)
    bottom = max(box.y + box.h for box in boxes)
    return BBox(x=x, y=y, w=right - x, h=bottom - y)


def _merge_once(boxes: Sequence[BBox]) -> list[BBox]:
    blocks: list[BBox] = []
    for box in boxes:
        kept = [block for block in blocks if not _touching(block, box)]
        joined = [block for block in blocks if _touching(block, box)]
        blocks = [*kept, _hull([box, *joined])]
    return blocks


def _areas(slots: Sequence[NavSlot]) -> tuple[BBox, ...]:
    if not slots:
        return ()
    pad = max(1, round(median(slot.bbox.h for slot in slots)))
    blocks = [_grow(slot.bbox, pad) for slot in slots]
    while True:
        merged = _merge_once(blocks)
        if len(merged) == len(blocks):
            return tuple(sorted(merged, key=lambda box: (box.y, box.x)))
        blocks = merged


# PR-024
SHARED_LABEL_SHARE = 0.5


# PR-024
def shared_labels(
    places: Sequence[LabelPlace], anchors: Mapping[int, Sequence[str]]
) -> frozenset[str]:
    texts: dict[int, set[str]] = {}
    for place in places:
        text = normalize_text(place.text)
        if text:
            texts.setdefault(place.screen_id, set()).add(text)
    windows: list[tuple[Sequence[str], set[str]]] = []
    for screen_id in sorted(texts):
        stored = anchors.get(screen_id, ())
        twin = next((w for w in windows if anchors_agree(w[0], stored)), None)
        if twin is None:
            windows.append((stored, set(texts[screen_id])))
        else:
            twin[1].update(texts[screen_id])
    if len(windows) < NAV_MIN_WINDOWS:
        return frozenset()
    counts = Counter(text for _anchors, seen in windows for text in seen)
    return frozenset(
        text for text, count in counts.items() if count >= len(windows) * SHARED_LABEL_SHARE
    )


# PR-023
def derive_navigation(
    places: Sequence[LabelPlace], anchors: Mapping[int, Sequence[str]] | None = None
) -> Navigation:
    windows = {place.screen_id for place in places}
    shared = frozenset() if anchors is None else shared_labels(places, anchors)
    if len(windows) < NAV_MIN_WINDOWS:
        return Navigation(windows=len(windows), shared=shared)
    quorum = max(NAV_MIN_WINDOWS, math.ceil(len(windows) * NAV_MIN_SHARE))

    grouped: dict[str, list[LabelPlace]] = {}
    for place in places:
        text = normalize_text(place.text)
        if text:
            grouped.setdefault(text, []).append(place)

    slots: list[NavSlot] = []
    for text, occurrences in sorted(grouped.items()):
        for cluster in place_clusters(occurrences):
            if len({place.screen_id for place in cluster}) >= quorum:
                slots.append(NavSlot(text=text, bbox=_median_box(cluster)))
    return Navigation(
        slots=tuple(slots),
        areas=_areas(slots),
        windows=len(windows),
        quorum=quorum,
        shared=shared,
    )


# PR-038
BAND_SEARCH_PT = 160

# PR-038
BAND_SEPARATOR_REACH_PT = 48

# PR-038
BAND_SEPARATOR_COVERAGE = 0.95

# PR-038
BAND_STEP_RGB = 6

_HOST_RE = re.compile(
    r"(?<![\w.@])(?:https?://)?"
    r"((?:[^\W_][\w-]*\.)+[^\W\d_]{2,}|\d{1,3}(?:\.\d{1,3}){3}|localhost)"
    r"(?::(\d{1,5}))?(?=$|[/?#\s])",
    re.IGNORECASE,
)

_PAGE_SCHEME_RE = re.compile(r"\b([a-z][a-z-]*)://([a-z0-9-]+)", re.IGNORECASE)

_WEB_SCHEMES = frozenset({"http", "https"})

_WWW = "www."


@dataclass(frozen=True)
class SiteLine:
    site: str
    line: OcrLine
    addressed: bool


@dataclass(frozen=True)
class BrowserBand:
    content_top: int = 0
    probe_from: int = 0
    site: str = ""

    def __bool__(self) -> bool:
        return self.content_top > 0

    def holds(self, line: OcrLine, top: int) -> bool:
        return bool(self) and line.bbox.center[1] < top + self.content_top


NO_BAND = BrowserBand()


# PR-037
def site_of(text: str) -> tuple[str, bool] | None:
    for scheme in _PAGE_SCHEME_RE.finditer(text):
        if scheme.group(1).lower() not in _WEB_SCHEMES:
            return f"{scheme.group(1).lower()}://{scheme.group(2).lower()}", True
    found = _HOST_RE.search(text)
    if found is None:
        return None
    host = found.group(1).lower().removeprefix(_WWW)
    port = found.group(2)
    rest = text[found.end() :]
    addressed = bool(port) or rest.startswith(("/", "?", "#")) or "://" in found.group(0)
    return (f"{host}:{port}" if port else host), addressed


# PR-037
def address_line(lines: Sequence[OcrLine], limit: int) -> SiteLine | None:
    found: list[SiteLine] = []
    for line in lines:
        if line.bbox.center[1] >= limit:
            continue
        parsed = site_of(line.text)
        if parsed is not None:
            found.append(SiteLine(site=parsed[0], line=line, addressed=parsed[1]))
    if not found:
        return None
    return max(found, key=lambda item: (item.addressed, item.line.bbox.w, -item.line.bbox.y))


def _row_coverage(image: np.ndarray, start: int, stop: int) -> np.ndarray:
    rows = image[max(0, start - 1) : stop].astype(np.int16)
    if len(rows) < 2:
        return np.zeros(0, dtype=np.float32)
    steps = np.abs(rows[1:] - rows[:-1]).max(axis=2) >= BAND_STEP_RGB
    return steps.mean(axis=1)


# PR-038
def separator_below(image: np.ndarray, start: int, reach: int) -> int | None:
    stop = min(int(image.shape[0]), start + reach)
    first = max(1, start)
    coverage = _row_coverage(image, first, stop)
    hits = np.flatnonzero(coverage >= BAND_SEPARATOR_COVERAGE)
    if not len(hits):
        return None
    run = int(hits[0])
    while run + 1 < len(coverage) and coverage[run + 1] >= BAND_SEPARATOR_COVERAGE:
        run += 1
    return first + run


# PR-038
def browser_band(image: np.ndarray, lines: Sequence[OcrLine], scale: float) -> BrowserBand:
    address = address_line(lines, round(BAND_SEARCH_PT * scale))
    if address is None:
        return NO_BAND
    probe_from = address.line.bbox.y + address.line.bbox.h
    top = separator_below(image, probe_from, round(BAND_SEPARATOR_REACH_PT * scale))
    if top is None:
        return NO_BAND
    return BrowserBand(content_top=top, probe_from=probe_from, site=address.site)


# PR-038
def band_still_holds(image: np.ndarray, band: BrowserBand, scale: float) -> bool:
    if not band:
        return True
    found = separator_below(image, band.probe_from, round(BAND_SEPARATOR_REACH_PT * scale))
    return found == band.content_top


def _side_by_side(one: BBox, other: BBox) -> bool:
    return one.x < other.x + other.w and other.x < one.x + one.w


# PR-058
def address_typed(
    lines: Sequence[OcrLine], scale: float, *, titled: bool, band: BrowserBand
) -> bool:
    limit = round(BAND_SEARCH_PT * scale)
    sited: list[SiteLine] = []
    for line in lines:
        parsed = site_of(line.text) if line.bbox.center[1] < limit else None
        if parsed is not None:
            sited.append(SiteLine(site=parsed[0], line=line, addressed=parsed[1]))
    if not sited:
        return False
    if not titled:
        return True
    if band:
        return False
    reach = round(BAND_SEPARATOR_REACH_PT * scale)
    for field in sited:
        if not field.addressed:
            continue
        top = field.line.bbox
        for echo in sited:
            below = echo.line.bbox
            if (
                echo.site == field.site
                and echo is not field
                and 0 <= below.y - (top.y + top.h) <= reach
                and _side_by_side(top, below)
            ):
                return True
    return False
