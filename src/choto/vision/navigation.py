from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from choto.graph import normalize_text
from choto.models import BBox, OcrLine

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

    def __bool__(self) -> bool:
        return bool(self.slots)

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(slot.text for slot in self.slots)

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


# PR-023
def derive_navigation(places: Sequence[LabelPlace]) -> Navigation:
    windows = {place.screen_id for place in places}
    if len(windows) < NAV_MIN_WINDOWS:
        return Navigation(windows=len(windows))
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
    )
