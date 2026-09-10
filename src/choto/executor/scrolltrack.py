from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from choto.graph.map_models import (
    OffscreenHint,
    ScrollEdge,
    ScrollExtent,
    ScrollPlacement,
    ScrollRole,
)
from choto.graph.repository import normalize_text
from choto.log import get_logger
from choto.models import BBox, OcrLine

_log = get_logger(__name__)

SHIFT_TOLERANCE_PX = 3

MIN_ANCHORS = 3

MIN_SHIFT_PX = 4

MIN_AGREEMENT = 0.6


@dataclass
class _Sighting:
    line: OcrLine
    offset: int
    doc_y: int | None
    role: ScrollRole | None


def _once_each(lines: Iterable[OcrLine]) -> dict[str, OcrLine]:
    seen: dict[str, OcrLine] = {}
    twice: set[str] = set()
    for line in lines:
        key = normalize_text(line.text)
        if not key:
            continue
        if key in seen:
            twice.add(key)
            continue
        seen[key] = line
    for key in twice:
        del seen[key]
    return seen


def _densest(votes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    heaviest = 0
    for pivot, _ in votes:
        near = [vote for vote in votes if abs(vote[0] - pivot) <= SHIFT_TOLERANCE_PX]
        weight = _weight(near)
        if weight > heaviest:
            best, heaviest = near, weight
    return best


def _weight(votes: Sequence[tuple[int, int]]) -> int:
    return sum(weight for _, weight in votes)


# PR-030
def _agreed(votes: Sequence[tuple[int, int]]) -> int | None:
    if len(votes) < MIN_ANCHORS:
        return None
    best = _densest(votes)
    if len(best) < MIN_ANCHORS or _weight(best) < MIN_AGREEMENT * _weight(votes):
        return None
    return sorted(value for value, _ in best)[len(best) // 2]


# PR-025
class ScrollPass:
    def __init__(self, viewport: BBox) -> None:
        if viewport.h <= 0:
            raise ValueError(f"A scrolled viewport must have height, got {viewport.h}.")
        self._viewport = viewport
        self._known: dict[str, _Sighting] = {}
        self._muddled: set[str] = set()
        self._current: set[str] = set()
        self._offset = 0
        self._offsets = [0]
        self._pages = 0
        self._seen = 0
        self._at_rest = False

    @property
    def at_rest(self) -> bool:
        return self._at_rest

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def pages(self) -> int:
        return self._pages

    @property
    def seen(self) -> int:
        return self._seen

    def page(self, lines: Sequence[OcrLine]) -> bool:
        seen = _once_each(lines)
        self._seen += 1
        if self._pages == 0:
            self._pages = 1
            self._at_rest = True
            self._absorb(seen, 0)
            return True
        offset = self._locate(seen)
        self._at_rest = offset is not None
        if offset is None:
            _log.info(
                "scroll_pass.unlocated",
                pages=self._pages,
                seen=self._seen,
                offset=self._offset,
                known=len(self._known),
                matched=sum(1 for key in seen if key in self._known),
            )
            return False
        self._pages += 1
        self._offset = offset
        self._offsets.append(offset)
        self._absorb(seen, offset)
        return True

    def extent(self, ends: Collection[ScrollEdge] = ()) -> ScrollExtent | None:
        if not self._at_rest or min(self._offsets) == max(self._offsets):
            return None
        return ScrollExtent(
            covered_top=min(self._offsets) + self._viewport.y,
            covered_bottom=max(self._offsets) + self._viewport.y + self._viewport.h,
            viewport_h=self._viewport.h,
            ends=frozenset(ends),
        )

    # PR-026
    def placements(self) -> tuple[ScrollPlacement, ...]:
        if not self._at_rest:
            return ()
        top = self._viewport.y + self._offset
        bottom = top + self._viewport.h
        placed: list[ScrollPlacement] = []
        for key, known in self._known.items():
            if known.role is None:
                continue
            if known.role is ScrollRole.PINNED:
                placed.append(
                    ScrollPlacement(
                        text=known.line.text,
                        bbox=known.line.bbox,
                        confidence=known.line.confidence,
                        role=ScrollRole.PINNED,
                    )
                )
                continue
            doc_y = known.doc_y
            if doc_y is None:
                continue
            hint = None
            if key not in self._current:
                hint = self._past_the_edge(doc_y, known.line.bbox.h, top, bottom)
                if hint is None:
                    continue
            placed.append(
                ScrollPlacement(
                    text=known.line.text,
                    bbox=known.line.bbox,
                    confidence=known.line.confidence,
                    role=ScrollRole.DOCUMENT,
                    doc_x=known.line.bbox.x,
                    doc_y=doc_y,
                    offscreen=hint,
                )
            )
        return tuple(placed)

    @staticmethod
    def _past_the_edge(doc_y: int, height: int, top: int, bottom: int) -> OffscreenHint | None:
        above = top - doc_y
        below = doc_y + height - bottom
        if above > 0 and above >= below:
            return OffscreenHint(side=ScrollEdge.TOP, distance=above)
        if below > 0:
            return OffscreenHint(side=ScrollEdge.BOTTOM, distance=below)
        return None

    def _locate(self, seen: Mapping[str, OcrLine]) -> int | None:
        anchored: list[tuple[int, int]] = []
        loose: list[tuple[int, int]] = []
        for key, line in seen.items():
            known = self._known.get(key)
            if known is None or known.doc_y is None:
                continue
            # PR-030
            vote = (known.doc_y - line.bbox.y, max(line.bbox.w, 1))
            if known.role is ScrollRole.DOCUMENT:
                anchored.append(vote)
            elif known.role is None:
                loose.append(vote)
        offset = _agreed(anchored)
        if offset is not None:
            return offset
        return _agreed([vote for vote in loose if abs(vote[0] - self._offset) >= MIN_SHIFT_PX])

    def _absorb(self, seen: Mapping[str, OcrLine], offset: int) -> None:
        self._current = set(seen)
        for key, line in seen.items():
            if key in self._muddled:
                continue
            known = self._known.get(key)
            if known is None:
                self._known[key] = _Sighting(
                    line=line, offset=offset, doc_y=line.bbox.y + offset, role=None
                )
                continue
            if known.offset == offset:
                known.line = line
                continue
            rides = (
                known.doc_y is not None
                and abs(line.bbox.y + offset - known.doc_y) <= SHIFT_TOLERANCE_PX
            )
            stays = abs(line.bbox.y - known.line.bbox.y) <= SHIFT_TOLERANCE_PX
            if rides and not stays:
                known.role = ScrollRole.DOCUMENT
            elif stays and not rides:
                known.role = ScrollRole.PINNED
                known.doc_y = None
            else:
                del self._known[key]
                self._muddled.add(key)
                continue
            known.line = line
            known.offset = offset
