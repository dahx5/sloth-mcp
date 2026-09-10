from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from choto.models import BBox, ElementKind, OcrLine

MIN_LABEL_CONFIDENCE = 0.5

MIN_LABEL_CHARS = 2


class ScrollEdge(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class ScrollRole(str, Enum):
    DOCUMENT = "document"
    PINNED = "pinned"


@dataclass(frozen=True)
class OffscreenHint:
    side: ScrollEdge
    distance: int

    def __post_init__(self) -> None:
        if not isinstance(self.side, ScrollEdge):
            raise TypeError(f"side must be a ScrollEdge, got {type(self.side).__name__}.")
        if self.distance < 0:
            raise ValueError(
                f"An off-screen distance must not be negative, got {self.distance}: "
                "an element that far away is in view."
            )


@dataclass(frozen=True)
class ScrollExtent:
    covered_top: int
    covered_bottom: int
    viewport_h: int
    ends: frozenset[ScrollEdge] = frozenset()

    def __post_init__(self) -> None:
        if self.viewport_h <= 0:
            raise ValueError(f"A scrollable region must have height, got {self.viewport_h}.")
        if self.length < self.viewport_h:
            raise ValueError(
                f"A covered stretch of {self.length} px is shorter than the {self.viewport_h} px "
                "viewport it was read through: the first reading alone covers a viewport."
            )
        for edge in self.ends:
            if not isinstance(edge, ScrollEdge):
                raise TypeError(f"An end must be a ScrollEdge, got {type(edge).__name__}.")
        object.__setattr__(self, "ends", frozenset(self.ends))

    @property
    def length(self) -> int:
        return self.covered_bottom - self.covered_top

    @property
    def viewports(self) -> float:
        return self.length / self.viewport_h

    @property
    def unreached(self) -> tuple[ScrollEdge, ...]:
        return tuple(edge for edge in (ScrollEdge.TOP, ScrollEdge.BOTTOM) if edge not in self.ends)

    @property
    def exhausted(self) -> bool:
        return not self.unreached


# PR-026
@dataclass(frozen=True)
class ScrollPlacement:
    text: str
    bbox: BBox
    confidence: float
    role: ScrollRole
    doc_x: int | None = None
    doc_y: int | None = None
    offscreen: OffscreenHint | None = None

    def __post_init__(self) -> None:
        placed = self.doc_x is not None and self.doc_y is not None
        if self.role is ScrollRole.PINNED and (placed or self.offscreen is not None):
            raise ValueError(
                f"{self.text!r} is pinned to the window, so it has no place in the document "
                "and no edge it drove past."
            )
        if self.role is ScrollRole.DOCUMENT and not placed:
            raise ValueError(
                f"{self.text!r} rides with the content but carries no document coordinates: "
                "that is the one thing a document element is for."
            )


@dataclass(frozen=True)
class AppSummary:
    app_name: str
    window_count: int
    visit_count: int
    visits_cut: int
    last_seen_at: datetime


@dataclass(frozen=True)
class WindowSummary:
    screen_id: int
    app_name: str
    window_title: str
    last_seen_at: datetime
    is_stale: bool
    element_count: int
    visit_count: int
    width: int
    height: int
    scroll: ScrollExtent | None = None


@dataclass(frozen=True)
class Label:
    text: str
    kind: ElementKind
    confidence: float
    bbox: BBox
    seen_at: datetime
    offscreen: OffscreenHint | None = None

    @property
    def targetable(self) -> bool:
        stripped = self.text.strip()
        if not stripped:
            return False
        if self.confidence < MIN_LABEL_CONFIDENCE:
            return False
        if len(stripped) >= MIN_LABEL_CHARS:
            return True
        return stripped.isalnum()

    @property
    def line(self) -> OcrLine:
        return OcrLine(
            text=self.text,
            bbox=self.bbox,
            confidence=min(max(self.confidence, 0.0), 1.0),
        )


@dataclass(frozen=True)
class Transition:
    transition_id: int
    from_window: int
    to_window: int
    action: dict[str, Any]
    success_count: int
    fail_count: int
