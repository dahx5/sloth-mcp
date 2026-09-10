from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from choto.graph.db_models import Element
from choto.graph.icons import GlyphLabel, GlyphPosition, UnlabeledGlyph, decode_icon_png
from choto.graph.repository import GraphRepository
from choto.log import get_logger
from choto.models import BBox, ElementKind, IconLabelSource
from choto.vision.iconsheet import SheetEntry, SheetResult, build_icon_sheet

_log = get_logger(__name__)

MAX_GLYPHS_PER_CALL = 60

_MAX_NEIGHBOURS = 3

_MAX_EXTRA_POSITIONS = 2

_MAX_CONTEXT_CHARS = 28

_ELLIPSIS = "…"

_TOP_BAND = 0.15
_BOTTOM_BAND = 0.85
_LEFT_BAND = 0.2
_RIGHT_BAND = 0.8

_MIN_CONFIDENCE = 0.5


class AnnotationError(ValueError): ...


class _ScreenElements:
    def __init__(self, repo: GraphRepository) -> None:
        self._repo = repo
        self._by_screen: dict[int, list[Element]] = {}

    def of(self, screen_id: int) -> list[Element]:
        found = self._by_screen.get(screen_id)
        if found is None:
            found = self._repo.get_elements(screen_id)
            self._by_screen[screen_id] = found
        return found


@dataclass(frozen=True)
class AnnotationBatch:
    app_name: str
    sheets: tuple[SheetResult, ...]
    glyph_count: int
    unlabeled_in_app: int
    unlabeled_elsewhere: int
    glyphs_in_app: int
    glyphs_everywhere: int

    @property
    def empty(self) -> bool:
        return not self.sheets


@dataclass(frozen=True)
class LabelOutcome:
    app_name: str
    glyphs_labeled: int
    elements_updated: int
    remaining_unlabeled: int


class IconAnnotator:
    def __init__(self, repo: GraphRepository, max_glyphs: int = MAX_GLYPHS_PER_CALL) -> None:
        if max_glyphs < 1:
            raise ValueError(f"max_glyphs must be positive, got {max_glyphs}.")
        self._repo = repo
        self._max_glyphs = max_glyphs

    def build(self, app_name: str | None = None) -> AnnotationBatch:
        target = app_name if app_name is not None else self._busiest_app()
        if target is None:
            everywhere_total = self._repo.icon_counts(None).glyphs_total
            return AnnotationBatch(
                app_name="",
                sheets=(),
                glyph_count=0,
                unlabeled_in_app=0,
                unlabeled_elsewhere=0,
                glyphs_in_app=0,
                glyphs_everywhere=everywhere_total,
            )

        glyphs = self._repo.unlabeled_glyphs(app_name=target, limit=self._max_glyphs)
        counts = self._repo.icon_counts(target)
        all_counts = self._repo.icon_counts(None)
        in_app = counts.glyphs_total - counts.glyphs_labeled
        everywhere = all_counts.glyphs_total - all_counts.glyphs_labeled
        if not glyphs:
            return AnnotationBatch(
                app_name=target,
                sheets=(),
                glyph_count=0,
                unlabeled_in_app=0,
                unlabeled_elsewhere=everywhere,
                glyphs_in_app=counts.glyphs_total,
                glyphs_everywhere=all_counts.glyphs_total,
            )

        elements = _ScreenElements(self._repo)
        entries = [
            SheetEntry(
                crop=decode_icon_png(glyph.crop_png),
                key=glyph.phash,
                context_lines=self._context_lines(glyph, elements),
            )
            for glyph in glyphs
        ]
        sheets = build_icon_sheet(entries)
        _log.info(
            "icons.annotation_built",
            app_name=target,
            glyphs=len(glyphs),
            sheets=len(sheets),
            unlabeled_in_app=in_app,
        )
        return AnnotationBatch(
            app_name=target,
            sheets=tuple(sheets),
            glyph_count=len(glyphs),
            unlabeled_in_app=in_app,
            unlabeled_elsewhere=everywhere - in_app,
            glyphs_in_app=counts.glyphs_total,
            glyphs_everywhere=all_counts.glyphs_total,
        )

    def _busiest_app(self) -> str | None:
        found = self._repo.unlabeled_glyphs(limit=1)
        return found[0].app_name if found else None

    def _unlabeled_count(self, app_name: str | None) -> int:
        counts = self._repo.icon_counts(app_name)
        return counts.glyphs_total - counts.glyphs_labeled

    def _context_lines(self, glyph: UnlabeledGlyph, elements: _ScreenElements) -> list[str]:
        if not glyph.positions:
            return [
                f'"{glyph.app_name}" · no position on the map any more '
                "(the windows that drew it were forgotten; the picture was kept)"
            ]

        first = glyph.positions[0]
        found = elements.of(first.screen_id)
        lines = [
            f'"{glyph.app_name}" · {_titled(first.window_title)} · '
            f"{_zone(first.bbox, _content_box(found))} · "
            f"{first.bbox.w}x{first.bbox.h}px"
        ]
        neighbours = _neighbours(first.bbox, found, _MAX_NEIGHBOURS)
        if neighbours:
            lines.append("near: " + ", ".join(f'"{text}"' for text in neighbours))
        extra = self._other_positions(glyph.positions[1:], elements)
        if extra:
            lines.append(extra)
        return lines

    def _other_positions(
        self, positions: Sequence[GlyphPosition], elements: _ScreenElements
    ) -> str:
        if not positions:
            return ""
        named = [
            f"{_titled(position.window_title)} "
            f"{_zone(position.bbox, _content_box(elements.of(position.screen_id)))}"
            for position in positions[:_MAX_EXTRA_POSITIONS]
        ]
        hidden = len(positions) - len(named)
        tail = f", +{hidden} more" if hidden > 0 else ""
        return (
            f"also drawn in: {'; '.join(named)}{tail} — one name covers them all, "
            "so name what the control does, not what this screen lists"
        )

    def apply(self, labels: Mapping[str, str], app_name: str) -> LabelOutcome:
        if not app_name.strip():
            raise AnnotationError(
                "submit_icon_labels needs app_name: an icon glyph is keyed per application, "
                "and the same drawing means different things in different programs. Pass the "
                "app_name that annotate_icons reported for the sheet."
            )
        if not labels:
            raise AnnotationError(
                'submit_icon_labels was given no labels. Pass {"<glyph key>": "<what the '
                'control is called>"} using the keys annotate_icons listed for the cells.'
            )

        batch = [self._label(app_name, key, text) for key, text in labels.items()]
        try:
            result = self._repo.apply_glyph_labels(batch)
        except KeyError as exc:
            raise AnnotationError(
                f"{exc.args[0]} Nothing was written. The keys come from the legend of the "
                "most recent annotate_icons call for this application — call it again to "
                "get the current ones (a glyph may have been swept in the meantime)."
            ) from exc

        remaining = self._unlabeled_count(app_name)
        _log.info(
            "icons.labels_submitted",
            app_name=app_name,
            glyphs=result.glyphs_labeled,
            elements=result.elements_updated,
            remaining=remaining,
        )
        return LabelOutcome(
            app_name=app_name,
            glyphs_labeled=result.glyphs_labeled,
            elements_updated=result.elements_updated,
            remaining_unlabeled=remaining,
        )

    @staticmethod
    def _label(app_name: str, key: str, text: str) -> GlyphLabel:
        try:
            return GlyphLabel(app_name=app_name, phash=key, label=text, source=IconLabelSource.LLM)
        except ValueError as exc:
            raise AnnotationError(f"label for glyph {key!r} rejected: {exc}") from exc


def _clip(text: str, limit: int = _MAX_CONTEXT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + _ELLIPSIS


def _titled(title: str) -> str:
    return f'"{_clip(title)}"' if title.strip() else "untitled window"


def _content_box(elements: Sequence[Element]) -> BBox | None:
    boxes = [element for element in elements if element.w > 0 and element.h > 0]
    if not boxes:
        return None
    left = min(element.x for element in boxes)
    top = min(element.y for element in boxes)
    right = max(element.x + element.w for element in boxes)
    bottom = max(element.y + element.h for element in boxes)
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def _zone(bbox: BBox, content: BBox | None) -> str:
    if content is None or content.w <= 0 or content.h <= 0:
        return "position unknown"
    cx, cy = bbox.center
    rel_x = (cx - content.x) / content.w
    rel_y = (cy - content.y) / content.h

    vertical = "top" if rel_y < _TOP_BAND else "bottom" if rel_y > _BOTTOM_BAND else "middle"
    horizontal = "left" if rel_x < _LEFT_BAND else "right" if rel_x > _RIGHT_BAND else "center"
    where = (
        "center" if (vertical, horizontal) == ("middle", "center") else f"{vertical}-{horizontal}"
    )
    if vertical == "top":
        return f"{where} (toolbar strip)"
    if horizontal == "left" and vertical == "middle":
        return f"{where} (sidebar column)"
    return where


def _neighbours(bbox: BBox, elements: Sequence[Element], limit: int) -> list[str]:
    usable = [
        element
        for element in elements
        if element.kind is ElementKind.TEXT
        and element.text.strip()
        and element.ocr_confidence >= _MIN_CONFIDENCE
    ]
    ranked = sorted(usable, key=lambda element: _rect_distance(bbox, element))
    texts: list[str] = []
    seen: set[str] = set()
    for element in ranked:
        text = _clip(element.text)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        if len(texts) == limit:
            break
    return texts


def _rect_distance(bbox: BBox, element: Element) -> float:
    dx = max(bbox.x - (element.x + element.w), element.x - (bbox.x + bbox.w), 0)
    dy = max(bbox.y - (element.y + element.h), element.y - (bbox.y + bbox.h), 0)
    return float(dx * dx + dy * dy) ** 0.5
