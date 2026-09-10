from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from choto.graph.db_models import (
    AppJournal,
    Edge,
    Element,
    ExecutionRun,
    IconGlyph,
    IconLabelScreen,
    Screen,
    ScreenVisit,
    ScrollRole,
)
from choto.graph.db_models import (
    Mission as MissionRow,
)
from choto.graph.db_models import (
    MissionItem as MissionItemRow,
)
from choto.graph.icons import (
    ICON_ELEMENT_CONFIDENCE,
    GlyphLabel,
    GlyphLabelResult,
    GlyphPosition,
    IconCounts,
    IconRecord,
    IconSaveResult,
    UnlabeledGlyph,
    icon_element_text,
)
from choto.graph.map_models import (
    AppSummary,
    Label,
    OffscreenHint,
    ScrollEdge,
    ScrollExtent,
    ScrollPlacement,
    Transition,
    WindowSummary,
)
from choto.graph.missions import (
    ActiveMissionExists,
    Mission,
    MissionItem,
    MissionItemDraft,
    MissionItemStatus,
    MissionStatus,
    WorldFingerprint,
    normalize_recipe,
    validate_mark,
)
from choto.log import get_logger
from choto.models import BBox, ElementKind, IconLabelSource, OcrLine, Plan, RunStatus, ScreenState

if TYPE_CHECKING:
    from choto.vision.navigation import LabelPlace

_log = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ScreenSighting:
    screen_id: int
    app_name: str
    window_title: str
    last_seen_at: datetime
    visit_count: int
    last_visit_at: datetime | None


@dataclass(frozen=True)
class JournalSpan:
    app_name: str
    retained: int
    cut: int
    first_seen_at: datetime
    oldest_retained_at: datetime | None

    @property
    def total(self) -> int:
        return self.retained + self.cut


@dataclass(frozen=True)
class VisitRotation:
    cut: tuple[tuple[str, int], ...] = ()

    @property
    def total(self) -> int:
        return sum(count for _app, count in self.cut)

    def __bool__(self) -> bool:
        return bool(self.cut)


@dataclass(frozen=True)
class GraphCounts:
    screens: int
    elements: int
    edges: int
    visits: int
    orphan_visits: int


def _decode_ends(stored: Any) -> frozenset[ScrollEdge]:
    if stored is None:
        return frozenset()
    if not isinstance(stored, list):
        raise ValueError(f"Reached edges must be stored as a list, got {type(stored).__name__}.")
    edges = set()
    for value in stored:
        try:
            edges.add(ScrollEdge(value))
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a scrollable edge: {exc}") from exc
    return frozenset(edges)


def normalize_text(text: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return collapsed.casefold()


def hamming_hex(a: str, b: str) -> int:
    try:
        xor = int(a, 16) ^ int(b, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex hash: {a!r} / {b!r}") from exc
    return xor.bit_count()


def _stored_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _scroll_extent(
    covered_top: int | None, covered_bottom: int | None, viewport_h: int | None, ends: Any
) -> ScrollExtent | None:
    present = [value is not None for value in (covered_top, covered_bottom, viewport_h)]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "A node's scrolling extent is three numbers written together, got "
            f"top={covered_top}, bottom={covered_bottom}, viewport={viewport_h}."
        )
    return ScrollExtent(
        covered_top=int(covered_top or 0),
        covered_bottom=int(covered_bottom or 0),
        viewport_h=int(viewport_h or 0),
        ends=_decode_ends(ends),
    )


def _icon_element(screen_id: int, record: IconRecord, label: str) -> Element:
    return Element(
        screen_id=screen_id,
        kind=ElementKind.ICON,
        icon_phash=record.phash,
        text=label,
        norm_text=normalize_text(label),
        x=record.bbox.x,
        y=record.bbox.y,
        w=record.bbox.w,
        h=record.bbox.h,
        ocr_confidence=ICON_ELEMENT_CONFIDENCE,
    )


# PR-026
def _offscreen_element(screen_id: int, placement: ScrollPlacement) -> Element:
    if placement.offscreen is None:
        raise ValueError(f"{placement.text!r} is on screen; it has screen coordinates to keep.")
    return Element(
        screen_id=screen_id,
        kind=ElementKind.TEXT,
        text=placement.text,
        norm_text=normalize_text(placement.text),
        x=None,
        y=None,
        w=placement.bbox.w,
        h=placement.bbox.h,
        ocr_confidence=placement.confidence,
        scroll_role=placement.role,
        doc_x=placement.doc_x,
        doc_y=placement.doc_y,
        offscreen_side=placement.offscreen.side,
        offscreen_distance=placement.offscreen.distance,
    )


def _window_query(
    *,
    app_name: str | None,
    title_query: str | None,
    screen_ids: Sequence[int] | None,
) -> Any:
    element_count = (
        select(func.count(Element.id))
        .where(Element.screen_id == Screen.id)
        .scalar_subquery()
        .label("element_count")
    )
    visit_count = (
        select(func.count(ScreenVisit.id))
        .where(ScreenVisit.screen_id == Screen.id)
        .scalar_subquery()
        .label("visit_count")
    )
    stmt = select(
        Screen.id,
        Screen.app_name,
        Screen.window_title,
        Screen.last_seen_at,
        Screen.is_stale,
        element_count,
        visit_count,
        Screen.width,
        Screen.height,
        Screen.scroll_covered_top,
        Screen.scroll_covered_bottom,
        Screen.scroll_viewport_h,
        Screen.scroll_ends,
    )
    if app_name is not None:
        stmt = stmt.where(func.lower(Screen.app_name) == app_name.lower())
    if title_query is not None:
        stmt = stmt.where(
            func.lower(Screen.window_title).contains(title_query.lower(), autoescape=True)
        )
    if screen_ids is not None:
        stmt = stmt.where(Screen.id.in_(list(screen_ids)))
    return stmt.order_by(Screen.last_seen_at.desc(), Screen.id)


def _window_summary(row: Any) -> WindowSummary:
    return WindowSummary(
        screen_id=int(row.id),
        app_name=str(row.app_name),
        window_title=str(row.window_title),
        last_seen_at=row.last_seen_at,
        is_stale=bool(row.is_stale),
        element_count=int(row.element_count),
        visit_count=int(row.visit_count),
        width=int(row.width),
        height=int(row.height),
        scroll=_scroll_extent(
            row.scroll_covered_top,
            row.scroll_covered_bottom,
            row.scroll_viewport_h,
            row.scroll_ends,
        ),
    )


def _label_value(element: Element) -> Label:
    x = element.x if element.x is not None else element.doc_x
    y = element.y if element.y is not None else element.doc_y
    return Label(
        text=element.text,
        kind=element.kind,
        confidence=element.ocr_confidence,
        bbox=BBox(x=int(x or 0), y=int(y or 0), w=max(element.w, 0), h=max(element.h, 0)),
        seen_at=_stored_utc(element.updated_at),
        offscreen=None
        if element.offscreen_side is None
        else OffscreenHint(
            side=element.offscreen_side, distance=int(element.offscreen_distance or 0)
        ),
    )


def _transition_value(edge: Edge) -> Transition:
    return Transition(
        transition_id=int(edge.id),
        from_window=int(edge.from_screen),
        to_window=int(edge.to_screen),
        action=dict(edge.action),
        success_count=int(edge.success_count),
        fail_count=int(edge.fail_count),
    )


def _forget_scroll_extent(screen: Screen) -> None:
    screen.scroll_covered_top = None
    screen.scroll_covered_bottom = None
    screen.scroll_viewport_h = None
    screen.scroll_ends = []


def _nearest_screens(session: Session, app_name: str, phash: str) -> list[tuple[int, Any]]:
    candidates = session.execute(
        select(Screen.id, Screen.phash, Screen.anchors).where(Screen.app_name == app_name)
    ).all()
    return sorted(
        ((hamming_hex(phash, candidate.phash), candidate) for candidate in candidates),
        key=lambda pair: (pair[0], pair[1].id),
    )


def _detached_screen(session: Session, screen_id: int) -> Screen | None:
    screen = session.get(Screen, screen_id)
    if screen is not None:
        session.expunge(screen)
    return screen


def _homed_glyph_ids(session: Session, screen_id: int, glyph_ids: set[int]) -> set[int]:
    if not glyph_ids:
        return set()
    return set(
        session.scalars(
            select(IconLabelScreen.glyph_id).where(
                IconLabelScreen.screen_id == screen_id,
                IconLabelScreen.glyph_id.in_(glyph_ids),
            )
        ).all()
    )


def _mission_item_value(row: MissionItemRow) -> MissionItem:
    recipe = row.recipe_json
    fingerprint = row.fingerprint_json
    return MissionItem(
        item_id=int(row.id),
        seq=int(row.seq),
        title=row.title,
        intent=row.intent,
        acceptance=row.acceptance,
        status=row.status,
        fail_reason=row.fail_reason,
        recipe=None if recipe is None else tuple(dict(step) for step in recipe),
        fingerprint=None if fingerprint is None else WorldFingerprint.from_json(fingerprint),
        updated_at=_stored_utc(row.updated_at),
    )


def _mission_value(row: MissionRow) -> Mission:
    return Mission(
        mission_id=int(row.id),
        goal=row.goal,
        status=row.status,
        items=tuple(_mission_item_value(item) for item in row.items),
        created_at=_stored_utc(row.created_at),
        updated_at=_stored_utc(row.updated_at),
    )


class GraphRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def upsert_screen(
        self, state: ScreenState, anchors: list[str], *, glyph_scope: str, window_title: str
    ) -> Screen:
        now = datetime.now(UTC)
        with self._session_factory() as session:
            screen = session.scalar(
                select(Screen).where(Screen.phash == state.phash, Screen.app_name == state.app_name)
            )
            if screen is None:
                screen = Screen(
                    app_name=state.app_name,
                    phash=state.phash,
                    window_title=window_title,
                    anchors=list(anchors),
                    width=state.width,
                    height=state.height,
                    is_stale=False,
                    glyph_scope=glyph_scope,
                    last_seen_at=now,
                )
                session.add(screen)
                _log.info(
                    "screen.created",
                    phash=state.phash,
                    app_name=state.app_name,
                    window_title=window_title,
                )
            else:
                screen.window_title = window_title
                screen.anchors = list(anchors)
                screen.width = state.width
                screen.height = state.height
                screen.is_stale = False
                screen.glyph_scope = glyph_scope
                screen.last_seen_at = now
                _log.info("screen.refreshed", screen_id=screen.id, phash=state.phash)
            session.commit()
            session.refresh(screen)
            session.expunge(screen)
            return screen

    # PR-027
    def get_screen(self, screen_id: int) -> Screen | None:
        with self._session_factory() as session:
            return _detached_screen(session, screen_id)

    def find_screen_by_phash(
        self,
        phash: str,
        max_hamming: int,
        *,
        app_name: str,
        anchors: Sequence[str] | None = None,
        navigation: Collection[str] = (),
    ) -> Screen | None:
        from choto.vision.anchors import anchors_agree, anchors_confirm

        with self._session_factory() as session:
            for distance, candidate in _nearest_screens(session, app_name, phash):
                # PR-024
                stored_anchors = [
                    text
                    for text in candidate.anchors
                    if normalize_text(str(text)) not in navigation
                ]
                # PR-004
                if anchors is not None and anchors_agree(stored_anchors, anchors):
                    if distance > max_hamming:
                        _log.info(
                            "screen.anchors_confirmed",
                            screen_id=candidate.id,
                            app_name=app_name,
                            distance=distance,
                            stored_anchors=stored_anchors,
                            fresh_anchors=list(anchors),
                        )
                    return _detached_screen(session, candidate.id)
                if distance > max_hamming:
                    continue
                confirmable = distance and anchors is not None
                if confirmable and not anchors_confirm(stored_anchors, anchors or ()):
                    _log.info(
                        "screen.anchors_refused",
                        screen_id=candidate.id,
                        app_name=app_name,
                        distance=distance,
                        stored_anchors=stored_anchors,
                        fresh_anchors=list(anchors or ()),
                    )
                    continue
                return _detached_screen(session, candidate.id)
            return None

    def app_summaries(self) -> list[AppSummary]:
        with self._session_factory() as session:
            visits = dict(
                session.execute(
                    select(ScreenVisit.app_name, func.count(ScreenVisit.id)).group_by(
                        ScreenVisit.app_name
                    )
                ).all()
            )
            rows = session.execute(
                select(
                    Screen.app_name,
                    func.count(Screen.id),
                    func.max(Screen.last_seen_at),
                ).group_by(Screen.app_name)
            ).all()
            cut = dict(session.execute(select(AppJournal.app_name, AppJournal.visits_cut)).all())
        summaries = [
            AppSummary(
                app_name=str(name),
                window_count=int(count),
                visit_count=int(visits.get(name, 0)),
                visits_cut=int(cut.get(name, 0)),
                last_seen_at=last_seen,
            )
            for name, count, last_seen in rows
        ]
        summaries.sort(key=lambda item: (item.last_seen_at, item.visit_count), reverse=True)
        return summaries

    def window_summaries(
        self,
        *,
        app_name: str | None = None,
        title_query: str | None = None,
        screen_ids: Sequence[int] | None = None,
    ) -> list[WindowSummary]:
        if screen_ids is not None and not screen_ids:
            return []
        stmt = _window_query(app_name=app_name, title_query=title_query, screen_ids=screen_ids)
        with self._session_factory() as session:
            rows = session.execute(stmt).all()
        return [_window_summary(row) for row in rows]

    def screen_sightings(self) -> list[ScreenSighting]:
        stmt = (
            select(
                Screen.id,
                Screen.app_name,
                Screen.window_title,
                Screen.last_seen_at,
                func.count(ScreenVisit.id),
                func.max(ScreenVisit.seen_at),
            )
            .outerjoin(ScreenVisit, ScreenVisit.screen_id == Screen.id)
            .group_by(Screen.id)
            .order_by(Screen.id)
        )
        with self._session_factory() as session:
            rows = session.execute(stmt).all()
        return [
            ScreenSighting(
                screen_id=int(screen_id),
                app_name=str(app),
                window_title=str(title),
                last_seen_at=last_seen,
                visit_count=int(visits),
                last_visit_at=last_visit,
            )
            for screen_id, app, title, last_seen, visits, last_visit in rows
        ]

    def delete_screens(self, screen_ids: Sequence[int]) -> int:
        if not screen_ids:
            return 0
        with self._session_factory() as session:
            result = session.execute(delete(Screen).where(Screen.id.in_(list(screen_ids))))
            session.commit()
            deleted = int(result.rowcount)
            _log.info("screens.deleted", count=deleted, screen_ids=list(screen_ids))
            return deleted

    def mark_stale(self, screen_id: int) -> None:
        with self._session_factory() as session:
            screen = session.get(Screen, screen_id)
            if screen is None:
                raise KeyError(f"Screen {screen_id} not found")
            screen.is_stale = True
            session.commit()
            _log.info("screen.marked_stale", screen_id=screen_id)

    def replace_elements(self, screen_id: int, lines: list[OcrLine]) -> None:
        fresh = {normalize_text(line.text) for line in lines if line.text.strip()}
        with self._session_factory() as session:
            screen = session.get(Screen, screen_id)
            if screen is None:
                raise KeyError(f"Screen {screen_id} not found")
            same_position = self._scroll_position_holds(session, screen, fresh)
            # PR-027
            held = self._scroll_roles(session, screen_id) if same_position else {}
            doomed = delete(Element).where(Element.screen_id == screen_id)
            if same_position:
                doomed = doomed.where(Element.offscreen_side.is_(None))
            else:
                _forget_scroll_extent(screen)
            session.execute(doomed)
            for line in lines:
                norm_text = normalize_text(line.text)
                role, doc_x, doc_y = held.get(norm_text, (None, None, None))
                session.add(
                    Element(
                        screen_id=screen_id,
                        kind=ElementKind.TEXT,
                        text=line.text,
                        norm_text=norm_text,
                        x=line.bbox.x,
                        y=line.bbox.y,
                        w=line.bbox.w,
                        h=line.bbox.h,
                        ocr_confidence=line.confidence,
                        scroll_role=role,
                        doc_x=doc_x,
                        doc_y=doc_y,
                    )
                )
            session.commit()
            _log.info(
                "elements.replaced",
                screen_id=screen_id,
                count=len(lines),
                scroll_kept=same_position,
            )

    # PR-025
    def record_scroll_pass(
        self, screen_id: int, extent: ScrollExtent, placements: Sequence[ScrollPlacement]
    ) -> None:
        with self._session_factory() as session:
            screen = session.get(Screen, screen_id)
            if screen is None:
                raise KeyError(f"Screen {screen_id} not found")
            session.execute(
                delete(Element).where(
                    Element.screen_id == screen_id, Element.offscreen_side.is_not(None)
                )
            )
            here = self._blank_scroll_roles(session, screen_id)
            marked = 0
            for placement in placements:
                key = normalize_text(placement.text)
                element = here.get(key)
                # PR-026
                if placement.offscreen is not None:
                    if key in here:
                        continue
                    session.add(_offscreen_element(screen_id, placement))
                    marked += 1
                    continue
                if element is None:
                    continue
                element.scroll_role = placement.role
                element.doc_x = placement.doc_x
                element.doc_y = placement.doc_y
                marked += 1
            screen.scroll_covered_top = extent.covered_top
            screen.scroll_covered_bottom = extent.covered_bottom
            screen.scroll_viewport_h = extent.viewport_h
            screen.scroll_ends = sorted(edge.value for edge in extent.ends)
            session.commit()
            _log.info(
                "screen.scroll_recorded",
                screen_id=screen_id,
                covered_top=extent.covered_top,
                covered_bottom=extent.covered_bottom,
                viewport_h=extent.viewport_h,
                ends=screen.scroll_ends,
                placements=len(placements),
                marked=marked,
            )

    # PR-027
    @staticmethod
    def _scroll_roles(
        session: Session, screen_id: int
    ) -> dict[str, tuple[ScrollRole | None, int | None, int | None]]:
        held: dict[str, tuple[ScrollRole | None, int | None, int | None]] = {}
        twice: set[str] = set()
        for element in session.scalars(
            select(Element).where(Element.screen_id == screen_id, Element.offscreen_side.is_(None))
        ):
            if element.norm_text in held:
                twice.add(element.norm_text)
                continue
            held[element.norm_text] = (element.scroll_role, element.doc_x, element.doc_y)
        for key in twice:
            del held[key]
        return held

    @staticmethod
    def _blank_scroll_roles(session: Session, screen_id: int) -> dict[str, Element | None]:
        here: dict[str, Element | None] = {}
        for element in session.scalars(select(Element).where(Element.screen_id == screen_id)):
            element.scroll_role = None
            element.doc_x = None
            element.doc_y = None
            here[element.norm_text] = None if element.norm_text in here else element
        session.flush()
        return here

    @staticmethod
    def _scroll_position_holds(session: Session, screen: Screen, fresh: set[str]) -> bool:
        if screen.scroll_viewport_h is None:
            return False
        anchored = session.scalars(
            select(Element.norm_text).where(
                Element.screen_id == screen.id,
                Element.scroll_role == ScrollRole.DOCUMENT,
                Element.offscreen_side.is_(None),
            )
        ).all()
        if not anchored:
            return False
        return all(text in fresh for text in anchored)

    def get_elements(self, screen_id: int, *, include_offscreen: bool = False) -> list[Element]:
        stmt = select(Element).where(Element.screen_id == screen_id)
        if not include_offscreen:
            stmt = stmt.where(Element.offscreen_side.is_(None))
        with self._session_factory() as session:
            elements = list(session.scalars(stmt.order_by(Element.id)).all())
            for element in elements:
                session.expunge(element)
            return elements

    # PR-023
    def app_label_places(self, app_name: str) -> list[LabelPlace]:
        from choto.vision.navigation import LabelPlace

        stmt = (
            select(Element.screen_id, Element.text, Element.x, Element.y, Element.w, Element.h)
            .join(Screen, Screen.id == Element.screen_id)
            .where(
                Screen.app_name == app_name,
                Element.kind == ElementKind.TEXT,
                Element.x.is_not(None),
                Element.y.is_not(None),
            )
        )
        with self._session_factory() as session:
            return [
                LabelPlace(
                    screen_id=int(screen_id),
                    text=text,
                    bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                )
                for screen_id, text, x, y, w, h in session.execute(stmt).all()
            ]

    def window_labels(
        self, screen_ids: Sequence[int], *, include_offscreen: bool = False
    ) -> dict[int, list[Label]]:
        wanted = list(dict.fromkeys(screen_ids))
        if not wanted:
            return {}
        stmt = select(Element).where(Element.screen_id.in_(wanted))
        if not include_offscreen:
            stmt = stmt.where(Element.offscreen_side.is_(None))
        grouped: dict[int, list[Label]] = {}
        with self._session_factory() as session:
            for element in session.scalars(stmt.order_by(Element.screen_id, Element.id)).all():
                grouped.setdefault(int(element.screen_id), []).append(_label_value(element))
        return grouped

    def save_icon_candidates(
        self, screen_id: int, app_name: str, candidates: Sequence[IconRecord]
    ) -> IconSaveResult:
        with self._session_factory() as session:
            screen = session.get(Screen, screen_id)
            if screen is None:
                raise KeyError(f"Screen {screen_id} not found")
            if screen.app_name != app_name:
                raise ValueError(
                    f"Screen {screen_id} belongs to {screen.app_name!r}, not {app_name!r}: "
                    "an icon glyph is keyed per application and must not cross that line."
                )

            session.execute(
                delete(Element).where(
                    Element.screen_id == screen_id,
                    Element.kind == ElementKind.ICON,
                    Element.offscreen_side.is_(None),
                )
            )

            wanted = {record.phash for record in candidates}
            glyphs: dict[str, IconGlyph] = {
                glyph.phash: glyph
                for glyph in session.scalars(
                    select(IconGlyph).where(
                        IconGlyph.app_name == app_name, IconGlyph.phash.in_(wanted)
                    )
                ).all()
            }

            home = _homed_glyph_ids(
                session, screen_id, {glyph.id for glyph in glyphs.values() if glyph.label}
            )

            created = 0
            named = 0
            for record in candidates:
                glyph = glyphs.get(record.phash)
                if glyph is None:
                    glyph = IconGlyph(
                        app_name=app_name,
                        phash=record.phash,
                        crop_png=record.crop_png,
                        label="",
                        label_source=IconLabelSource.UNLABELED,
                    )
                    session.add(glyph)
                    glyphs[record.phash] = glyph
                    created += 1
                label = icon_element_text(glyph.label, at_home=glyph.id in home)
                if label:
                    named += 1
                session.add(_icon_element(screen_id, record, label))
            session.commit()

        _log.info(
            "icons.saved",
            screen_id=screen_id,
            app_name=app_name,
            elements=len(candidates),
            new_glyphs=created,
            named=named,
        )
        return IconSaveResult(
            elements_created=len(candidates), glyphs_created=created, elements_named=named
        )

    def unlabeled_glyphs(
        self, app_name: str | None = None, limit: int | None = None
    ) -> list[UnlabeledGlyph]:
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be positive when given, got {limit}.")

        positions = (
            select(func.count(Element.id))
            .select_from(Element)
            .join(Screen, Screen.id == Element.screen_id)
            .where(
                Element.kind == ElementKind.ICON,
                Element.icon_phash == IconGlyph.phash,
                Screen.app_name == IconGlyph.app_name,
            )
            .scalar_subquery()
        )
        stmt = (
            select(IconGlyph).where(IconGlyph.label == "").order_by(positions.desc(), IconGlyph.id)
        )
        if app_name is not None:
            stmt = stmt.where(IconGlyph.app_name == app_name)
        if limit is not None:
            stmt = stmt.limit(limit)

        with self._session_factory() as session:
            glyphs = list(session.scalars(stmt).all())
            if not glyphs:
                return []
            found = self._glyph_positions(session, glyphs)
            return [
                UnlabeledGlyph(
                    glyph_id=int(glyph.id),
                    app_name=glyph.app_name,
                    phash=glyph.phash,
                    crop_png=glyph.crop_png,
                    positions=found.get((glyph.app_name, glyph.phash), ()),
                )
                for glyph in glyphs
            ]

    @staticmethod
    def _glyph_positions(
        session: Session, glyphs: Sequence[IconGlyph]
    ) -> dict[tuple[str, str], tuple[GlyphPosition, ...]]:
        rows = session.execute(
            select(
                Screen.app_name,
                Element.icon_phash,
                Element.screen_id,
                Screen.window_title,
                Element.x,
                Element.y,
                Element.w,
                Element.h,
            )
            .join(Screen, Screen.id == Element.screen_id)
            .where(
                Element.kind == ElementKind.ICON,
                Element.icon_phash.in_({glyph.phash for glyph in glyphs}),
                Screen.app_name.in_({glyph.app_name for glyph in glyphs}),
                Element.offscreen_side.is_(None),
            )
            .order_by(Element.screen_id, Element.y, Element.x)
        ).all()
        grouped: dict[tuple[str, str], list[GlyphPosition]] = {}
        for app, phash, screen_id, title, x, y, w, h in rows:
            grouped.setdefault((str(app), str(phash)), []).append(
                GlyphPosition(
                    screen_id=int(screen_id),
                    window_title=str(title),
                    bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    def apply_glyph_labels(self, labels: Sequence[GlyphLabel]) -> GlyphLabelResult:
        if not labels:
            return GlyphLabelResult(glyphs_labeled=0, elements_updated=0)

        keys = [(label.app_name, label.phash) for label in labels]
        if len(set(keys)) != len(keys):
            raise ValueError("A labelling batch must name each glyph at most once.")

        updated = 0
        with self._session_factory() as session:
            for label in labels:
                glyph = session.scalar(
                    select(IconGlyph).where(
                        IconGlyph.app_name == label.app_name, IconGlyph.phash == label.phash
                    )
                )
                if glyph is None:
                    raise KeyError(
                        f"Icon glyph {label.phash} of {label.app_name!r} not found; "
                        "the batch was not applied."
                    )
                glyph.label = label.text
                glyph.label_source = label.source
                home = self._record_label_home(session, glyph)
                updated += self._retag_drawings(session, label, home)
            session.commit()

        _log.info("icons.labeled", glyphs=len(labels), elements=updated)
        return GlyphLabelResult(glyphs_labeled=len(labels), elements_updated=updated)

    @staticmethod
    def _retag_drawings(session: Session, label: GlyphLabel, home: set[int]) -> int:
        updated = 0
        for at_home in (True, False):
            shown = icon_element_text(label.text, at_home=at_home)
            result = session.execute(
                update(Element)
                .where(
                    Element.kind == ElementKind.ICON,
                    Element.icon_phash == label.phash,
                    Element.screen_id.in_(
                        select(Screen.id).where(Screen.app_name == label.app_name)
                    ),
                    Element.screen_id.in_(home) if at_home else Element.screen_id.not_in(home),
                )
                .values(text=shown, norm_text=normalize_text(shown))
                .execution_options(synchronize_session=False)
            )
            updated += int(result.rowcount)
        return updated

    @staticmethod
    def _record_label_home(session: Session, glyph: IconGlyph) -> set[int]:
        home = set(
            session.scalars(
                select(Element.screen_id)
                .join(Screen, Screen.id == Element.screen_id)
                .where(
                    Element.kind == ElementKind.ICON,
                    Element.icon_phash == glyph.phash,
                    Element.offscreen_side.is_(None),
                    Screen.app_name == glyph.app_name,
                )
                .distinct()
            ).all()
        )
        session.execute(delete(IconLabelScreen).where(IconLabelScreen.glyph_id == glyph.id))
        if home:
            session.execute(
                insert(IconLabelScreen),
                [{"glyph_id": glyph.id, "screen_id": screen_id} for screen_id in sorted(home)],
            )
        return home

    def icon_counts(self, app_name: str | None = None) -> IconCounts:
        glyphs = select(func.count(IconGlyph.id))
        labeled = select(func.count(IconGlyph.id)).where(IconGlyph.label != "")
        elements = (
            select(func.count(Element.id))
            .select_from(Element)
            .join(Screen, Screen.id == Element.screen_id)
            .where(Element.kind == ElementKind.ICON)
        )
        if app_name is not None:
            glyphs = glyphs.where(IconGlyph.app_name == app_name)
            labeled = labeled.where(IconGlyph.app_name == app_name)
            elements = elements.where(Screen.app_name == app_name)
        with self._session_factory() as session:
            return IconCounts(
                glyphs_total=int(session.scalar(glyphs) or 0),
                glyphs_labeled=int(session.scalar(labeled) or 0),
                icon_elements=int(session.scalar(elements) or 0),
            )

    def orphan_unlabeled_glyphs(self, ignoring_screen_ids: Sequence[int] = ()) -> tuple[int, ...]:
        drawn = (
            select(Element.id)
            .join(Screen, Screen.id == Element.screen_id)
            .where(
                Element.kind == ElementKind.ICON,
                Element.icon_phash == IconGlyph.phash,
                Screen.app_name == IconGlyph.app_name,
            )
        )
        if ignoring_screen_ids:
            drawn = drawn.where(Screen.id.not_in(list(ignoring_screen_ids)))
        stmt = (
            select(IconGlyph.id)
            .where(IconGlyph.label == "", ~drawn.exists())
            .order_by(IconGlyph.id)
        )
        with self._session_factory() as session:
            return tuple(int(glyph_id) for glyph_id in session.scalars(stmt).all())

    def delete_icon_glyphs(self, glyph_ids: Sequence[int]) -> int:
        if not glyph_ids:
            return 0
        with self._session_factory() as session:
            result = session.execute(delete(IconGlyph).where(IconGlyph.id.in_(list(glyph_ids))))
            session.commit()
            deleted = int(result.rowcount)
            _log.info("icon_glyphs.deleted", count=deleted)
            return deleted

    def record_edge(self, from_id: int, action: dict, to_id: int, success: bool) -> Edge:
        with self._session_factory() as session:
            existing = session.scalars(
                select(Edge).where(Edge.from_screen == from_id, Edge.to_screen == to_id)
            ).all()
            edge = next((candidate for candidate in existing if candidate.action == action), None)
            if edge is None:
                edge = Edge(
                    from_screen=from_id,
                    to_screen=to_id,
                    action=dict(action),
                    success_count=1 if success else 0,
                    fail_count=0 if success else 1,
                )
                session.add(edge)
            elif success:
                edge.success_count += 1
            else:
                edge.fail_count += 1
            session.commit()
            session.refresh(edge)
            session.expunge(edge)
            return edge

    def transitions(self, *, from_screen: int | None = None) -> list[Transition]:
        stmt = select(Edge).order_by(Edge.id)
        if from_screen is not None:
            stmt = stmt.where(Edge.from_screen == from_screen)
        with self._session_factory() as session:
            return [_transition_value(edge) for edge in session.scalars(stmt).all()]

    def record_screen_visit(
        self, run_id: int, screen_id: int, seq: int, step_index: int, action: str
    ) -> ScreenVisit:
        if seq < 0 or step_index < 0:
            raise ValueError(f"seq and step_index must be non-negative, got {seq}/{step_index}")
        with self._session_factory() as session:
            if session.get(ExecutionRun, run_id) is None:
                raise KeyError(f"ExecutionRun {run_id} not found")
            screen = session.get(Screen, screen_id)
            if screen is None:
                raise KeyError(f"Screen {screen_id} not found")
            visit = ScreenVisit(
                run_id=run_id,
                screen_id=screen_id,
                app_name=screen.app_name,
                window_title=screen.window_title,
                seq=seq,
                step_index=step_index,
                action=action,
                seen_at=datetime.now(UTC),
            )
            session.add(visit)
            session.commit()
            session.refresh(visit)
            session.expunge(visit)
            _log.info(
                "visit.recorded",
                run_id=run_id,
                screen_id=screen_id,
                seq=seq,
                step=step_index,
            )
            return visit

    def count_visits_since(self, app_name: str, moment: datetime) -> int:
        with self._session_factory() as session:
            return int(
                session.scalar(
                    select(func.count(ScreenVisit.id)).where(
                        ScreenVisit.app_name == app_name, ScreenVisit.seen_at > moment
                    )
                )
                or 0
            )

    def journal_spans(self) -> dict[str, JournalSpan]:
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    ScreenVisit.app_name,
                    func.count(ScreenVisit.id),
                    func.min(ScreenVisit.seen_at),
                ).group_by(ScreenVisit.app_name)
            ).all()
            marks = {
                str(app): (int(cut), first_seen)
                for app, cut, first_seen in session.execute(
                    select(AppJournal.app_name, AppJournal.visits_cut, AppJournal.first_seen_at)
                ).all()
            }

        spans = {}
        for app, retained, oldest in rows:
            cut, first_seen = marks.pop(str(app), (0, None))
            spans[str(app)] = JournalSpan(
                app_name=str(app),
                retained=int(retained),
                cut=cut,
                first_seen_at=first_seen if first_seen is not None else oldest,
                oldest_retained_at=oldest,
            )
        for app, (cut, first_seen) in marks.items():
            spans[app] = JournalSpan(
                app_name=app,
                retained=0,
                cut=cut,
                first_seen_at=first_seen,
                oldest_retained_at=None,
            )
        return spans

    def count_visits_over_cap(self, per_app_limit: int) -> int:
        if per_app_limit < 1:
            raise ValueError(f"per_app_limit must be positive, got {per_app_limit}.")
        stored = func.count(ScreenVisit.id)
        with self._session_factory() as session:
            rows = session.execute(
                select(stored).group_by(ScreenVisit.app_name).having(stored > per_app_limit)
            ).all()
        return sum(int(count) - per_app_limit for (count,) in rows)

    def rotate_visits(self, per_app_limit: int) -> VisitRotation:
        if per_app_limit < 1:
            raise ValueError(f"per_app_limit must be positive, got {per_app_limit}.")

        stored = func.count(ScreenVisit.id)
        cut: list[tuple[str, int]] = []
        with self._session_factory() as session:
            over_cap = session.execute(
                select(ScreenVisit.app_name, stored, func.min(ScreenVisit.seen_at))
                .group_by(ScreenVisit.app_name)
                .having(stored > per_app_limit)
                .order_by(ScreenVisit.app_name)
            ).all()
            for app_name, count, earliest in over_cap:
                surplus = int(count) - per_app_limit
                removed = self._drop_oldest_visits(session, str(app_name), surplus)
                if not removed:
                    continue
                self._mark_visits_cut(session, str(app_name), removed, earliest)
                cut.append((str(app_name), removed))
            session.commit()

        rotation = VisitRotation(tuple(cut))
        if rotation:
            _log.info(
                "journal.rotated",
                apps=len(rotation.cut),
                visits=rotation.total,
                per_app_limit=per_app_limit,
            )
        return rotation

    @staticmethod
    def _drop_oldest_visits(session: Session, app_name: str, surplus: int) -> int:
        doomed = (
            select(ScreenVisit.id)
            .where(ScreenVisit.app_name == app_name)
            .order_by(ScreenVisit.seen_at, ScreenVisit.id)
            .limit(surplus)
            .scalar_subquery()
        )
        return int(
            session.execute(
                delete(ScreenVisit)
                .where(ScreenVisit.id.in_(doomed))
                .execution_options(synchronize_session=False)
            ).rowcount
        )

    # PR-011
    @staticmethod
    def _mark_visits_cut(session: Session, app_name: str, removed: int, earliest: datetime) -> None:
        mark = session.scalar(select(AppJournal).where(AppJournal.app_name == app_name))
        if mark is None:
            session.add(AppJournal(app_name=app_name, visits_cut=removed, first_seen_at=earliest))
        else:
            mark.visits_cut += removed

    def graph_counts(self) -> GraphCounts:
        with self._session_factory() as session:
            return GraphCounts(
                screens=int(session.scalar(select(func.count(Screen.id))) or 0),
                elements=int(session.scalar(select(func.count(Element.id))) or 0),
                edges=int(session.scalar(select(func.count(Edge.id))) or 0),
                visits=int(session.scalar(select(func.count(ScreenVisit.id))) or 0),
                orphan_visits=int(
                    session.scalar(
                        select(func.count(ScreenVisit.id)).where(ScreenVisit.screen_id.is_(None))
                    )
                    or 0
                ),
            )

    def create_run(self, plan: Plan) -> ExecutionRun:
        with self._session_factory() as session:
            run = ExecutionRun(
                plan=plan.model_dump(mode="json"),
                status=RunStatus.RUNNING,
                journey=[],
                started_at=datetime.now(UTC),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            session.expunge(run)
            _log.info("run.created", run_id=run.id, steps=len(plan.steps))
            return run

    def finish_run(self, run_id: int, status: RunStatus, journey: list[str]) -> ExecutionRun:
        if status is RunStatus.RUNNING:
            raise ValueError("finish_run requires a terminal status, not 'running'.")
        with self._session_factory() as session:
            run = session.get(ExecutionRun, run_id)
            if run is None:
                raise KeyError(f"ExecutionRun {run_id} not found")
            run.status = status
            run.journey = list(journey)
            run.finished_at = datetime.now(UTC)
            session.commit()
            session.refresh(run)
            session.expunge(run)
            _log.info("run.finished", run_id=run_id, status=status.value)
            return run

    def count_runs_over_cap(self, keep: int) -> int:
        if keep < 1:
            raise ValueError(f"keep must be positive, got {keep}.")
        with self._session_factory() as session:
            total = int(session.scalar(select(func.count(ExecutionRun.id))) or 0)
        return max(0, total - keep)

    def rotate_runs(self, keep: int) -> int:
        if keep < 1:
            raise ValueError(f"keep must be positive, got {keep}.")
        with self._session_factory() as session:
            doomed = (
                select(ExecutionRun.id)
                .order_by(ExecutionRun.started_at.desc(), ExecutionRun.id.desc())
                .offset(keep)
                .scalar_subquery()
            )
            removed = int(
                session.execute(
                    delete(ExecutionRun)
                    .where(ExecutionRun.id.in_(doomed))
                    .execution_options(synchronize_session=False)
                ).rowcount
            )
            session.commit()
        if removed:
            _log.info("runs.rotated", removed=removed, keep=keep)
        return removed

    def create_mission(self, goal: str, items: Sequence[MissionItemDraft]) -> Mission:
        goal_text = goal.strip()
        if not goal_text:
            raise ValueError("A mission must state its goal.")
        drafts = list(items)
        if not drafts:
            raise ValueError(
                "A mission must have at least one item: a goal with no checklist is "
                "the state a mission exists to replace."
            )
        for position, draft in enumerate(drafts):
            if not isinstance(draft, MissionItemDraft):
                raise TypeError(
                    f"Mission item {position} must be a MissionItemDraft, "
                    f"got {type(draft).__name__}."
                )

        with self._session_factory() as session:
            running = session.scalar(
                select(MissionRow).where(MissionRow.status == MissionStatus.ACTIVE)
            )
            if running is not None:
                raise ActiveMissionExists(
                    f"Mission #{running.id} is still active ({running.goal!r}); finish the "
                    f"current one (done or abandoned) before starting another."
                )
            mission = MissionRow(goal=goal_text, status=MissionStatus.ACTIVE)
            mission.items = [
                MissionItemRow(
                    seq=seq,
                    title=draft.title,
                    intent=draft.intent,
                    acceptance=draft.acceptance,
                    status=MissionItemStatus.PENDING,
                    fail_reason="",
                )
                for seq, draft in enumerate(drafts)
            ]
            session.add(mission)
            session.commit()
            value = _mission_value(mission)
        _log.info("mission.created", mission_id=value.mission_id, items=len(value.items))
        return value

    def active_mission(self) -> Mission | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(MissionRow).where(MissionRow.status == MissionStatus.ACTIVE)
            )
            return None if row is None else _mission_value(row)

    def mark_item(
        self,
        item_id: int,
        status: MissionItemStatus,
        fail_reason: str = "",
        recipe: Sequence[Mapping[str, Any]] | None = None,
        fingerprint: WorldFingerprint | None = None,
    ) -> Mission:
        reason = validate_mark(status, fail_reason)
        steps = None if recipe is None else normalize_recipe(recipe)

        with self._session_factory() as session:
            item = session.get(MissionItemRow, item_id)
            if item is None:
                raise KeyError(f"Mission item {item_id} not found")
            mission = item.mission
            if mission.status is not MissionStatus.ACTIVE:
                raise ValueError(
                    f"Mission #{mission.id} is {mission.status.value}, not active: its items "
                    f"are a record of what happened and are not marked any more."
                )
            item.status = status
            item.fail_reason = reason
            if steps is not None:
                item.recipe_json = list(steps)
            if fingerprint is not None:
                item.fingerprint_json = fingerprint.to_json()
            mission.updated_at = datetime.now(UTC)
            session.commit()
            value = _mission_value(mission)
        _log.info(
            "mission.item_marked",
            mission_id=value.mission_id,
            item_id=item_id,
            status=status.value,
            recipe_steps=0 if steps is None else len(steps),
        )
        return value

    def finish_mission(self, status: MissionStatus) -> Mission:
        if status is MissionStatus.ACTIVE:
            raise ValueError("finish_mission requires a terminal status: 'done' or 'abandoned'.")
        with self._session_factory() as session:
            mission = session.scalar(
                select(MissionRow).where(MissionRow.status == MissionStatus.ACTIVE)
            )
            if mission is None:
                raise KeyError("No active mission to finish")
            mission.status = status
            session.commit()
            value = _mission_value(mission)
        _log.info(
            "mission.finished",
            mission_id=value.mission_id,
            status=status.value,
            resolved=value.resolved,
            items=len(value.items),
        )
        return value
