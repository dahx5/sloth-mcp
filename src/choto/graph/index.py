from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from choto.graph.eviction import ROUTE_KINDS
from choto.graph.map_models import (
    Label,
    OffscreenHint,
    ScrollEdge,
    Transition,
    WindowSummary,
)
from choto.graph.repository import GraphRepository, normalize_text
from choto.log import get_logger
from choto.models import Action, BBox, ElementKind, OcrLine, Step, Target
from choto.vision.anchors import select_anchors
from choto.vision.navigation import LabelPlace, Navigation, derive_navigation

if TYPE_CHECKING:  # pragma: no cover - typing only, see the module docstring
    from choto.matching.matcher import MatchResult, TargetMatcher

_log = get_logger(__name__)

DEFAULT_FIND_LIMIT = 5

_CLICKS = frozenset({Action.CLICK.value, Action.DOUBLE_CLICK.value, Action.RIGHT_CLICK.value})

# PR-049
_ANCHOR_CANDIDATES = 8

# PR-049
_MAX_ANCHOR_CHARS = 40

# PR-049
_MIN_ANCHOR_LETTERS = 3

# PR-052
_MIN_NAV_NAME_CHARS = 3


def _target_text(target: Any) -> str | None:
    if isinstance(target, dict):
        target = target.get("text")
    if isinstance(target, str) and target.strip():
        return target
    return None


def action_target_texts(action: dict[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    text = _target_text(action.get("target"))
    if text is not None:
        texts.append(text)
    path = action.get("path")
    if isinstance(path, Sequence) and not isinstance(path, str | bytes):
        for point in path:
            if not isinstance(point, dict):
                continue
            waypoint = point.get("target")
            waypoint_text = _target_text(waypoint)
            if waypoint_text is not None:
                texts.append(waypoint_text)
    return tuple(texts)


# PR-023, PR-052
@dataclass(frozen=True)
class AppNavigation:
    navigation: Navigation = field(default_factory=Navigation)
    names: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.navigation)

    def holds(self, line: OcrLine) -> bool:
        if not self.navigation:
            return False
        if self.navigation.holds(line):
            return True
        cx, cy = line.bbox.center
        return any(
            area.x <= cx < area.x + area.w and area.y <= cy < area.y + area.h
            for area in self.navigation.areas
        )


def app_navigation(labels: Mapping[int, Sequence[Label]]) -> AppNavigation:
    on_screen = [
        (screen_id, label)
        for screen_id, rows in labels.items()
        for label in rows
        if label.kind is ElementKind.TEXT and label.offscreen is None
    ]
    navigation = derive_navigation(
        [LabelPlace(screen_id=sid, text=label.text, bbox=label.bbox) for sid, label in on_screen]
    )
    if not navigation:
        return AppNavigation(navigation=navigation)
    spelled: dict[str, str] = {}
    for _sid, label in on_screen:
        spelled.setdefault(normalize_text(label.text), label.text.strip())
    slots = sorted(navigation.slots, key=lambda slot: (slot.bbox.y, slot.bbox.x))
    names = (spelled.get(slot.text, slot.text) for slot in slots)
    return AppNavigation(
        navigation=navigation,
        names=tuple(dict.fromkeys(name for name in names if len(name) >= _MIN_NAV_NAME_CHARS)),
    )


@dataclass(frozen=True)
class RouteStep:
    origin: WindowSummary
    action: dict[str, Any]
    target: WindowSummary
    success_count: int
    fail_count: int
    # PR-049
    step: dict[str, Any] = field(default_factory=dict)
    # PR-047
    via_navigation: bool = False


@dataclass(frozen=True)
class Route:
    steps: tuple[RouteStep, ...]
    alternatives: int
    # PR-048
    instead_of: WindowSummary | None = None

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def crosses_apps(self) -> bool:
        apps = {step.origin.app_name for step in self.steps}
        apps.update(step.target.app_name for step in self.steps)
        return len(apps) > 1


class NoRouteReason(str, Enum):
    ALREADY_THERE = "already there"
    ORIGIN_UNKNOWN = "origin unknown"
    TARGET_UNKNOWN = "target unknown"
    NEVER_ENTERED = "never entered"
    NOT_CONNECTED = "not connected"


@dataclass(frozen=True)
class Entrance:
    window: WindowSummary
    has_way_in: bool


@dataclass(frozen=True)
class RouteOrigin:
    window: WindowSummary
    outgoing: int
    reached: int


# PR-013
@dataclass(frozen=True)
class NoRoute:
    reason: NoRouteReason
    entrances: tuple[Entrance, ...] = ()
    origin: RouteOrigin | None = None


class FindWhere(str, Enum):
    ELEMENT = "element"
    ICON = "icon"
    TITLE = "window title"
    NAVIGATION = "app navigation"


@dataclass(frozen=True)
class FindHit:
    window: WindowSummary
    where: FindWhere
    text: str
    score: float
    tier: str
    route: Route | NoRoute | None
    offscreen: OffscreenHint | None = None
    # PR-050
    windows: int = 1


@dataclass(frozen=True)
class _ScoredHit:
    window: WindowSummary
    where: FindWhere
    text: str
    score: float
    tier: str
    offscreen: OffscreenHint | None
    windows: int = 1


@dataclass(frozen=True)
class _Ground:
    windows: dict[int, WindowSummary]
    recorded: tuple[Transition, ...]
    moves: tuple[Transition, ...]
    labels: dict[int, list[Label]]
    navigation: dict[str, AppNavigation]
    virtual: frozenset[tuple[int, int]] = frozenset()


@dataclass(frozen=True)
class FindReport:
    query: str
    hits: tuple[FindHit, ...]
    apps_searched: int
    windows_searched: int
    elements_searched: int
    truncated: int


class ScrollProgress(str, Enum):
    UNKNOWN = "unknown"
    PARTIAL = "partial"
    THROUGH = "through"


@dataclass(frozen=True)
class WindowCoverage:
    window: WindowSummary
    targets: int
    activated: int
    outgoing: int
    incoming: int
    unlabeled_icons: int
    offscreen_targets: int = 0

    @property
    def unexplored(self) -> bool:
        return self.outgoing == 0

    @property
    def scroll_progress(self) -> ScrollProgress:
        extent = self.window.scroll
        if extent is None:
            return ScrollProgress.UNKNOWN
        return ScrollProgress.THROUGH if extent.exhausted else ScrollProgress.PARTIAL

    @property
    def unreached_edges(self) -> tuple[ScrollEdge, ...]:
        extent = self.window.scroll
        return () if extent is None else extent.unreached

    @property
    def walked(self) -> bool:
        return (
            self.outgoing > 0
            and self.targets > 0
            and self.activated >= self.targets
            and self.unlabeled_icons == 0
            and self.scroll_progress is not ScrollProgress.PARTIAL
        )

    @property
    def seen_once(self) -> bool:
        return self.window.visit_count <= 1


@dataclass(frozen=True)
class AppOutline:
    app_name: str
    windows: int
    unexplored: int
    seen_once: int
    no_way_in: int
    scroll_pending: int = 0


@dataclass(frozen=True)
class AppCoverage:
    outline: AppOutline
    windows: tuple[WindowCoverage, ...]
    unlabeled_glyphs: int

    @property
    def targets(self) -> int:
        return sum(item.targets for item in self.windows)

    @property
    def activated(self) -> int:
        return sum(item.activated for item in self.windows)


class InterfaceIndex:
    def __init__(self, repo: GraphRepository, matcher: TargetMatcher) -> None:
        self._repo = repo
        self._matcher = matcher

    def route(self, to_screen: int, from_screen: int) -> Route | NoRoute:
        windows = {item.screen_id: item for item in self._repo.window_summaries()}
        return self._route(to_screen, from_screen, self._ground(windows))

    def navigation(self, app_name: str) -> AppNavigation:
        windows = self._repo.window_summaries(app_name=app_name)
        return app_navigation(self._repo.window_labels([item.screen_id for item in windows]))

    def _ground(self, windows: dict[int, WindowSummary]) -> _Ground:
        recorded = tuple(
            move
            for move in self._repo.transitions()
            if move.from_window in windows and move.to_window in windows
        )
        usable = [move for move in recorded if _routable(move)]
        labels = self._repo.window_labels(list(windows))
        grouped: dict[str, dict[int, list[Label]]] = {}
        for screen_id, window in windows.items():
            grouped.setdefault(window.app_name, {})[screen_id] = labels.get(screen_id, [])
        navigation = {app: app_navigation(members) for app, members in grouped.items()}
        extra = self._navigation_moves(usable, recorded, windows, labels, navigation)
        return _Ground(
            windows=windows,
            recorded=recorded,
            moves=(*usable, *extra),
            labels=labels,
            navigation=navigation,
            virtual=frozenset((move.transition_id, move.from_window) for move in extra),
        )

    # PR-047
    def _navigation_moves(
        self,
        usable: Sequence[Transition],
        recorded: Sequence[Transition],
        windows: dict[int, WindowSummary],
        labels: Mapping[int, Sequence[Label]],
        navigation: Mapping[str, AppNavigation],
    ) -> list[Transition]:
        # PR-043
        tried = {(move.from_window, _action_key(move.action)) for move in recorded}
        extra: list[Transition] = []
        for move in usable:
            origin = windows[move.from_window]
            nav = navigation.get(origin.app_name)
            target = _click_target(move.action)
            if not nav or target is None or windows[move.to_window].app_name != origin.app_name:
                continue
            if not self._through_navigation(target, origin, labels, nav):
                continue
            for screen_id, window in windows.items():
                if screen_id in (move.from_window, move.to_window):
                    continue
                if window.app_name != origin.app_name:
                    continue
                if (screen_id, _action_key(move.action)) in tried:
                    continue
                if self._through_navigation(target, window, labels, nav):
                    extra.append(replace(move, from_window=screen_id))
        return extra

    # PR-029
    def _through_navigation(
        self,
        target: Target,
        window: WindowSummary,
        labels: Mapping[int, Sequence[Label]],
        nav: AppNavigation,
    ) -> bool:
        rows = [label.line for label in labels.get(window.screen_id, ()) if label.targetable]
        content = [line for line in rows if not nav.holds(line)]
        held = [line for line in rows if nav.holds(line)]
        if not held or self._matcher.resolve(target, content, window.width, window.height):
            return False
        return self._matcher.resolve(target, held, window.width, window.height) is not None

    def _route(self, to_screen: int, from_screen: int, ground: _Ground) -> Route | NoRoute:
        windows = ground.windows
        if from_screen not in windows:
            return NoRoute(NoRouteReason.ORIGIN_UNKNOWN)
        if to_screen not in windows:
            return NoRoute(NoRouteReason.TARGET_UNKNOWN)
        if from_screen == to_screen:
            return NoRoute(NoRouteReason.ALREADY_THERE)

        spread = _spread(from_screen, ground.moves)
        if spread.reaches(to_screen):
            return self._walk(spread, to_screen, ground)
        missing = _no_route(to_screen, windows, ground.recorded, spread)
        # PR-048
        start = _nearest_start(to_screen, from_screen, ground)
        if start is None:
            return missing
        found = self._walk(_spread(start, ground.moves), to_screen, ground)
        return replace(found, instead_of=windows[from_screen])

    def _walk(self, spread: _Spread, to_screen: int, ground: _Ground) -> Route:
        return Route(
            steps=tuple(self._route_step(move, ground) for move in spread.best_path(to_screen)),
            alternatives=spread.alternatives(to_screen),
        )

    def _route_step(self, move: Transition, ground: _Ground) -> RouteStep:
        return RouteStep(
            origin=ground.windows[move.from_window],
            action=dict(move.action),
            target=ground.windows[move.to_window],
            success_count=move.success_count,
            fail_count=move.fail_count,
            step={**move.action, "expect": self._arrival(move, ground)},
            via_navigation=(move.transition_id, move.from_window) in ground.virtual,
        )

    # PR-049
    def _arrival(self, move: Transition, ground: _Ground) -> dict[str, Any]:
        origin = ground.windows[move.from_window]
        nav = ground.navigation.get(ground.windows[move.to_window].app_name, AppNavigation())
        own = [
            label.line
            for label in ground.labels.get(move.to_window, ())
            if label.targetable and label.kind is ElementKind.TEXT and not nav.holds(label.line)
        ]
        before = [
            label.line for label in ground.labels.get(move.from_window, ()) if label.targetable
        ]
        for text in select_anchors(own, n=_ANCHOR_CANDIDATES):
            if len(text) > _MAX_ANCHOR_CHARS or not _wordy(text):
                continue
            if self._matcher.resolve(Target(text=text), before, origin.width, origin.height):
                continue
            return {"appears": text}
        return {"screen_changes": True}

    def last_seen(self, app_name: str) -> WindowSummary | None:
        windows = self._repo.window_summaries(app_name=app_name)
        return latest_window(windows)

    def find(
        self,
        query: str,
        app_name: str | None = None,
        origin: int | None = None,
        limit: int = DEFAULT_FIND_LIMIT,
    ) -> FindReport:
        text = query.strip()
        if not text:
            raise ValueError("find needs something to look for; the query is empty.")
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}.")

        windows = self._repo.window_summaries(app_name=app_name)
        labels = self._repo.window_labels(
            [item.screen_id for item in windows], include_offscreen=True
        )
        target = Target(text=text)
        by_id = {
            item.screen_id: item
            for item in (windows if app_name is None else self._repo.window_summaries())
        }
        ground = self._ground(by_id)

        scored: list[_ScoredHit] = []
        searched = 0
        for window in windows:
            usable = [label for label in labels.get(window.screen_id, ()) if label.targetable]
            searched += len(usable)
            nav = ground.navigation.get(window.app_name, AppNavigation())
            best = self._best_hit(target, window, usable, nav)
            if best is not None:
                scored.append(best)

        ranked = sorted(
            _one_per_navigation(scored),
            key=lambda row: (
                row.where is FindWhere.NAVIGATION,
                -row.score,
                -row.window.last_seen_at.timestamp(),
                row.window.screen_id,
            ),
        )
        kept = ranked[:limit]

        hits = tuple(
            FindHit(
                window=row.window,
                where=row.where,
                text=row.text,
                score=row.score,
                tier=row.tier,
                route=None
                if row.where is FindWhere.NAVIGATION
                else self._route_to(row.window, origin, ground),
                offscreen=row.offscreen,
                windows=row.windows,
            )
            for row in kept
        )
        report = FindReport(
            query=text,
            hits=hits,
            apps_searched=len({item.app_name for item in windows}),
            windows_searched=len(windows),
            elements_searched=searched,
            truncated=max(len(ranked) - len(kept), 0),
        )
        _log.info(
            "index.find",
            query=text,
            app_name=app_name,
            hits=len(hits),
            windows=report.windows_searched,
            elements=report.elements_searched,
        )
        return report

    # PR-050
    def _best_hit(
        self, target: Target, window: WindowSummary, labels: Sequence[Label], nav: AppNavigation
    ) -> _ScoredHit | None:
        lines = [label.line for label in labels]
        by_line = {id(line): label for line, label in zip(lines, labels, strict=True)}
        held = {
            id(line)
            for line, label in zip(lines, labels, strict=True)
            if label.offscreen is None and nav.holds(line)
        }
        own = self._own_hit(
            target, window, [line for line in lines if id(line) not in held], by_line
        )
        if own is not None:
            return own
        hit = self._matcher.resolve(
            target, [line for line in lines if id(line) in held], window.width, window.height
        )
        if hit is None:
            return None
        return _ScoredHit(
            window=window,
            where=FindWhere.NAVIGATION,
            text=by_line[id(hit.line)].text.strip(),
            score=hit.score,
            tier=hit.tier,
            offscreen=None,
        )

    def _own_hit(
        self,
        target: Target,
        window: WindowSummary,
        lines: list[OcrLine],
        by_line: Mapping[int, Label],
    ) -> _ScoredHit | None:
        candidates: list[_ScoredHit] = []
        hit = self._matcher.resolve(target, lines, window.width, window.height) if lines else None
        if hit is not None:
            label = by_line[id(hit.line)]
            candidates.append(
                _ScoredHit(
                    window=window,
                    where=FindWhere.ICON if label.kind is ElementKind.ICON else FindWhere.ELEMENT,
                    text=label.text.strip(),
                    score=hit.score,
                    tier=hit.tier,
                    offscreen=label.offscreen,
                )
            )
        title = self._match_title(target, window)
        if title is not None:
            candidates.append(
                _ScoredHit(
                    window=window,
                    where=FindWhere.TITLE,
                    text=window.window_title,
                    score=title.score,
                    tier=title.tier,
                    offscreen=None,
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda row: row.score)

    def _match_title(self, target: Target, window: WindowSummary) -> MatchResult | None:
        title = window.window_title.strip()
        if not title:
            return None
        line = OcrLine(
            text=title,
            bbox=BBox(x=0, y=0, w=max(window.width, 0), h=0),
            confidence=1.0,
        )
        return self._matcher.resolve(target, [line], window.width, window.height)

    def _route_to(
        self, window: WindowSummary, origin: int | None, ground: _Ground
    ) -> Route | NoRoute:
        if origin is None:
            start = latest_window(
                [item for item in ground.windows.values() if item.app_name == window.app_name]
            )
            if start is None:
                return NoRoute(NoRouteReason.ORIGIN_UNKNOWN)
            origin = start.screen_id
        return self._route(window.screen_id, origin, ground)

    def map_outline(self) -> dict[str, AppOutline]:
        outgoing, incoming = _degrees(self._repo.transitions())
        grouped: dict[str, list[WindowSummary]] = {}
        for window in self._repo.window_summaries():
            grouped.setdefault(window.app_name, []).append(window)
        return {
            app_name: AppOutline(
                app_name=app_name,
                windows=len(windows),
                unexplored=sum(1 for item in windows if not outgoing.get(item.screen_id)),
                seen_once=sum(1 for item in windows if item.visit_count <= 1),
                no_way_in=sum(1 for item in windows if not incoming.get(item.screen_id)),
                scroll_pending=sum(
                    1 for item in windows if item.scroll is not None and not item.scroll.exhausted
                ),
            )
            for app_name, windows in grouped.items()
        }

    def app_coverage(self, app_name: str) -> AppCoverage:
        windows = self._repo.window_summaries(app_name=app_name)
        covered = self._window_coverage(windows, self._repo.transitions())
        outline = AppOutline(
            app_name=app_name,
            windows=len(covered),
            unexplored=sum(1 for item in covered if item.unexplored),
            seen_once=sum(1 for item in covered if item.seen_once),
            no_way_in=sum(1 for item in covered if item.incoming == 0),
            scroll_pending=sum(
                1 for item in covered if item.scroll_progress is ScrollProgress.PARTIAL
            ),
        )
        counts = self._repo.icon_counts(app_name)
        return AppCoverage(
            outline=outline,
            windows=covered,
            unlabeled_glyphs=counts.glyphs_total - counts.glyphs_labeled,
        )

    def window_coverage(self, screen_id: int) -> WindowCoverage | None:
        windows = self._repo.window_summaries(screen_ids=[screen_id])
        if not windows:
            return None
        covered = self._window_coverage(windows, self._repo.transitions())
        return covered[0]

    def _window_coverage(
        self, windows: Sequence[WindowSummary], moves: Sequence[Transition]
    ) -> tuple[WindowCoverage, ...]:
        if not windows:
            return ()
        wanted = {window.screen_id for window in windows}
        labels = self._repo.window_labels(sorted(wanted), include_offscreen=True)
        outgoing: dict[int, list[Transition]] = {}
        incoming: dict[int, int] = {}
        for move in moves:
            if move.from_window in wanted:
                outgoing.setdefault(move.from_window, []).append(move)
            if move.to_window in wanted:
                incoming[move.to_window] = incoming.get(move.to_window, 0) + 1

        return tuple(
            self._cover_window(
                window,
                labels.get(window.screen_id, []),
                outgoing.get(window.screen_id, ()),
                incoming.get(window.screen_id, 0),
            )
            for window in windows
        )

    def _cover_window(
        self,
        window: WindowSummary,
        stored: Sequence[Label],
        outgoing: Sequence[Transition],
        incoming: int,
    ) -> WindowCoverage:
        usable = [label for label in stored if label.targetable]
        return WindowCoverage(
            window=window,
            targets=len(usable),
            activated=self._activated(window, usable, outgoing),
            outgoing=len(outgoing),
            incoming=incoming,
            unlabeled_icons=sum(
                1 for label in stored if label.kind is ElementKind.ICON and not label.text.strip()
            ),
            offscreen_targets=sum(1 for label in usable if label.offscreen is not None),
        )

    def _activated(
        self, window: WindowSummary, labels: Sequence[Label], moves: Iterable[Transition]
    ) -> int:
        if not labels:
            return 0
        wanted: list[str] = []
        for move in moves:
            for text in action_target_texts(move.action):
                if text not in wanted:
                    wanted.append(text)
        if not wanted:
            return 0
        lines = [label.line for label in labels]
        by_line = {id(line): index for index, line in enumerate(lines)}
        used: set[int] = set()
        for text in wanted:
            hit = self._matcher.resolve(Target(text=text), lines, window.width, window.height)
            if hit is not None:
                used.add(by_line[id(hit.line)])
        return len(used)


@dataclass(frozen=True)
class _Spread:
    origin: int
    outgoing: dict[int, list[Transition]]
    distance: dict[int, int]
    ways: dict[int, int]
    arrivals: dict[int, list[Transition]]

    def reaches(self, screen_id: int) -> bool:
        return screen_id in self.distance

    @property
    def reached(self) -> int:
        return len(self.distance) - 1

    def outgoing_from(self, screen_id: int) -> int:
        return len(self.outgoing.get(screen_id, ()))

    def alternatives(self, screen_id: int) -> int:
        return max(self.ways[screen_id] - 1, 0)

    def best_path(self, screen_id: int) -> tuple[Transition, ...]:
        walked: list[Transition] = []
        node = screen_id
        while node != self.origin:
            move = max(self.arrivals[node], key=transition_quality)
            walked.append(move)
            node = move.from_window
        walked.reverse()
        return tuple(walked)


def _spread(origin: int, moves: Sequence[Transition]) -> _Spread:
    outgoing: dict[int, list[Transition]] = {}
    for move in moves:
        outgoing.setdefault(move.from_window, []).append(move)

    distance = {origin: 0}
    ways = {origin: 1}
    arrivals: dict[int, list[Transition]] = {}
    queue = deque([origin])
    while queue:
        node = queue.popleft()
        for move in outgoing.get(node, ()):
            nxt = move.to_window
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                ways[nxt] = 0
                arrivals[nxt] = []
                queue.append(nxt)
            if distance[nxt] == distance[node] + 1:
                ways[nxt] += ways[node]
                arrivals[nxt].append(move)
    return _Spread(origin, outgoing, distance, ways, arrivals)


def _action_key(action: Mapping[str, Any]) -> str:
    return json.dumps(action, sort_keys=True)


def _click_target(action: Mapping[str, Any]) -> Target | None:
    if action.get("action") not in _CLICKS:
        return None
    target = action.get("target")
    if not isinstance(target, dict) or _target_text(target) is None:
        return None
    return Target.model_validate(target)


# PR-041, PR-043, PR-045
def _routable(move: Transition) -> bool:
    if move.action.get("action") not in ROUTE_KINDS or move.fail_count > move.success_count:
        return False
    try:
        Step.model_validate(move.action)
    except ValidationError:
        return False
    return True


def _wordy(text: str) -> bool:
    letters = sum(1 for char in text if char.isalnum())
    shown = sum(1 for char in text if not char.isspace())
    return letters >= _MIN_ANCHOR_LETTERS and letters * 2 > shown


# PR-048
def _nearest_start(to_screen: int, from_screen: int, ground: _Ground) -> int | None:
    incoming: dict[int, list[Transition]] = {}
    for move in ground.moves:
        incoming.setdefault(move.to_window, []).append(move)
    distance = {to_screen: 0}
    queue = deque([to_screen])
    while queue:
        node = queue.popleft()
        for move in incoming.get(node, ()):
            if move.from_window not in distance:
                distance[move.from_window] = distance[node] + 1
                queue.append(move.from_window)
    app_name = ground.windows[to_screen].app_name
    starts = [
        screen_id
        for screen_id, hops in distance.items()
        if hops and screen_id != from_screen and ground.windows[screen_id].app_name == app_name
    ]
    return min(
        starts,
        key=lambda screen_id: (
            distance[screen_id],
            -ground.windows[screen_id].last_seen_at.timestamp(),
            screen_id,
        ),
        default=None,
    )


# PR-050
def _one_per_navigation(scored: Sequence[_ScoredHit]) -> list[_ScoredHit]:
    kept = [row for row in scored if row.where is not FindWhere.NAVIGATION]
    grouped: dict[str, list[_ScoredHit]] = {}
    for row in scored:
        if row.where is FindWhere.NAVIGATION:
            grouped.setdefault(row.window.app_name, []).append(row)
    for rows in grouped.values():
        best = max(rows, key=lambda row: (row.score, row.window.last_seen_at.timestamp()))
        kept.append(replace(best, windows=len(rows)))
    return kept


# PR-013
def _no_route(
    to_screen: int,
    windows: dict[int, WindowSummary],
    linked: Sequence[Transition],
    spread: _Spread,
) -> NoRoute:
    entered = {move.to_window for move in linked}
    found = sorted({move.from_window for move in linked if move.to_window == to_screen})
    return NoRoute(
        NoRouteReason.NEVER_ENTERED if not found else NoRouteReason.NOT_CONNECTED,
        entrances=tuple(
            Entrance(window=windows[screen_id], has_way_in=screen_id in entered)
            for screen_id in found
        ),
        origin=RouteOrigin(
            window=windows[spread.origin],
            outgoing=spread.outgoing_from(spread.origin),
            reached=spread.reached,
        ),
    )


def transition_quality(move: Transition) -> tuple[int, int, int]:
    return (move.success_count - move.fail_count, -move.fail_count, -move.transition_id)


def _degrees(moves: Iterable[Transition]) -> tuple[dict[int, int], dict[int, int]]:
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for move in moves:
        outgoing[move.from_window] = outgoing.get(move.from_window, 0) + 1
        incoming[move.to_window] = incoming.get(move.to_window, 0) + 1
    return outgoing, incoming


# PR-012
def latest_window(windows: Sequence[WindowSummary]) -> WindowSummary | None:
    return max(
        windows,
        key=lambda window: (window.last_seen_at.timestamp(), window.screen_id),
        default=None,
    )
