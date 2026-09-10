from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from choto.graph.map_models import (
    Label,
    OffscreenHint,
    ScrollEdge,
    Transition,
    WindowSummary,
)
from choto.graph.repository import GraphRepository
from choto.log import get_logger
from choto.models import BBox, ElementKind, OcrLine, Target

if TYPE_CHECKING:  # pragma: no cover - typing only, see the module docstring
    from choto.matching.matcher import MatchResult, TargetMatcher

_log = get_logger(__name__)

DEFAULT_FIND_LIMIT = 5


def action_target_texts(action: dict[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    target = action.get("target")
    if isinstance(target, str) and target.strip():
        texts.append(target)
    path = action.get("path")
    if isinstance(path, Sequence) and not isinstance(path, str | bytes):
        for point in path:
            if not isinstance(point, dict):
                continue
            waypoint = point.get("target")
            if isinstance(waypoint, dict):
                text = waypoint.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    return tuple(texts)


@dataclass(frozen=True)
class RouteStep:
    origin: WindowSummary
    action: dict[str, Any]
    target: WindowSummary
    success_count: int
    fail_count: int


@dataclass(frozen=True)
class Route:
    steps: tuple[RouteStep, ...]
    alternatives: int

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


@dataclass(frozen=True)
class FindHit:
    window: WindowSummary
    where: FindWhere
    text: str
    score: float
    tier: str
    route: Route | NoRoute
    offscreen: OffscreenHint | None = None


@dataclass(frozen=True)
class _ScoredHit:
    window: WindowSummary
    where: FindWhere
    text: str
    score: float
    tier: str
    offscreen: OffscreenHint | None


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
        return self._route(to_screen, from_screen, windows, self._repo.transitions())

    def _route(
        self,
        to_screen: int,
        from_screen: int,
        windows: dict[int, WindowSummary],
        moves: Sequence[Transition],
    ) -> Route | NoRoute:
        if from_screen not in windows:
            return NoRoute(NoRouteReason.ORIGIN_UNKNOWN)
        if to_screen not in windows:
            return NoRoute(NoRouteReason.TARGET_UNKNOWN)
        if from_screen == to_screen:
            return NoRoute(NoRouteReason.ALREADY_THERE)

        linked = [
            move for move in moves if move.from_window in windows and move.to_window in windows
        ]
        spread = _spread(from_screen, linked)
        if not spread.reaches(to_screen):
            return _no_route(to_screen, windows, linked, spread)
        return Route(
            steps=tuple(_route_step(move, windows) for move in spread.best_path(to_screen)),
            alternatives=spread.alternatives(to_screen),
        )

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

        scored: list[_ScoredHit] = []
        searched = 0
        for window in windows:
            usable = [label for label in labels.get(window.screen_id, ()) if label.targetable]
            searched += len(usable)
            best = self._best_hit(target, window, usable)
            if best is not None:
                scored.append(best)

        scored.sort(
            key=lambda row: (-row.score, -row.window.last_seen_at.timestamp(), row.window.screen_id)
        )
        kept = scored[:limit]

        moves = self._repo.transitions()
        by_id = {
            item.screen_id: item
            for item in (windows if app_name is None else self._repo.window_summaries())
        }

        hits = tuple(
            FindHit(
                window=row.window,
                where=row.where,
                text=row.text,
                score=row.score,
                tier=row.tier,
                route=self._route_to(row.window, origin, by_id, moves),
                offscreen=row.offscreen,
            )
            for row in kept
        )
        report = FindReport(
            query=text,
            hits=hits,
            apps_searched=len({item.app_name for item in windows}),
            windows_searched=len(windows),
            elements_searched=searched,
            truncated=max(len(scored) - len(kept), 0),
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

    def _best_hit(
        self, target: Target, window: WindowSummary, labels: Sequence[Label]
    ) -> _ScoredHit | None:
        candidates: list[_ScoredHit] = []
        lines = [label.line for label in labels]
        by_line = {id(line): label for line, label in zip(lines, labels, strict=True)}
        hit = self._matcher.resolve(target, lines, window.width, window.height)
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
        self,
        window: WindowSummary,
        origin: int | None,
        windows: dict[int, WindowSummary],
        moves: Sequence[Transition],
    ) -> Route | NoRoute:
        if origin is None:
            start = latest_window(
                [item for item in windows.values() if item.app_name == window.app_name]
            )
            if start is None:
                return NoRoute(NoRouteReason.ORIGIN_UNKNOWN)
            origin = start.screen_id
        return self._route(window.screen_id, origin, windows, moves)

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


def _route_step(move: Transition, windows: dict[int, WindowSummary]) -> RouteStep:
    return RouteStep(
        origin=windows[move.from_window],
        action=dict(move.action),
        target=windows[move.to_window],
        success_count=move.success_count,
        fail_count=move.fail_count,
    )


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
