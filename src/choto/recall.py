from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from choto.graph.index import (
    AppCoverage,
    AppOutline,
    Entrance,
    FindHit,
    FindReport,
    FindWhere,
    InterfaceIndex,
    NoRoute,
    NoRouteReason,
    Route,
    RouteOrigin,
    RouteStep,
    ScrollProgress,
    WindowCoverage,
    latest_window,
    transition_quality,
)
from choto.graph.map_models import (
    AppSummary,
    Label,
    OffscreenHint,
    ScrollEdge,
    Transition,
    WindowSummary,
)
from choto.graph.repository import GraphRepository, normalize_text
from choto.log import get_logger
from choto.matching.matcher import TargetMatcher

_log = get_logger(__name__)


_CHAR_BUDGET = 2400

_MAX_APPS = 12

_MAX_WINDOWS = 8

_MAX_ELEMENTS = 24

_MAX_TRANSITIONS = 8

_MAX_ALTERNATIVES = 4

_MAX_APP_HINTS = 8

_MAX_FIND_HITS = 5

_MAX_ROUTE_STEPS = 6

_MAX_ENTRANCES = 3

MAX_DEPTH = 3
DEFAULT_DEPTH = 1

_MAX_TEXT_CHARS = 40

_ELLIPSIS = "…"

_TRUNCATION_NOTE = "(reply cut at the size cap — narrow it: name a window, or use a smaller depth)"

_FOCUS_MODEL_NOTE = (
    "map is by window: each entry is one window's own elements; chrome (menu bar, "
    'status items, open menus) is never remembered — a step reaches it with scope="chrome".'
)

_UNTITLED = "<untitled>"

_NO_OUTLINE = AppOutline(app_name="", windows=0, unexplored=0, seen_once=0, no_way_in=0)

_SCROLL_DIRECTIONS = {
    ScrollEdge.TOP: "up",
    ScrollEdge.BOTTOM: "down",
    ScrollEdge.LEFT: "left",
    ScrollEdge.RIGHT: "right",
}

_INDENT = "  "

_LABEL_PUNCTUATION = " &.,'\"-()/:!?…+%"

_MINUTE = 60
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR
_MONTH = 30 * _DAY


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


_RELATIVE_SCALES = (
    (_HOUR, _MINUTE, "m"),
    (_DAY, _HOUR, "h"),
    (_MONTH, _DAY, "d"),
)


def format_relative(moment: datetime, now: datetime) -> str:
    seconds = (now - _as_utc(moment)).total_seconds()
    if seconds < _MINUTE:
        return "just now"
    for limit, unit, suffix in _RELATIVE_SCALES:
        if seconds < limit:
            return f"{int(seconds // unit)}{suffix} ago"
    return f"{int(seconds // _MONTH)}mo ago"


def format_stamp(moment: datetime, now: datetime) -> str:
    moment = _as_utc(moment)
    if moment.year == now.year:
        return moment.strftime("%m-%d %H:%MZ")
    return moment.strftime("%Y-%m-%d %H:%MZ")


def format_when(moment: datetime, now: datetime) -> str:
    return f"{format_stamp(moment, now)} ({format_relative(moment, now)})"


def _clauses(*parts: tuple[object, str]) -> str:
    return "".join(text for shown, text in parts if shown)


def _more_note(total: int, limit: int) -> str:
    hidden = max(total - limit, 0)
    return f" (+{hidden} more)" if hidden else ""


def _clip(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + _ELLIPSIS


def _quoted_title(title: str) -> str:
    return f'"{_clip(title)}"' if title.strip() else _UNTITLED


def _node_ref(summary: WindowSummary) -> str:
    return f"#{summary.screen_id} {_quoted_title(summary.window_title)}"


def _path_label(path: Sequence[Any]) -> str:
    ends = [path[0], path[-1]]
    rendered: list[str] = []
    for point in ends:
        target = point.get("target") if isinstance(point, dict) else None
        if isinstance(target, dict) and target.get("text"):
            rendered.append(f'"{_clip(str(target["text"]))}"')
        elif isinstance(point, dict) and point.get("x") is not None:
            rendered.append(f"({point.get('x')},{point.get('y')})")
        else:
            rendered.append("?")
    return " -> ".join(rendered)


def action_label(action: dict[str, Any]) -> str:
    name = str(action.get("action", "?"))
    target = action.get("target")
    if target:
        return f'{name} "{_clip(str(target))}"'
    path = action.get("path")
    if isinstance(path, Sequence) and not isinstance(path, str | bytes) and path:
        return f"{name} {_path_label(path)}"
    keys = action.get("keys")
    if isinstance(keys, Sequence) and not isinstance(keys, str | bytes) and keys:
        return f"{name} {'+'.join(str(key) for key in keys)}"
    text = action.get("text")
    if text:
        return f'{name} "{_clip(str(text))}"'
    app_name = action.get("app_name")
    if app_name:
        return f'{name} "{_clip(str(app_name))}"'
    amount = action.get("amount")
    if amount is not None:
        return f"{name} {amount}"
    return name


def _label_noise(text: str) -> int:
    return sum(1 for char in text if not char.isalnum() and char not in _LABEL_PUNCTUATION)


def _pick_labels(labels: Sequence[Label], limit: int) -> list[Label]:
    seen: set[str] = set()
    usable: list[Label] = []
    for label in labels:
        if not label.targetable:
            continue
        key = normalize_text(label.text.strip())
        if key in seen:
            continue
        seen.add(key)
        usable.append(label)
    ranked = sorted(
        usable,
        key=lambda item: (
            _label_noise(item.text.strip()),
            len(item.text.strip()),
            -item.confidence,
        ),
    )
    return sorted(ranked[:limit], key=lambda item: (item.bbox.y, item.bbox.x))


# PR-014
def _labels_seen_at(labels: Sequence[Label], window: WindowSummary) -> datetime:
    return max(
        (label.seen_at for label in labels),
        default=_as_utc(window.last_seen_at),
    )


def _entrance_label(entrance: Entrance) -> str:
    mark = "" if entrance.has_way_in else " (never entered either)"
    return f"{_node_ref(entrance.window)}{mark}"


# PR-013
def _origin_clause(origin: RouteOrigin | None) -> str:
    if origin is None:
        return ""
    where = f"{_node_ref(origin.window)}, the window routes start from,"
    if origin.outgoing == 0:
        return f"; nothing has ever been clicked in {where} so nothing is reachable from it yet"
    return f"; {where} reaches {origin.reached} window(s), and this is not one of them"


# PR-010
def _step_confidence(step: RouteStep) -> str:
    if step.success_count:
        return ""
    if step.fail_count:
        return f" (unconfirmed, {step.fail_count}x failed)"
    return " (unconfirmed)"


def _route_text(
    route: Route | NoRoute, origin: int | None, target: WindowSummary, now: datetime
) -> str:
    if isinstance(route, Route):
        shown = route.steps[:_MAX_ROUTE_STEPS]
        moves = "; ".join(
            f"in {_node_ref(step.origin)} {action_label(step.action)} "
            f"-> #{step.target.screen_id}{_step_confidence(step)}"
            for step in shown
        )
        hidden = route.hops - len(shown)
        if hidden > 0:
            moves += f"; (+{hidden} more step(s))"
        shape = f"{route.hops} step(s), "
        shape += (
            f"shortest of {route.alternatives + 1} equally short"
            if route.alternatives
            else "shortest known"
        )
        if route.crosses_apps:
            shape += ", leaves the app"
        return f"route ({shape}): {moves}"

    # PR-015
    if route.reason is NoRouteReason.ALREADY_THERE:
        return (
            "route: none needed — this is the window routes start from: it was seen "
            f"{format_when(target.last_seen_at, now)}, so the app is here unless something "
            "moved it since."
        )
    if route.reason is NoRouteReason.ORIGIN_UNKNOWN:
        named = f" (#{origin})" if origin is not None else ""
        return f"route: cannot be measured — the window to start from{named} is not in memory."
    if route.reason is NoRouteReason.TARGET_UNKNOWN:
        return f"route: none — #{target.screen_id} is no longer in memory."
    if route.reason is NoRouteReason.NEVER_ENTERED:
        return (
            f"route: none known — nothing recorded leads into #{target.screen_id}: it has been "
            "seen, but no action was ever credited with opening it, so the way in is unknown "
            f"rather than absent{_origin_clause(route.origin)}."
        )
    entrances = ", ".join(_entrance_label(item) for item in route.entrances[:_MAX_ENTRANCES])
    tail = _more_note(len(route.entrances), _MAX_ENTRANCES)
    return (
        f"route: none known from here — the only recorded way into #{target.screen_id} is from "
        f"{entrances}{tail}{_origin_clause(route.origin)}."
    )


def _scroll_direction(side: ScrollEdge) -> str:
    return _SCROLL_DIRECTIONS[side]


def _offscreen_clause(offscreen: OffscreenHint) -> str:
    return (
        f"— off screen: scroll at least {offscreen.distance} px "
        f"{_scroll_direction(offscreen.side)} first; there is no coordinate to click yet"
    )


def _hit_line(hit: FindHit, now: datetime) -> str:
    stale = " STALE" if hit.window.is_stale else ""
    where = hit.where.value
    if hit.where is FindWhere.TITLE:
        matched = f"{where} {_quoted_title(hit.text)}"
    else:
        matched = f'{where} "{_clip(hit.text)}"'
    away = f" {_offscreen_clause(hit.offscreen)}" if hit.offscreen is not None else ""
    return (
        f'"{hit.window.app_name}" > {_node_ref(hit.window)}{stale} — {matched} '
        f"({hit.tier} {hit.score:.2f}), seen {format_when(hit.window.last_seen_at, now)}{away}"
    )


def _searched(report: FindReport) -> str:
    return (
        f"Searched {report.windows_searched} window(s) of {report.apps_searched} app(s), "
        f"{report.elements_searched} element(s) and their titles."
    )


def _map_coverage_line(outlines: Iterable[AppOutline], windows: int) -> str:
    unexplored = sum(item.unexplored for item in outlines)
    seen_once = sum(item.seen_once for item in outlines)
    return (
        f"coverage: {unexplored} of {windows} window(s) never clicked in, {seen_once} seen "
        "once — those are holes in the map, not empty rooms in the app."
    )


def _app_coverage_line(coverage: AppCoverage) -> str:
    outline = coverage.outline
    parts = [
        f"coverage: {coverage.activated} of {coverage.targets} known control(s) ever clicked",
        f"{outline.unexplored} of {outline.windows} window(s) never clicked in",
    ]
    parts += [
        f"{count} {phrase}"
        for count, phrase in (
            (outline.seen_once, "seen once"),
            (outline.no_way_in, "with no recorded way in"),
            (coverage.unlabeled_glyphs, "unnamed icon glyph(s)"),
            (outline.scroll_pending, "not scrolled to the end"),
        )
        if count
    ]
    return ", ".join(parts) + " (of what is remembered, not of what the app has)"


def _scroll_clause(coverage: WindowCoverage) -> str:
    extent = coverage.window.scroll
    if extent is None:
        return ""
    seen = f"{extent.viewports:.1f} screenful(s) of its own content seen"
    if coverage.scroll_progress is ScrollProgress.THROUGH:
        return f"; scrolled end to end — {seen}, so this is the whole document"
    unreached = coverage.unreached_edges
    missing = " and ".join(edge.value for edge in unreached)
    verb = "were" if len(unreached) > 1 else "was"
    return (
        f"; {seen} and the {missing} {verb} never reached — what is past it was never read, "
        "so this window is not a finished map"
    )


def _window_coverage_line(coverage: WindowCoverage) -> str:
    if coverage.walked:
        tail = "every control here has been used at least once"
    elif coverage.targets:
        tail = f"{coverage.activated} of {coverage.targets} control(s) here have ever been clicked"
    else:
        tail = "no control here is nameable yet"
    extras = _clauses(
        (
            coverage.offscreen_targets,
            f", {coverage.offscreen_targets} of them past the fold and not clickable "
            "until something scrolls",
        ),
        (
            coverage.unlabeled_icons,
            f", plus {coverage.unlabeled_icons} unnamed icon(s) nothing can aim at",
        ),
        (coverage.seen_once, " (this window rests on a single sighting)"),
    )
    return f"coverage: {tail}{extras}{_scroll_clause(coverage)}"


class _Page:
    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._lines: list[str] = []
        self._used = 0
        self.truncated = False

    def add(self, line: str) -> None:
        if self.truncated:
            return
        cost = len(line) + 1
        if self._lines and self._used + cost > self._budget:
            self.truncated = True
            return
        self._lines.append(line)
        self._used += cost

    def render(self) -> str:
        lines = list(self._lines)
        if self.truncated:
            lines.append(_TRUNCATION_NOTE)
        return "\n".join(lines)


def _add_more(page: _Page, indent: str, total: int, limit: int, noun: str) -> None:
    hidden = max(total - limit, 0)
    if hidden:
        page.add(f"{indent}(+{hidden} more {noun})")


class RecallService:
    def __init__(
        self,
        repo: GraphRepository,
        matcher: TargetMatcher,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repo = repo
        self._clock = clock or (lambda: datetime.now(UTC))
        self._index = InterfaceIndex(repo, matcher)

    def render(
        self,
        app_name: str | None = None,
        window: str | None = None,
        depth: int = DEFAULT_DEPTH,
        query: str | None = None,
        from_window: str | None = None,
    ) -> str:
        now = self._clock()
        page = _Page(_CHAR_BUDGET)
        effective_depth = max(DEFAULT_DEPTH, min(depth, MAX_DEPTH))
        page.add(
            f"now {_as_utc(now).strftime('%Y-%m-%d %H:%MZ')} (server time; "
            "stamps below are UTC, MM-DD HH:MM within this year)"
        )
        page.add(_FOCUS_MODEL_NOTE)
        self._add_icon_header(page)
        if query is None and depth != effective_depth:
            page.add(f"note: depth {depth} clamped to {effective_depth} (allowed 1..{MAX_DEPTH})")
        origin = self._resolve_origin(page, from_window)

        resolved: AppSummary | str | None = None
        if app_name is not None:
            resolved = self._resolve_app(app_name)
        scope = resolved.app_name if isinstance(resolved, AppSummary) else None

        if isinstance(resolved, str):
            page.add(resolved)
        elif query is not None:
            self._render_find(page, now, query=query, app_name=scope, origin=origin)
        elif window is not None:
            self._render_window_search(
                page, now, app_name=scope, query=window, depth=effective_depth, origin=origin
            )
        elif resolved is None:
            self._render_overview(page, now)
        else:
            self._render_app(page, now, resolved, depth=effective_depth)
        reply = page.render()
        _log.info(
            "recall.rendered",
            app_name=app_name,
            window=window,
            depth=effective_depth,
            query=query,
            chars=len(reply),
            truncated=page.truncated,
        )
        return reply

    def _resolve_origin(self, page: _Page, from_window: str | None) -> int | None:
        if from_window is None:
            return None
        matches = self._find_windows(None, from_window)
        if not matches:
            page.add(
                f'note: no window matching "{_clip(from_window)}" in memory, so routes below '
                "start at each app's most recently seen window instead."
            )
            return None
        return matches[0].screen_id

    def _add_icon_header(self, page: _Page) -> None:
        counts = self._repo.icon_counts()
        unlabeled = counts.glyphs_total - counts.glyphs_labeled
        if unlabeled <= 0:
            return
        page.add(
            f"icons: {unlabeled} glyph(s) in memory have no name yet, so they are clickable "
            "places nothing below can name — annotate_icons names them for good."
        )

    def _render_overview(self, page: _Page, now: datetime) -> None:
        summaries = self._repo.app_summaries()
        if not summaries:
            page.add(
                "memory is empty — no windows learned yet. Run observe, or a short "
                '"go and look" plan (execute_plan reports every window it walks '
                "through), and they will show up here."
            )
            return
        windows = sum(item.window_count for item in summaries)
        visits = sum(item.visit_count for item in summaries)
        outlines = self._index.map_outline()
        page.add(
            f"known: {len(summaries)} app(s), {windows} window(s), {visits} visit(s) "
            "— freshest first"
        )
        for summary in summaries[:_MAX_APPS]:
            outline = outlines.get(summary.app_name, _NO_OUTLINE)
            page.add(f"{_INDENT}{self._app_line(summary, now, outline)}")
        _add_more(page, _INDENT, len(summaries), _MAX_APPS, "app(s)")
        page.add(_map_coverage_line(outlines.values(), windows))
        page.add(
            'next: recall(app_name="<app>") for its windows and what leads where, or '
            'recall(query="<what you are after>") to search everything at once.'
        )

    def _app_line(self, summary: AppSummary, now: datetime, outline: AppOutline) -> str:
        journal = _clauses(
            (summary.visit_count, f", {summary.visit_count} visit(s)"),
            (
                summary.visits_cut,
                f" (+{summary.visits_cut} older sighting(s) trimmed from the journal)",
            ),
        )
        unexplored = f"; {outline.unexplored} never clicked in" if outline.unexplored else ""
        return (
            f'"{summary.app_name}" — {summary.window_count} window(s){journal}, '
            f"last seen {format_when(summary.last_seen_at, now)}{unexplored}"
        )

    def _resolve_app(self, query: str) -> AppSummary | str:
        summaries = self._repo.app_summaries()
        wanted = query.strip().casefold()
        if not wanted:
            return "app_name is empty — call recall() with no arguments to list what is known."
        exact = [item for item in summaries if item.app_name.casefold() == wanted]
        if exact:
            return exact[0]
        partial = [item for item in summaries if wanted in item.app_name.casefold()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = ", ".join(f'"{item.app_name}"' for item in partial[:_MAX_APP_HINTS])
            return f'"{query}" matches several known apps: {names} — name one.'
        known = ", ".join(f'"{item.app_name}"' for item in summaries[:_MAX_APP_HINTS])
        if not known:
            return (
                f'nothing known about "{query}" — memory is empty. Observe it once, or run a '
                'short "go and look" plan, and it will be remembered.'
            )
        return (
            f'nothing known about "{query}". Known apps: {known}. '
            "Everything else has to be observed once before it can be recalled."
        )

    def _render_app(self, page: _Page, now: datetime, app: AppSummary, depth: int) -> None:
        coverage = self._index.app_coverage(app.app_name)
        windows = [item.window for item in coverage.windows]
        root = latest_window(windows)
        if root is None:
            page.add(f'"{app.app_name}" has no windows in memory.')
            return
        page.add(self._app_line(app, now, coverage.outline))
        icons = self._icon_line(app.app_name, root.last_seen_at, now)
        if icons:
            page.add(icons)
        page.add(_app_coverage_line(coverage))
        page.add(f"windows ({self._counted(len(windows), _MAX_WINDOWS)}), freshest first:")
        for item in coverage.windows[:_MAX_WINDOWS]:
            page.add(f"{_INDENT}{self._window_line(item.window, now, item.unexplored)}")
        _add_more(page, _INDENT, len(windows), _MAX_WINDOWS, "window(s)")

        # PR-015
        page.add(
            f"routes start at {_node_ref(root)}, last seen "
            f"{format_when(root.last_seen_at, now)} — still there unless something moved it:"
        )
        self._render_node(page, now, root, depth=depth, shown={root.screen_id}, level=1)
        page.add(
            f'next: recall(app_name="{app.app_name}", window="<title or #id>") to open another '
            'window, or recall(query="…") to search every app at once.'
        )

    # PR-014
    def _icon_line(self, app_name: str, seen_at: datetime, now: datetime) -> str:
        counts = self._repo.icon_counts(app_name)
        if counts.glyphs_total <= 0:
            return ""
        unlabeled = counts.glyphs_total - counts.glyphs_labeled
        tail = " — annotate_icons names them" if unlabeled else ""
        return (
            f"icons: {counts.glyphs_labeled} labeled / {unlabeled} unlabeled "
            f"glyph(s), drawn in {counts.icon_elements} place(s) in windows read up to "
            f"{format_stamp(seen_at, now)}{tail}"
        )

    def _window_line(self, window: WindowSummary, now: datetime, unexplored: bool = False) -> str:
        visits = f", {window.visit_count} visit(s)" if window.visit_count else ""
        marks = _clauses(
            (window.is_stale, " STALE (marked for re-reading; treat its contents as unconfirmed)"),
            (unexplored, ", never clicked in"),
        )
        return (
            f"{_node_ref(window)} — {window.element_count} element(s){visits}, "
            f"seen {format_when(window.last_seen_at, now)}{marks}"
        )

    def _render_window_search(
        self,
        page: _Page,
        now: datetime,
        app_name: str | None,
        query: str,
        depth: int,
        origin: int | None = None,
    ) -> None:
        matches = self._find_windows(app_name, query)
        scope = f' in "{app_name}"' if app_name else ""
        if not matches:
            page.add(f'no window matching "{query}"{scope} in memory.')
            self._render_window_hints(page, now, app_name)
            return
        target = matches[0]
        coverage = self._index.window_coverage(target.screen_id)
        unexplored = coverage is not None and coverage.unexplored
        page.add(f'"{target.app_name}" > {self._window_line(target, now, unexplored)}')
        page.add(self._route_line(target, origin, now))
        if coverage is not None:
            page.add(_window_coverage_line(coverage))
        self._render_node(page, now, target, depth=depth, shown={target.screen_id}, level=0)
        others = matches[1:]
        if others:
            named = ", ".join(
                f"#{item.screen_id} {_quoted_title(item.window_title)} "
                f"({format_relative(item.last_seen_at, now)})"
                for item in others[:_MAX_ALTERNATIVES]
            )
            tail = _more_note(len(others), _MAX_ALTERNATIVES)
            page.add(f'also matching "{query}": {named}{tail}')

    def _find_windows(self, app_name: str | None, query: str) -> list[WindowSummary]:
        stripped = query.strip()
        if not stripped:
            return []
        candidate_id = stripped.removeprefix("#")
        if candidate_id.isdigit():
            by_id = self._repo.window_summaries(app_name=app_name, screen_ids=[int(candidate_id)])
            if by_id:
                return by_id
        return self._repo.window_summaries(app_name=app_name, title_query=stripped)

    def _render_window_hints(self, page: _Page, now: datetime, app_name: str | None) -> None:
        if app_name is None:
            page.add("call recall() with no arguments to see which apps are known.")
            return
        windows = self._repo.window_summaries(app_name=app_name)
        if not windows:
            page.add(f'"{app_name}" has no windows in memory.')
            return
        page.add(f'windows known for "{app_name}":')
        for window in windows[:_MAX_WINDOWS]:
            page.add(f"{_INDENT}{self._window_line(window, now)}")
        _add_more(page, _INDENT, len(windows), _MAX_WINDOWS, "window(s)")

    def _route_line(self, target: WindowSummary, origin: int | None, now: datetime) -> str:
        start = origin
        if start is None:
            here = self._index.last_seen(target.app_name)
            if here is None:
                return "route: none — nothing else of this app is in memory."
            start = here.screen_id
        return _route_text(self._index.route(target.screen_id, start), start, target, now)

    def _render_find(
        self,
        page: _Page,
        now: datetime,
        query: str,
        app_name: str | None,
        origin: int | None,
    ) -> None:
        text = query.strip()
        if not text:
            page.add(
                'query is empty — recall(query="font size") searches element labels, icon '
                "names and window titles at once."
            )
            return
        report = self._index.find(text, app_name=app_name, origin=origin, limit=_MAX_FIND_HITS)
        scope = f' in "{app_name}"' if app_name else ""
        if not report.hits:
            page.add(f'nothing matching "{_clip(text)}"{scope} in memory. {_searched(report)}')
            page.add(self._holes_line(app_name))
            return
        more = f" (+{report.truncated} more)" if report.truncated else ""
        page.add(
            f'found "{_clip(text)}"{scope} — {len(report.hits)} place(s){more}, best match '
            f"first. {_searched(report)}"
        )
        for hit in report.hits:
            page.add(f"{_INDENT}{_hit_line(hit, now)}")
            page.add(f"{_INDENT * 2}{_route_text(hit.route, None, hit.window, now)}")

    def _holes_line(self, app_name: str | None) -> str:
        outlines = self._index.map_outline()
        if app_name is not None:
            outlines = {name: item for name, item in outlines.items() if name == app_name}
        unexplored = sum(item.unexplored for item in outlines.values())
        windows = sum(item.windows for item in outlines.values())
        if not windows:
            return (
                "nothing is mapped here yet, so this is not evidence of absence — observe the "
                "app once, or walk it with a short plan, and ask again."
            )
        if not unexplored:
            return (
                f"every one of those {windows} window(s) has been clicked in, so the map is as "
                "walked as it gets — but it still only holds what was on screen when it was "
                "read: anything a scroll away was never seen."
            )
        return (
            f"read this as 'never seen', not 'not there': {unexplored} of {windows} window(s) "
            "have never been clicked in, so whatever they lead to was never opened."
        )

    def _render_node(
        self,
        page: _Page,
        now: datetime,
        window: WindowSummary,
        depth: int,
        shown: set[int],
        level: int,
    ) -> None:
        if page.truncated:
            return
        self._render_labels(page, now, window, level)
        self._render_transitions(page, now, window, depth, shown, level)

    def _render_labels(self, page: _Page, now: datetime, window: WindowSummary, level: int) -> None:
        indent = _INDENT * (level + 1)
        labels = self._repo.window_labels([window.screen_id]).get(window.screen_id, [])
        picked = _pick_labels(labels, _MAX_ELEMENTS)
        if not picked:
            stamp = format_stamp(_labels_seen_at(labels, window), now)
            page.add(f"{indent}elements: none readable (last read {stamp})")
            return
        read_at = format_stamp(_labels_seen_at(picked, window), now)
        page.add(
            f"{indent}elements ({self._counted(window.element_count, len(picked))}, "
            f"read {read_at}): " + ", ".join(_clip(item.text.strip()) for item in picked)
        )

    def _render_transitions(
        self,
        page: _Page,
        now: datetime,
        window: WindowSummary,
        depth: int,
        shown: set[int],
        level: int,
    ) -> None:
        indent = _INDENT * (level + 1)
        moves = self._ranked_transitions(window.screen_id)
        if not moves:
            page.add(f"{indent}transitions: none recorded (nothing has been clicked here yet)")
            return
        page.add(f"{indent}transitions ({self._counted(len(moves), _MAX_TRANSITIONS)}):")
        targets = {
            summary.screen_id: summary
            for summary in self._repo.window_summaries(
                screen_ids=[move.to_window for move in moves[:_MAX_TRANSITIONS]]
            )
        }
        for move in moves[:_MAX_TRANSITIONS]:
            target = targets.get(move.to_window)
            if target is None:
                page.add(
                    f"{indent}{_INDENT}{action_label(move.action)} -> #{move.to_window} "
                    "(window no longer in memory)"
                )
                continue
            repeated = target.screen_id in shown
            rendered_target = self._transition_target(move, target, now, repeated)
            page.add(f"{indent}{_INDENT}{action_label(move.action)} -> {rendered_target}")
            if repeated or depth <= 1:
                continue
            shown.add(target.screen_id)
            self._render_node(page, now, target, depth - 1, shown, level + 2)
        _add_more(page, indent + _INDENT, len(moves), _MAX_TRANSITIONS, "transition(s)")

    def _ranked_transitions(self, screen_id: int) -> list[Transition]:
        return sorted(
            self._repo.transitions(from_screen=screen_id), key=transition_quality, reverse=True
        )

    def _transition_target(
        self, move: Transition, target: WindowSummary, now: datetime, repeated: bool
    ) -> str:
        counts = f"{move.success_count}x ok" if move.success_count else "never confirmed"
        if move.fail_count:
            counts += f", {move.fail_count}x failed"
        mark = " [shown above]" if repeated else ""
        stale = " STALE" if target.is_stale else ""
        return (
            f"{_node_ref(target)}{stale} — seen {format_when(target.last_seen_at, now)}, "
            f"{counts}{mark}"
        )

    @staticmethod
    def _counted(total: int, limit: int) -> str:
        shown = min(total, limit)
        return f"{shown} of {total}" if shown < total else str(total)
