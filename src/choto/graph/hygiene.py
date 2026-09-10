from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from choto.config import Settings
from choto.graph.eviction import UNCONFIRMED_MAX_VISITS, plan_eviction
from choto.graph.repository import GraphRepository, ScreenSighting
from choto.log import get_logger
from choto.recall import format_when

_log = get_logger(__name__)

BACKUP_STAMP_FORMAT = "%Y%m%d-%H%M%S"
BACKUP_SUFFIX = ".backup-"

_INDENT = "  "


class MapError(RuntimeError): ...


@dataclass(frozen=True)
class AppMapStat:
    app_name: str
    nodes: int
    unconfirmed: int
    visits: int
    sightings_ever: int
    last_seen_at: datetime | None


@dataclass(frozen=True)
class MapStats:
    nodes: int
    elements: int
    edges: int
    visits: int
    orphan_visits: int
    unconfirmed: int
    evictable_unconfirmed: int
    evictable_capped: int
    glyphs: int
    glyphs_labeled: int
    glyph_positions: int
    evictable_glyphs: int
    visits_cut: int
    evictable_visits: int
    journal_starts_at: datetime | None
    oldest_retained_at: datetime | None
    apps: tuple[AppMapStat, ...]


def collect_stats(repo: GraphRepository, settings: Settings) -> MapStats:
    counts = repo.graph_counts()
    nodes = repo.screen_sightings()
    journal = repo.journal_spans()
    plan = plan_eviction(repo, settings)
    icons = repo.icon_counts()

    grouped: dict[str, list[ScreenSighting]] = defaultdict(list)
    for node in nodes:
        grouped[node.app_name].append(node)

    apps = [
        AppMapStat(
            app_name=app_name,
            nodes=len(group),
            unconfirmed=sum(1 for node in group if node.visit_count <= UNCONFIRMED_MAX_VISITS),
            visits=sum(node.visit_count for node in group),
            sightings_ever=span.total if (span := journal.get(app_name)) else 0,
            last_seen_at=max(node.last_seen_at for node in group),
        )
        for app_name, group in grouped.items()
    ]
    apps.extend(
        AppMapStat(
            app_name=app_name,
            nodes=0,
            unconfirmed=0,
            visits=0,
            sightings_ever=span.total,
            last_seen_at=None,
        )
        for app_name, span in journal.items()
        if app_name not in grouped
    )
    apps.sort(key=lambda stat: (stat.nodes, stat.sightings_ever, stat.app_name), reverse=True)

    starts = [span.first_seen_at for span in journal.values()]
    stored = [
        span.oldest_retained_at for span in journal.values() if span.oldest_retained_at is not None
    ]

    return MapStats(
        nodes=counts.screens,
        elements=counts.elements,
        edges=counts.edges,
        visits=counts.visits,
        orphan_visits=counts.orphan_visits,
        unconfirmed=sum(stat.unconfirmed for stat in apps),
        evictable_unconfirmed=len(plan.unconfirmed),
        evictable_capped=len(plan.capped),
        glyphs=icons.glyphs_total,
        glyphs_labeled=icons.glyphs_labeled,
        glyph_positions=icons.icon_elements,
        evictable_glyphs=len(plan.orphan_glyphs),
        visits_cut=sum(span.cut for span in journal.values()),
        evictable_visits=plan.visits_over_cap,
        journal_starts_at=min(starts) if starts else None,
        oldest_retained_at=min(stored) if stored else None,
        apps=tuple(apps),
    )


def _journal_history_line(stats: MapStats, settings: Settings, now: datetime) -> str:
    cap = f"cap {settings.map_visits_per_app} sighting(s) per app"
    if stats.journal_starts_at is None:
        return f"journal history: nothing recorded yet ({cap})"
    first = format_when(stats.journal_starts_at, now)
    if not stats.visits_cut:
        pending = f", {stats.evictable_visits} over it right now" if stats.evictable_visits else ""
        return f"journal history: reaches back to {first}, nothing trimmed yet ({cap}{pending})"
    stored = (
        format_when(stats.oldest_retained_at, now)
        if stats.oldest_retained_at is not None
        else "nothing"
    )
    return (
        f"journal history: first sighting {first}, but {stats.visits_cut} older sighting(s) "
        f"have been trimmed ({cap}) — stored rows start at {stored}"
    )


def render_stats(stats: MapStats, settings: Settings, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    lines = [
        f"map: {stats.nodes} node(s), {stats.elements} element(s), {stats.edges} edge(s)",
        f"journal: {stats.visits} sighting(s), {stats.orphan_visits} of them orphaned "
        "(their node is gone; the sighting is not)",
        _journal_history_line(stats, settings, now),
        f"unconfirmed: {stats.unconfirmed} node(s) met at most once — of those, "
        f"{stats.evictable_unconfirmed} have already been outlived by "
        f"{settings.map_unconfirmed_after_visits} later sighting(s) of their app",
        f"over the per-window cap of {settings.map_max_nodes_per_window}: "
        f"{stats.evictable_capped} node(s)",
        f"next eviction would remove {stats.evictable_unconfirmed + stats.evictable_capped} "
        f"of {stats.nodes} node(s)"
        + (f" and trim {stats.evictable_visits} sighting(s)" if stats.evictable_visits else ""),
        f"glyphs: {stats.glyphs} icon drawing(s), {stats.glyphs_labeled} named, drawn in "
        f"{stats.glyph_positions} place(s) — {stats.evictable_glyphs} nameless and orphaned "
        "(nothing draws them any more; the next eviction sweeps them, named ones never)",
    ]
    if not stats.apps:
        lines.append("by app: nothing known yet")
        return "\n".join(lines) + "\n"

    lines.append("by app (nodes / unconfirmed / sightings now / sightings ever / last seen):")
    for stat in stats.apps:
        last_seen = (
            stat.last_seen_at.strftime("%Y-%m-%d %H:%MZ")
            if stat.last_seen_at is not None
            else "no nodes left"
        )
        lines.append(
            f'{_INDENT}"{stat.app_name}" — {stat.nodes} / {stat.unconfirmed} / '
            f"{stat.visits} / {stat.sightings_ever} / {last_seen}"
        )
    return "\n".join(lines) + "\n"


def backup_database(db_path: Path, *, now: datetime | None = None) -> Path:
    if str(db_path) == ":memory:":
        raise MapError("an in-memory database cannot be backed up; set CHOTO_DB_PATH to a file")
    if not db_path.is_file():
        raise MapError(f"no database at {db_path} — there is nothing to back up or change")

    stamp = (now or datetime.now()).strftime(BACKUP_STAMP_FORMAT)
    base = f"{db_path.name}{BACKUP_SUFFIX}{stamp}"
    target = db_path.with_name(base)
    attempt = 1
    while target.exists():
        attempt += 1
        target = db_path.with_name(f"{base}-{attempt}")

    try:
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise MapError(f"cannot open {db_path} for backup: {exc}") from exc
    try:
        destination = sqlite3.connect(target)
        try:
            with destination:
                source.backup(destination)
        finally:
            destination.close()
    except sqlite3.Error as exc:
        target.unlink(missing_ok=True)
        raise MapError(f"backing up {db_path} to {target} failed: {exc}") from exc
    finally:
        source.close()

    _log.info("map.backed_up", source=str(db_path), backup=str(target))
    return target


def nodes_of_app(repo: GraphRepository, app_name: str) -> list[int]:
    windows = repo.window_summaries(app_name=app_name)
    if not windows:
        known = ", ".join(f'"{summary.app_name}"' for summary in repo.app_summaries())
        detail = f" Known: {known}." if known else " The map is empty."
        raise MapError(f'no application named "{app_name}" has any nodes in the map.{detail}')
    return [window.screen_id for window in windows]


def all_nodes(repo: GraphRepository) -> list[int]:
    return [window.screen_id for window in repo.window_summaries()]


def forget_nodes(repo: GraphRepository, screen_ids: list[int]) -> int:
    deleted = repo.delete_screens(screen_ids)
    _log.info("map.forgotten", nodes=deleted)
    return deleted
