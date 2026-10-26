from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from choto.config import Settings
from choto.graph.repository import GraphRepository, ScreenSighting
from choto.log import get_logger
from choto.models import Action

_log = get_logger(__name__)

UNCONFIRMED_MAX_VISITS = 1

RUNS_KEPT = 5000

ROUTE_ACTIONS = frozenset(
    {
        Action.CLICK,
        Action.DOUBLE_CLICK,
        Action.RIGHT_CLICK,
        Action.HOTKEY,
        Action.DRAG,
    }
)

ROUTE_KINDS = tuple(sorted(action.value for action in ROUTE_ACTIONS))


@dataclass(frozen=True)
class EvictionPlan:
    unconfirmed: tuple[int, ...] = ()
    capped: tuple[int, ...] = ()
    orphan_glyphs: tuple[int, ...] = ()
    surviving: int = 0
    spared: int = 0
    stray_edges: int = 0
    visits_over_cap: int = 0
    runs_over_cap: int = 0

    @property
    def doomed(self) -> tuple[int, ...]:
        return tuple(sorted(self.unconfirmed + self.capped))

    def __bool__(self) -> bool:
        return bool(
            self.unconfirmed
            or self.capped
            or self.orphan_glyphs
            or self.stray_edges
            or self.visits_over_cap
            or self.runs_over_cap
        )


def _reference_moment(node: ScreenSighting) -> datetime:
    if node.last_visit_at is None:
        return node.last_seen_at
    return max(node.last_visit_at, node.last_seen_at)


def plan_eviction(repo: GraphRepository, settings: Settings) -> EvictionPlan:
    over_cap = repo.count_visits_over_cap(settings.map_visits_per_app)
    runs_over_cap = repo.count_runs_over_cap(RUNS_KEPT)
    nodes = repo.screen_sightings()
    # PR-041
    stray_edges = repo.count_edges_except(ROUTE_KINDS)
    if not nodes:
        return EvictionPlan(
            orphan_glyphs=repo.orphan_unlabeled_glyphs(),
            stray_edges=stray_edges,
            visits_over_cap=over_cap,
            runs_over_cap=runs_over_cap,
        )

    routed = repo.screens_with_edges_of(ROUTE_KINDS)
    unconfirmed: list[int] = []
    for node in nodes:
        if node.visit_count > UNCONFIRMED_MAX_VISITS or node.screen_id in routed:  # PR-032
            continue
        later = repo.count_visits_since(node.app_name, node.window_title, _reference_moment(node))
        if later >= settings.map_unconfirmed_after_visits:
            unconfirmed.append(node.screen_id)

    refuted = set(unconfirmed)
    windows: dict[tuple[str, str], list[ScreenSighting]] = defaultdict(list)
    for node in nodes:
        if node.screen_id not in refuted and node.screen_id not in routed:  # PR-034
            windows[(node.app_name, node.window_title)].append(node)

    capped: list[int] = []
    for kept in windows.values():
        if len(kept) <= settings.map_max_nodes_per_window:
            continue
        kept.sort(key=lambda node: (node.last_seen_at, node.screen_id), reverse=True)
        capped.extend(node.screen_id for node in kept[settings.map_max_nodes_per_window :])

    return EvictionPlan(
        unconfirmed=tuple(sorted(unconfirmed)),
        capped=tuple(sorted(capped)),
        orphan_glyphs=repo.orphan_unlabeled_glyphs(sorted(unconfirmed + capped)),
        surviving=len(nodes) - len(unconfirmed) - len(capped),
        spared=sum(1 for node in nodes if node.screen_id in routed),  # PR-035
        stray_edges=stray_edges,
        visits_over_cap=over_cap,
        runs_over_cap=runs_over_cap,
    )


def evict(repo: GraphRepository, settings: Settings) -> EvictionPlan:
    plan = plan_eviction(repo, settings)
    if not plan:
        return plan
    edges_dropped = repo.delete_edges_except(ROUTE_KINDS) if plan.stray_edges else 0
    if plan.doomed:
        repo.delete_screens(plan.doomed)
    if plan.orphan_glyphs:
        repo.delete_icon_glyphs(plan.orphan_glyphs)
    rotation = repo.rotate_visits(settings.map_visits_per_app)
    runs_dropped = repo.rotate_runs(RUNS_KEPT)
    _log.info(
        "map.evicted",
        unconfirmed=len(plan.unconfirmed),
        capped=len(plan.capped),
        orphan_glyphs=len(plan.orphan_glyphs),
        remaining=plan.surviving,
        spared=plan.spared,
        edges_dropped=edges_dropped,
        visits_rotated=rotation.total,
        runs_dropped=runs_dropped,
    )
    return plan
