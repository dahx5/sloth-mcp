from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from choto.executor.applauncher import AppLauncher
from choto.graph.missions import (
    ActiveMissionExists,
    MissionItemDraft,
    MissionItemStatus,
    MissionStatus,
    WorldFingerprint,
    compare_fingerprints,
    render_checklist,
    render_status,
)
from choto.graph.repository import GraphRepository
from choto.log import get_logger
from choto.models import Action, ExecutionReport, Plan, RunStatus, SeenScreen
from choto.vision.windowsource import WindowSource

_log = get_logger(__name__)

_ITEM_FIELDS = ("title", "intent", "acceptance")

_MARK_STATUSES = {
    "passed": MissionItemStatus.PASSED,
    "failed": MissionItemStatus.FAILED,
    "skipped": MissionItemStatus.SKIPPED,
}

_FINISH_STATUSES = {
    "done": MissionStatus.DONE,
    "abandoned": MissionStatus.ABANDONED,
}

_ROUTE_HASH_CHARS = 6

_NO_ACTIVE_MISSION = (
    "no active mission. mission_start(goal, items) writes one down — the goal, and per item a "
    "title, the intent behind it and the acceptance that settles it — after which every reply "
    "carries where you are in it."
)

_DIVERGENCE_RULE = (
    "A difference is NOT evidence that the application was updated, and this server will "
    "never read it that way: it is material for you. Decide yourself — run the steps "
    "deliberately with execute_plan(mission_item_id=...), which records the world it "
    "succeeds in as the item's new truth, or mark the item by hand with mission_mark."
)

PlanRunner = Callable[[Plan], ExecutionReport]


@dataclass(frozen=True)
class ReplayOutcome:
    report: ExecutionReport | None
    text: str


class MissionService:
    def __init__(self, repo: GraphRepository, windows: WindowSource, apps: AppLauncher) -> None:
        self._repo = repo
        self._windows = windows
        self._apps = apps

    def start(self, goal: str, items: Sequence[Mapping[str, Any]]) -> str:
        try:
            drafts = _drafts(items)
        except (TypeError, ValueError) as exc:
            return f"mission_start did nothing: {exc}"
        try:
            mission = self._repo.create_mission(goal, drafts)
        except ActiveMissionExists as exc:
            return str(exc)
        except (TypeError, ValueError) as exc:
            return f"mission_start did nothing: {exc}"
        return render_status(mission)

    def status(self, full: bool = False) -> str:
        mission = self._repo.active_mission()
        if mission is None:
            return _NO_ACTIVE_MISSION
        return render_checklist(mission) if full else render_status(mission)

    def mark(self, item_id: int, status: str, reason: str = "") -> str:
        item_status = _MARK_STATUSES.get(status.strip().lower())
        if item_status is None:
            return (
                f"mission_mark did nothing: status must be one of "
                f"{', '.join(sorted(_MARK_STATUSES))}, got {status!r}."
            )
        return self._mark(item_id, item_status, fail_reason=reason)

    def finish(self, status: str) -> str:
        mission_status = _FINISH_STATUSES.get(status.strip().lower())
        if mission_status is None:
            return (
                f"mission_finish did nothing: status must be one of "
                f"{', '.join(sorted(_FINISH_STATUSES))}, got {status!r}."
            )
        try:
            mission = self._repo.finish_mission(mission_status)
        except KeyError:
            return _NO_ACTIVE_MISSION
        except ValueError as exc:
            return f"mission_finish did nothing: {exc}"
        return render_checklist(mission)

    def record_run(
        self,
        item_id: int,
        report: ExecutionReport,
        steps: Sequence[Mapping[str, Any]],
        llm_version_tag: str = "",
    ) -> str:
        if report.status is RunStatus.SUCCESS:
            fingerprint = _fingerprint(report, llm_version_tag, self._apps)
            block = self._mark(
                item_id,
                MissionItemStatus.PASSED,
                recipe=list(steps),
                fingerprint=fingerprint,
            )
            if fingerprint is None:
                return (
                    "no fingerprint was stored with this pass: the run walked through no "
                    "window that could be attributed to an application (an unscoped read "
                    "belongs to no window), so there was nothing to record the world as. "
                    "The recipe was kept; mission_replay will run it without a preflight.\n"
                    f"{block}"
                )
            return block
        return self._mark(item_id, MissionItemStatus.FAILED, fail_reason=_failure_reason(report))

    def replay(self, item_id: int, runner: PlanRunner, llm_version_tag: str = "") -> ReplayOutcome:
        mission = self._repo.active_mission()
        if mission is None:
            return ReplayOutcome(None, _NO_ACTIVE_MISSION)

        item = next((entry for entry in mission.items if entry.item_id == item_id), None)
        if item is None:
            known = ", ".join(f"#{entry.item_id}" for entry in mission.items)
            return ReplayOutcome(
                None,
                f"mission_replay did nothing: mission #{mission.mission_id} has no item "
                f"#{item_id}. Its items are {known}.",
            )
        if item.recipe is None:
            return ReplayOutcome(
                None,
                f"mission_replay did nothing: item #{item_id} has no recipe — nothing has been "
                "recorded as working for it yet. Run it once with "
                f"execute_plan(steps=[...], mission_item_id={item_id}); a successful run keeps "
                "its steps here, and that is what this tool repeats.",
            )
        try:
            plan = Plan.model_validate({"steps": list(item.recipe)})
        except ValidationError as exc:
            return ReplayOutcome(
                None,
                f"mission_replay did nothing: the recipe stored for item #{item_id} is no longer "
                f"a valid plan:\n{_validation_lines(exc)}\nRe-run the item with "
                f"execute_plan(steps=[...], mission_item_id={item_id}) to record a fresh one.",
            )

        notes: list[str] = []
        saved = item.fingerprint
        if saved is None:
            notes.append(
                "preflight: none — no fingerprint was stored with this recipe, so there was "
                "nothing to compare the world against before running it."
            )
        else:
            current = self._current_world(saved, item.recipe, llm_version_tag)
            verdict = compare_fingerprints(saved, current)
            if verdict.diverged:
                _log.info(
                    "mission.replay_refused",
                    item_id=item_id,
                    diverged=list(verdict.diverged),
                )
                return ReplayOutcome(
                    None,
                    f"{_divergence_text(item_id, saved, current, verdict.diverged)}\n\n"
                    f"{render_status(mission)}",
                )
            notes.append(_preflight_note(verdict.unverifiable))

        report = runner(plan)

        if report.status is RunStatus.SUCCESS:
            notes.extend(_route_notes(item_id, saved, _route(report.seen_screens)))
            block = self._mark(item_id, MissionItemStatus.PASSED)
        else:
            block = self._mark(
                item_id, MissionItemStatus.FAILED, fail_reason=_failure_reason(report)
            )
        _log.info("mission.replayed", item_id=item_id, status=report.status.value)
        return ReplayOutcome(report, "\n".join([*notes, block]))

    def _mark(
        self,
        item_id: int,
        status: MissionItemStatus,
        fail_reason: str = "",
        recipe: Sequence[Mapping[str, Any]] | None = None,
        fingerprint: WorldFingerprint | None = None,
    ) -> str:
        try:
            mission = self._repo.mark_item(
                item_id,
                status,
                fail_reason=fail_reason,
                recipe=recipe,
                fingerprint=fingerprint,
            )
        except KeyError:
            return (
                f"the mission was not updated: there is no mission item #{item_id}. "
                "mission_status(full=true) lists the items and their ids."
            )
        except (TypeError, ValueError) as exc:
            return f"the mission was not updated: {exc}"
        return render_status(mission)

    def _current_world(
        self,
        saved: WorldFingerprint,
        recipe: Sequence[Mapping[str, Any]],
        llm_version_tag: str,
    ) -> WorldFingerprint:
        observed = _leading_focus_app(recipe) or self._windows.frontmost_app_name()
        app_name = saved.app_name if self._apps.same_app(observed, saved.app_name) else observed
        return WorldFingerprint(
            app_name=app_name or saved.app_name,
            app_version=self._apps.app_version(app_name or saved.app_name),
            llm_version_tag=llm_version_tag,
            route_phashes=(),
        )


def _drafts(items: Sequence[Mapping[str, Any]]) -> list[MissionItemDraft]:
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError(f"items must be a list of checklist items, got {type(items).__name__}.")
    entries = list(items)
    if not entries:
        raise ValueError(
            "items is empty: a goal with no checklist is the state a mission exists to replace."
        )
    drafts: list[MissionItemDraft] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"item {position} must be an object with {', '.join(_ITEM_FIELDS)}, "
                f"got {type(entry).__name__}."
            )
        unknown = sorted(set(entry) - set(_ITEM_FIELDS))
        if unknown:
            raise ValueError(
                f"item {position} has unknown field(s) {', '.join(unknown)}; an item is "
                f"{{{', '.join(_ITEM_FIELDS)}}}."
            )
        missing = [field for field in _ITEM_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"item {position} is missing {', '.join(missing)}.")
        for field in _ITEM_FIELDS:
            if not isinstance(entry[field], str):
                raise TypeError(
                    f"item {position}: {field} must be a string, got {type(entry[field]).__name__}."
                )
        try:
            drafts.append(
                MissionItemDraft(
                    title=entry["title"],
                    intent=entry["intent"],
                    acceptance=entry["acceptance"],
                )
            )
        except ValueError as exc:
            raise ValueError(f"item {position}: {exc}") from exc
    return drafts


def _validation_lines(exc: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
        for error in exc.errors()
    )


def _route(seen: Sequence[SeenScreen]) -> tuple[str, ...]:
    return tuple(screen.phash for screen in seen if screen.scoped)


def _fingerprint(
    report: ExecutionReport, llm_version_tag: str, apps: AppLauncher
) -> WorldFingerprint | None:
    app_name = next(
        (screen.app_name for screen in report.seen_screens if screen.scoped and screen.app_name),
        "",
    )
    if not app_name.strip():
        return None
    return WorldFingerprint(
        app_name=app_name,
        app_version=apps.app_version(app_name),
        llm_version_tag=llm_version_tag,
        route_phashes=_route(report.seen_screens),
    )


def _failure_reason(report: ExecutionReport) -> str:
    where = "" if report.failed_step is None else f" at step {report.failed_step}"
    why = f": {report.reason}" if report.reason else ""
    return f"run {report.status.value}{where}{why}"


def _leading_focus_app(recipe: Sequence[Mapping[str, Any]]) -> str:
    if not recipe:
        return ""
    first = recipe[0]
    if not isinstance(first, Mapping) or first.get("action") != Action.FOCUS_APP.value:
        return ""
    app_name = first.get("app_name")
    return app_name.strip() if isinstance(app_name, str) else ""


def _field_value(fingerprint: WorldFingerprint, field: str) -> str:
    value = getattr(fingerprint, field)
    if isinstance(value, tuple):
        return _short_route(value)
    return f'"{value}"' if value else "(unknown)"


def _short_route(route: Sequence[str]) -> str:
    return " -> ".join(phash[:_ROUTE_HASH_CHARS] for phash in route) if route else "(none)"


def _divergence_text(
    item_id: int,
    saved: WorldFingerprint,
    current: WorldFingerprint,
    diverged: Sequence[str],
) -> str:
    lines = [
        f"mission_replay did NOT run: the world differs from the one item #{item_id} passed in, "
        f"so the recipe was not executed.",
    ]
    lines.extend(
        f"  {field}: recorded {_field_value(saved, field)}, now {_field_value(current, field)}"
        for field in diverged
    )
    lines.append(_DIVERGENCE_RULE)
    return "\n".join(lines)


def _preflight_note(unverifiable: Sequence[str]) -> str:
    names = [field for field in unverifiable if field != "route_phashes"]
    if not names:
        return (
            "preflight: every recorded field matches the world as it is now. The route is "
            "compared after the run, not before."
        )
    return (
        f"preflight: nothing comparable differs, but {', '.join(names)} could not be compared "
        "(one side does not know them), so this is not evidence that the world is unchanged. "
        "The route is compared after the run, not before."
    )


def _route_notes(item_id: int, saved: WorldFingerprint | None, walked: Sequence[str]) -> list[str]:
    if saved is None:
        return []
    if not saved.route_phashes:
        return [
            f"route: not checked — no route was recorded with item #{item_id}'s recipe, so this "
            "run's path could not be held against anything."
        ]
    if not walked:
        return [
            f"route: not checked — this run recorded no scoped window, so there was nothing to "
            f"compare with item #{item_id}'s recorded path."
        ]
    if tuple(walked) == saved.route_phashes:
        return []
    return [
        f"route CHANGED, and the recipe worked anyway: item #{item_id} recorded "
        f"{_short_route(saved.route_phashes)}, this run went "
        f"{_short_route(walked)}. The steps passed, so the item is marked passed and its stored "
        f"fingerprint is left exactly as it was. {_DIVERGENCE_RULE}"
    ]


__all__ = [
    "MissionService",
    "PlanRunner",
    "ReplayOutcome",
]
