from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

_MAX_GOAL_CHARS = 140
_MAX_TITLE_CHARS = 80
_MAX_INTENT_CHARS = 220
_MAX_ACCEPTANCE_CHARS = 180
_MAX_REASON_CHARS = 120

_MAX_FAILURES_SHOWN = 5

_ELLIPSIS = "…"

_INDENT = "  "


class MissionStatus(str, Enum):
    ACTIVE = "active"
    DONE = "done"
    ABANDONED = "abandoned"


class MissionItemStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActiveMissionExists(RuntimeError): ...


def _clip(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + _ELLIPSIS


def _require_text(value: str, field: str, subject: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"A mission {subject} must have a non-empty {field}.")
    return stripped


@dataclass(frozen=True)
class WorldFingerprint:
    app_name: str
    app_version: str = ""
    llm_version_tag: str = ""
    route_phashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.app_name.strip():
            raise ValueError("A world fingerprint must name its application.")
        object.__setattr__(self, "app_name", self.app_name.strip())
        object.__setattr__(self, "app_version", self.app_version.strip())
        object.__setattr__(self, "llm_version_tag", self.llm_version_tag.strip())
        route = tuple(self.route_phashes)
        for phash in route:
            if not isinstance(phash, str) or not phash.strip():
                raise ValueError(f"A route hash must be a non-empty string, got {phash!r}.")
        object.__setattr__(self, "route_phashes", tuple(phash.strip() for phash in route))

    def to_json(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "llm_version_tag": self.llm_version_tag,
            "route_phashes": list(self.route_phashes),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> WorldFingerprint:
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"A fingerprint payload must be a mapping, got {type(payload).__name__}."
            )
        unknown = set(payload) - {"app_name", "app_version", "llm_version_tag", "route_phashes"}
        if unknown:
            raise ValueError(f"Unknown fingerprint fields: {sorted(unknown)}.")
        route = payload.get("route_phashes", ())
        if isinstance(route, str) or not isinstance(route, Sequence):
            raise TypeError(f"route_phashes must be a list of hashes, got {type(route).__name__}.")
        for field_name in ("app_name", "app_version", "llm_version_tag"):
            value = payload.get(field_name, "")
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string, got {type(value).__name__}.")
        return cls(
            app_name=str(payload.get("app_name", "")),
            app_version=str(payload.get("app_version", "")),
            llm_version_tag=str(payload.get("llm_version_tag", "")),
            route_phashes=tuple(route),
        )


@dataclass(frozen=True)
class FingerprintVerdict:
    match: bool
    diverged: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()


def compare_fingerprints(saved: WorldFingerprint, current: WorldFingerprint) -> FingerprintVerdict:
    diverged: list[str] = []
    unverifiable: list[str] = []

    if saved.app_name != current.app_name:
        diverged.append("app_name")
    for field_name in ("app_version", "llm_version_tag"):
        was = getattr(saved, field_name)
        now = getattr(current, field_name)
        if not was or not now:
            unverifiable.append(field_name)
        elif was != now:
            diverged.append(field_name)
    if not saved.route_phashes or not current.route_phashes:
        unverifiable.append("route_phashes")
    elif saved.route_phashes != current.route_phashes:
        diverged.append("route_phashes")

    return FingerprintVerdict(
        match=not diverged,
        diverged=tuple(diverged),
        unverifiable=tuple(unverifiable),
    )


@dataclass(frozen=True)
class MissionItemDraft:
    title: str
    intent: str
    acceptance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _require_text(self.title, "title", "item"))
        object.__setattr__(self, "intent", _require_text(self.intent, "intent", "item"))
        object.__setattr__(self, "acceptance", _require_text(self.acceptance, "acceptance", "item"))


@dataclass(frozen=True)
class MissionItem:
    item_id: int
    seq: int
    title: str
    intent: str
    acceptance: str
    status: MissionItemStatus
    fail_reason: str
    recipe: tuple[dict[str, Any], ...] | None
    fingerprint: WorldFingerprint | None
    updated_at: datetime


@dataclass(frozen=True)
class Mission:
    mission_id: int
    goal: str
    status: MissionStatus
    items: tuple[MissionItem, ...]
    created_at: datetime
    updated_at: datetime

    @property
    def current(self) -> MissionItem | None:
        return next((item for item in self.items if item.status is MissionItemStatus.PENDING), None)

    @property
    def failed(self) -> tuple[MissionItem, ...]:
        return tuple(item for item in self.items if item.status is MissionItemStatus.FAILED)

    @property
    def resolved(self) -> int:
        return sum(1 for item in self.items if item.status is not MissionItemStatus.PENDING)

    def count(self, status: MissionItemStatus) -> int:
        return sum(1 for item in self.items if item.status is status)


def normalize_recipe(recipe: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(recipe, (str, bytes, Mapping)):
        raise TypeError(f"A recipe must be a list of steps, got {type(recipe).__name__}.")
    steps = list(recipe)
    if not steps:
        raise ValueError("A recipe must hold at least one step; pass None for 'no recipe'.")
    for position, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"Recipe step {position} must be a mapping, got {type(step).__name__}.")
        if not step:
            raise ValueError(f"Recipe step {position} is empty.")
    normalized = tuple(dict(step) for step in steps)
    try:
        json.dumps(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"A recipe must be JSON-serializable: {exc}") from exc
    return normalized


def validate_mark(status: MissionItemStatus, fail_reason: str) -> str:
    if not isinstance(status, MissionItemStatus):
        raise TypeError(f"status must be a MissionItemStatus, got {type(status).__name__}.")
    reason = fail_reason.strip()
    if status is MissionItemStatus.FAILED:
        if not reason:
            raise ValueError(
                "A failed item must say why it failed: 'failed' with no reason gives a "
                "re-planner nothing to act on."
            )
        return reason
    if reason:
        raise ValueError(
            f"Only a failed item carries a fail_reason; got one with status '{status.value}'."
        )
    return ""


def _headline(mission: Mission) -> str:
    goal = _clip(mission.goal, _MAX_GOAL_CHARS)
    if mission.status is MissionStatus.ACTIVE:
        return f"Mission #{mission.mission_id}: {goal}"
    return f"Mission #{mission.mission_id} ({mission.status.value}): {goal}"


def _progress_line(mission: Mission) -> str:
    total = len(mission.items)
    parts = [
        f"{mission.count(status)} {status.value}"
        for status in (
            MissionItemStatus.PASSED,
            MissionItemStatus.FAILED,
            MissionItemStatus.SKIPPED,
        )
        if mission.count(status)
    ]
    line = f"Progress {mission.resolved}/{total}"
    if parts:
        line += " — " + ", ".join(parts)
    return line


def _current_block(mission: Mission) -> list[str]:
    total = len(mission.items)
    if not total:
        return ["Now: nothing to do — this mission has no items."]
    item = mission.current
    if item is None:
        return [f"Now: nothing pending — all {total} items are resolved."]
    return [
        f"Now #{item.item_id} ({item.seq + 1}/{total}): {_clip(item.title, _MAX_TITLE_CHARS)}",
        f"{_INDENT}why: {_clip(item.intent, _MAX_INTENT_CHARS)}",
        f"{_INDENT}done when: {_clip(item.acceptance, _MAX_ACCEPTANCE_CHARS)}",
    ]


def render_status(mission: Mission) -> str:
    lines = [_headline(mission), _progress_line(mission), *_current_block(mission)]

    failed = mission.failed
    if failed:
        lines.append("Failed:")
        for item in failed[:_MAX_FAILURES_SHOWN]:
            lines.append(
                f"{_INDENT}#{item.item_id} {_clip(item.title, _MAX_TITLE_CHARS)} — "
                f"{_clip(item.fail_reason, _MAX_REASON_CHARS)}"
            )
        remainder = len(failed) - _MAX_FAILURES_SHOWN
        if remainder > 0:
            lines.append(f"{_INDENT}…and {remainder} more failed (the full checklist lists them).")
    return "\n".join(lines)


def render_checklist(mission: Mission) -> str:
    lines = [_headline(mission), _progress_line(mission), *_current_block(mission)]
    for item in mission.items:
        line = (
            f"{_INDENT}{item.seq + 1}. #{item.item_id} [{item.status.value}] "
            f"{_clip(item.title, _MAX_TITLE_CHARS)}"
        )
        if item.fail_reason:
            line += f" — {_clip(item.fail_reason, _MAX_REASON_CHARS)}"
        lines.append(line)
    return "\n".join(lines)
