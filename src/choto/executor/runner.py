from __future__ import annotations

import math
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from choto.capture.framing import clamp_box, crop_frame
from choto.capture.grabber import CaptureError
from choto.capture.monitor import MonitorInfo
from choto.capture.stability import changed_regions, diff_frames
from choto.capture.tracking import TargetTracker, TrackedTarget, TrackStatus
from choto.config import Settings
from choto.executor.applauncher import AppActivationError, AppLauncher
from choto.executor.screen_reader import (
    ReadResult,
    ScreenReader,
    layout_lines,
    lines_between,
    split_surfaces,
    stitch_lines,
)
from choto.executor.scrolltrack import ScrollPass
from choto.executor.workarea import (
    BlindArea,
    WindowRectRefiner,
    WindowSelectionError,
    WorkArea,
    resolve_work_area,
)
from choto.graph.map_models import ScrollEdge
from choto.graph.repository import GraphRepository, hamming_hex
from choto.inputctl.errors import FailsafeTriggeredError, InputControlError
from choto.inputctl.gestures import in_failsafe_corner
from choto.inputctl.protocol import InputBackend
from choto.log import get_logger
from choto.matching.matcher import MatchResult, TargetMatcher, region_bounds
from choto.models import (
    Action,
    BBox,
    ExecutionReport,
    Expect,
    ExtractedSource,
    ExtractedText,
    OcrLine,
    PathPoint,
    Plan,
    Region,
    RunStatus,
    Scope,
    ScreenPoint,
    SeenElement,
    SeenScreen,
    ServoReadout,
    Step,
    Target,
)
from choto.ocr.engine import OcrError
from choto.overlay.backend import NullOverlayBackend, OverlayBackend
from choto.overlay.geometry import Rect
from choto.vision.calibration import DEFAULT_PARSERS, TickParser
from choto.vision.imaging import screenshot_png
from choto.vision.windowsource import WindowSource

_log = get_logger(__name__)

_SCREENSHOT_MAX_WIDTH = 1280

_EXPECT_POLL_INTERVAL_S = 0.1

_PERMISSION_REASON = "input permission missing — user must grant Accessibility in System Settings"

_STOP_BUTTON_REASON = "kill-switch: stop button pressed; plan aborted"

_NO_STOP_BUTTON_NOTE = (
    "kill-switch: this overlay backend has no stop button, so the failsafe corner "
    "(move the mouse to the top-left pixel of the screen) is the only way to stop a run"
)

_CLICK_ACTIONS = frozenset({Action.CLICK, Action.DOUBLE_CLICK, Action.RIGHT_CLICK})

_COORDINATE_AIM_NOTE = (
    "aimed by coordinate, not by an element: nothing was matched or verified there, "
    "so this click is only as good as the pixels the plan was written against"
)

_REFOCUS_ACTIONS = frozenset(
    {
        Action.CLICK,
        Action.DOUBLE_CLICK,
        Action.RIGHT_CLICK,
        Action.DRAG,
        Action.TYPE,
        Action.HOTKEY,
        Action.SCROLL,
    }
)

_FOCUS_CONTENTION_HINT = "is the user interacting with the screen?"

_NO_TARGET_APP_NOTE = (
    "target app not identified: the window server named no frontmost application, so no "
    "step can check that its input goes where the plan meant; steps that click or type "
    'will refuse until a "focus_app" step names an application'
)

_NO_TARGET_APP_REASON = (
    "no target application: the window server named no frontmost application when this run "
    "started, so there is no way to tell whether this step's input reaches the application "
    'the plan meant — put a "focus_app" step in front of it and re-run'
)


def _unverified_target_note(front: str, why: str) -> str:
    return (
        f'target app pinned to "{front}" as the window server spells it ({why}), so focus '
        "drift will be noticed but cannot be repaired: a step that loses the front will "
        "escalate instead of taking it back"
    )


_SUMMARY_MAX_ELEMENTS_PER_SCREEN = 12

_SUMMARY_MAX_ELEMENT_CHARS = 32

_SUMMARY_MIN_CONFIDENCE = 0.5

_SUMMARY_MIN_ELEMENT_CHARS = 2

_SUMMARY_MAX_SCREENS = 8

_SUMMARY_CHAR_BUDGET = 1800

_SUMMARY_HEADER = "windows seen during this run:"

_LISTING_CHAR_BUDGET = 2800

_PRE_RUN_ACTION = "start"

_ELLIPSIS = "…"

_NEAR_MISS_CANDIDATES = 3

_NEAR_MISS_MIN_SCORE = 0.3


SCROLL_SEARCH_AMOUNT = 3

MAX_SCROLL_SEARCH_STEPS = 6

_SCROLL_ANCHOR_GRID = 3

# PR-031
SCROLL_STUCK_NOTE = (
    "the page did not move, so either everything it holds is already on screen or the wheel "
    "landed somewhere that does not scroll"
)


_TRACK_CYCLE_MS = 14.9

_TRACK_SEARCH_MARGIN_PX = 64

_TRACK_STILL_SPEED_PX_PER_MS = 0.02

_TRACK_STILL_UPDATES = 3

_TRACK_STEADY_UPDATES = 5

_TRACK_MAX_UPDATES = 40


_SERVO_READOUT_MARGIN_X_PX = 20

_SERVO_READOUT_MARGIN_Y_PX = 4

_SERVO_MAX_MOVES = 24

_SERVO_MAX_BLIND_LOOKS = 3

_SERVO_MIN_MOVE_PX = 1.0

_SERVO_FLAT_COMPONENT = 1e-9

_SERVO_REDRAW_SETTLE_S = 0.033

_AREA_NOT_RESOLVED = "the run had not resolved a window yet"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _prefixed(prefix: str, note: str) -> str:
    return f"{prefix}; {note}" if prefix else note


def _held(modifiers: list[str] | None) -> str:
    return f" holding {'+'.join(modifiers)}" if modifiers else ""


def _step_window(step: Step) -> str | None:
    if step.target is not None:
        return step.target.window
    for point in step.path or ():
        if point.target is not None and point.target.window:
            return point.target.window
    return None


def _format_elements(lines: Sequence[OcrLine]) -> tuple[list[str], int]:
    rendered: list[str] = []
    spent = 0
    for index, line in enumerate(lines):
        cx, cy = line.bbox.center
        entry = f"{line.text} @ ({cx},{cy}) [conf {line.confidence:.2f}]"
        if rendered and spent + len(entry) > _LISTING_CHAR_BUDGET:
            return rendered, len(lines) - index
        rendered.append(entry)
        spent += len(entry) + 1
    return rendered, 0


def _dropped_note(dropped: int) -> str:
    return (
        f"… {_plural(dropped, 'more element')} on this surface, not listed (the listing is "
        f"capped at {_LISTING_CHAR_BUDGET} characters per surface); narrow the step with "
        'target.region, or use action "read" to bring the text back in full'
    )


def _empty_chrome_note(area: WorkArea, from_cache: bool) -> str:
    if not area.scoped:
        return " no window scope — everything read is listed in the block above"
    if from_cache:
        return (
            " not read — this window was served from memory (window elements only); "
            "act, or open the menu, and observe again to see the chrome"
        )
    if not area.chrome:
        return " the window server reported no chrome windows"
    return " no readable text on it"


def format_surface_listing(
    area: WorkArea, lines: Sequence[OcrLine], from_cache: bool = False
) -> list[str]:
    surfaces = split_surfaces(area, lines)
    window_elements, window_dropped = _format_elements(surfaces.window)
    chrome_elements, chrome_dropped = _format_elements(surfaces.chrome)
    rendered: list[str] = []
    if surfaces.elsewhere:
        rendered.append(
            f"off_surface: {len(surfaces.elsewhere)} element(s) elsewhere on screen "
            "(other applications' windows, the desktop) — not listed, not targetable"
        )
    if area.scoped:
        rendered.append(f"elements in focus window ({len(window_elements)}):")
    else:
        rendered.append(
            f"elements on screen, no window scope ({len(window_elements)}) — "
            "they may belong to several applications and none of them is targetable "
            "by target.window:"
        )
    rendered.extend(window_elements)
    if window_dropped:
        rendered.append(_dropped_note(window_dropped))
    rendered.append(
        f"chrome (menu bar / system, {len(chrome_elements)}):"
        f"{_empty_chrome_note(area, from_cache) if not chrome_elements else ''}"
    )
    rendered.extend(chrome_elements)
    if chrome_dropped:
        rendered.append(_dropped_note(chrome_dropped))
    return rendered


def _inside_box(box: BBox, x: int, y: int) -> bool:
    return box.x <= x < box.x + box.w and box.y <= y < box.y + box.h


def _settle_region(area: WorkArea) -> BBox | None:
    return area.window if area.scoped else None


def _intersect(first: BBox, second: BBox) -> BBox:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.w, second.x + second.w)
    y1 = min(first.y + first.h, second.y + second.h)
    return BBox(x=x0, y=y0, w=max(x1 - x0, 0), h=max(y1 - y0, 0))


def _bounding(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        return BBox(x=0, y=0, w=0, h=0)
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x + box.w for box in boxes)
    y1 = max(box.y + box.h for box in boxes)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _cell_center(rect: BBox, cell: tuple[int, int]) -> tuple[int, int]:
    row, col = cell
    x0 = rect.x + rect.w * col // _SCROLL_ANCHOR_GRID
    x1 = rect.x + rect.w * (col + 1) // _SCROLL_ANCHOR_GRID
    y0 = rect.y + rect.h * row // _SCROLL_ANCHOR_GRID
    y1 = rect.y + rect.h * (row + 1) // _SCROLL_ANCHOR_GRID
    return ((x0 + x1) // 2, (y0 + y1) // 2)


# PR-030
def _densest_cell_center(
    rect: BBox, lines: Sequence[OcrLine], keep_out: Sequence[BBox] = ()
) -> tuple[int, int] | None:
    if rect.w <= 0 or rect.h <= 0:
        return None

    def clear(point: tuple[int, int]) -> bool:
        return not any(_inside_box(box, *point) for box in keep_out)

    if not lines:
        return rect.center if clear(rect.center) else None
    counts: dict[tuple[int, int], int] = {}
    for line in lines:
        cx, cy = line.bbox.center
        col = min((cx - rect.x) * _SCROLL_ANCHOR_GRID // rect.w, _SCROLL_ANCHOR_GRID - 1)
        row = min((cy - rect.y) * _SCROLL_ANCHOR_GRID // rect.h, _SCROLL_ANCHOR_GRID - 1)
        counts[(row, col)] = counts.get((row, col), 0) + 1
    cells = [cell for cell in counts if clear(_cell_center(rect, cell))]
    if not cells:
        return None
    return _cell_center(rect, min(cells, key=lambda cell: (-counts[cell], cell)))


def _scroll_note(moves: list[str], notes: list[str]) -> str:
    if moves:
        head = "not on screen, scrolled " + ", ".join(moves)
    else:
        head = "not on screen, did not scroll"
    seen: list[str] = []
    for note in notes:
        if note not in seen:
            seen.append(note)
    return "; ".join([head, *seen])


def _surface_note(area: WorkArea, scope: Scope) -> str:
    if not area.scoped:
        return ""
    if scope is Scope.CHROME:
        return (
            "searched chrome only (menu bar, open menus and popovers); the focus window "
            'was not searched — if the target is in the window, retry with scope="window"'
        )
    return (
        "searched the focus window only; chrome (menu bar, open menus and popovers) was "
        'not searched — if you meant a menu item, retry with scope="chrome"'
    )


def _matched(match: MatchResult, scope: Scope) -> str:
    surface = f", {scope.value}" if scope is Scope.CHROME else ""
    return f'"{match.line.text}" ({match.tier} {match.score:.2f}{surface})'


def _point_label(action: Action, modifiers: list[str] | None, point: ScreenPoint) -> str:
    return f"{action.value}{_held(modifiers)} at ({point.x},{point.y})"


def _target_label(action: Action, modifiers: list[str] | None, text: str) -> str:
    return f'{action.value}{_held(modifiers)} "{text}"'


def _absent_note(label: str, text: str, note: str) -> str:
    return f'{label} "{text}": {note}'


def _coordinate_note(label: str, area: WorkArea, lines: Sequence[OcrLine]) -> str:
    searched = area.describe(len(area.select(lines, Scope.WINDOW)), len(lines), Scope.WINDOW)
    return f"{label}: {_COORDINATE_AIM_NOTE}; {searched}"


def _click_note(
    action: Action, modifiers: list[str] | None, target: Target, found: str, match: MatchResult
) -> str:
    cx, cy = match.line.bbox.center
    return (
        f"{_target_label(action, modifiers, target.text)}: {found} -> "
        f"matched {_matched(match, target.scope)} at ({cx},{cy})"
    )


def _noted(resolution: _Resolution, note: str) -> _Resolution:
    return replace(resolution, note=f"{resolution.note}; {note}")


def _searched(resolution: _Resolution, moves: list[str], notes: list[str]) -> _Resolution:
    return replace(
        resolution, note=f"{resolution.note}; {_scroll_note(moves, notes)}", moved=bool(moves)
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + _ELLIPSIS


def _summary_elements(lines: list[OcrLine]) -> list[SeenElement]:
    usable = [
        line
        for line in lines
        if line.confidence >= _SUMMARY_MIN_CONFIDENCE
        and len(line.text.strip()) >= _SUMMARY_MIN_ELEMENT_CHARS
    ]
    ranked = sorted(
        usable,
        key=lambda line: (len(line.text.strip()), -line.confidence, line.bbox.y, line.bbox.x),
    )
    picked = sorted(
        ranked[:_SUMMARY_MAX_ELEMENTS_PER_SCREEN],
        key=lambda line: (line.bbox.y, line.bbox.x),
    )
    elements: list[SeenElement] = []
    for line in picked:
        cx, cy = line.bbox.center
        elements.append(
            SeenElement(text=_clip(line.text.strip(), _SUMMARY_MAX_ELEMENT_CHARS), x=cx, y=cy)
        )
    return elements


def _format_seen_screen(screen: SeenScreen) -> str:
    steps = ",".join(str(step) for step in screen.steps)
    label = f"step{'s' if len(screen.steps) > 1 else ''} {steps} {screen.action}"
    body = ", ".join(element.text for element in screen.elements) or "(no readable text)"
    hidden = screen.element_count - len(screen.elements)
    if hidden > 0:
        body += f" (+{hidden} more)"
    if not screen.scoped:
        return f"  [{label}] whole screen, window scope unavailable ({screen.phash[:6]}) — {body}"
    window = f'"{screen.window_title}"' if screen.window_title else "<untitled>"
    return f'  [{label}] "{screen.app_name}" window {window} ({screen.phash[:6]}) — {body}'


def format_seen_screens(screens: list[SeenScreen]) -> str:
    if not screens:
        return ""

    selected: list[tuple[SeenScreen, str]] = []
    used = len(_SUMMARY_HEADER)
    for screen in reversed(screens[-_SUMMARY_MAX_SCREENS:]):
        rendered = _format_seen_screen(screen)
        if selected and used + len(rendered) + 1 > _SUMMARY_CHAR_BUDGET:
            break
        used += len(rendered) + 1
        selected.append((screen, rendered))

    lines = [rendered for _, rendered in reversed(selected)]
    hidden = len(screens) - len(selected)
    if hidden > 0:
        lines.append(f"  (+{hidden} earlier windows seen, not shown)")
    return "\n".join([_SUMMARY_HEADER, *lines])


def _read_source(area: WorkArea, scope: Scope) -> str:
    if not area.scoped:
        return "the whole screen"
    window = f'window "{area.title}"' if area.title else "an untitled window"
    if scope is Scope.CHROME:
        return f"the chrome around {window}"
    return window


def _extracted_source(item: ExtractedText) -> str:
    if item.source is ExtractedSource.CLIPBOARD:
        return "clipboard"
    if item.window_title:
        return f'window "{item.window_title}"'
    return "an untitled window"


def format_extracted(extracted: Sequence[ExtractedText]) -> str:
    if not extracted:
        return ""
    blocks = [
        f"extracted (step {item.step_index}, {_extracted_source(item)}):\n{item.text}"
        for item in extracted
    ]
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class _Resolution:
    match: MatchResult | None
    read: ReadResult
    candidates: list[OcrLine]
    note: str
    surface: str = ""
    moved: bool = False


@dataclass(frozen=True)
class _Aim:
    x: int
    y: int
    note: str = ""
    failure: tuple[str, RunStatus] | None = None


def _aim_stopped(x: int, y: int, updates: int) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        note=f"target was moving, tracked {_plural(updates, 'update')}, stopped at ({x},{y})",
    )


def _aim_ahead(x: int, y: int, last: tuple[int, int], speed: float, updates: int) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        note=(
            f"target moving {speed:.2f} px/ms, tracked {_plural(updates, 'update')}, clicked "
            f"{round(math.hypot(x - last[0], y - last[1]))}px ahead at ({x},{y})"
        ),
    )


def _aim_unsettled(x: int, y: int, updates: int) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        failure=(
            f"the target never stopped moving and never moved steadily enough to aim ahead "
            f"of: tracked {_plural(updates, 'update')}, last seen at ({x},{y})",
            RunStatus.ESCALATED,
        ),
    )


def _aim_off_display(x: int, y: int, updates: int) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        failure=(
            f"the target moved to the edge of the display: tracked "
            f"{_plural(updates, 'update')}, last seen at ({x},{y})",
            RunStatus.ESCALATED,
        ),
    )


def _aim_unseen(x: int, y: int, updates: int, why: str) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        failure=(
            f"the moving target could not be looked at again: tracked "
            f"{_plural(updates, 'update')}, last seen at ({x},{y}) ({why})",
            RunStatus.ESCALATED,
        ),
    )


def _aim_lost(x: int, y: int, updates: int, score: float) -> _Aim:
    return _Aim(
        x=x,
        y=y,
        failure=(
            f"the target was moving and the tracker lost it: tracked "
            f"{_plural(updates, 'update')}, lost at ({x},{y}), best correlation {score:.2f}",
            RunStatus.ESCALATED,
        ),
    )


# PR-028
def _unscrolled(points: Sequence[PathPoint]) -> list[PathPoint]:
    return [
        point
        if point.target is None
        else point.model_copy(
            update={"target": point.target.model_copy(update={"scroll_to_find": False})}
        )
        for point in points
    ]


@dataclass(frozen=True)
class _PathResolution:
    points: list[tuple[int, int]]
    labels: list[str]
    read: ReadResult
    note: str
    failed_target: Target | None = None
    failed_resolution: _Resolution | None = None
    reason: str | None = None
    status: RunStatus = RunStatus.ESCALATED

    @property
    def ok(self) -> bool:
        return self.failed_resolution is None and self.reason is None


@dataclass(frozen=True)
class _ServoGoal:
    text: str
    value: float
    tolerance: float
    parser: TickParser


def _servo_goal(readout: ServoReadout) -> _ServoGoal | None:
    for parser in DEFAULT_PARSERS:
        value = parser.parse(readout.value)
        if value is not None:
            return _ServoGoal(
                text=readout.value, value=value, tolerance=readout.tolerance, parser=parser
            )
    return None


@dataclass(frozen=True)
class _ServoSetup:
    goal: _ServoGoal
    anchor: BBox
    read: ReadResult


@dataclass(frozen=True)
class _Reading:
    text: str
    value: float | None = None


@dataclass(frozen=True)
class _ServoMove:
    offset: float = 0.0
    stride: float = 0.0
    forward: float = 1.0
    stopped: str = ""


@dataclass(frozen=True)
class _Servo:
    note: str
    failure: tuple[str, RunStatus] | None = None


@dataclass(frozen=True)
class _ServoAim:
    origin: tuple[int, int]
    span: float
    direction: tuple[float, float]
    refusal: _Servo | None = None


def _servo_aim(points: Sequence[tuple[int, int]], goal: _ServoGoal) -> _ServoAim:
    origin, towards = points[0], points[1]
    span = math.hypot(towards[0] - origin[0], towards[1] - origin[1])
    if span < _SERVO_MIN_MOVE_PX:
        return _ServoAim(
            origin=origin,
            span=span,
            direction=(0.0, 0.0),
            refusal=_Servo(
                note="servo: not attempted",
                failure=(
                    f'a drag steered by the readout "{goal.text}" was given both of its path '
                    f"points at ({origin[0]},{origin[1]}): the second one says which way to "
                    "move, so it has to be somewhere else",
                    RunStatus.ESCALATED,
                ),
            ),
        )
    return _ServoAim(
        origin=origin,
        span=span,
        direction=((towards[0] - origin[0]) / span, (towards[1] - origin[1]) / span),
    )


def _servo_aborted(moves: int, why: str) -> _Servo:
    return _Servo(
        note=f"servo: stopped after {_plural(moves, 'move')}",
        failure=(f"kill-switch during the drag: {why}", RunStatus.ABORTED),
    )


def _servo_released(moves: int, goal: _ServoGoal, seen: str, why: str) -> _Servo:
    return _Servo(
        note=f"servo: stopped after {_plural(moves, 'move')}",
        failure=(
            f'the drag steered by the readout "{goal.text}" was released after '
            f'{_plural(moves, "move")} showing "{seen}": {why}',
            RunStatus.ESCALATED,
        ),
    )


def _servo_outcome(
    moves: int, goal: _ServoGoal, seen: str, failure: tuple[str, RunStatus] | None
) -> _Servo:
    if failure is not None:
        return _Servo(note=f"servo: released after {_plural(moves, 'move')}", failure=failure)
    note = f'servo: readout "{goal.text}" reached in {_plural(moves, "move")}'
    if seen != goal.text:
        note = f'{note}, showing "{seen}"'
    return _Servo(note=note)


@dataclass
class _Sweep:
    read: ReadResult
    resolution: _Resolution | None = None
    steps: int = 0
    moved: int = 0
    aborted: str | None = None
    stopped: str | None = None
    edge: ScrollEdge | None = None

    @property
    def found(self) -> bool:
        return self.resolution is not None and self.resolution.match is not None


@dataclass(frozen=True)
class _Harvest:
    read: ReadResult
    note: str = ""
    outcome: _StepOutcome | None = None


@dataclass(frozen=True)
class _SettleWatch:
    timeout_ms: int
    grace_ms: int
    acted_at: tuple[int, int] | None


@dataclass(frozen=True)
class _Verified:
    ok: bool
    detail: str


@dataclass
class _StepOutcome:
    ok: bool
    reason: str | None = None
    frame: np.ndarray | None = None
    lines: list[OcrLine] = field(default_factory=list)
    from_cache: bool = False
    status: RunStatus = RunStatus.ESCALATED
    note: str = ""


def _step_refused(
    reason: str,
    read: ReadResult,
    status: RunStatus = RunStatus.ESCALATED,
    note: str = "",
) -> _StepOutcome:
    return _StepOutcome(
        ok=False,
        reason=reason,
        frame=read.frame.image,
        lines=read.state.lines,
        from_cache=read.from_cache,
        status=status,
        note=note,
    )


@dataclass(frozen=True)
class _ScreenChange:
    phash_distance: int
    phash_limit: int
    pixel_fraction: float
    pixel_limit: float
    reframed: str | None = None

    @property
    def by_phash(self) -> bool:
        return self.reframed is None and self.phash_distance > self.phash_limit

    @property
    def by_pixels(self) -> bool:
        return self.reframed is None and self.pixel_fraction > self.pixel_limit

    @property
    def changed(self) -> bool:
        return self.reframed is not None or self.by_phash or self.by_pixels

    def describe(self) -> str:
        if self.reframed is not None:
            return (
                f"the two reads covered different pixels ({self.reframed}), so nothing was "
                "comparable and the screen is called changed"
            )
        return (
            f"phash {self.phash_distance} {'>' if self.by_phash else '<='} {self.phash_limit}, "
            f"pixels {self.pixel_fraction:.5f} "
            f"{'>' if self.by_pixels else '<='} {self.pixel_limit:.5f}"
        )


@dataclass(frozen=True)
class _Transition:
    from_id: int | None
    to_id: int | None
    change: _ScreenChange

    @property
    def remembered(self) -> bool:
        return self.from_id is not None and self.to_id is not None

    @property
    def new_node(self) -> bool:
        return self.remembered and self.from_id != self.to_id

    def describe(self) -> str:
        if self.new_node:
            said = f"screen changed to another window #{self.from_id}->#{self.to_id}"
        elif self.change.changed:
            said = f"screen changed inside the same window ({self.change.describe()})"
        else:
            said = f"screen unchanged ({self.change.describe()})"
        if not self.remembered:
            said += "; not remembered (no window scope)"
        return said


class Executor:
    def __init__(
        self,
        reader: ScreenReader,
        matcher: TargetMatcher,
        input_controller: InputBackend,
        windows: WindowSource,
        apps: AppLauncher,
        repo: GraphRepository,
        settings: Settings,
        overlay: OverlayBackend | None = None,
    ) -> None:
        self._reader = reader
        self._matcher = matcher
        self._input = input_controller
        self._windows = windows
        self._apps = apps
        self._repo = repo
        self._settings = settings
        self._overlay: OverlayBackend = overlay if overlay is not None else NullOverlayBackend()
        self._rect_refiner = WindowRectRefiner(
            enabled=settings.window_rect_refinement_enabled,
        )
        self._target_app: str | None = None
        self._begin_run(None)

    def _begin_run(self, run_id: int | None) -> None:
        self._monitor: MonitorInfo | None = None
        self._screen_dirty = False
        self._seen: dict[str, SeenScreen] = {}
        self._run_id = run_id
        self._visit_seq = 0
        self._step_index = 0
        self._step_action = _PRE_RUN_ACTION
        self._area: WorkArea = BlindArea(_AREA_NOT_RESOLVED)
        self._actions_used = 0
        self._extracted: list[ExtractedText] = []
        self._step_deadline = 0.0
        # PR-025
        self._scroll_pass: ScrollPass | None = None
        # PR-027
        self._pass_screen_id: int | None = None

    def execute(self, plan: Plan) -> ExecutionReport:
        self._overlay.begin()
        try:
            return self._execute(plan)
        finally:
            self._overlay.hide()

    def _execute(self, plan: Plan) -> ExecutionReport:
        run = self._repo.create_run(plan)
        journey: list[str] = []
        completed: list[int] = []
        self._begin_run(run.id)

        budget = self._settings.max_actions_per_plan
        if len(plan.steps) > budget:
            reason = (
                f"this plan has {_plural(len(plan.steps), 'step')} and a run may perform "
                f"{_plural(budget, 'action')} (CHOTO_MAX_ACTIONS_PER_PLAN); send a shorter "
                "plan, or raise the budget"
            )
            journey.append(f"plan refused: {reason}")
            _log.warning("run.plan_too_long", run_id=run.id, steps=len(plan.steps), budget=budget)
            return self._terminal_report(run.id, RunStatus.ESCALATED, [], None, reason, journey)

        if not self._ensure_input_permission():
            return self._permission_escalation(run.id, journey)

        try:
            self._target_app, target_note = self._detect_target_app()
            if target_note is not None:
                journey.append(target_note)
            self._begin_step_area(None)

            for index, step in enumerate(plan.steps, start=1):
                self._step_index = index
                self._step_action = step.action.value

                reason = self._abort_requested()
                if reason is not None:
                    journey.append(f"step {index} {step.action.value}: aborted by kill-switch")
                    _log.warning("run.aborted", run_id=run.id, step=index)
                    return self._terminal_report(
                        run.id, RunStatus.ABORTED, completed, index, reason, journey
                    )

                self._actions_used += 1
                outcome = self._run_step(index, step)
                if outcome.note:
                    journey.append(outcome.note)
                if not outcome.ok:
                    _log.warning(
                        "run.stopped",
                        run_id=run.id,
                        step=index,
                        status=outcome.status.value,
                        reason=outcome.reason,
                    )
                    return self._terminal_report(
                        run.id,
                        outcome.status,
                        completed,
                        index,
                        outcome.reason,
                        journey,
                        outcome.frame,
                        outcome.lines,
                        outcome.from_cache,
                    )
                completed.append(index)
        except (CaptureError, OcrError) as exc:
            return self._unreadable_screen_report(run.id, completed, journey, exc)

        _log.info("run.success", run_id=run.id, steps=len(plan.steps))
        return self._report(run.id, RunStatus.SUCCESS, completed, None, None, journey)

    def _read(self, force_ocr: bool = False) -> ReadResult:
        # PR-025, PR-027
        pin = self._pin_to_pass if self._scroll_pass is not None else None
        result = self._reader.read(self._area, force_ocr=force_ocr, pin=pin)
        self._screen_dirty = False
        self._record_seen(result)
        return result

    # PR-027
    def _pin_to_pass(self, lines: Sequence[OcrLine]) -> int | None:
        walked = self._scroll_pass
        if walked is None:
            return None
        return self._pass_screen_id if walked.page(list(lines)) else None

    # PR-017
    def _probe(self) -> ReadResult:
        return self._reader.read(self._area, force_ocr=True, persist=False)

    def _record_seen(self, read: ReadResult) -> None:
        state = read.state
        seen = self._seen.get(state.phash)
        if seen is not None:
            if self._step_index not in seen.steps:
                seen.steps.append(self._step_index)
                self._persist_visit(read.screen_db_id)
            return
        window_lines = self._area.in_window(state.lines)
        self._seen[state.phash] = SeenScreen(
            phash=state.phash,
            app_name=state.app_name,
            window_title=read.window_title,
            scoped=read.screen_db_id is not None,
            steps=[self._step_index],
            action=self._step_action,
            elements=_summary_elements(window_lines),
            element_count=len(window_lines),
        )
        self._persist_visit(read.screen_db_id)

    def _persist_visit(self, screen_id: int | None) -> None:
        if self._run_id is None:
            raise RuntimeError("a window was read outside of a run; no ledger to append to")
        if screen_id is None:
            return
        self._repo.record_screen_visit(
            run_id=self._run_id,
            screen_id=screen_id,
            seq=self._visit_seq,
            step_index=self._step_index,
            action=self._step_action,
        )
        self._visit_seq += 1

    def _ensure_input_permission(self) -> bool:
        if self._input.ensure_permission():
            return True
        return self._input.request_permission()

    def _permission_escalation(self, run_id: int, journey: Sequence[str]) -> ExecutionReport:
        told = [
            *journey,
            f"input permission check failed: {_PERMISSION_REASON}",
            "hint: enable this process under System Settings -> Privacy & Security -> "
            "Accessibility, then re-run the plan",
        ]
        return self._terminal_report(
            run_id, RunStatus.ESCALATED, [], None, _PERMISSION_REASON, told
        )

    def _unreadable_screen_report(
        self,
        run_id: int,
        completed: list[int],
        journey: Sequence[str],
        exc: Exception,
    ) -> ExecutionReport:
        reason = (
            f"the screen could not be read: {exc}. Nothing was captured to show for it — "
            "re-plan against a fresh observe"
        )
        where = (
            f"step {self._step_index} {self._step_action}"
            if self._step_index
            else f"before step 1 ({self._step_action})"
        )
        told = [*journey, f"{where}: {reason}"]
        _log.warning(
            "run.screen_unreadable",
            run_id=run_id,
            step=self._step_index,
            action=self._step_action,
            error=str(exc),
        )
        return self._report(
            run_id, RunStatus.ESCALATED, completed, self._step_index or None, reason, told
        )

    def _abort_requested(self, during: str = "") -> str | None:
        if in_failsafe_corner(self._input.mouse_position()):
            return f"kill-switch: mouse moved to the failsafe corner{during}; plan aborted"
        if self._overlay.stop_requested():
            return _STOP_BUTTON_REASON
        return None

    def _kill_switch_note(self) -> str | None:
        return None if self._overlay.capabilities.stop_button else _NO_STOP_BUTTON_NOTE

    def _finish(self, run_id: int, status: RunStatus, journey: Sequence[str]) -> list[str]:
        note = self._kill_switch_note()
        told = list(journey) if note is None else [*journey, note]
        if note is not None:
            _log.warning("run.kill_switch_degraded", run_id=run_id, note=note)
        self._repo.finish_run(run_id, status, told)
        return told

    def _show_overlay(self) -> None:
        area = self._area
        if not area.scoped:
            self._overlay.hide()
            return
        window = area.window
        scale = self._pixel_scale()
        self._overlay.show(
            Rect(x=window.x / scale, y=window.y / scale, w=window.w / scale, h=window.h / scale)
        )

    def _detect_target_app(self) -> tuple[str | None, str | None]:
        front = self._windows.frontmost_app_name().strip()
        if not front:
            _log.warning("focus.target_nameless")
            return None, _NO_TARGET_APP_NOTE
        try:
            identity = self._apps.resolve(front)
        except AppActivationError as exc:
            _log.warning("focus.target_ambiguous", frontmost=front, error=str(exc))
            return front, _unverified_target_note(front, str(exc))
        if identity is None:
            _log.warning("focus.target_unknown", frontmost=front)
            return front, _unverified_target_note(
                front, "no running application answers to that name"
            )
        _log.info("focus.target_pinned", target_app=identity.display_name)
        return identity.display_name, None

    def _ensure_target_frontmost(self, timeout_ms: int) -> tuple[str, str | None]:
        target = self._target_app
        if target is None:
            _log.warning("focus.no_target_app", step=self._step_index)
            return "", _NO_TARGET_APP_REASON

        front = self._windows.frontmost_app_name()
        if self._apps.same_app(front, target):
            return "", None

        _log.warning("focus.drift", target_app=target, frontmost=front)
        try:
            resolved = self._apps.activate(target)
        except (AppActivationError, ValueError) as exc:
            reason = (
                f'lost focus to "{front}": could not bring "{target}" frontmost — '
                f"{_FOCUS_CONTENTION_HINT} (activation failed: {exc})"
            )
            _log.warning(
                "focus.recovery_failed", target_app=target, frontmost=front, error=str(exc)
            )
            return "", reason

        self._target_app = resolved
        self._screen_dirty = True
        self._reader.wait_until_stable(
            timeout_ms, quiet_grace_ms=self._settings.settle_quiet_grace_ms
        )
        _log.info("focus.restored", target_app=resolved, was=front)
        return f'refocused "{resolved}" (was "{front}")', None

    def _run_step(self, index: int, step: Step) -> _StepOutcome:
        action = step.action
        self._scroll_pass = None
        # PR-027
        self._pass_screen_id = None
        self._step_deadline = time.monotonic() + step.timeout_ms / 1000.0

        focus_note = ""
        if action in _REFOCUS_ACTIONS:
            focus_note, focus_failure = self._ensure_target_frontmost(self._remaining_ms())
            if focus_failure is not None:
                return _StepOutcome(
                    ok=False,
                    reason=focus_failure,
                    note=f"step {index} {action.value}: {focus_failure}",
                )

        area_failure = self._begin_step_area(_step_window(step))
        if area_failure is not None:
            return _StepOutcome(
                ok=False,
                reason=area_failure,
                note=f"step {index} {action.value}: {area_failure}",
            )
        area = self._area
        before = self._read(force_ocr=self._screen_dirty)

        counted_text = step.expect.appears_count_increases if step.expect is not None else None
        before, baseline_count = self._count_baseline(counted_text, before)
        acted_at: tuple[int, int] | None = None

        if step.at is not None:
            label = _point_label(action, step.modifiers, step.at)
            refusal = self._coordinate_refusal(step.at, before, area)
            if refusal is not None:
                return self._gesture_failed(index, _prefixed(focus_note, label), refusal, before)
            note = _coordinate_note(label, area, before.state.lines)
            acted_at = (step.at.x, step.at.y)
            failure = self._perform_click(action, step.at.x, step.at.y, step.modifiers)
            if failure is not None:
                reason, status = failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
        elif action in _CLICK_ACTIONS:
            resolution, aborted = self._locate(step.target, before, area)
            before = resolution.read
            if aborted is not None:
                label = _target_label(action, step.modifiers, step.target.text)
                return self._gesture_failed(
                    index,
                    _prefixed(focus_note, f"{label}: {resolution.note}"),
                    aborted,
                    before,
                    RunStatus.ABORTED,
                )
            if resolution.match is None:
                if step.skip_if_absent:
                    return self._skipped(
                        index,
                        _prefixed(
                            focus_note,
                            _absent_note(action.value, step.target.text, resolution.note),
                        ),
                    )
                return self._target_not_found(index, step, step.target, resolution, focus_note)
            match = resolution.match
            if resolution.moved:
                before, baseline_count = self._count_baseline(counted_text, before)
            note = _click_note(action, step.modifiers, step.target, resolution.note, match)
            aim = self._aim(match.line.bbox, before, self._remaining_ms())
            if aim.failure is not None:
                reason, status = aim.failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
            if aim.note:
                note = f"{note}; {aim.note}"
            acted_at = (aim.x, aim.y)
            failure = self._perform_click(action, aim.x, aim.y, step.modifiers)
            if failure is not None:
                reason, status = failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
        elif action is Action.DRAG:
            label = f"{action.value}{_held(step.modifiers)}"
            resolved = self._resolve_path(step.path or [], before, area)
            before = resolved.read
            if resolved.failed_resolution is not None:
                return self._target_not_found(
                    index,
                    step,
                    resolved.failed_target,
                    resolved.failed_resolution,
                    focus_note,
                )
            if resolved.reason is not None:
                return self._gesture_failed(
                    index, _prefixed(focus_note, label), resolved.reason, before, resolved.status
                )
            path = "path: " + " => ".join(resolved.labels)
            servo, servo_failure = self._begin_servo(index, step, before, area, focus_note)
            if servo_failure is not None:
                return servo_failure
            if servo is None:
                failure = self._perform_drag(resolved.points, step.modifiers)
                if failure is not None:
                    reason, status = failure
                    return self._gesture_failed(
                        index,
                        _prefixed(focus_note, f"{label} {path}"),
                        reason,
                        before,
                        status,
                    )
                note = f"{label}: {resolved.note} -> {path}"
            else:
                before = servo.read
                steered = self._servo_drag(resolved.points, step, servo.goal, servo.anchor, before)
                if steered.failure is not None:
                    reason, status = steered.failure
                    return self._gesture_failed(
                        index,
                        _prefixed(focus_note, f"{label} {path}"),
                        reason,
                        before,
                        status,
                    )
                note = f"{label}: {resolved.note} -> {path}; {steered.note}"
        elif action is Action.TYPE:
            note = f'type "{step.text}"'
            failure = self._synthesize(note, lambda: self._input.type_text(step.text or ""))
            if failure is not None:
                reason, status = failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
        elif action is Action.HOTKEY:
            note = f"hotkey {'+'.join(step.keys or [])}"
            failure = self._synthesize(note, lambda: self._input.hotkey(list(step.keys or [])))
            if failure is not None:
                reason, status = failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
        elif action is Action.SCROLL:
            if step.target is not None:
                resolution, aborted = self._locate(step.target, before, area)
                before = resolution.read
                if aborted is not None:
                    return self._gesture_failed(
                        index,
                        _prefixed(focus_note, f'scroll over "{step.target.text}"'),
                        aborted,
                        before,
                        RunStatus.ABORTED,
                    )
                if resolution.match is None:
                    if step.skip_if_absent:
                        return self._skipped(
                            index,
                            _prefixed(
                                focus_note,
                                _absent_note("scroll over", step.target.text, resolution.note),
                            ),
                        )
                    return self._target_not_found(index, step, step.target, resolution, focus_note)
                if resolution.moved:
                    before, baseline_count = self._count_baseline(counted_text, before)
                px, py = self._to_points(*resolution.match.line.bbox.center)
                aimed = self._synthesize("scroll", lambda: self._input.move(px, py))
                if aimed is not None:
                    reason, status = aimed
                    return self._gesture_failed(
                        index,
                        _prefixed(focus_note, f'scroll over "{step.target.text}"'),
                        reason,
                        before,
                        status,
                    )
            note = f"scroll {step.amount}"
            failure = self._synthesize(note, lambda: self._input.scroll(step.amount or 0))
            if failure is not None:
                reason, status = failure
                return self._gesture_failed(
                    index, _prefixed(focus_note, note), reason, before, status
                )
        elif action is Action.READ:
            harvest = self._harvest_screen(index, step, before, area, focus_note)
            before = harvest.read
            if harvest.outcome is not None:
                return harvest.outcome
            note = harvest.note
        elif action is Action.READ_CLIPBOARD:
            note = self._harvest_clipboard(index)
        elif action is Action.WAIT:
            note = f"wait up to {step.timeout_ms}ms"
        elif action is Action.FOCUS_APP:
            try:
                resolved = self._apps.activate(step.app_name or "")
            except (AppActivationError, ValueError) as exc:
                reason = f"focus_app failed: {exc}"
                return _step_refused(
                    reason, before, note=f'step {index} focus_app "{step.app_name}": {reason}'
                )
            self._target_app = resolved
            self._begin_step_area(None)
            note = f'focus_app "{step.app_name}": activated "{resolved}"'
        else:  # pragma: no cover - Action enum is exhaustive above.
            reason = f'unsupported action "{action.value}"'
            return _step_refused(reason, before)

        return self._finish_step(
            index, step, before, _prefixed(focus_note, note), baseline_count, acted_at
        )

    def _skipped(self, index: int, note: str) -> _StepOutcome:
        _log.info("step.skipped_absent", step=index)
        return _StepOutcome(ok=True, note=f"step {index} {note} -> skipped: target absent")

    def _readable_lines(
        self, read: ReadResult, area: WorkArea, scope: Scope, region: Region | None
    ) -> list[OcrLine]:
        lines = area.select(read.state.lines, scope)
        if region is None:
            return lines
        bounds = self._region_rect(region, read, area, scope)
        return [line for line in lines if _inside_box(bounds, *line.bbox.center)]

    # PR-029
    def _anchor_line(
        self, text: str, scope: Scope, lines: Sequence[OcrLine], read: ReadResult
    ) -> OcrLine | None:
        target = Target(text=text, scope=scope)
        content = [line for line in lines if not read.navigation.holds(line)]
        match = None
        if len(content) < len(lines):
            match = self._matcher.resolve(target, content, read.state.width, read.state.height)
        if match is None:
            match = self._matcher.resolve(target, list(lines), read.state.width, read.state.height)
        return match.line if match is not None else None

    def _harvest_screen(
        self,
        index: int,
        step: Step,
        read: ReadResult,
        area: WorkArea,
        focus_note: str,
    ) -> _Harvest:
        target = step.target
        scope = target.scope if target is not None else Scope.WINDOW
        region = target.region if target is not None else None
        if read.from_cache:
            read = self._read(force_ocr=True)

        collected: list[str] = []
        started = step.from_text is None
        reached_end_anchor = False
        anchor: tuple[int, int] | None = None
        moved = 0
        scrolls = 0
        notes: list[str] = []
        page = 0
        # PR-028
        may_hunt = target.scroll_to_find if target is not None else True
        # PR-025
        ends: set[ScrollEdge] = set()
        if step.scroll or may_hunt:
            self._open_scroll_pass(region, read, area)

        while True:
            visible = self._readable_lines(read, area, scope, region)
            page += 1
            first = None
            if not started:
                first = self._anchor_line(step.from_text or "", scope, visible, read)
                started = first is not None
            gained = 0
            if started:
                last = None
                if step.to_text is not None:
                    last = self._anchor_line(step.to_text, scope, visible, read)
                    reached_end_anchor = last is not None
                before_count = len(collected)
                wanted = lines_between(visible, first, last)
                collected = stitch_lines(collected, layout_lines(wanted))
                gained = len(collected) - before_count

            # PR-028
            hunting = (
                may_hunt and self._missing_anchor(step, started, reached_end_anchor) is not None
            )
            if not hunting and (reached_end_anchor or not step.scroll):
                break
            if started and page > 1 and gained == 0:
                notes.append("no new content")
                break
            if not area.scoped:
                notes.append("not scrolled (no window scope)")
                break
            if anchor is None:
                # PR-030
                anchor, refused = self._scroll_anchor(region, read, area)
                if anchor is None:
                    notes.append(refused)
                    break
            if not self._can_afford_scroll(moved):
                notes.append("action budget reached")
                break
            if self._out_of_time():
                notes.append("step timeout reached")
                break

            aborted = self._abort_requested(" while reading the document")
            if aborted is not None:
                return self._reading_stopped(index, collected, read, aborted, focus_note)
            self._screen_dirty = True
            failure = self._scroll_at(anchor, SCROLL_SEARCH_AMOUNT)
            if failure is not None:
                notes.append(failure)
                break
            scrolls += 1
            self._settle_after_scroll(area)
            fresh = self._read(force_ocr=True)
            if not self._screen_changed(read, fresh).changed:
                read = fresh
                # PR-031
                if moved:
                    ends.add(ScrollEdge.BOTTOM)
                    notes.append("reached the end of the content")
                else:
                    notes.append(SCROLL_STUCK_NOTE)
                break
            moved += 1
            read = fresh

        moves = [f"{scrolls}x down"] if scrolls else []
        if moved and anchor is not None:
            back, rewind_note, aborted = self._rewind(anchor, 1, moved, area)
            if back:
                moves.append(f"{back}x back")
            if rewind_note is not None:
                notes.append(rewind_note)
            if aborted is not None:
                return self._reading_stopped(index, collected, read, aborted, focus_note)
            read = self._read(force_ocr=True)
        self._close_scroll_pass(ends)

        text = self._record_reading(index, collected, read)

        missing = self._missing_anchor(step, started, reached_end_anchor)
        if missing is not None:
            # PR-028
            searched = _scroll_note(moves, notes) if moves or notes else ""
            return _Harvest(
                read=read,
                outcome=self._anchor_not_found(
                    index, step, missing, scope, read, area, focus_note, searched
                ),
            )

        _log.info(
            "step.screen_read",
            step=index,
            lines=len(collected),
            chars=len(text),
            scrolls=scrolls,
            window_title=read.window_title,
        )
        return _Harvest(
            read=read,
            note=self._harvest_note(step, area, scope, len(collected), len(text), moves, notes),
        )

    def _record_reading(self, index: int, collected: Sequence[str], read: ReadResult) -> str:
        text = "\n".join(collected)
        if not text:
            return ""
        self._extracted.append(
            ExtractedText(
                step_index=index,
                source=ExtractedSource.SCREEN,
                window_title=read.window_title,
                text=text,
            )
        )
        return text

    def _reading_stopped(
        self,
        index: int,
        collected: Sequence[str],
        read: ReadResult,
        reason: str,
        focus_note: str,
    ) -> _Harvest:
        text = self._record_reading(index, collected, read)
        label = _prefixed(focus_note, f"read: {_plural(len(text), 'char')} so far")
        return _Harvest(
            read=read,
            outcome=self._gesture_failed(index, label, reason, read, RunStatus.ABORTED),
        )

    @staticmethod
    def _missing_anchor(step: Step, started: bool, reached_end_anchor: bool) -> str | None:
        if not started and step.from_text is not None:
            return step.from_text
        if step.to_text is not None and not reached_end_anchor:
            return step.to_text
        return None

    def _anchor_not_found(
        self,
        index: int,
        step: Step,
        text: str,
        scope: Scope,
        read: ReadResult,
        area: WorkArea,
        focus_note: str,
        searched: str = "",
    ) -> _StepOutcome:
        region = step.target.region if step.target is not None else None
        candidates = self._readable_lines(read, area, scope, region)
        note = area.describe(len(candidates), len(read.state.lines), scope)
        resolution = _Resolution(
            match=None,
            read=read,
            candidates=candidates,
            note=note if not searched else f"{note}; {searched}",
            surface=_surface_note(area, scope),
        )
        if step.skip_if_absent:
            return self._skipped(
                index, _prefixed(focus_note, _absent_note("read", text, resolution.note))
            )
        return self._target_not_found(
            index, step, Target(text=text, scope=scope), resolution, focus_note
        )

    @staticmethod
    def _harvest_note(
        step: Step,
        area: WorkArea,
        scope: Scope,
        line_count: int,
        char_count: int,
        moves: Sequence[str],
        notes: Sequence[str],
    ) -> str:
        parts = [
            f"read: {_plural(line_count, 'line')} ({_plural(char_count, 'char')}) "
            f"from {_read_source(area, scope)}"
        ]
        bounds = []
        if step.from_text is not None:
            bounds.append(f'from "{step.from_text}"')
        if step.to_text is not None:
            bounds.append(f'to "{step.to_text}"')
        if bounds:
            parts.append(" ".join(bounds))
        if moves:
            parts.append("scrolled " + ", ".join(moves))
        seen: list[str] = []
        for note in notes:
            if note not in seen:
                seen.append(note)
        parts.extend(seen)
        return "; ".join(parts)

    def _harvest_clipboard(self, index: int) -> str:
        text = self._input.read_clipboard()
        if text is None:
            return "read_clipboard: nothing on the clipboard (empty, or holding no text)"
        if not text:
            return "read_clipboard: 0 chars (the clipboard holds an empty string)"
        self._extracted.append(
            ExtractedText(
                step_index=index,
                source=ExtractedSource.CLIPBOARD,
                window_title=self._area.title if self._area.scoped else "",
                text=text,
            )
        )
        _log.info("step.clipboard_read", step=index, chars=len(text))
        return f"read_clipboard: {_plural(len(text), 'char')}"

    def _coordinate_refusal(
        self, point: ScreenPoint, read: ReadResult, area: WorkArea
    ) -> str | None:
        width, height = read.state.width, read.state.height
        if not (0 <= point.x < width and 0 <= point.y < height):
            return (
                f"click at ({point.x},{point.y}) is outside the captured screen "
                f"({width}x{height} px)"
            )
        if area.scoped:
            window = area.window
            if not _inside_box(window, point.x, point.y):
                return (
                    f"click at ({point.x},{point.y}) is outside {area.label()}, which covers "
                    f"({window.x},{window.y}) to ({window.x + window.w},{window.y + window.h}) — "
                    "the window has moved or the coordinate came from another screen"
                )
            if area.covered.hides(point.x, point.y):
                return (
                    f"click at ({point.x},{point.y}) would land under {area.covered.named}, "
                    "which is drawn over that part of the window: the click would go to it, "
                    "not to what is underneath"
                )
        if in_failsafe_corner(self._to_points(point.x, point.y)):
            return (
                f"click at ({point.x},{point.y}) is in the failsafe corner, which is the "
                "user's kill-switch: a run may not put the cursor there"
            )
        return None

    def _resolve_target(self, target: Target, read: ReadResult, area: WorkArea) -> _Resolution:
        candidates = area.select(read.state.lines, target.scope)
        match, region_note = self._match_here(target, candidates, read, area)
        if read.from_cache and (match is None or self._cache_drifted(read)):
            read = self._read(force_ocr=True)
            candidates = area.select(read.state.lines, target.scope)
            match, region_note = self._match_here(target, candidates, read, area)
        if match is not None:
            _log.info(
                "target.resolved",
                target=target.text,
                matched=match.line.text,
                tier=match.tier,
                score=round(match.score, 3),
                scope=target.scope.value,
            )
        note = area.describe(len(candidates), len(read.state.lines), target.scope)
        return _Resolution(
            match=match,
            read=read,
            candidates=candidates,
            note=note if region_note is None else f"{note}; {region_note}",
            surface=_surface_note(area, target.scope),
        )

    def _cache_drifted(self, read: ReadResult) -> bool:
        if not read.from_cache:
            return False
        drift = hamming_hex(read.phash, read.state.phash)
        if drift:
            _log.info("target.cache_drifted", drift=drift, phash=read.state.phash)
        return drift > 0

    # PR-029
    def _match_here(
        self, target: Target, candidates: list[OcrLine], read: ReadResult, area: WorkArea
    ) -> tuple[MatchResult | None, str | None]:
        content = [line for line in candidates if not read.navigation.holds(line)]
        if len(content) < len(candidates):
            match, region_note = self._match_among(target, content, read, area)
            if match is not None:
                return match, region_note
            _log.info(
                "target.not_in_content",
                target=target.text,
                content=len(content),
                navigation=len(candidates) - len(content),
            )
        return self._match_among(target, candidates, read, area)

    def _match_among(
        self, target: Target, candidates: list[OcrLine], read: ReadResult, area: WorkArea
    ) -> tuple[MatchResult | None, str | None]:
        plain = target if target.region is None else target.model_copy(update={"region": None})
        match = self._matcher.resolve(plain, candidates, read.state.width, read.state.height)
        if target.region is None or match is None:
            return match, None
        rect = self._region_rect(target.region, read, area, target.scope)
        inside = [line for line in candidates if _inside_box(rect, *line.bbox.center)]
        preferred = (
            self._matcher.resolve(plain, inside, read.state.width, read.state.height)
            if inside
            else None
        )
        if preferred is not None and preferred.score >= match.score:
            return preferred, None
        cx, cy = match.line.bbox.center
        return match, (
            f'region "{target.region.value}" did not decide: nothing in that part of the '
            f"surface matched as well as the line at ({cx},{cy}), which was taken instead"
        )

    def _locate(
        self, target: Target, read: ReadResult, area: WorkArea
    ) -> tuple[_Resolution, str | None]:
        resolution = self._resolve_target(target, read, area)
        if resolution.match is not None or not target.scroll_to_find:
            return resolution, None
        return self._scroll_search(target, resolution, area)

    def _scroll_search(
        self, target: Target, missed: _Resolution, area: WorkArea
    ) -> tuple[_Resolution, str | None]:
        # PR-030
        anchor, refused = self._scroll_anchor(target.region, missed.read, area)
        if anchor is None:
            return _noted(missed, f"not on screen, {refused}"), None
        # PR-025
        ends: set[ScrollEdge] = set()
        self._open_scroll_pass(target.region, missed.read, area)
        resolution, aborted = self._sweep_both_ways(target, missed, area, anchor, ends)
        self._close_scroll_pass(ends)
        return resolution, aborted

    def _sweep_both_ways(
        self,
        target: Target,
        missed: _Resolution,
        area: WorkArea,
        anchor: tuple[int, int],
        ends: set[ScrollEdge],
    ) -> tuple[_Resolution, str | None]:
        self._screen_dirty = True
        resolution = missed
        moves: list[str] = []
        notes: list[str] = []
        at_rest = True
        # PR-031
        stirred = False
        for direction, name in ((1, "down"), (-1, "up")):
            sweep = self._sweep(target, area, anchor, direction, resolution.read, stirred)
            stirred = stirred or sweep.moved > 0
            if sweep.resolution is not None:
                resolution = sweep.resolution
            if sweep.steps:
                moves.append(f"{sweep.steps}x {name}")
                at_rest = False
            if sweep.edge is not None:
                ends.add(sweep.edge)
            if sweep.stopped is not None:
                notes.append(sweep.stopped)
            if sweep.aborted is not None:
                notes.append("not scrolled back (kill-switch)")
                return _searched(resolution, moves, notes), sweep.aborted
            if sweep.found:
                return _searched(resolution, moves, notes), None

            back, note, aborted = self._rewind(anchor, direction, sweep.moved, area)
            if back:
                moves.append(f"{back}x back")
            if note is not None:
                notes.append(note)
            if aborted is not None:
                return _searched(resolution, moves, notes), aborted
            if note is not None:
                break
            if moves:
                resolution = self._resolve_target(target, self._read(force_ocr=True), area)
                at_rest = True

        if moves and not at_rest:
            resolution = self._resolve_target(target, self._read(force_ocr=True), area)
        return _searched(resolution, moves, notes), None

    def _sweep(
        self,
        target: Target,
        area: WorkArea,
        anchor: tuple[int, int],
        direction: int,
        read: ReadResult,
        stirred: bool = False,
    ) -> _Sweep:
        sweep = _Sweep(read=read)
        while sweep.steps < MAX_SCROLL_SEARCH_STEPS:
            aborted = self._abort_requested(" while scrolling to find the target")
            if aborted is not None:
                sweep.aborted = aborted
                return sweep
            if not self._can_afford_scroll(sweep.moved):
                sweep.stopped = "action budget reached"
                return sweep
            if self._out_of_time():
                sweep.stopped = "step timeout reached"
                return sweep
            failure = self._scroll_at(anchor, direction * SCROLL_SEARCH_AMOUNT)
            if failure is not None:
                sweep.stopped = failure
                return sweep
            sweep.steps += 1

            self._settle_after_scroll(area)
            fresh = self._read(force_ocr=True)
            moved = self._screen_changed(sweep.read, fresh).changed
            sweep.read = fresh
            if not moved:
                # PR-031
                if not sweep.moved and not stirred:
                    sweep.stopped = SCROLL_STUCK_NOTE
                    return sweep
                # PR-025
                sweep.edge = ScrollEdge.BOTTOM if direction > 0 else ScrollEdge.TOP
                sweep.stopped = f"reached the {'bottom' if direction > 0 else 'top'} of the content"
                return sweep
            sweep.moved += 1
            sweep.resolution = self._resolve_target(target, fresh, area)
            if sweep.found:
                return sweep
        return sweep

    def _rewind(
        self,
        anchor: tuple[int, int],
        direction: int,
        steps: int,
        area: WorkArea,
    ) -> tuple[int, str | None, str | None]:
        done = 0
        for _ in range(steps):
            aborted = self._abort_requested(" while scrolling back after the search")
            if aborted is not None:
                return done, "not fully scrolled back (kill-switch)", aborted
            if not self._can_afford_scroll(steps - done - 1):
                return done, "not fully scrolled back (action budget reached)", None
            failure = self._scroll_at(anchor, -direction * SCROLL_SEARCH_AMOUNT)
            if failure is not None:
                return done, f"could not scroll back: {failure}", None
            done += 1
            self._settle_after_scroll(area)
        return done, None, None

    # PR-025
    def _open_scroll_pass(self, region: Region | None, read: ReadResult, area: WorkArea) -> None:
        self._scroll_pass = None
        # PR-027
        self._pass_screen_id = None
        if not area.scoped:
            return
        rect = self._content_rect(region, read, area)
        if rect.h <= 0:
            return
        self._scroll_pass = ScrollPass(rect)
        self._scroll_pass.page(area.in_window(read.state.lines))
        self._pass_screen_id = read.screen_db_id

    # PR-025, PR-027
    def _close_scroll_pass(self, ends: Collection[ScrollEdge]) -> None:
        walked, screen_id = self._scroll_pass, self._pass_screen_id
        if walked is None or screen_id is None:
            return
        extent = walked.extent(ends)
        if extent is None:
            _log.info(
                "scroll_pass.unmeasured",
                screen_id=screen_id,
                at_rest=walked.at_rest,
            )
            return
        self._repo.record_scroll_pass(screen_id, extent, walked.placements())

    def _can_afford_scroll(self, owed: int) -> bool:
        return self._actions_used + owed + 2 <= self._settings.max_actions_per_plan

    def _scroll_at(self, anchor: tuple[int, int], amount: int) -> str | None:
        px, py = self._to_points(*anchor)
        try:
            self._input.move(px, py)
            self._input.scroll(amount)
        except (InputControlError, ValueError) as exc:
            _log.warning("scroll_search.failed", x=px, y=py, amount=amount, error=str(exc))
            return f"scroll failed: {exc}"
        self._actions_used += 1
        return None

    def _settle_after_scroll(self, area: WorkArea) -> None:
        self._reader.wait_until_stable(
            self._remaining_ms(),
            quiet_grace_ms=self._settings.settle_quiet_grace_ms,
            region=_settle_region(area),
        )

    # PR-030
    def _scroll_anchor(
        self, region: Region | None, read: ReadResult, area: WorkArea
    ) -> tuple[tuple[int, int] | None, str]:
        rect = self._content_rect(region, read, area)
        if rect.w <= 0 or rect.h <= 0:
            _log.info("scroll_search.no_anchor", region=region, area=area.label())
            return None, "nothing scrollable here"
        lines = [
            line
            for line in area.in_window(read.state.lines)
            if _inside_box(rect, *line.bbox.center)
        ]
        rides = [line for line in lines if not read.navigation.holds(line)]
        anchor = (
            None
            if lines and not rides
            else _densest_cell_center(rect, rides, read.navigation.areas)
        )
        if anchor is None:
            _log.info(
                "scroll_search.only_navigation",
                region=region,
                area=area.label(),
                lines=len(lines),
                rides=len(rides),
            )
            return None, "only the application's navigation is here, and it does not scroll"
        if in_failsafe_corner(self._to_points(*anchor)):
            _log.info("scroll_search.anchor_in_failsafe_corner", x=anchor[0], y=anchor[1])
            return None, "nothing scrollable here outside the failsafe corner"
        return anchor, ""

    def _content_rect(self, region: Region | None, read: ReadResult, area: WorkArea) -> BBox:
        width, height = read.state.width, read.state.height
        if width <= 0 or height <= 0:
            return BBox(x=0, y=0, w=0, h=0)
        if region is None:
            return self._surface_rect(Scope.WINDOW, area, width, height)
        return self._region_rect(region, read, area, Scope.WINDOW)

    def _surface_rect(self, scope: Scope, area: WorkArea, width: int, height: int) -> BBox:
        frame = BBox(x=0, y=0, w=width, h=height)
        if not area.scoped:
            if scope is Scope.CHROME or area.blind_region is None:
                return frame
            return clamp_box(area.blind_region, width, height)
        if scope is Scope.CHROME:
            return clamp_box(_bounding(area.chrome), width, height) if area.chrome else frame
        return clamp_box(area.window, width, height)

    def _region_rect(
        self, region: Region, read: ReadResult, area: WorkArea, scope: Scope = Scope.WINDOW
    ) -> BBox:
        base = self._surface_rect(scope, area, read.state.width, read.state.height)
        if base.w <= 0 or base.h <= 0:
            return base
        bounds = region_bounds(region, base.w, base.h)
        return BBox(x=base.x + bounds.x, y=base.y + bounds.y, w=bounds.w, h=bounds.h)

    def _begin_step_area(self, window_query: str | None) -> str | None:
        monitor = self._monitor_info()
        try:
            area = resolve_work_area(
                self._windows.frontmost_windows(),
                monitor=monitor,
                target_app=self._target_app,
                window_query=window_query,
                same_app=self._apps.same_app,
                frame=lambda: self._reader.grab_region(
                    BBox(x=0, y=0, w=monitor.width_px, h=monitor.height_px)
                ),
                refiner=self._rect_refiner,
                obstructions=getattr(self._windows, "obstructions_over", None),
            )
        except WindowSelectionError as exc:
            _log.warning("workarea.not_found", target_app=self._target_app, query=window_query)
            return str(exc)
        self._area = area
        self._show_overlay()
        return None

    def _resolve_path(
        self, points: list[PathPoint], read: ReadResult, area: WorkArea
    ) -> _PathResolution:
        resolved: list[tuple[int, int]] = []
        labels: list[str] = []
        moved = False
        note = area.describe(
            len(area.select(read.state.lines, Scope.WINDOW)), len(read.state.lines), Scope.WINDOW
        )
        for number, point in enumerate(points, start=1):
            if point.target is not None:
                # PR-028
                resolution, aborted = self._locate(point.target, read, area)
                read = resolution.read
                note = resolution.note
                if aborted is not None:
                    return _PathResolution(
                        points=[],
                        labels=labels,
                        read=read,
                        note=note,
                        reason=aborted,
                        status=RunStatus.ABORTED,
                    )
                if resolution.moved:
                    moved = True
                if resolution.match is None:
                    return _PathResolution(
                        points=[],
                        labels=labels,
                        read=read,
                        note=note,
                        failed_target=point.target,
                        failed_resolution=resolution,
                    )
                match = resolution.match
                cx, cy = match.line.bbox.center
                label = _matched(match, point.target.scope)
                if point.offset is not None:
                    cx += point.offset.dx
                    cy += point.offset.dy
                    label += f" offset {point.offset.dx:+d},{point.offset.dy:+d}"
                label += f" at ({cx},{cy})"
            else:
                cx, cy = int(point.x or 0), int(point.y or 0)
                label = f"({cx},{cy})"

            if not (0 <= cx < read.state.width and 0 <= cy < read.state.height):
                return _PathResolution(
                    points=[],
                    labels=labels,
                    read=read,
                    note=note,
                    reason=(
                        f"drag path point {number} lands at ({cx},{cy}), outside the captured "
                        f"screen ({read.state.width}x{read.state.height} px)"
                    ),
                )
            resolved.append((cx, cy))
            labels.append(label)
        # PR-028
        if moved:
            return self._resolve_path(_unscrolled(points), read, area)
        return _PathResolution(points=resolved, labels=labels, read=read, note=note)

    def _gesture_failed(
        self,
        index: int,
        note: str,
        reason: str,
        read: ReadResult,
        status: RunStatus = RunStatus.ESCALATED,
    ) -> _StepOutcome:
        return _step_refused(reason, read, status, note=f"step {index} {note}: {reason}")

    def _perform_drag(
        self, points: list[tuple[int, int]], modifiers: list[str] | None
    ) -> tuple[str, RunStatus] | None:
        path = [self._to_points(x, y) for x, y in points]
        try:
            self._input.drag(path, modifiers=modifiers)
        except FailsafeTriggeredError as exc:
            _log.warning("drag.aborted", points=len(path), error=str(exc))
            return f"kill-switch during the drag: {exc}", RunStatus.ABORTED
        except (InputControlError, ValueError) as exc:
            _log.warning("drag.failed", points=len(path), error=str(exc))
            return f"drag failed: {exc}", RunStatus.ESCALATED
        return None

    def _begin_servo(
        self,
        index: int,
        step: Step,
        read: ReadResult,
        area: WorkArea,
        focus_note: str,
    ) -> tuple[_ServoSetup | None, _StepOutcome | None]:
        readout = step.until_readout
        if readout is None:
            return None, None
        goal = _servo_goal(readout)
        if goal is None:
            forms = ", ".join(parser.name for parser in DEFAULT_PARSERS)
            reason = (
                f'the readout value "{readout.value}" is not a reading anything can compare '
                f"against: it has to be written the way the application writes it, as one of "
                f'{forms} (e.g. "00:12:00", "50%", "-6 dB", "1 250")'
            )
            return None, self._gesture_failed(
                index, _prefixed(focus_note, step.action.value), reason, read
            )
        # PR-028
        watched, _ = self._locate(readout.watch, read, area)
        if watched.match is None:
            return None, self._target_not_found(index, step, readout.watch, watched, focus_note)
        return _ServoSetup(goal=goal, anchor=watched.match.line.bbox, read=watched.read), None

    def _servo_drag(
        self,
        points: list[tuple[int, int]],
        step: Step,
        goal: _ServoGoal,
        anchor: BBox,
        read: ReadResult,
    ) -> _Servo:
        aim = _servo_aim(points, goal)
        if aim.refusal is not None:
            return aim.refusal
        origin, direction = aim.origin, aim.direction
        region = self._readout_region(anchor, read)
        reach = self._servo_reach(origin, direction, read)
        deadline = self._step_deadline

        moves = 0
        blind = 0
        seen = ""
        offset = 0.0
        stride = min(aim.span, max(reach[1], _SERVO_MIN_MOVE_PX))
        forward = 1.0
        latest: tuple[float, float] | None = None
        previous: tuple[float, float] | None = None
        below: tuple[float, float] | None = None
        above: tuple[float, float] | None = None
        failure: tuple[str, RunStatus] | None = None

        try:
            with self._input.drag_hold(
                *self._to_points(*origin), modifiers=step.modifiers
            ) as steer:
                while True:
                    aborted = self._abort_requested(" while the drag was steering to the readout")
                    if aborted is not None:
                        failure = (aborted, RunStatus.ABORTED)
                        break

                    reading = self._look_at_readout(region, anchor, goal.parser)
                    if reading.text:
                        seen = reading.text
                    if reading.value is None:
                        blind += 1
                        if blind >= _SERVO_MAX_BLIND_LOOKS:
                            failure = (
                                self._unreadable_reason(goal, region, blind, moves, seen),
                                RunStatus.ESCALATED,
                            )
                            break
                        time.sleep(_SERVO_REDRAW_SETTLE_S)
                        continue
                    blind = 0

                    error = goal.value - reading.value
                    if abs(error) <= goal.tolerance:
                        _log.info("servo.reached", value=goal.text, reading=seen, moves=moves)
                        break
                    if moves >= _SERVO_MAX_MOVES or time.monotonic() >= deadline:
                        failure = (
                            self._unreached_reason(goal, moves, seen, "spent its budget"),
                            RunStatus.ESCALATED,
                        )
                        break

                    if error > 0:
                        below = (offset, error)
                    else:
                        above = (offset, error)
                    previous, latest = latest, (offset, error)
                    next_move = self._servo_move(
                        latest, previous, below, above, stride, forward, reach
                    )
                    if next_move.stopped:
                        failure = (
                            self._unreached_reason(goal, moves, seen, next_move.stopped),
                            RunStatus.ESCALATED,
                        )
                        break

                    offset = next_move.offset
                    stride, forward = next_move.stride, next_move.forward
                    steer(*self._along(origin, direction, offset))
                    moves += 1
                    time.sleep(_SERVO_REDRAW_SETTLE_S)
        except FailsafeTriggeredError as exc:
            _log.warning("servo.aborted", moves=moves, error=str(exc))
            return _servo_aborted(moves, str(exc))
        except (InputControlError, CaptureError, OcrError, ValueError) as exc:
            _log.warning("servo.failed", moves=moves, error=str(exc))
            return _servo_released(moves, goal, seen, str(exc))

        return _servo_outcome(moves, goal, seen, failure)

    @staticmethod
    def _servo_move(
        latest: tuple[float, float],
        previous: tuple[float, float] | None,
        below: tuple[float, float] | None,
        above: tuple[float, float] | None,
        stride: float,
        forward: float,
        reach: tuple[float, float],
    ) -> _ServoMove:
        offset, error = latest
        secant: float | None = None
        if previous is not None and previous[1] != error:
            secant = offset - error * (offset - previous[0]) / (error - previous[1])

        bracket = sorted((below[0], above[0])) if below is not None and above is not None else None
        if bracket is not None:
            low, high = bracket
            candidate = secant if secant is not None and low < secant < high else (low + high) / 2.0
        elif secant is not None:
            candidate = secant
        else:
            candidate = offset + forward * stride
            stride *= 2.0

        candidate = min(max(candidate, reach[0]), reach[1])
        if abs(candidate - offset) >= _SERVO_MIN_MOVE_PX:
            return _ServoMove(
                offset=candidate, stride=stride, forward=math.copysign(1.0, candidate - offset)
            )
        if bracket is not None:
            return _ServoMove(
                stopped=(
                    f"the readout skips the value: it was seen on both sides of it within "
                    f"{bracket[1] - bracket[0]:.0f} px, at offsets {bracket[0]:.0f} and "
                    f"{bracket[1]:.0f} from where the button went down"
                )
            )
        return _ServoMove(
            stopped=(
                f"the pointer reached the edge of the area it may be dragged in, "
                f"{offset:.0f} px from where the button went down"
            )
        )

    def _along(
        self, origin: tuple[int, int], direction: tuple[float, float], offset: float
    ) -> tuple[float, float]:
        return self._to_points(
            round(origin[0] + direction[0] * offset), round(origin[1] + direction[1] * offset)
        )

    def _readout_region(self, anchor: BBox, read: ReadResult) -> BBox:
        width, height = read.state.width, read.state.height
        grown = BBox(
            x=anchor.x - _SERVO_READOUT_MARGIN_X_PX,
            y=anchor.y - _SERVO_READOUT_MARGIN_Y_PX,
            w=anchor.w + 2 * _SERVO_READOUT_MARGIN_X_PX,
            h=anchor.h + 2 * _SERVO_READOUT_MARGIN_Y_PX,
        )
        return clamp_box(grown, width, height)

    def _servo_reach(
        self, origin: tuple[int, int], direction: tuple[float, float], read: ReadResult
    ) -> tuple[float, float]:
        width, height = read.state.width, read.state.height
        bounds = self._pointer_bounds(width, height)
        if not _inside_box(bounds, origin[0], origin[1]):
            bounds = BBox(x=0, y=0, w=width, h=height)
        low, high = -math.inf, math.inf
        for position, component, first, last in (
            (origin[0], direction[0], bounds.x, bounds.x + bounds.w - 1),
            (origin[1], direction[1], bounds.y, bounds.y + bounds.h - 1),
        ):
            if abs(component) < _SERVO_FLAT_COMPONENT:
                continue
            near, far = (first - position) / component, (last - position) / component
            low = max(low, min(near, far))
            high = min(high, max(near, far))
        return low, high

    def _look_at_readout(self, region: BBox, anchor: BBox, parser: TickParser) -> _Reading:
        lines = self._reader.read_region_text(region)
        if not lines:
            return _Reading(text="")
        centre = anchor.center

        def distance(line: OcrLine) -> float:
            x, y = line.bbox.center
            return math.hypot(x - centre[0], y - centre[1])

        readable = [(line, parser.parse(line.text)) for line in lines]
        parsed = [(line, value) for line, value in readable if value is not None]
        if parsed:
            line, value = min(parsed, key=lambda pair: distance(pair[0]))
            return _Reading(text=line.text, value=value)
        return _Reading(text=min(lines, key=distance).text)

    @staticmethod
    def _unreadable_reason(
        goal: _ServoGoal, region: BBox, looks: int, moves: int, seen: str
    ) -> str:
        showed = f'the nearest text was "{_clip(seen, _SUMMARY_MAX_ELEMENT_CHARS)}"'
        if not seen:
            showed = "the region held no text at all"
        return (
            f"the readout this drag was steering by could not be read as {goal.parser.name}: "
            f"{_plural(looks, 'look')} over the {region.w}x{region.h} px at "
            f"({region.x},{region.y}) parsed nothing and {showed}; the button was released "
            f"where it stood, after {_plural(moves, 'move')}"
        )

    @staticmethod
    def _unreached_reason(goal: _ServoGoal, moves: int, seen: str, why: str) -> str:
        wanted = f'"{goal.text}"'
        if goal.tolerance:
            wanted = f"{wanted} (±{goal.tolerance:g})"
        showing = f'"{_clip(seen, _SUMMARY_MAX_ELEMENT_CHARS)}"' if seen else "nothing"
        return (
            f"the drag never reached the readout {wanted}: {why} after "
            f"{_plural(moves, 'move')} and was released showing {showing}"
        )

    def _aim(self, box: BBox, read: ReadResult, timeout_ms: int) -> _Aim:
        x, y = box.center
        if not self._settings.motion_tracking_enabled:
            return _Aim(x=x, y=y)

        tracker = TargetTracker()
        try:
            tracker.lock(read.frame.image, box, at_ms=read.frame.captured_at_ms)
        except (TypeError, ValueError) as exc:
            _log.info("aim.not_trackable", box=(box.x, box.y, box.w, box.h), reason=str(exc))
            return _Aim(x=x, y=y, note=f"no movement recheck ({exc})")

        region = self._search_region(box, read)
        try:
            grabbed = self._reader.grab_frame(region)
            pixels = grabbed.image
            seen = tracker.update(pixels, (region.x, region.y), at_ms=grabbed.captured_at_ms)
        except (CaptureError, ValueError) as exc:
            _log.warning(
                "aim.recheck_failed",
                region=(region.x, region.y, region.w, region.h),
                error=str(exc),
            )
            return _Aim(x=x, y=y, note=f"no movement recheck ({exc})")

        if seen.status is TrackStatus.LOST:
            return self._aim_after_settle(tracker, box, read, region, timeout_ms, seen)
        if self._target_is_moving(read, region, box, pixels):
            return self._track(tracker, box, read, seen, timeout_ms)
        return self._aim_at(box, seen.bbox, "")

    def _aim_after_settle(
        self,
        tracker: TargetTracker,
        box: BBox,
        read: ReadResult,
        region: BBox,
        timeout_ms: int,
        first: TrackedTarget,
    ) -> _Aim:
        self._reader.wait_until_stable(
            timeout_ms,
            quiet_grace_ms=self._settings.settle_quiet_grace_watched_ms,
            region=_settle_region(self._area),
        )
        x, y = box.center
        try:
            grabbed = self._reader.grab_frame(region)
            seen = tracker.update(grabbed.image, (region.x, region.y), at_ms=grabbed.captured_at_ms)
        except (CaptureError, ValueError) as exc:
            _log.warning("aim.second_look_failed", error=str(exc))
            return _Aim(
                x=x,
                y=y,
                failure=(
                    f"the target was no longer at ({x},{y}) and the area around it could not "
                    f"be captured again: {exc}",
                    RunStatus.ESCALATED,
                ),
            )
        if seen.status is TrackStatus.LOST:
            _log.warning("aim.target_vanished", box=(box.x, box.y, box.w, box.h))
            return _Aim(
                x=x,
                y=y,
                failure=(
                    f"the target matched at ({x},{y}) was no longer there when the click was "
                    f"about to land: two looks over the {region.w}x{region.h} px around it "
                    f"scored {first.score:.2f} and {seen.score:.2f}, both below the "
                    "threshold that makes a match believable",
                    RunStatus.ESCALATED,
                ),
            )
        return self._aim_at(box, seen.bbox, "target was gone for one look, then reappeared")

    @staticmethod
    def _aim_at(matched: BBox, found: BBox | None, prefix: str) -> _Aim:
        if found is None:  # pragma: no cover - a tracked observation has a box.
            raise RuntimeError("a tracked target reported no position")
        mx, my = matched.center
        fx, fy = found.center
        dx, dy = fx - mx, fy - my
        if dx == 0 and dy == 0:
            return _Aim(x=fx, y=fy, note=prefix)
        moved = (
            f"target moved {round(math.hypot(dx, dy))}px ({dx:+d},{dy:+d}), "
            f"corrected to ({fx},{fy})"
        )
        return _Aim(x=fx, y=fy, note=_prefixed(prefix, moved))

    def _target_is_moving(
        self, read: ReadResult, region: BBox, box: BBox, pixels: np.ndarray
    ) -> bool:
        try:
            reference = crop_frame(read.frame.image, region)
        except ValueError as exc:
            _log.warning("aim.reference_unavailable", error=str(exc))
            return False
        if reference.shape[:2] != pixels.shape[:2]:
            _log.warning(
                "aim.reference_mismatch",
                reference=tuple(reference.shape[:2]),
                region=tuple(pixels.shape[:2]),
            )
            return False
        for changed in changed_regions(reference, pixels):
            moved = BBox(x=region.x + changed.x, y=region.y + changed.y, w=changed.w, h=changed.h)
            overlap = _intersect(moved, box)
            if overlap.w > 0 and overlap.h > 0:
                return True
        return False

    def _track(
        self,
        tracker: TargetTracker,
        box: BBox,
        read: ReadResult,
        first: TrackedTarget,
        timeout_ms: int,
    ) -> _Aim:
        deadline = time.monotonic() + timeout_ms / 1000.0
        seen = first
        updates = 1
        still = 0
        while True:
            last = seen.bbox if seen.bbox is not None else box
            lx, ly = last.center
            speed = self._tracked_speed(tracker)
            if speed is not None:
                if speed < _TRACK_STILL_SPEED_PX_PER_MS:
                    still += 1
                    if still >= _TRACK_STILL_UPDATES:
                        _log.info("track.stopped", updates=updates, x=lx, y=ly)
                        return _aim_stopped(lx, ly, updates)
                else:
                    still = 0
                    ahead = (
                        tracker.predict(self._settings.click_lead_ms)
                        if updates >= _TRACK_STEADY_UPDATES
                        else None
                    )
                    if ahead is not None:
                        ax, ay = self._within_reach(ahead, read)
                        _log.info(
                            "track.predicted", updates=updates, speed=round(speed, 3), x=ax, y=ay
                        )
                        return _aim_ahead(ax, ay, (lx, ly), speed, updates)

            if updates >= _TRACK_MAX_UPDATES or time.monotonic() >= deadline:
                _log.warning("track.unsettled", updates=updates, x=lx, y=ly)
                return _aim_unsettled(lx, ly, updates)

            aborted = self._abort_requested(" while following the moving target")
            if aborted is not None:
                return _Aim(x=lx, y=ly, failure=(aborted, RunStatus.ABORTED))

            expected = tracker.predict(_TRACK_CYCLE_MS)
            region = self._search_region(expected if expected is not None else last, read)
            if region.w < box.w or region.h < box.h:
                _log.warning("track.off_display", updates=updates, x=lx, y=ly)
                return _aim_off_display(lx, ly, updates)
            try:
                grabbed = self._reader.grab_frame(region)
                seen = tracker.update(
                    grabbed.image, (region.x, region.y), at_ms=grabbed.captured_at_ms
                )
            except (CaptureError, ValueError) as exc:
                _log.warning("track.look_failed", updates=updates, error=str(exc))
                return _aim_unseen(lx, ly, updates, str(exc))
            updates += 1
            if seen.status is TrackStatus.LOST:
                _log.warning("track.lost", updates=updates, x=lx, y=ly, score=round(seen.score, 3))
                return _aim_lost(lx, ly, updates, seen.score)

    @staticmethod
    def _tracked_speed(tracker: TargetTracker) -> float | None:
        velocity = tracker.velocity()
        if velocity is None:
            return None
        return math.hypot(velocity.vx, velocity.vy)

    @staticmethod
    def _search_region(box: BBox, read: ReadResult) -> BBox:
        height, width = int(read.frame.image.shape[0]), int(read.frame.image.shape[1])
        grown = BBox(
            x=box.x - _TRACK_SEARCH_MARGIN_PX,
            y=box.y - _TRACK_SEARCH_MARGIN_PX,
            w=box.w + 2 * _TRACK_SEARCH_MARGIN_PX,
            h=box.h + 2 * _TRACK_SEARCH_MARGIN_PX,
        )
        return clamp_box(grown, width, height)

    def _within_reach(self, box: BBox, read: ReadResult) -> tuple[int, int]:
        height, width = int(read.frame.image.shape[0]), int(read.frame.image.shape[1])
        bounds = self._pointer_bounds(width, height)
        cx, cy = box.center
        return (
            min(max(cx, bounds.x), bounds.x + bounds.w - 1),
            min(max(cy, bounds.y), bounds.y + bounds.h - 1),
        )

    def _pointer_bounds(self, width: int, height: int) -> BBox:
        whole = BBox(x=0, y=0, w=width, h=height)
        area = self._area
        if not area.scoped:
            return whole
        bounds = clamp_box(area.window, width, height)
        return whole if bounds.w == 0 or bounds.h == 0 else bounds

    def _perform_click(
        self, action: Action, x: int, y: int, modifiers: list[str] | None = None
    ) -> tuple[str, RunStatus] | None:
        px, py = self._to_points(x, y)

        def send() -> None:
            if action is Action.CLICK:
                self._input.click(px, py, modifiers=modifiers)
            elif action is Action.DOUBLE_CLICK:
                self._input.click(px, py, count=2, modifiers=modifiers)
            else:
                self._input.right_click(px, py, modifiers=modifiers)

        return self._synthesize(action.value, send)

    def _synthesize(self, label: str, send: Callable[[], None]) -> tuple[str, RunStatus] | None:
        try:
            send()
        except FailsafeTriggeredError as exc:
            _log.warning("input.aborted", action=label, error=str(exc))
            return f"kill-switch during {label}: {exc}", RunStatus.ABORTED
        except (InputControlError, ValueError) as exc:
            _log.warning("input.failed", action=label, error=str(exc))
            return f"{label} failed: {exc}", RunStatus.ESCALATED
        return None

    def _remaining_ms(self) -> int:
        return max(0, round((self._step_deadline - time.monotonic()) * 1000))

    def _out_of_time(self) -> bool:
        return self._remaining_ms() <= 0

    def _count_baseline(self, text: str | None, read: ReadResult) -> tuple[ReadResult, int]:
        if text is None:
            return read, 0
        if read.from_cache:
            read = self._read(force_ocr=True)
        return read, self._count_text(text, read)

    def _finish_step(
        self,
        index: int,
        step: Step,
        before: ReadResult,
        note: str,
        baseline_count: int,
        acted_at: tuple[int, int] | None,
    ) -> _StepOutcome:
        self._screen_dirty = True
        acted = time.monotonic()
        anchor = self._reader.latest_frame()
        region = _settle_region(self._area)
        watch = self._settle_watch(step, before, acted_at)
        settled = self._reader.wait_until_stable(
            watch.timeout_ms,
            quiet_grace_ms=watch.grace_ms,
            region=region,
            acted_at=watch.acted_at,
            baseline=anchor,
        )
        expect = step.expect
        verified = _Verified(ok=True, detail="")
        if expect is not None:
            verified = self._verify_expect(expect, before, self._remaining_ms(), baseline_count)
            # PR-010
            settled = self._reader.wait_until_stable(
                self._remaining_ms(),
                quiet_grace_ms=self._settings.settle_quiet_grace_ms,
                region=region,
                acted_at=watch.acted_at,
                baseline=anchor,
                since_ms=round((time.monotonic() - acted) * 1000),
            )
        after = self._read(force_ocr=expect is not None)

        # PR-006
        crossed = _Transition(
            before.screen_db_id, after.screen_db_id, self._screen_changed(before, after)
        )
        transition = f"{settled.describe()}; {crossed.describe()}"
        if crossed.new_node:
            # PR-021
            self._repo.record_edge(
                crossed.from_id, self._action_payload(step), crossed.to_id, verified.ok
            )

        if expect is None:
            line = f"step {index} {note} -> {transition}"
        elif not verified.ok:
            reason = f"expectation not met: {verified.detail}"
            line = f"step {index} {note} -> {transition}; {reason}"
            appears = expect.appears
            if appears is not None and self._resolve_text(appears, after) is None:
                closest = self._closest_note(appears, after)
                if closest:
                    reason = f"{reason}; {closest}"
            return _step_refused(reason, after, note=line)
        else:
            line = f"step {index} {note} -> {transition}; expect ok ({verified.detail})"

        return _StepOutcome(ok=True, note=line)

    def _settle_watch(
        self, step: Step, before: ReadResult, acted_at: tuple[int, int] | None
    ) -> _SettleWatch:
        expect = step.expect
        budget = self._remaining_ms()
        if step.action is Action.WAIT and expect is None:
            return _SettleWatch(timeout_ms=budget, grace_ms=budget, acted_at=None)
        # PR-017
        if expect is None or self._expect_holds(expect, before):
            return _SettleWatch(
                timeout_ms=budget,
                grace_ms=self._settings.settle_quiet_grace_ms,
                acted_at=acted_at,
            )
        return _SettleWatch(
            timeout_ms=min(budget, self._settings.settle_quiet_grace_ms),
            grace_ms=self._settings.settle_quiet_grace_watched_ms,
            acted_at=acted_at,
        )

    def _expect_holds(self, expect: Expect, before: ReadResult) -> bool:
        if expect.appears is not None and self._resolve_text(expect.appears, before) is None:
            return False
        if (
            expect.disappears is not None
            and self._resolve_text(expect.disappears, before) is not None
        ):
            return False
        if expect.appears_count_increases is not None:
            return False
        return not expect.screen_changes

    def _verify_expect(
        self,
        expect: Expect,
        before: ReadResult,
        timeout_ms: int,
        baseline_count: int,
    ) -> _Verified:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            # PR-018
            current = self._probe()
            ok, detail = self._check_expect(expect, before, current, baseline_count)
            if ok:
                return _Verified(ok=True, detail=detail)
            if time.monotonic() >= deadline:
                return _Verified(ok=False, detail=detail)
            time.sleep(_EXPECT_POLL_INTERVAL_S)

    def _check_expect(
        self, expect: Expect, before: ReadResult, current: ReadResult, baseline_count: int
    ) -> tuple[bool, str]:
        satisfied: list[bool] = []
        details: list[str] = []

        if expect.appears is not None:
            found = self._resolve_text(expect.appears, current) is not None
            satisfied.append(found)
            details.append(f'appears "{expect.appears}"={found}')
        if expect.disappears is not None:
            gone = self._resolve_text(expect.disappears, current) is None
            satisfied.append(gone)
            details.append(f'disappears "{expect.disappears}"={gone}')
        if expect.appears_count_increases is not None:
            text = expect.appears_count_increases
            now = self._count_text(text, current)
            increased = now > baseline_count
            satisfied.append(increased)
            details.append(
                f'appears_count_increases "{text}"={increased} ({baseline_count}->{now})'
            )
        if expect.screen_changes is not None:
            change = self._screen_changed(before, current)
            satisfied.append(change.changed == expect.screen_changes)
            details.append(
                f"screen_changes={expect.screen_changes} "
                f"(actual={change.changed}; {change.describe()})"
            )

        return (all(satisfied), "; ".join(details))

    def _resolve_text(self, text: str, read: ReadResult) -> MatchResult | None:
        return self._matcher.resolve(
            Target(text=text), read.state.lines, read.state.width, read.state.height
        )

    def _count_text(self, text: str, read: ReadResult) -> int:
        return self._matcher.count(
            Target(text=text), read.state.lines, read.state.width, read.state.height
        )

    # PR-016
    def _screen_changed(self, before: ReadResult, current: ReadResult) -> _ScreenChange:
        first, second = before.crop.box, current.crop.box
        reframed = (
            None
            if (first.w, first.h) == (second.w, second.h)
            else f"{first.w}x{first.h} -> {second.w}x{second.h}"
        )
        return _ScreenChange(
            phash_distance=hamming_hex(before.phash, current.phash),
            phash_limit=self._settings.expect_screen_change_hamming,
            pixel_fraction=0.0 if reframed is not None else diff_frames(before.crop, current.crop),
            pixel_limit=self._settings.expect_screen_change_pixel_fraction,
            reframed=reframed,
        )

    def _monitor_info(self) -> MonitorInfo:
        if self._monitor is None:
            self._monitor = self._reader.monitor_info()
        return self._monitor

    def _pixel_scale(self) -> float:
        return self._monitor_info().scale or 1.0

    def _to_points(self, px: int, py: int) -> tuple[float, float]:
        scale = self._pixel_scale()
        return (px / scale, py / scale)

    @staticmethod
    def _action_payload(step: Step) -> dict:
        payload: dict = {"action": step.action.value}
        if step.at is not None:
            payload["at"] = [step.at.x, step.at.y]
        if step.target is not None:
            payload["target"] = step.target.text
            if step.target.region is not None:
                payload["region"] = step.target.region.value
            if step.target.window is not None:
                payload["window"] = step.target.window
        if step.text is not None:
            payload["text"] = step.text
        if step.keys is not None:
            payload["keys"] = list(step.keys)
        if step.amount is not None:
            payload["amount"] = step.amount
        if step.app_name is not None:
            payload["app_name"] = step.app_name
        if step.path is not None:
            payload["path"] = [
                point.model_dump(mode="json", exclude_none=True) for point in step.path
            ]
        if step.modifiers is not None:
            payload["modifiers"] = list(step.modifiers)
        if step.until_readout is not None:
            payload["until_readout"] = step.until_readout.value
        return payload

    def _target_not_found(
        self,
        index: int,
        step: Step,
        target: Target,
        resolution: _Resolution,
        focus_note: str = "",
    ) -> _StepOutcome:
        read = resolution.read
        reason = f'could not resolve target "{target.text}"'
        line = f"step {index} " + _prefixed(
            focus_note,
            f'{step.action.value} "{target.text}": {resolution.note} -> {reason}',
        )
        reason = f"{reason}; {resolution.note}"
        if resolution.surface:
            reason = f"{reason}; {resolution.surface}"
        hidden = self._hidden_note(target, read)
        if hidden:
            reason = f"{reason}; {hidden}"
        closest = self._closest_note(target.text, read, resolution.candidates)
        if closest:
            reason = f"{reason}; {closest}"
        return _step_refused(reason, read, note=line)

    def _hidden_note(self, target: Target, read: ReadResult) -> str:
        area = self._area
        if not area.scoped or target.scope is not Scope.WINDOW or not area.covered.any:
            return ""
        in_window = area.in_window(read.state.lines)
        visible = {id(line) for line in area.visible(in_window)}
        buried = [line for line in in_window if id(line) not in visible]
        if not buried:
            return ""
        plain = target if target.region is None else target.model_copy(update={"region": None})
        match = self._matcher.resolve(plain, buried, read.state.width, read.state.height)
        if match is None:
            return ""
        cx, cy = match.line.bbox.center
        return (
            f'"{_clip(match.line.text.strip(), _SUMMARY_MAX_ELEMENT_CHARS)}" is on this window '
            f"at ({cx},{cy}) but {area.covered.named} is drawn over it — close or move what is "
            "on top, then retry"
        )

    def _closest_note(self, text: str, read: ReadResult, lines: list[OcrLine] | None = None) -> str:
        candidates = self._matcher.rank(
            Target(text=text),
            read.state.lines if lines is None else lines,
            read.state.width,
            read.state.height,
            _NEAR_MISS_CANDIDATES,
        )
        rendered: list[str] = []
        for candidate in candidates:
            if candidate.score < _NEAR_MISS_MIN_SCORE:
                continue
            label = _clip(candidate.line.text.strip(), _SUMMARY_MAX_ELEMENT_CHARS)
            note = f"{candidate.score:.2f}"
            if candidate.opposed is not None:
                note += (
                    f', rejected: says "{candidate.opposed.candidate_word}" where the target '
                    f'says "{candidate.opposed.target_word}" — the opposite of what was asked'
                )
            rendered.append(f'"{label}" ({note})')
        if not rendered:
            return ""
        return "closest on screen: " + ", ".join(rendered)

    def _terminal_report(
        self,
        run_id: int,
        status: RunStatus,
        completed: list[int],
        failed_step: int | None,
        reason: str | None,
        journey: Sequence[str],
        frame: np.ndarray | None = None,
        lines: list[OcrLine] | None = None,
        from_cache: bool = False,
    ) -> ExecutionReport:
        told = list(journey)
        if frame is None:
            try:
                fresh = self._read(force_ocr=True)
            except (CaptureError, OcrError) as exc:
                told.append(f"no picture to show for it: the screen could not be read: {exc}")
                _log.warning("run.report_screen_unreadable", run_id=run_id, error=str(exc))
            else:
                frame = fresh.frame.image
                lines = fresh.state.lines
                from_cache = fresh.from_cache
        return self._report(
            run_id,
            status,
            completed,
            failed_step,
            reason,
            told,
            frame=frame,
            elements=format_surface_listing(self._area, lines or [], from_cache),
        )

    def _report(
        self,
        run_id: int,
        status: RunStatus,
        completed: list[int],
        failed_step: int | None,
        reason: str | None,
        journey: Sequence[str],
        *,
        frame: np.ndarray | None = None,
        elements: list[str] | None = None,
    ) -> ExecutionReport:
        final = self._finish(run_id, status, journey)
        return ExecutionReport(
            status=status,
            completed_steps=completed,
            failed_step=failed_step,
            reason=reason,
            journey=final,
            screenshot_png=(
                None if frame is None else screenshot_png(frame, max_width=_SCREENSHOT_MAX_WIDTH)
            ),
            elements=elements or [],
            seen_screens=list(self._seen.values()),
            extracted=list(self._extracted),
        )
