from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from anyio import to_thread
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from choto.annotate import AnnotationBatch, AnnotationError, IconAnnotator, LabelOutcome
from choto.capture.grabber import CaptureError
from choto.capture.monitor import MonitorInfo
from choto.capture.overlaymask import mask_overlay
from choto.capture.source import ScreenSource
from choto.config import Settings, get_settings
from choto.executor.applauncher import AppActivationError, AppLauncher
from choto.executor.runner import (
    Executor,
    format_extracted,
    format_seen_screens,
    format_surface_listing,
)
from choto.executor.screen_reader import ReadResult, ScreenReader
from choto.executor.tooltipprobe import (
    DEFAULT_TOOLTIP_PROBE_LIMIT,
    TooltipProbe,
    TooltipProbeReport,
    prose_ocr_settings,
    validate_probe_limit,
)
from choto.executor.workarea import (
    WindowRectRefiner,
    WorkArea,
    resolve_work_area,
    window_titles,
    windows_on_frame,
)
from choto.graph.engine import create_engine_from_settings, create_session_factory
from choto.graph.eviction import evict
from choto.graph.repository import GraphRepository
from choto.inputctl.protocol import InputBackend
from choto.log import get_logger
from choto.matching.matcher import TargetMatcher
from choto.mcpserver.toolspec import (
    ANNOTATE_DESCRIPTION,
    EXECUTE_DESCRIPTION,
    FOCUS_DESCRIPTION,
    MISSION_FINISH_DESCRIPTION,
    MISSION_MARK_DESCRIPTION,
    MISSION_REPLAY_DESCRIPTION,
    MISSION_START_DESCRIPTION,
    MISSION_STATUS_DESCRIPTION,
    OBSERVE_DESCRIPTION,
    PROBE_TOOLTIPS_DESCRIPTION,
    RECALL_DESCRIPTION,
    SUBMIT_DESCRIPTION,
)
from choto.missionsvc import MissionService
from choto.models import BBox, ElementKind, ExecutionReport, Plan, RunStatus
from choto.ocr.engine import OcrError, create_ocr_engine, resolve_ocr_factory
from choto.overlay.backend import OverlayBackend
from choto.platforms import (
    PlatformBackends,
    PlatformError,
    create_overlay_backend,
    resolve_platform,
)
from choto.recall import DEFAULT_DEPTH, RecallService
from choto.vision.imaging import screenshot_png
from choto.vision.windowsource import WindowList, WindowSource

_log = get_logger(__name__)

_SCREENSHOT_MAX_WIDTH = 1280


def create_capture_source(backends: PlatformBackends, overlay: OverlayBackend) -> ScreenSource:
    capabilities = overlay.capabilities
    reaches_the_frame = capabilities.draws_frame and not capabilities.excluded_from_capture
    return mask_overlay(backends.screen_source(), overlay if reaches_the_frame else None)


@dataclass
class AppContext:
    settings: Settings
    backends: PlatformBackends
    session_factory: sessionmaker[Session]
    repo: GraphRepository
    recall: RecallService
    annotator: IconAnnotator
    matcher: TargetMatcher
    input_controller: InputBackend
    window_source: WindowSource
    app_launcher: AppLauncher
    overlay: OverlayBackend
    lock: threading.Lock
    mission_lock: threading.Lock = field(default_factory=threading.Lock)

    def new_reader(self) -> ScreenReader:
        source = create_capture_source(self.backends, self.overlay)
        ocr = create_ocr_engine(self.settings)
        return ScreenReader(source, ocr, self.repo, self.settings, self.window_source)

    def missions(self) -> MissionService:
        return MissionService(self.repo, self.window_source, self.app_launcher)

    def close(self) -> None:
        for name, close in (
            ("input_backend", self.input_controller.close),
            ("window_source", self.window_source.close),
            ("app_launcher", self.app_launcher.close),
            ("capture_stream", self.backends.close),
        ):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail on cleanup
                _log.warning("daemon.seam_close_failed", seam=name, error=str(exc))


def build_context(settings: Settings | None = None) -> AppContext:
    settings = settings or get_settings()
    resolve_ocr_factory(settings.ocr_engine)
    backends = resolve_platform(settings.platform_backend)
    engine = create_engine_from_settings(settings)
    session_factory = create_session_factory(engine)
    repo = GraphRepository(session_factory)
    matcher = TargetMatcher(settings)
    overlay = create_overlay_backend(backends, settings)
    input_controller = backends.input_backend(
        overlay.stop_requested if overlay.capabilities.stop_button else None
    )
    _log.info(
        "daemon.kill_switch",
        stop_button=overlay.capabilities.stop_button,
        overlay_frame=overlay.capabilities.draws_frame,
        overlay_excluded_from_capture=overlay.capabilities.excluded_from_capture,
    )
    return AppContext(
        settings=settings,
        backends=backends,
        session_factory=session_factory,
        repo=repo,
        recall=RecallService(repo, matcher=matcher),
        annotator=IconAnnotator(repo),
        matcher=matcher,
        input_controller=input_controller,
        window_source=backends.window_source(),
        app_launcher=backends.app_launcher(),
        overlay=overlay,
        lock=threading.Lock(),
    )


class ResourceBusyError(RuntimeError):
    def __init__(self, what: str, waited: float, detail: str) -> None:
        super().__init__(
            f"{what} is held by another Choto session and did not come free within "
            f"{waited:g}s. {detail}"
        )
        self.what = what
        self.waited = waited


@contextlib.contextmanager
def held(lock: threading.Lock, timeout: float, what: str, detail: str) -> Iterator[None]:
    if not lock.acquire(timeout=timeout):
        raise ResourceBusyError(what, timeout, detail)
    try:
        yield
    finally:
        lock.release()


def screen(context: AppContext) -> AbstractContextManager[None]:
    return held(
        context.lock,
        context.settings.contention_timeout_seconds,
        "the screen",
        "Only one call may drive the screen at a time — two plans clicking at once would "
        "fight over one pointer — so this one waited instead of interleaving, and the plan, "
        "observation or tooltip pass in front of it is still going. Nothing was captured and "
        "nothing was clicked here. Try again; if it keeps happening, another conversation is "
        "running long plans on this desktop.",
    )


def checklist(context: AppContext) -> AbstractContextManager[None]:
    return held(
        context.mission_lock,
        context.settings.contention_timeout_seconds,
        "the mission checklist",
        "One mission is active at a time and its items are marked by reading them and then "
        "writing them back, so those decisions are taken one at a time. The likeliest holder "
        "is a mission_replay, which keeps the checklist for its whole run. Nothing was "
        "written here.",
    )


def _on_checklist(context: AppContext, refusal: str, work: Callable[[MissionService], str]) -> str:
    try:
        with checklist(context):
            return work(context.missions())
    except ResourceBusyError as exc:
        return f"{refusal}: {exc}"


def _icon_note(context: AppContext, screen_db_id: int | None) -> str:
    if screen_db_id is None:
        return ""
    icons = [
        element
        for element in context.repo.get_elements(screen_db_id)
        if element.kind is ElementKind.ICON
    ]
    if not icons:
        return ""
    labeled = sum(1 for element in icons if element.text.strip())
    unlabeled = len(icons) - labeled
    hint = " (annotate_icons to label)" if unlabeled else ""
    return f"icons: {labeled} labeled, {unlabeled} unlabeled{hint}"


@dataclass(frozen=True)
class _Observation:
    monitor: MonitorInfo
    listing: WindowList
    area: WorkArea
    result: ReadResult


def _observe(context: AppContext) -> _Observation:
    with screen(context):
        reader = context.new_reader()
        try:
            monitor = reader.monitor_info()
            # PR-007
            listing = context.window_source.frontmost_windows()
            area = resolve_work_area(
                listing,
                monitor=monitor,
                target_app=None,
                window_query=None,
                same_app=context.app_launcher.same_app,
                frame=lambda: reader.grab_region(
                    BBox(x=0, y=0, w=monitor.width_px, h=monitor.height_px)
                ),
                refiner=WindowRectRefiner(enabled=context.settings.window_rect_refinement_enabled),
            )
            return _Observation(monitor, listing, area, reader.read(area))
        finally:
            reader.close()


def observe_impl(context: AppContext) -> list[Any]:
    try:
        observation = _observe(context)
    except ResourceBusyError as exc:
        return [f"observe did nothing: {exc}"]
    except (CaptureError, OcrError) as exc:
        _log.warning("observe.screen_unreadable", error=str(exc))
        return [f"observe failed: the screen could not be read: {exc}"]
    return _render_observation(context, observation)


def _render_observation(context: AppContext, observation: _Observation) -> list[Any]:
    monitor, area, result = observation.monitor, observation.area, observation.result
    state = result.state
    scale = monitor.scale or 1.0
    open_windows = windows_on_frame(
        observation.listing.app_windows,
        scale=scale,
        frame_width=monitor.width_px,
        frame_height=monitor.height_px,
    )
    width_pt = round(state.width / scale)
    height_pt = round(state.height / scale)
    summary_lines = [
        f"app: {state.app_name}",
        f"window: {area.window_line()}",
        f"open_windows: {window_titles(open_windows)}",
        f"known_window: {result.from_cache}",
        f"size_px: {state.width}x{state.height}",
        f"size_pt: {width_pt}x{height_pt}",
        f"single_character_scan: {area.scan_note()}",
    ]
    icon_note = _icon_note(context, result.screen_db_id)
    if icon_note:
        summary_lines.append(icon_note)
    summary_lines.extend(format_surface_listing(area, state.lines, result.from_cache))
    png = screenshot_png(result.frame.image, max_width=_SCREENSHOT_MAX_WIDTH)
    return ["\n".join(summary_lines), Image(data=png, format="png")]


def _evict_after_run(context: AppContext) -> None:
    try:
        evict(context.repo, context.settings)
    except SQLAlchemyError as exc:
        _log.error("map.eviction_failed", error=str(exc))


def execute_plan_impl(
    context: AppContext,
    steps: list[dict[str, Any]],
    mission_item_id: int | None = None,
    llm_version_tag: str = "",
) -> list[Any]:
    try:
        plan = Plan.model_validate({"steps": steps})
    except ValidationError as exc:
        reply: list[Any] = [f"Invalid plan — fix and resend:\n{_format_validation_error(exc)}"]
        if mission_item_id is not None:
            reply.append(_status_block(context))
        return reply

    try:
        report = _run_plan(context, plan)
    except ResourceBusyError as exc:
        refused: list[Any] = [f"execute_plan did nothing: {exc}"]
        if mission_item_id is not None:
            refused.append(_status_block(context))
        return refused

    parts = _render_report(report)
    if mission_item_id is not None:
        try:
            with checklist(context):
                missions = context.missions()
                parts.append(missions.record_run(mission_item_id, report, steps, llm_version_tag))
        except ResourceBusyError as exc:
            parts.append(
                f"the verdict was NOT filed onto mission item #{mission_item_id}: {exc} The run "
                "itself is above and stands; once the checklist is free, record it with "
                f"mission_mark(item_id={mission_item_id}, status=…)."
            )
    return parts


def _status_block(context: AppContext) -> str:
    return _on_checklist(
        context, "the mission status could not be read", lambda missions: missions.status()
    )


def _run_plan(context: AppContext, plan: Plan) -> ExecutionReport:
    with screen(context):
        reader = context.new_reader()
        try:
            executor = Executor(
                reader,
                context.matcher,
                context.input_controller,
                context.window_source,
                context.app_launcher,
                context.repo,
                context.settings,
                context.overlay,
            )
            report = executor.execute(plan)
        finally:
            reader.close()

    _evict_after_run(context)
    return report


def _render_report(report: ExecutionReport) -> list[Any]:
    header = (
        f"status: {report.status.value}\n"
        f"completed_steps: {report.completed_steps}\n"
        + (f"failed_step: {report.failed_step}\n" if report.failed_step is not None else "")
        + (f"reason: {report.reason}\n" if report.reason else "")
        + "journey:\n"
        + "\n".join(report.journey)
    )
    parts: list[Any] = [header]
    harvested = format_extracted(report.extracted)
    if harvested:
        parts.append(harvested)
    if report.status in (RunStatus.ESCALATED, RunStatus.ABORTED):
        if report.screenshot_png is not None:
            parts.append(Image(data=report.screenshot_png, format="png"))
        if report.elements:
            parts.append("\n".join(report.elements))
    summary = format_seen_screens(report.seen_screens)
    if summary:
        parts.append(summary)
    return parts


def focus_app_impl(context: AppContext, app_name: str) -> str:
    try:
        with screen(context):
            try:
                resolved = context.app_launcher.activate(app_name)
            except (AppActivationError, ValueError) as exc:
                return f"focus_app failed: {exc}"
    except ResourceBusyError as exc:
        return f"focus_app did nothing: {exc}"
    return f'focused "{resolved}"'


def recall_impl(
    context: AppContext,
    app_name: str | None = None,
    window: str | None = None,
    depth: int = DEFAULT_DEPTH,
    query: str | None = None,
    from_window: str | None = None,
) -> str:
    return context.recall.render(
        app_name=app_name,
        window=window,
        depth=depth,
        query=query,
        from_window=from_window,
    )


def annotate_icons_impl(context: AppContext, app_name: str | None = None) -> list[Any]:
    batch = context.annotator.build(app_name)
    if batch.empty:
        return [_nothing_to_annotate(batch, app_name)]

    header = [
        f'{batch.glyph_count} unnamed icon(s) of "{batch.app_name}", numbered on the '
        f"sheet{'s' if len(batch.sheets) > 1 else ''} below.",
        "Read each numbered crop, decide what the control is, then call "
        f'submit_icon_labels(app_name="{batch.app_name}", labels={{"<key>": "<name>"}}) '
        "using the key printed for that cell.",
    ]
    if batch.unlabeled_in_app > batch.glyph_count:
        header.append(
            f"{batch.unlabeled_in_app - batch.glyph_count} more unnamed glyph(s) of this app "
            "did not fit — call annotate_icons again after submitting these."
        )
    if batch.unlabeled_elsewhere:
        header.append(
            f"{batch.unlabeled_elsewhere} unnamed glyph(s) belong to other applications; "
            "name one app at a time (a drawing means what its own program says it means)."
        )

    parts: list[Any] = ["\n".join(header)]
    for index, sheet in enumerate(batch.sheets, start=1):
        parts.append(f"sheet {index} of {len(batch.sheets)} — cell: context\n{sheet.legend}")
        parts.append(_annotated_keys(sheet.cell_keys))
        parts.append(Image(data=sheet.png, format="png"))
    _log.info(
        "icons.annotate_requested",
        app_name=batch.app_name,
        glyphs=batch.glyph_count,
        sheets=len(batch.sheets),
    )
    return parts


def _annotated_keys(cell_keys: dict[int, str]) -> str:
    pairs = ", ".join(f"#{number}={key}" for number, key in sorted(cell_keys.items()))
    return f"keys (use these in submit_icon_labels, not the numbers): {pairs}"


def _nothing_to_annotate(batch: AnnotationBatch, asked_for: str | None) -> str:
    if asked_for is not None:
        elsewhere = (
            f" {batch.unlabeled_elsewhere} unnamed glyph(s) are waiting under other "
            "applications — call annotate_icons() with no app_name to be pointed at one."
            if batch.unlabeled_elsewhere
            else ""
        )
        if not batch.glyphs_in_app:
            return (
                f'nothing to annotate for "{asked_for}": the map holds no icon of that '
                "application at all — not a named one and not an unnamed one. Either it has "
                "never been read (observe it, or run a short plan through it, and call this "
                "again), or that is not the name the map files it under — recall() with no "
                f"arguments lists the applications it knows.{elsewhere}"
            )
        return (
            f'nothing to annotate for "{asked_for}": all {batch.glyphs_in_app} icon glyph(s) '
            f"stored for it already have a name.{elsewhere}"
        )
    if not batch.glyphs_everywhere:
        return (
            "nothing to annotate: the map holds no icon glyph at all yet. They are learned "
            "when a window is freshly read, so observe an app with a toolbar (or run a short "
            "plan through it) and call this again."
        )
    return (
        f"nothing to annotate: all {batch.glyphs_everywhere} icon glyph(s) in memory already "
        "have a name."
    )


def submit_icon_labels_impl(context: AppContext, labels: dict[str, str], app_name: str) -> str:
    try:
        outcome = context.annotator.apply(labels, app_name)
    except AnnotationError as exc:
        _log.info("icons.labels_rejected", app_name=app_name, reason=str(exc))
        return f"submit_icon_labels did nothing: {exc}"
    return _label_report(outcome)


def _label_report(outcome: LabelOutcome) -> str:
    remaining = (
        f"{outcome.remaining_unlabeled} glyph(s) of this app still unnamed "
        "(annotate_icons for the next sheet)"
        if outcome.remaining_unlabeled
        else "every icon glyph of this app now has a name"
    )
    return (
        f'named {outcome.glyphs_labeled} glyph(s) of "{outcome.app_name}", '
        f"updating {outcome.elements_updated} place(s) on the map — they are clickable by "
        f"the names you gave them now, here and in every future reading of these windows. "
        f"{remaining}."
    )


def probe_tooltips_impl(context: AppContext, limit: int = DEFAULT_TOOLTIP_PROBE_LIMIT) -> str:
    try:
        limit = validate_probe_limit(limit)
    except ValueError as exc:
        return f"probe_tooltips did nothing: {exc}"

    try:
        report = _probe_tooltips(context, limit)
    except ResourceBusyError as exc:
        return f"probe_tooltips did nothing: {exc}"
    return report.render()


def _probe_tooltips(context: AppContext, limit: int) -> TooltipProbeReport:
    with screen(context), contextlib.ExitStack() as handles:
        reader = handles.enter_context(contextlib.closing(context.new_reader()))
        grabber = handles.enter_context(contextlib.closing(context.backends.screen_grabber()))
        probe = TooltipProbe(
            reader=reader,
            grabber=grabber,
            ocr=create_ocr_engine(prose_ocr_settings(context.settings)),
            input_controller=context.input_controller,
            windows=context.window_source,
            apps=context.app_launcher,
            repo=context.repo,
            settings=context.settings,
            overlay=context.overlay,
        )
        report = probe.probe(limit)
    return report


def mission_start_impl(context: AppContext, goal: str, items: list[dict[str, Any]]) -> str:
    return _on_checklist(
        context, "mission_start did nothing", lambda missions: missions.start(goal, items)
    )


def mission_status_impl(context: AppContext, full: bool = False) -> str:
    return _on_checklist(
        context, "mission_status could not answer", lambda missions: missions.status(full)
    )


def mission_mark_impl(context: AppContext, item_id: int, status: str, reason: str = "") -> str:
    return _on_checklist(
        context,
        "mission_mark did nothing",
        lambda missions: missions.mark(item_id, status, reason),
    )


def mission_finish_impl(context: AppContext, status: str) -> str:
    return _on_checklist(
        context, "mission_finish did nothing", lambda missions: missions.finish(status)
    )


def mission_replay_impl(context: AppContext, item_id: int, llm_version_tag: str = "") -> list[Any]:
    try:
        with checklist(context):
            outcome = context.missions().replay(
                item_id,
                lambda plan: _run_plan(context, plan),
                llm_version_tag,
            )
    except ResourceBusyError as exc:
        return [f"mission_replay did nothing: {exc}"]
    if outcome.report is None:
        return [outcome.text]
    parts = _render_report(outcome.report)
    parts.append(outcome.text)
    return parts


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


async def _in_worker_thread[T](
    func: Callable[..., T], context: AppContext | PlatformError, *args: Any
) -> T:
    if isinstance(context, PlatformError):
        raise ToolError(str(context))
    return await to_thread.run_sync(func, context, *args)


def create_mcp(context: AppContext | PlatformError) -> FastMCP:
    mcp = FastMCP("choto")

    @mcp.tool(description=OBSERVE_DESCRIPTION, structured_output=False)
    async def observe() -> list[Any]:
        return await _in_worker_thread(observe_impl, context)

    @mcp.tool(description=EXECUTE_DESCRIPTION, structured_output=False)
    async def execute_plan(
        steps: list[dict[str, Any]],
        mission_item_id: int | None = None,
        llm_version_tag: str = "",
    ) -> list[Any]:
        return await _in_worker_thread(
            execute_plan_impl, context, steps, mission_item_id, llm_version_tag
        )

    @mcp.tool(description=FOCUS_DESCRIPTION)
    async def focus_app(app_name: str) -> str:
        return await _in_worker_thread(focus_app_impl, context, app_name)

    @mcp.tool(description=RECALL_DESCRIPTION)
    async def recall(
        app_name: str | None = None,
        window: str | None = None,
        depth: int = DEFAULT_DEPTH,
        query: str | None = None,
        from_window: str | None = None,
    ) -> str:
        return await _in_worker_thread(
            recall_impl, context, app_name, window, depth, query, from_window
        )

    @mcp.tool(description=ANNOTATE_DESCRIPTION, structured_output=False)
    async def annotate_icons(app_name: str | None = None) -> list[Any]:
        return await _in_worker_thread(annotate_icons_impl, context, app_name)

    @mcp.tool(description=SUBMIT_DESCRIPTION)
    async def submit_icon_labels(labels: dict[str, str], app_name: str) -> str:
        return await _in_worker_thread(submit_icon_labels_impl, context, labels, app_name)

    @mcp.tool(description=PROBE_TOOLTIPS_DESCRIPTION)
    async def probe_tooltips(limit: int = DEFAULT_TOOLTIP_PROBE_LIMIT) -> str:
        return await _in_worker_thread(probe_tooltips_impl, context, limit)

    @mcp.tool(description=MISSION_START_DESCRIPTION)
    async def mission_start(goal: str, items: list[dict[str, Any]]) -> str:
        return await _in_worker_thread(mission_start_impl, context, goal, items)

    @mcp.tool(description=MISSION_STATUS_DESCRIPTION)
    async def mission_status(full: bool = False) -> str:
        return await _in_worker_thread(mission_status_impl, context, full)

    @mcp.tool(description=MISSION_MARK_DESCRIPTION)
    async def mission_mark(item_id: int, status: str, reason: str = "") -> str:
        return await _in_worker_thread(mission_mark_impl, context, item_id, status, reason)

    @mcp.tool(description=MISSION_FINISH_DESCRIPTION)
    async def mission_finish(status: str) -> str:
        return await _in_worker_thread(mission_finish_impl, context, status)

    @mcp.tool(description=MISSION_REPLAY_DESCRIPTION, structured_output=False)
    async def mission_replay(item_id: int, llm_version_tag: str = "") -> list[Any]:
        return await _in_worker_thread(mission_replay_impl, context, item_id, llm_version_tag)

    return mcp
