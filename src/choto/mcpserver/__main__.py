from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from types import ModuleType

from alembic import command
from alembic.config import Config
from sqlalchemy.exc import SQLAlchemyError

from choto.config import Settings, get_settings
from choto.graph import hygiene
from choto.graph.eviction import evict
from choto.log import get_logger, setup_logging
from choto.matching import embeddings
from choto.mcpserver import appbundle, launchdsocket
from choto.mcpserver.daemonprobe import DaemonHealth
from choto.mcpserver.server import AppContext, build_context, create_mcp
from choto.mcpserver.service import (
    PROJECT_ROOT,
    ServiceError,
    ServiceManager,
    resolve_service_manager,
)
from choto.mcpserver.socket_path import (
    LAUNCHD_SOCKET_OPTION,
    default_socket_path,
    resolve_socket_path,
)
from choto.mcpserver.socket_transport import (
    SocketUnavailableError,
    serve_activated_socket,
    serve_socket,
)
from choto.platforms import resolve_platform

_PROJECT_ROOT = PROJECT_ROOT
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"

_EXIT_OK = 0
_EXIT_FAILURE = 1
_EXIT_PERMISSION_DENIED = 2

_log = get_logger(__name__)


def _anchor_db_path() -> None:
    db_path = Settings().db_path
    if str(db_path) == ":memory:" or db_path.is_absolute():
        return
    absolute = (_PROJECT_ROOT / db_path).resolve()
    os.environ["CHOTO_DB_PATH"] = str(absolute)
    get_settings.cache_clear()
    _log.info("db_path.anchored", db_path=str(absolute))


def _run_migrations() -> None:
    if not _ALEMBIC_INI.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {_ALEMBIC_INI}")
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_PROJECT_ROOT / "ops" / "alembic"))
    with contextlib.redirect_stdout(sys.stderr):
        command.upgrade(config, "head")
    _log.info("migrations.applied", target="head")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_socket = default_socket_path()
    parser = argparse.ArgumentParser(
        prog="choto",
        description=(
            "Choto MCP server: Claude Desktop's eyes and hands. "
            "Serves stdio by default, or a unix socket in daemon mode. "
            "The `service` commands manage the user service that keeps the "
            "daemon running across logins."
        ),
    )
    parser.add_argument(
        "--socket",
        dest="socket_path",
        nargs="?",
        const=default_socket,
        default=None,
        metavar="PATH",
        help=(
            "Run as a daemon listening on a unix socket instead of stdio. "
            f"Without a value the default path is used ({default_socket}). "
            "Clients attach with choto-bridge."
        ),
    )
    parser.add_argument(
        LAUNCHD_SOCKET_OPTION,
        dest="launchd_socket",
        default=None,
        metavar="NAME",
        help=(
            "Serve the socket launchd already bound under this name in the "
            "agent's Sockets dictionary, instead of binding one. Written into "
            "the launch agent by `choto service install`; passing it by hand is "
            "only meaningful inside a launchd job. The daemon then leaves once "
            "it has served nobody for CHOTO_IDLE_EXIT_SECONDS, since launchd "
            "starts it again on the next connection."
        ),
    )
    parser.add_argument(
        appbundle.CHECK_PERMISSION_FLAG,
        dest="check_input_permission",
        action="store_true",
        help=(
            "Report whether this process may post input events and exit. Run "
            "through Choto.app, this answers for the bundle's own TCC identity, "
            "which is what the daemon runs as under launchd."
        ),
    )
    parser.add_argument(
        appbundle.REPORT_TO_OPTION,
        dest="report_to",
        default=None,
        metavar="PATH",
        help=(
            f"With {appbundle.CHECK_PERMISSION_FLAG}, also write the verdict to "
            "PATH as JSON. Used by `choto service status`, which cannot read the "
            "stderr of a LaunchServices-started process."
        ),
    )
    parser.set_defaults(
        command=None,
        action=None,
        service_socket_path=None,
        dry_run=False,
        app_name=None,
        icon_detector=None,
    )

    commands = parser.add_subparsers(dest="command")
    service = commands.add_parser(
        "service",
        help="Manage the user service that runs the daemon.",
        description=(
            "Install, remove, inspect or restart the user service that keeps the "
            "daemon alive: a launchd agent on macOS, which runs in the Aqua "
            "session and therefore carries the TCC grants the daemon needs. "
            "User-scoped, no sudo involved."
        ),
    )
    actions = service.add_subparsers(dest="action", required=True)

    install = actions.add_parser(
        "install", help="Write the service definition and load it (idempotent)."
    )
    _add_service_socket_option(install)
    install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated service definition to stderr and change nothing.",
    )

    actions.add_parser("uninstall", help="Unload the service and delete it (idempotent).")

    status = actions.add_parser(
        "status", help="Report the service's supervisor state and probe the daemon over MCP."
    )
    _add_service_socket_option(status)

    actions.add_parser("restart", help="Kill and restart the running daemon.")

    _add_map_commands(commands)
    _add_model_commands(commands)

    parsed = parser.parse_args(argv)
    if parsed.launchd_socket is not None and parsed.socket_path is None:
        parsed.socket_path = default_socket
    return parsed


def _add_model_commands(commands: argparse._SubParsersAction) -> None:
    model = commands.add_parser(
        "model",
        help="Install or inspect the models the matcher and the icon pass need.",
        description=(
            "Two models run locally and neither is ever fetched mid-plan. The "
            "matcher's third tier compares meanings with a local model2vec model "
            "held in the Hugging Face cache; the icon pass finds clickable glyphs "
            "with a CoreML detector held in Choto's own model directory. The "
            "daemon loads both from disk and from nowhere else — it never waits "
            "on the network to be told which revision it already has — so putting "
            "them there happens once, here, by a command you type. Without either "
            "Choto still runs: targets resolve on exact and fuzzy text, unnamed "
            "glyphs stay unfound, and every startup says so."
        ),
    )
    model_actions = model.add_subparsers(dest="action", required=True)

    install = model_actions.add_parser(
        "install",
        help="Put the models on this machine (idempotent).",
        description=(
            "Fetches the preferred multilingual embedding model and the smaller "
            "fallback. Both, deliberately: the fallback exists for the day the "
            "preferred one cannot be read, and one that is not in the cache could "
            "never be reached. Needs the network; downloads nothing that is "
            "already there. The icon detector is installed in the same breath, "
            "but from a path you name: there is no server publishing the CoreML "
            "build Choto runs (see --icon-detector)."
        ),
    )
    install.add_argument(
        "--icon-detector",
        dest="icon_detector",
        default=None,
        metavar="PATH",
        help=(
            "Install the icon detector from this .mlpackage directory, replacing "
            "any installed copy. The package is verified by loading it before it "
            "is accepted, and lands in Choto's model directory. A local path "
            "rather than a download because nothing publishes it: upstream "
            "(microsoft/OmniParser-v2.0, icon_detect/model.pt) ships PyTorch "
            "weights, and the CoreML build made from them is what runs here."
        ),
    )

    model_actions.add_parser(
        "status",
        help="Report which models are on this machine, and where.",
        description=(
            "Names each model, says whether it is present and where it sits on "
            "disk. Exits non-zero when the semantic tier has nothing to load or "
            "the icon detector is missing, so it can be used as a check."
        ),
    )


def _add_map_commands(commands: argparse._SubParsersAction) -> None:
    graph = commands.add_parser(
        "map",
        help="Inspect or prune the remembered interface map.",
        description=(
            "Report what the interface graph holds, or drop part of it. The map "
            "is derived data — every window it forgets is re-learned by the next "
            "visit, at the cost of one parse — but the visit journal is history "
            "and no command here touches it, so a machine that has forgotten every "
            "window still knows where it has been. (The journal's own far end is "
            "trimmed by the eviction pass once an application passes its ceiling; "
            "stats reports how much and since when.) Both destructive commands "
            "back the database up first and print the path."
        ),
    )
    graph_actions = graph.add_subparsers(dest="action", required=True)

    graph_actions.add_parser(
        "stats",
        help="Print node/element/edge/sighting counts, per application.",
        description=(
            "Also reports how many nodes have been met at most once and how many "
            "of those the eviction policy would remove on the next run, and how "
            "far the visit journal reaches back — including the sightings its "
            "per-application ceiling has already trimmed."
        ),
    )

    forget = graph_actions.add_parser(
        "forget",
        help="Delete every window node of one application.",
        description=(
            "For an application that was rearranged by an update: its stored "
            "windows describe a layout that no longer exists, and re-learning "
            "them is one parse each. The name is matched in full, "
            "case-insensitively — no substrings, because this deletes."
        ),
    )
    forget.add_argument(
        "--app",
        dest="app_name",
        required=True,
        metavar="NAME",
        help='Application whose nodes to delete, e.g. --app "System Settings".',
    )
    _add_service_socket_option(forget)

    reset = graph_actions.add_parser(
        "reset",
        help="Delete every window node in the map.",
        description=(
            "Rebuilds the map from nothing on subsequent visits. The visit "
            "journal, the execution history and the runs are untouched."
        ),
    )
    _add_service_socket_option(reset)


def _add_service_socket_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        dest="service_socket_path",
        default=None,
        metavar="PATH",
        help=f"Socket path the daemon binds (default: {default_socket_path()}).",
    )


def _write_stderr(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _run_permission_check(report_to: str | None) -> int:
    backend = resolve_platform(get_settings().platform_backend).input_backend(None)
    try:
        granted = backend.ensure_permission()
    finally:
        backend.close()
    if report_to is not None:
        try:
            appbundle.write_permission_report(Path(report_to), granted=granted)
        except appbundle.BundleError as exc:
            _log.error("input_permission.report_failed", error=str(exc))
            return _EXIT_FAILURE
    _log.info(
        "input_permission.checked",
        granted=granted,
        executable=sys.executable,
        identifier=appbundle.BUNDLE_IDENTIFIER,
    )
    return _EXIT_OK if granted else _EXIT_PERMISSION_DENIED


_MODEL_MISSING_GUIDANCE = (
    "\nThe semantic tier is off: neither embedding model is in the local cache.\n"
    "Targets will resolve on exact and fuzzy text only, so a control the\n"
    "application spells differently than your plan does will be missed.\n\n"
    "  choto model install         # download them once (needs the network)\n"
    "  choto model status          # what is present, and where\n"
)

_ICON_MODEL_MISSING_GUIDANCE = (
    "\nThe icon pass is off: no icon detector is installed on this machine.\n"
    "Windows are still read, but a control drawn as a symbol and never spelled\n"
    "out stays invisible — there is nothing to detect it with, so nothing to\n"
    "name (annotate_icons) and nothing to click by name.\n\n"
    "  choto model install --icon-detector PATH/icon_detect.mlpackage\n\n"
    "PATH is a local one because no server publishes this file: upstream ships\n"
    "PyTorch weights (microsoft/OmniParser-v2.0, icon_detect/model.pt) and what\n"
    "Choto runs is the CoreML build made from them.\n"
)


def _run_model_command(args: argparse.Namespace) -> int:
    if args.action == "install":
        return _install_models(args.icon_detector)
    if args.action == "status":
        return _report_models()
    _log.error("model.unhandled_action", action=args.action)
    return _EXIT_FAILURE


def _install_models(icon_detector: str | None) -> int:
    lines: list[str] = []
    for presence in embeddings.installed_models():
        if presence.folder is not None:
            lines.append(f"{presence.model_id}: already in the cache at {presence.folder}")
            _log.info("model.install.present", model=presence.model_id)
            continue
        try:
            folder = embeddings.install_semantic_model(presence.model_id)
        except OSError as exc:
            lines.append(f"{presence.model_id}: download failed — {exc}")
            _write_stderr("\n".join(lines) + "\n")
            _log.error("model.install.failed", model=presence.model_id, error=str(exc))
            return _EXIT_FAILURE
        lines.append(f"{presence.model_id}: downloaded to {folder}")
    lines.append("the semantic tier will be available to the next daemon start.")

    exit_code, icon_lines = _install_icon_detector(icon_detector)
    _write_stderr("\n".join([*lines, *icon_lines]) + "\n")
    return exit_code


def _iconmodel() -> ModuleType:
    from choto.vision import iconmodel

    return iconmodel


def _install_icon_detector(source: str | None) -> tuple[int, list[str]]:
    iconmodel = _iconmodel()
    presence = iconmodel.installed_icon_model()
    if source is None:
        if presence.installed:
            return _EXIT_OK, [f"icon detector: already installed at {presence.path}"]
        return _EXIT_OK, ["icon detector: not installed", _ICON_MODEL_MISSING_GUIDANCE]
    try:
        path = iconmodel.install_icon_model(Path(source))
    except iconmodel.IconModelError as exc:
        _log.error("model.install.icon_detector_failed", source=source, error=str(exc))
        return _EXIT_FAILURE, [f"icon detector: install failed — {exc}"]
    return _EXIT_OK, [f"icon detector: installed to {path}"]


def _report_models() -> int:
    presences = embeddings.installed_models()
    lines = [
        f"{presence.model_id}: {presence.folder}"
        if presence.folder is not None
        else f"{presence.model_id}: not in the cache"
        for presence in presences
    ]
    semantic_ready = any(presence.folder is not None for presence in presences)
    icon = _iconmodel().installed_icon_model()
    lines.append(
        f"icon detector: {icon.path}" if icon.installed else "icon detector: not installed"
    )
    report = "\n".join(lines) + "\n"
    if not semantic_ready:
        report += _MODEL_MISSING_GUIDANCE
    if not icon.installed:
        report += _ICON_MODEL_MISSING_GUIDANCE
    _write_stderr(report)
    _log.info(
        "model.status",
        installed=sum(p.folder is not None for p in presences),
        icon_detector=icon.installed,
    )
    return _EXIT_OK if semantic_ready and icon.installed else _EXIT_FAILURE


def _report_missing_models() -> None:
    presences = embeddings.installed_models()
    semantic_ready = any(presence.folder is not None for presence in presences)
    if not semantic_ready:
        _write_stderr(_MODEL_MISSING_GUIDANCE)

    icon = _iconmodel().installed_icon_model()
    if not icon.installed:
        _write_stderr(_ICON_MODEL_MISSING_GUIDANCE)

    _log.info(
        "models.present",
        semantic=semantic_ready,
        icon_detector=icon.installed,
        loaded="on first use",
    )


def _run_service_command(args: argparse.Namespace) -> int:
    try:
        socket_path = resolve_socket_path(args.service_socket_path)
    except ValueError as exc:
        _log.error("service.socket_path_invalid", error=str(exc))
        return _EXIT_FAILURE

    try:
        manager = resolve_service_manager()
        if args.action == "install":
            if args.dry_run:
                _write_stderr(manager.render_definition(socket_path))
                _log.info("service.install.dry_run", **manager.dry_run_fields(socket_path))
                return _EXIT_OK
            _write_stderr(manager.install_guidance(manager.install(socket_path)))
        elif args.action == "uninstall":
            manager.uninstall()
        elif args.action == "status":
            _log.info("service.status", **manager.status_fields(manager.status(socket_path)))
        elif args.action == "restart":
            manager.restart()
        else:
            raise ServiceError(f"unhandled service action: {args.action!r}")
    except (ServiceError, appbundle.BundleError) as exc:
        _log.error("service.command_failed", action=args.action, error=str(exc))
        return _EXIT_FAILURE

    return _EXIT_OK


def _running_daemon(manager: ServiceManager, socket_path: Path) -> str | None:
    supervised = manager.supervisor_description()
    if supervised is not None:
        return supervised
    if manager.starts_on_demand(socket_path):
        return None
    health, _detail = manager.probe_daemon(socket_path)
    if health in (DaemonHealth.RESPONSIVE, DaemonHealth.BUSY):
        return f"a daemon is answering on {socket_path} ({health.value})"
    return None


def _daemon_refusal(running: str, command: str) -> str:
    return (
        f"\nRefusing to change the map: the Choto daemon is running ({running}).\n"
        "It writes to this same database, so a backup taken now would not be a\n"
        "consistent snapshot and a run finishing mid-command would re-create what\n"
        "was just deleted.\n\n"
        "  choto service uninstall     # stop the daemon and unload the agent\n"
        f"  {command}\n"
        "  choto service install       # start it again\n\n"
        "`choto service restart` will not do: it kicks a running daemon, it cannot\n"
        "stop one. If the daemon was started by hand (`choto --socket`), quit that\n"
        "process instead of uninstalling the agent.\n"
    )


def _map_command_line(args: argparse.Namespace) -> str:
    if args.action == "forget":
        return f'choto map forget --app "{args.app_name}"'
    return f"choto map {args.action}"


def _open_graph() -> AppContext:
    _anchor_db_path()
    _run_migrations()
    return build_context()


def _run_map_command(args: argparse.Namespace) -> int:
    destructive = args.action in ("forget", "reset")
    if destructive:
        try:
            socket_path = resolve_socket_path(args.service_socket_path)
        except ValueError as exc:
            _log.error("map.socket_path_invalid", error=str(exc))
            return _EXIT_FAILURE
        try:
            running = _running_daemon(resolve_service_manager(), socket_path)
        except ServiceError as exc:
            _log.error("map.daemon_check_failed", action=args.action, error=str(exc))
            return _EXIT_FAILURE
        if running is not None:
            _write_stderr(_daemon_refusal(running, _map_command_line(args)))
            _log.error("map.refused_daemon_running", action=args.action, detail=running)
            return _EXIT_FAILURE

    context = _open_graph()
    try:
        if args.action == "stats":
            stats = hygiene.collect_stats(context.repo, context.settings)
            _write_stderr(hygiene.render_stats(stats, context.settings))
            _log.info(
                "map.stats",
                nodes=stats.nodes,
                elements=stats.elements,
                edges=stats.edges,
                visits=stats.visits,
                orphan_visits=stats.orphan_visits,
                visits_cut=stats.visits_cut,
                evictable_visits=stats.evictable_visits,
                unconfirmed=stats.unconfirmed,
                evictable=stats.evictable_unconfirmed + stats.evictable_capped,
                glyphs=stats.glyphs,
                glyphs_labeled=stats.glyphs_labeled,
                evictable_glyphs=stats.evictable_glyphs,
                apps=len(stats.apps),
            )
            return _EXIT_OK

        if args.action == "forget":
            doomed = hygiene.nodes_of_app(context.repo, args.app_name)
            scope = f'of "{args.app_name}"'
        else:
            doomed = hygiene.all_nodes(context.repo)
            scope = "— the whole map"

        backup = hygiene.backup_database(context.settings.db_path)
        _write_stderr(f"backup: {backup}\n")
        deleted = hygiene.forget_nodes(context.repo, doomed)
        _write_stderr(
            f"forgot {deleted} node(s) {scope}; the visit journal is untouched.\n"
            "They are re-learned on the next visit, one parse each.\n"
        )
        _log.info(
            "map.forgot",
            action=args.action,
            app_name=args.app_name,
            nodes=deleted,
            backup=str(backup),
        )
    except hygiene.MapError as exc:
        _write_stderr(f"{exc}\n")
        _log.error("map.command_failed", action=args.action, error=str(exc))
        return _EXIT_FAILURE
    return _EXIT_OK


def _activated_descriptor(name: str | None) -> int | None:
    if name is None:
        return None
    try:
        return launchdsocket.activate_socket(name)
    except launchdsocket.SocketActivationError as exc:
        _log.warning(
            "socket.activation_unavailable",
            launchd_socket=name,
            error=str(exc),
            remedy="choto service install (rewrites the agent's Sockets entry)",
            fallback="binding the socket path directly",
        )
        return None


def _run_startup_eviction(context: AppContext) -> None:
    try:
        evict(context.repo, context.settings)
    except SQLAlchemyError as exc:
        _log.error("map.eviction_failed", error=str(exc))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()

    if args.command == "service":
        return _run_service_command(args)

    if args.command == "map":
        return _run_map_command(args)

    if args.command == "model":
        return _run_model_command(args)

    if args.check_input_permission:
        return _run_permission_check(args.report_to)

    socket_path: Path | None = None
    if args.socket_path is not None:
        try:
            socket_path = resolve_socket_path(args.socket_path)
        except ValueError as exc:
            _log.error("socket.path_invalid", error=str(exc))
            return _EXIT_FAILURE

    _anchor_db_path()
    _run_migrations()
    _report_missing_models()

    if socket_path is None:
        context = build_context()
        try:
            _run_startup_eviction(context)
            mcp = create_mcp(context)
            _log.info("mcp.starting", transport="stdio")
            mcp.run()
        finally:
            context.close()
        return _EXIT_OK

    context = build_context()
    try:
        _run_startup_eviction(context)
        mcp = create_mcp(context)
        activated = _activated_descriptor(args.launchd_socket)
        _log.info(
            "mcp.starting",
            transport="unix-socket",
            socket=str(socket_path),
            max_sessions=context.settings.max_mcp_sessions,
            launchd_socket=args.launchd_socket,
            activated=activated is not None,
        )
        try:
            if activated is None:
                serve_socket(mcp, socket_path, context.settings.max_mcp_sessions)
            else:
                serve_activated_socket(
                    mcp,
                    activated,
                    socket_path,
                    context.settings.max_mcp_sessions,
                    context.settings.idle_exit_seconds,
                )
        except SocketUnavailableError as exc:
            _log.error("socket.unavailable", socket=str(socket_path), error=str(exc))
            return _EXIT_FAILURE
        except OSError as exc:
            _log.error("socket.bind_failed", socket=str(socket_path), error=str(exc))
            return _EXIT_FAILURE
    finally:
        context.close()

    _log.info("mcp.stopped", transport="unix-socket")
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
