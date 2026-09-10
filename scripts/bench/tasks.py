from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from bench.bench import (  # noqa: E402 - the path bootstrap above must run first
    DEFAULT_CALL_TIMEOUT_S,
    BenchError,
    SocketMcpClient,
)
from bench.taskdefs import TASKS, Task, get_task, tasks_by_id  # noqa: E402
from bench.taskops import OpError, sandbox_root  # noqa: E402
from bench.taskrun import (  # noqa: E402
    DEFAULT_PLANS_DIR,
    DEFAULT_REPORTS_DIR,
    PlanInvalid,
    PlanMissing,
    TaskOutcome,
    Verdict,
    load_plan,
    plan_path_for,
    print_outcome,
    print_summary,
    run_task,
    summarize,
    wait_between_tasks,
    write_json_report,
)
from choto.log import get_logger, setup_logging  # noqa: E402
from choto.mcpserver.socket_path import default_socket_path, resolve_socket_path  # noqa: E402
from choto.recall import DEFAULT_DEPTH  # noqa: E402

_log = get_logger(__name__)

RECALL_TOOL = "recall"

SETTLE_BETWEEN_TASKS_S = 2.0

EXIT_OK = 0
EXIT_FAILED = 1


def command_list(args: argparse.Namespace) -> int:
    plans_dir = Path(args.plans_dir).expanduser()
    print(f"{len(TASKS)} tasks   sandbox: {sandbox_root()}   plans: {plans_dir}")
    print()
    current_tier = ""
    for task in TASKS:
        if task.tier != current_tier:
            current_tier = task.tier
            print(f"[{current_tier}]")
        has_plan = "plan" if plan_path_for(task.id, plans_dir).is_file() else "no plan"
        print(f"  {task.id:<24} {has_plan:<8} {','.join(task.tags)}")
        print(f"      {task.instruction}")
        if args.verbose:
            print(f"      checks: {'; '.join(check.describe() for check in task.check)}")
            print(f"      why:    {task.rationale}")
    return EXIT_OK


def command_plan_template(args: argparse.Namespace) -> int:
    task = _resolve_task(args.task_id)
    if task is None:
        return EXIT_FAILED

    print(f"task:        {task.id}   [{task.tier}]   tags: {','.join(task.tags)}")
    print(f"instruction: {task.instruction}")
    print(f"sandbox:     {sandbox_root()}")
    print(f"apps:        {', '.join(task.apps)}")
    print()
    print("prepared before the plan runs:")
    for op in task.setup:
        print(f"  - {op.describe()}")
    print("verified after it (independently of Choto):")
    for check in task.check:
        print(f"  - {check.describe()}")
    print("cleaned up afterwards:")
    for op in task.teardown:
        print(f"  - {op.describe()}")
    print()
    print(f"why this task exists: {task.rationale}")
    print()
    print(f"write the plan to: {plan_path_for(task.id, Path(args.plans_dir).expanduser())}")
    print('shape: {"steps": [ {"action": "focus_app", "app_name": "..."} , ... ]}')
    print()
    _print_recall(task, args)
    return EXIT_OK


def _print_recall(task: Task, args: argparse.Namespace) -> None:
    try:
        socket_path = resolve_socket_path(args.socket_path)
    except ValueError as exc:
        print(f"recall unavailable: {exc}")
        return
    try:
        with SocketMcpClient(socket_path, args.timeout) as client:
            for app in task.apps:
                print(f"--- recall: {app} ---")
                reply = client.call_tool(RECALL_TOOL, {"app_name": app, "depth": args.depth})
                print(reply.text or "(nothing remembered)")
                print()
    except BenchError as exc:
        print(f"--- recall unavailable ({exc}) ---")
        print("Plan from the instruction alone, or start the daemon: uv run choto --socket")


def command_run(args: argparse.Namespace) -> int:
    task = _resolve_task(args.task_id)
    if task is None:
        return EXIT_FAILED

    plans_dir = Path(args.plans_dir).expanduser()
    plan_file = Path(args.plan).expanduser() if args.plan else plan_path_for(task.id, plans_dir)
    return _run_batch([task], plan_file_for=lambda _: plan_file, args=args)


def command_run_all(args: argparse.Namespace) -> int:
    plans_dir = Path(args.plans_dir).expanduser()
    selected = [task for task in TASKS if not args.tag or args.tag in task.tags]
    if not selected:
        print(f"no task carries the tag {args.tag!r}")
        return EXIT_FAILED
    return _run_batch(selected, plan_file_for=lambda t: plan_path_for(t.id, plans_dir), args=args)


def _run_batch(tasks: list[Task], *, plan_file_for: Any, args: argparse.Namespace) -> int:
    dry_run = args.dry_run
    print(f"choto task stand — {len(tasks)} task(s)" + ("  [dry run]" if dry_run else ""))
    print(f"sandbox: {sandbox_root()}")

    plans: dict[str, list[dict[str, Any]] | None] = {}
    if not dry_run:
        for task in tasks:
            path = plan_file_for(task)
            try:
                plans[task.id] = load_plan(path)
            except PlanMissing:
                plans[task.id] = None
            except PlanInvalid as exc:
                print(f"  {task.id}: unusable plan — {exc}")
                return EXIT_FAILED
        if not any(steps for steps in plans.values()):
            for task in tasks:
                print(f"  {task.id}: no plan at {plan_file_for(task)}")
            print("write a plan per task first (see plan-template), or pass --dry-run")
            return EXIT_FAILED

    client = None
    outcomes: list[TaskOutcome] = []
    try:
        if not dry_run:
            socket_path = resolve_socket_path(args.socket_path)
            print(f"socket:  {socket_path}")
            client = SocketMcpClient(socket_path, args.timeout)
            client.connect()
        print()
        for index, task in enumerate(tasks):
            steps = None if dry_run else plans[task.id]
            if not dry_run and steps is None:
                outcome = TaskOutcome(
                    task_id=task.id,
                    tier=task.tier,
                    tags=task.tags,
                    verdict=Verdict.NO_PLAN,
                    failure=f"no plan at {plan_file_for(task)}",
                )
                outcomes.append(outcome)
                print_outcome(outcome)
                continue
            if index and not dry_run:
                wait_between_tasks(SETTLE_BETWEEN_TASKS_S)
            outcome = run_task(
                task,
                plan_steps=steps,
                client=client,
                reports_dir=Path(args.reports_dir).expanduser(),
            )
            outcomes.append(outcome)
            print_outcome(outcome)
            if outcome.fatal:
                print("  stopping: the daemon connection is gone")
                break
    except BenchError as exc:
        _log.error("bench.tasks.daemon", error=str(exc))
        print(f"cannot talk to the daemon: {exc}")
        return EXIT_FAILED
    except ValueError as exc:
        print(f"bad socket path: {exc}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\ninterrupted; the current task's teardown has already run")
        return EXIT_FAILED
    finally:
        if client is not None:
            client.close()

    summary = summarize(outcomes)
    print_summary(summary)
    if args.json_path:
        try:
            write_json_report(Path(args.json_path).expanduser(), outcomes, summary)
        except OpError as exc:
            print(f"could not write the JSON report: {exc}")
            return EXIT_FAILED
        print(f"\nfull results written to {args.json_path}")

    return EXIT_OK if _batch_is_green(outcomes, dry_run=dry_run) else EXIT_FAILED


def _batch_is_green(outcomes: list[TaskOutcome], *, dry_run: bool) -> bool:
    if not outcomes:
        return False
    if dry_run:
        return all(outcome.verdict is Verdict.DRY_RUN and outcome.clean for outcome in outcomes)
    return all(outcome.verdict is Verdict.PASSED and outcome.clean for outcome in outcomes)


def _resolve_task(task_id: str) -> Task | None:
    try:
        return get_task(task_id)
    except KeyError:
        print(f"unknown task {task_id!r}. Known tasks:")
        for known in sorted(tasks_by_id()):
            print(f"  {known}")
        return None


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _add_daemon_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        dest="socket_path",
        metavar="PATH",
        default=None,
        help=f"Path of the daemon's unix socket (default: {default_socket_path()}).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_CALL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Per-call timeout (default: {DEFAULT_CALL_TIMEOUT_S:.0f}).",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plans",
        dest="plans_dir",
        metavar="DIR",
        default=str(DEFAULT_PLANS_DIR),
        help=f"Directory holding <task-id>.json plans (default: {DEFAULT_PLANS_DIR}).",
    )
    parser.add_argument(
        "--reports",
        dest="reports_dir",
        metavar="DIR",
        default=str(DEFAULT_REPORTS_DIR),
        help=f"Where escalation reports are written (default: {DEFAULT_REPORTS_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise setup, checks and teardown without running any plan.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        default=None,
        help="Also write every number to this file as JSON.",
    )
    _add_daemon_arguments(parser)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tasks.py",
        description="Choto's own task stand: real tasks, independent checks, measured cost.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="Show the catalog.")
    listing.add_argument(
        "--plans",
        dest="plans_dir",
        metavar="DIR",
        default=str(DEFAULT_PLANS_DIR),
        help="Directory to look for plans in.",
    )
    listing.add_argument(
        "-v", "--verbose", action="store_true", help="Also show checks and rationale."
    )
    listing.set_defaults(func=command_list)

    template = subparsers.add_parser(
        "plan-template", help="Print a task's brief, plus what recall remembers."
    )
    template.add_argument("task_id")
    template.add_argument(
        "--plans",
        dest="plans_dir",
        metavar="DIR",
        default=str(DEFAULT_PLANS_DIR),
        help="Directory the plan should be written to.",
    )
    template.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"How many levels of remembered transitions to unfold (default: {DEFAULT_DEPTH}).",
    )
    _add_daemon_arguments(template)
    template.set_defaults(func=command_plan_template)

    single = subparsers.add_parser("run", help="Run one task.")
    single.add_argument("task_id")
    single.add_argument("--plan", metavar="PATH", default=None, help="Plan file for this task.")
    _add_run_arguments(single)
    single.set_defaults(func=command_run)

    batch = subparsers.add_parser("run-all", help="Run the whole catalog.")
    batch.add_argument("--tag", default=None, help="Only tasks carrying this tag.")
    _add_run_arguments(batch)
    batch.set_defaults(func=command_run_all)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
