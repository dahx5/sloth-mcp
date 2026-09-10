from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from bench.bench import (
    BYTES_PER_TEXT_TOKEN,
    IMAGE_PIXELS_PER_TOKEN,
    STATUS_SUCCESS,
    BenchError,
    ToolResult,
    estimate_image_tokens,
    estimate_text_tokens,
    failure_detail,
    plan_status,
)
from bench.taskdefs import Task
from bench.taskops import (
    CheckResult,
    MacSystemBridge,
    OpError,
    PreconditionError,
    SystemBridge,
    TaskContext,
    sandbox_root,
)
from choto.log import get_logger

_log = get_logger(__name__)

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_PLANS_DIR = BENCH_DIR / "plans"
DEFAULT_REPORTS_DIR = BENCH_DIR / "reports"

EXECUTE_PLAN_TOOL = "execute_plan"

_COMPLETED_PREFIX = "completed_steps: "

_EXECUTED = frozenset({"passed", "failed_check", "escalated", "aborted"})


class Verdict(str, Enum):
    PASSED = "passed"
    FAILED_CHECK = "failed_check"
    ESCALATED = "escalated"
    ABORTED = "aborted"
    ERROR = "error"
    DRY_RUN = "dry_run"
    NO_PLAN = "no_plan"


class PlanMissing(RuntimeError): ...


class PlanInvalid(RuntimeError): ...


class ToolClient(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


def load_plan(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PlanMissing(f"no plan at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanInvalid(f"{path}: {exc}") from exc

    steps = payload.get("steps") if isinstance(payload, dict) else payload
    if not isinstance(steps, list) or not steps:
        raise PlanInvalid(f"{path}: expected a non-empty list of steps, or {{'steps': [...]}}")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise PlanInvalid(f"{path}: step {index} is not an object")
    return steps


def plan_path_for(task_id: str, plans_dir: Path = DEFAULT_PLANS_DIR) -> Path:
    return plans_dir / f"{task_id}.json"


@dataclass
class TaskOutcome:
    task_id: str
    tier: str
    tags: tuple[str, ...]
    verdict: Verdict
    llm_calls: int = 0
    plan_steps: int = 0
    completed_steps: int = 0
    execution_ms: float = 0.0
    text_tokens: int = 0
    image_tokens: int = 0
    check_passed: bool | None = None
    check_details: list[str] = field(default_factory=list)
    failure: str | None = None
    report_dir: str | None = None
    teardown_errors: list[str] = field(default_factory=list)
    residue: list[str] = field(default_factory=list)
    fatal: bool = False

    @property
    def total_tokens(self) -> int:
        return self.text_tokens + self.image_tokens

    @property
    def clean(self) -> bool:
        return not self.teardown_errors and not self.residue

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "tier": self.tier,
            "tags": list(self.tags),
            "verdict": self.verdict.value,
            "llm_calls": self.llm_calls,
            "plan_steps": self.plan_steps,
            "completed_steps": self.completed_steps,
            "execution_ms": round(self.execution_ms, 1),
            "text_tokens": self.text_tokens,
            "image_tokens": self.image_tokens,
            "total_tokens": self.total_tokens,
            "check_passed": self.check_passed,
            "check_details": self.check_details,
            "failure": self.failure,
            "report_dir": self.report_dir,
            "teardown_errors": self.teardown_errors,
            "residue": self.residue,
        }


def completed_step_count(reply: str) -> int:
    for line in reply.splitlines():
        if line.startswith(_COMPLETED_PREFIX):
            raw = line[len(_COMPLETED_PREFIX) :].strip()
            try:
                completed = json.loads(raw)
            except json.JSONDecodeError:
                return 0
            return len(completed) if isinstance(completed, list) else 0
    return 0


def classify(status: str, check_passed: bool | None) -> Verdict:
    if status == STATUS_SUCCESS:
        if check_passed is None:
            return Verdict.ERROR
        return Verdict.PASSED if check_passed else Verdict.FAILED_CHECK
    if status == "escalated":
        return Verdict.ESCALATED
    if status == "aborted":
        return Verdict.ABORTED
    return Verdict.ERROR


def run_task(
    task: Task,
    *,
    plan_steps: list[dict[str, Any]] | None,
    client: ToolClient | None,
    system: SystemBridge | None = None,
    sandbox: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> TaskOutcome:
    ctx = TaskContext(
        sandbox=sandbox if sandbox is not None else sandbox_root(),
        system=system if system is not None else MacSystemBridge(),
    )
    outcome = TaskOutcome(
        task_id=task.id,
        tier=task.tier,
        tags=task.tags,
        verdict=Verdict.ERROR,
        plan_steps=len(plan_steps) if plan_steps else 0,
    )

    try:
        for op in task.setup:
            op.apply(ctx)

        if plan_steps is None:
            outcome.verdict = Verdict.DRY_RUN
            outcome.check_passed, outcome.check_details = _evaluate(task, ctx)
        else:
            if client is None:
                raise OpError("a task run needs a daemon connection")
            outcome.llm_calls = 1
            status, reply = _execute(task, plan_steps, client, outcome, reports_dir)
            outcome.check_passed, outcome.check_details = _evaluate(task, ctx)
            outcome.verdict = classify(status, outcome.check_passed)
            if outcome.verdict in (Verdict.ESCALATED, Verdict.ABORTED, Verdict.ERROR):
                outcome.failure = failure_detail(reply)
            elif outcome.verdict is Verdict.FAILED_CHECK:
                outcome.failure = "plan reported success, the independent check disagreed"
    except PreconditionError as exc:
        outcome.verdict = Verdict.ERROR
        outcome.failure = f"precondition not met: {exc}"
        _log.warning("bench.task.precondition", task=task.id, error=str(exc))
    except BenchError as exc:
        outcome.verdict = Verdict.ERROR
        outcome.failure = f"daemon: {exc}"
        outcome.fatal = True
        _log.error("bench.task.transport", task=task.id, error=str(exc))
    except (OpError, OSError) as exc:
        outcome.verdict = Verdict.ERROR
        outcome.failure = f"{type(exc).__name__}: {exc}"
        _log.error("bench.task.failed", task=task.id, error=str(exc))
    finally:
        _teardown(task, ctx, outcome)

    return outcome


def _execute(
    task: Task,
    plan_steps: list[dict[str, Any]],
    client: ToolClient,
    outcome: TaskOutcome,
    reports_dir: Path,
) -> tuple[str, str]:
    _log.info("bench.task.execute", task=task.id, steps=len(plan_steps))
    result = client.call_tool(EXECUTE_PLAN_TOOL, {"steps": plan_steps})
    outcome.execution_ms = result.latency_ms
    outcome.text_tokens = estimate_text_tokens(result.text)
    outcome.image_tokens = sum(estimate_image_tokens(w, h) for w, h in result.images)
    outcome.completed_steps = completed_step_count(result.text)

    status = plan_status(result.text)
    if status != STATUS_SUCCESS:
        outcome.report_dir = str(_write_report(task, result, reports_dir))
    return status, result.text


def _write_report(task: Task, result: ToolResult, reports_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = reports_dir / task.id / stamp
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reply.txt").write_text(result.text, encoding="utf-8")
    (directory / "instruction.txt").write_text(task.instruction + "\n", encoding="utf-8")
    for index, payload in enumerate(result.png_payloads, start=1):
        (directory / f"screen-{index}.png").write_bytes(payload)
    _log.info("bench.task.report", task=task.id, path=str(directory))
    return directory


def _evaluate(task: Task, ctx: TaskContext) -> tuple[bool, list[str]]:
    details: list[str] = []
    passed = True
    for check in task.check:
        result: CheckResult = check.evaluate(ctx)
        details.append(f"{'ok' if result.passed else 'NO'}: {result.detail}")
        passed = passed and result.passed
    return passed, details


def _teardown(task: Task, ctx: TaskContext, outcome: TaskOutcome) -> None:
    for op in task.teardown:
        try:
            op.revert(ctx)
        except (OpError, OSError) as exc:
            message = f"{op.describe()}: {exc}"
            outcome.teardown_errors.append(message)
            _log.error("bench.task.teardown_failed", task=task.id, error=message)
    for op in task.teardown:
        try:
            outcome.residue.extend(op.residue(ctx))
        except (OpError, OSError) as exc:
            outcome.residue.append(f"{op.describe()}: could not verify cleanup ({exc})")


def print_outcome(outcome: TaskOutcome) -> None:
    head = f"  {outcome.task_id:<26}{outcome.verdict.value:<13}"
    if outcome.plan_steps:
        steps = f"{outcome.completed_steps}/{outcome.plan_steps}"
        print(
            f"{head}steps {steps:>7}  {outcome.execution_ms / 1000:>6.1f}s  "
            f"tokens {outcome.total_tokens:>7,}"
        )
    else:
        print(head.rstrip())
    for detail in outcome.check_details:
        print(f"      check {detail}")
    if outcome.failure:
        print(f"      -> {outcome.failure}")
    if outcome.report_dir:
        print(f"      report: {outcome.report_dir}")
    for message in outcome.teardown_errors:
        print(f"      TEARDOWN FAILED: {message}")
    for message in outcome.residue:
        print(f"      LEFT BEHIND: {message}")


def summarize(outcomes: Sequence[TaskOutcome]) -> dict[str, Any]:
    counts: dict[str, int] = {verdict.value: 0 for verdict in Verdict}
    for outcome in outcomes:
        counts[outcome.verdict.value] += 1
    executed = sum(counts[name] for name in _EXECUTED)
    passed = counts[Verdict.PASSED.value]
    false_success = counts[Verdict.FAILED_CHECK.value]
    return {
        "tasks": len(outcomes),
        "executed": executed,
        "verdicts": counts,
        "first_plan_pass_rate": (passed / executed) if executed else None,
        "false_success_rate": (false_success / executed) if executed else None,
        "total_tokens": sum(outcome.total_tokens for outcome in outcomes),
        "text_tokens": sum(outcome.text_tokens for outcome in outcomes),
        "image_tokens": sum(outcome.image_tokens for outcome in outcomes),
        "total_execution_ms": round(sum(outcome.execution_ms for outcome in outcomes), 1),
        "llm_calls": sum(outcome.llm_calls for outcome in outcomes),
        "unclean": [outcome.task_id for outcome in outcomes if not outcome.clean],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("summary")
    verdicts = summary["verdicts"]
    reported = ", ".join(f"{name} {count}" for name, count in verdicts.items() if count)
    print(f"  tasks {summary['tasks']} ({reported or 'none'})")
    if summary["executed"]:
        print(
            f"  passed on the first plan: {summary['first_plan_pass_rate']:.0%} "
            f"of {summary['executed']} executed"
        )
        print(
            f"  false successes (plan said success, check disagreed): "
            f"{summary['false_success_rate']:.0%}"
        )
    print(
        f"  tokens {summary['total_tokens']:,} "
        f"(text {summary['text_tokens']:,} + image {summary['image_tokens']:,})   "
        f"plan wall {summary['total_execution_ms'] / 1000:.1f}s"
    )
    if summary["unclean"]:
        print(f"  NOT CLEAN after teardown: {', '.join(summary['unclean'])}")
    print()
    print("notes")
    print(
        f"  tokens are estimates: text = ceil(utf-8 bytes / {BYTES_PER_TEXT_TOKEN}), "
        f"image = ceil(w*h / {IMAGE_PIXELS_PER_TOKEN})"
    )
    print("  llm_calls is 1 per task here: a first attempt at a plan written in advance")
    print("  an escalated task is not retried — correcting its plan is a separate pass")


def write_json_report(path: Path, outcomes: Sequence[TaskOutcome], summary: dict[str, Any]) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sandbox": str(sandbox_root()),
        "token_model": {
            "bytes_per_text_token": BYTES_PER_TEXT_TOKEN,
            "image_pixels_per_token": IMAGE_PIXELS_PER_TOKEN,
        },
        "tasks": [outcome.as_dict() for outcome in outcomes],
        "summary": summary,
    }
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise OpError(f"cannot write results to {path}: {exc}") from exc


def wait_between_tasks(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
