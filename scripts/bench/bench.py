from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import socket
import statistics
import struct
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.types import LATEST_PROTOCOL_VERSION

from choto.log import get_logger, setup_logging
from choto.mcpserver.socket_path import default_socket_path, resolve_socket_path

_log = get_logger(__name__)


CLIENT_NAME = "choto-bench"
CLIENT_VERSION = "1.0"

_RECV_CHUNK_BYTES = 65536

DEFAULT_CALL_TIMEOUT_S = 300.0

REQUIRED_TOOLS = frozenset({"observe", "execute_plan"})

_HANGUP_HINT = (
    "It serves one client at a time — quit Claude Desktop (or any other attached client) and retry."
)


BYTES_PER_TEXT_TOKEN = 4

IMAGE_PIXELS_PER_TOKEN = 750

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


TEXTEDIT_APP = "TextEdit"

MARKER_PREFIX = "Choto benchmark run"

PANGRAM = "the quick brown fox jumps over the lazy dog"

MARKER_FRAGMENT = "jumps over the lazy dog"

ARM_BATCHED = "batched"
ARM_PER_STEP = "per-step"
ARMS = (ARM_BATCHED, ARM_PER_STEP)

STATUS_SUCCESS = "success"

_SAVE_SHEET_MARKERS = (
    "do you want to save",
    "don't save",
    "dont save",
    "do you want to save",
    "don't save",
)

LAUNCH_TIMEOUT_MS = 15_000

MAX_CLEANUP_ROUNDS = 3

DEFAULT_RUNS = 3


class BenchError(RuntimeError): ...


def png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise ValueError("not a PNG image: signature or IHDR header missing")
    width, height = struct.unpack(">II", data[16:24])
    if width == 0 or height == 0:
        raise ValueError(f"PNG declares an empty image: {width}x{height}")
    return int(width), int(height)


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TEXT_TOKEN)


def estimate_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    return math.ceil(width * height / IMAGE_PIXELS_PER_TOKEN)


@dataclass(frozen=True)
class ToolResult:
    text: str
    images: list[tuple[int, int]]
    wire_bytes: int
    latency_ms: float
    is_error: bool
    png_payloads: tuple[bytes, ...] = ()


def parse_tool_blocks(result: dict[str, Any]) -> tuple[str, list[bytes]]:
    content = result.get("content")
    if not isinstance(content, list):
        raise BenchError(f"tools/call reply has no content list: {result!r}")

    texts: list[str] = []
    payloads: list[bytes] = []
    for block in content:
        if not isinstance(block, dict):
            raise BenchError(f"content block is not an object: {block!r}")
        kind = block.get("type")
        if kind == "text":
            value = block.get("text")
            if not isinstance(value, str):
                raise BenchError(f"text block carries no string text: {block!r}")
            texts.append(value)
        elif kind == "image":
            mime = block.get("mimeType")
            if mime != "image/png":
                raise BenchError(f"unexpected image mime type {mime!r}; only image/png is priced")
            data = block.get("data")
            if not isinstance(data, str):
                raise BenchError(f"image block carries no base64 data: {block.get('type')!r}")
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise BenchError(f"image block is not valid base64: {exc}") from exc
            try:
                png_dimensions(raw)
            except ValueError as exc:
                raise BenchError(f"image block is not a usable PNG: {exc}") from exc
            payloads.append(raw)
        else:
            raise BenchError(f"unsupported content block type {kind!r}")

    return "\n".join(texts), payloads


class SocketMcpClient:
    def __init__(self, path: Path, timeout_s: float = DEFAULT_CALL_TIMEOUT_S) -> None:
        self._path = path
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._buffer = b""
        self._next_id = 0
        self.server_info: dict[str, Any] = {}

    def __enter__(self) -> SocketMcpClient:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout_s)
        try:
            sock.connect(str(self._path))
        except OSError as exc:
            sock.close()
            raise BenchError(
                f"cannot reach the Choto daemon at {self._path}: {exc}. Start it in a "
                f"terminal that holds Accessibility permission: uv run choto --socket"
            ) from exc
        self._sock = sock
        self._handshake()

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError as exc:
            _log.warning("bench.socket.close_failed", error=str(exc))
        finally:
            self._sock = None

    def _handshake(self) -> None:
        result, _ = self._request(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        server_info = result.get("serverInfo")
        self.server_info = server_info if isinstance(server_info, dict) else {}
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        _log.info(
            "bench.connected",
            socket=str(self._path),
            server=self.server_info.get("name", "unknown"),
            protocol=result.get("protocolVersion", "unknown"),
        )

    def list_tools(self) -> set[str]:
        result, _ = self._request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise BenchError(f"tools/list reply has no tools list: {result!r}")
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise BenchError(f"tools/list entry is malformed: {tool!r}")
            names.add(tool["name"])
        return names

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        result, wire_bytes = self._request("tools/call", {"name": name, "arguments": arguments})
        latency_ms = (time.perf_counter() - started) * 1000.0
        text, payloads = parse_tool_blocks(result)
        return ToolResult(
            text=text,
            images=[png_dimensions(payload) for payload in payloads],
            wire_bytes=wire_bytes,
            latency_ms=latency_ms,
            is_error=bool(result.get("isError", False)),
            png_payloads=tuple(payloads),
        )

    def _request(self, method: str, params: dict[str, Any]) -> tuple[dict[str, Any], int]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        while True:
            message, size = self._read_message()
            if message.get("id") == request_id and ("result" in message or "error" in message):
                error = message.get("error")
                if error is not None:
                    raise BenchError(f"{method} failed: {error}")
                result = message.get("result")
                if not isinstance(result, dict):
                    raise BenchError(f"{method} returned a non-object result: {result!r}")
                return result, size
            if "method" in message and message.get("id") is not None:
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": f"{CLIENT_NAME} implements no client-side requests",
                        },
                    }
                )
                continue
            _log.debug("bench.message.ignored", method=message.get("method"), id=message.get("id"))

    def _send(self, message: dict[str, Any]) -> None:
        if self._sock is None:
            raise BenchError("not connected to the daemon")
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n"
        label = message.get("method", "response")
        try:
            self._sock.sendall(payload)
        except ConnectionError as exc:
            raise BenchError(
                f"the daemon closed the connection while sending {label}. {_HANGUP_HINT}"
            ) from exc
        except OSError as exc:
            raise BenchError(f"sending {label} failed: {exc}") from exc

    def _read_message(self) -> tuple[dict[str, Any], int]:
        if self._sock is None:
            raise BenchError("not connected to the daemon")

        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(_RECV_CHUNK_BYTES)
            except TimeoutError as exc:
                raise BenchError(
                    f"the daemon did not answer within {self._timeout_s:.0f}s; "
                    f"check its log (it may be waiting on a UI that never settled)"
                ) from exc
            except OSError as exc:
                raise BenchError(f"reading from the daemon failed: {exc}") from exc
            if not chunk:
                raise BenchError(f"the daemon closed the connection. {_HANGUP_HINT}")
            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchError(f"the daemon sent a malformed JSON-RPC frame: {exc}") from exc
        if not isinstance(message, dict):
            raise BenchError(f"the daemon sent a non-object JSON-RPC frame: {message!r}")
        return message, len(line) + 1


def build_steps(run_index: int) -> list[dict[str, Any]]:
    message = f"{MARKER_PREFIX} {run_index}: {PANGRAM}"
    return [
        {
            "action": "focus_app",
            "app_name": TEXTEDIT_APP,
            "expect": {"appears": TEXTEDIT_APP},
            "timeout_ms": LAUNCH_TIMEOUT_MS,
        },
        {
            "action": "hotkey",
            "keys": ["cmd", "n"],
            "expect": {"screen_changes": True},
            "timeout_ms": LAUNCH_TIMEOUT_MS,
        },
        {"action": "type", "text": message, "expect": {"appears": MARKER_FRAGMENT}},
        {"action": "hotkey", "keys": ["cmd", "a"]},
        {"action": "hotkey", "keys": ["delete"], "expect": {"disappears": MARKER_FRAGMENT}},
        {"action": "hotkey", "keys": ["cmd", "w"], "expect": {"screen_changes": True}},
        {"action": "hotkey", "keys": ["cmd", "delete"]},
    ]


def plan_status(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("status: "):
            return line[len("status: ") :].strip()
    if text.startswith("Invalid plan"):
        return "invalid"
    return "unknown"


def observed_app(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("app: "):
            return line[len("app: ") :].strip()
    return None


def failure_detail(text: str) -> str:
    lines = text.splitlines()
    prefix = ""
    reason = ""
    for line in lines:
        if line.startswith("failed_step: "):
            prefix = f"step {line[len('failed_step: ') :].strip()}: "
        elif line.startswith("reason: ") and not reason:
            reason = line[len("reason: ") :].strip()

    if reason:
        return prefix + reason
    journey = [line for line in lines if line.startswith("step ")]
    if journey:
        return prefix + journey[-1].strip()
    return prefix + (lines[0].strip() if lines else "")


@dataclass(frozen=True)
class CallMetrics:
    tool: str
    status: str
    latency_ms: float
    text_chars: int
    text_tokens: int
    image_tokens: int
    images: int
    wire_bytes: int

    @property
    def total_tokens(self) -> int:
        return self.text_tokens + self.image_tokens


@dataclass
class ArmResult:
    arm: str
    run_index: int
    calls: list[CallMetrics] = field(default_factory=list)
    ok: bool = True
    failure: str | None = None

    @property
    def roundtrips(self) -> int:
        return len(self.calls)

    @property
    def latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)

    @property
    def text_tokens(self) -> int:
        return sum(call.text_tokens for call in self.calls)

    @property
    def image_tokens(self) -> int:
        return sum(call.image_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.text_tokens + self.image_tokens

    @property
    def images(self) -> int:
        return sum(call.images for call in self.calls)

    @property
    def wire_bytes(self) -> int:
        return sum(call.wire_bytes for call in self.calls)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "run": self.run_index,
            "ok": self.ok,
            "failure": self.failure,
            "roundtrips": self.roundtrips,
            "images": self.images,
            "text_tokens": self.text_tokens,
            "image_tokens": self.image_tokens,
            "total_tokens": self.total_tokens,
            "wire_bytes": self.wire_bytes,
            "latency_ms": round(self.latency_ms, 1),
            "calls": [
                {
                    "tool": call.tool,
                    "status": call.status,
                    "latency_ms": round(call.latency_ms, 1),
                    "text_chars": call.text_chars,
                    "text_tokens": call.text_tokens,
                    "image_tokens": call.image_tokens,
                    "images": call.images,
                    "wire_bytes": call.wire_bytes,
                }
                for call in self.calls
            ],
        }


def measure(
    client: SocketMcpClient, tool: str, arguments: dict[str, Any]
) -> tuple[CallMetrics, ToolResult]:
    result = client.call_tool(tool, arguments)
    image_tokens = sum(estimate_image_tokens(w, h) for w, h in result.images)
    if tool == "execute_plan":
        status = plan_status(result.text)
    else:
        status = "error" if result.is_error else "ok"
    metrics = CallMetrics(
        tool=tool,
        status=status,
        latency_ms=result.latency_ms,
        text_chars=len(result.text),
        text_tokens=estimate_text_tokens(result.text),
        image_tokens=image_tokens,
        images=len(result.images),
        wire_bytes=result.wire_bytes,
    )
    return metrics, result


def run_batched_arm(client: SocketMcpClient, run_index: int) -> ArmResult:
    arm = ArmResult(arm=ARM_BATCHED, run_index=run_index)

    observation, _ = measure(client, "observe", {})
    arm.calls.append(observation)

    execution, result = measure(client, "execute_plan", {"steps": build_steps(run_index)})
    arm.calls.append(execution)
    if execution.status != STATUS_SUCCESS:
        arm.ok = False
        arm.failure = f"execute_plan status={execution.status}: {failure_detail(result.text)}"
    return arm


def run_per_step_arm(client: SocketMcpClient, run_index: int) -> ArmResult:
    arm = ArmResult(arm=ARM_PER_STEP, run_index=run_index)

    for position, step in enumerate(build_steps(run_index), start=1):
        observation, _ = measure(client, "observe", {})
        arm.calls.append(observation)

        execution, result = measure(client, "execute_plan", {"steps": [step]})
        arm.calls.append(execution)
        if execution.status != STATUS_SUCCESS:
            arm.ok = False
            arm.failure = (
                f"step {position} ({step['action']}) status={execution.status}: "
                f"{failure_detail(result.text)}"
            )
            break
    return arm


def cleanup(client: SocketMcpClient) -> None:
    for round_index in range(1, MAX_CLEANUP_ROUNDS + 1):
        text = client.call_tool("observe", {}).text
        if observed_app(text) != TEXTEDIT_APP:
            return

        lowered = text.lower()
        if any(marker in lowered for marker in _SAVE_SHEET_MARKERS):
            _log.info("bench.cleanup.discard_sheet", round=round_index)
            client.call_tool(
                "execute_plan",
                {"steps": [{"action": "hotkey", "keys": ["cmd", "delete"]}]},
            )
            continue

        if MARKER_PREFIX in text:
            _log.info("bench.cleanup.close_document", round=round_index)
            client.call_tool(
                "execute_plan",
                {
                    "steps": [
                        {"action": "hotkey", "keys": ["cmd", "a"]},
                        {"action": "hotkey", "keys": ["delete"]},
                        {"action": "hotkey", "keys": ["cmd", "w"]},
                    ]
                },
            )
            continue

        return

    _log.warning(
        "bench.cleanup.incomplete",
        rounds=MAX_CLEANUP_ROUNDS,
        hint="TextEdit may still show a benchmark document or a save sheet",
    )


def _cleanup_quietly(client: SocketMcpClient) -> None:
    try:
        cleanup(client)
    except BenchError as exc:
        _log.warning("bench.cleanup.failed", error=str(exc))


def _median(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.median(data) if data else 0.0


def _ratio(baseline: float, choto: float) -> float | None:
    return baseline / choto if choto > 0 else None


def _format_ratio(value: float | None) -> str:
    return f"x{value:.1f}" if value is not None else "n/a"


def summarize(results: Sequence[ArmResult], arm: str) -> dict[str, float]:
    successful = [r for r in results if r.arm == arm and r.ok]
    if not successful:
        return {}
    return {
        "runs": float(len(successful)),
        "roundtrips": _median(r.roundtrips for r in successful),
        "images": _median(r.images for r in successful),
        "text_tokens": _median(r.text_tokens for r in successful),
        "image_tokens": _median(r.image_tokens for r in successful),
        "total_tokens": _median(r.total_tokens for r in successful),
        "wire_bytes": _median(r.wire_bytes for r in successful),
        "latency_ms": _median(r.latency_ms for r in successful),
    }


def compare(summaries: dict[str, dict[str, float]]) -> dict[str, float | None]:
    batched = summaries.get(ARM_BATCHED) or {}
    per_step = summaries.get(ARM_PER_STEP) or {}
    if not batched or not per_step:
        return {}
    return {
        metric: _ratio(per_step[metric], batched[metric])
        for metric in ("total_tokens", "latency_ms", "roundtrips", "wire_bytes")
    }


def _print_run_line(arm: ArmResult) -> None:
    verdict = "OK" if arm.ok else "FAILED"
    print(
        f"  run {arm.run_index}  {arm.arm:<9} "
        f"roundtrips {arm.roundtrips:>3}  images {arm.images:>2}  "
        f"tokens {arm.total_tokens:>7,}  wall {arm.latency_ms / 1000:>6.1f}s  {verdict}"
    )
    if not arm.ok and arm.failure:
        print(f"      -> {arm.failure}")


def print_report(results: Sequence[ArmResult], summaries: dict[str, dict[str, float]]) -> None:
    print()
    print("median over successful runs")
    header = (
        f"  {'arm':<9} {'runs':>5} {'roundtrips':>11} {'images':>7} "
        f"{'text tok':>9} {'image tok':>10} {'total tok':>10} {'wall':>8}"
    )
    print(header)
    for arm in ARMS:
        summary = summaries.get(arm)
        if not summary:
            if any(r.arm == arm for r in results):
                print(f"  {arm:<9} no successful run")
            continue
        print(
            f"  {arm:<9} {int(summary['runs']):>5} {int(summary['roundtrips']):>11} "
            f"{int(summary['images']):>7} {int(summary['text_tokens']):>9,} "
            f"{int(summary['image_tokens']):>10,} {int(summary['total_tokens']):>10,} "
            f"{summary['latency_ms'] / 1000:>7.1f}s"
        )

    factors = compare(summaries)
    if factors:
        print()
        print(
            "  per-step / batched:  "
            f"tokens {_format_ratio(factors['total_tokens'])}   "
            f"tool latency {_format_ratio(factors['latency_ms'])}   "
            f"round-trips {_format_ratio(factors['roundtrips'])}   "
            f"wire bytes {_format_ratio(factors['wire_bytes'])}"
        )

    print()
    print("notes")
    print(
        f"  tokens are estimates: text = ceil(utf-8 bytes / {BYTES_PER_TEXT_TOKEN}), "
        f"image = ceil(w*h / {IMAGE_PIXELS_PER_TOKEN})"
    )
    print("  wall time counts tool execution only; the model's own thinking time per")
    print("  round-trip is excluded, which understates the per-step arm's real latency")
    print("  the screen graph is shared and warms up, so run 1 is the cold-cache case")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description=(
            "Measure Choto's batched protocol against a screenshot-per-step loop "
            "on one real TextEdit scenario."
        ),
    )
    parser.add_argument(
        "--socket",
        dest="socket_path",
        metavar="PATH",
        default=None,
        help=f"Path of the daemon's unix socket (default: {default_socket_path()}).",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=DEFAULT_RUNS,
        help=f"How many times to run each arm (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--arms",
        choices=[*ARMS, "both"],
        default="both",
        help="Which arm(s) to run (default: both).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_CALL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Per-call timeout (default: {DEFAULT_CALL_TIMEOUT_S:.0f}).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        default=None,
        help="Also write the full per-call results to this file as JSON.",
    )
    return parser.parse_args(argv)


def _run_arms(client: SocketMcpClient, runs: int, arms: Sequence[str]) -> list[ArmResult]:
    runners = {ARM_BATCHED: run_batched_arm, ARM_PER_STEP: run_per_step_arm}
    results: list[ArmResult] = []
    for run_index in range(1, runs + 1):
        for arm in arms:
            try:
                result = runners[arm](client, run_index)
            finally:
                _cleanup_quietly(client)
            results.append(result)
            _print_run_line(result)
    return results


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise BenchError(f"cannot write results to {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _parse_args(argv)

    try:
        socket_path = resolve_socket_path(args.socket_path)
    except ValueError as exc:
        _log.error("bench.socket_path.invalid", error=str(exc))
        return 1

    arms = list(ARMS) if args.arms == "both" else [args.arms]
    steps = build_steps(1)

    print(f"Choto bench — TextEdit round-trip, {len(steps)} steps, {args.runs} run(s) per arm")
    print(f"socket: {socket_path}")
    print()

    try:
        with SocketMcpClient(socket_path, args.timeout) as client:
            missing = REQUIRED_TOOLS - client.list_tools()
            if missing:
                raise BenchError(f"the daemon does not expose {sorted(missing)}")

            client.call_tool("observe", {})
            _cleanup_quietly(client)

            results = _run_arms(client, args.runs, arms)
    except BenchError as exc:
        _log.error("bench.failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        _log.warning("bench.interrupted", hint="TextEdit may need manual cleanup")
        return 130

    summaries = {arm: summarize(results, arm) for arm in arms}
    print_report(results, summaries)

    if args.json_path:
        payload = {
            "socket": str(socket_path),
            "scenario": {"app": TEXTEDIT_APP, "steps": steps},
            "token_model": {
                "bytes_per_text_token": BYTES_PER_TEXT_TOKEN,
                "image_pixels_per_token": IMAGE_PIXELS_PER_TOKEN,
            },
            "results": [result.as_dict() for result in results],
            "medians": summaries,
            "ratios_per_step_over_batched": compare(summaries),
        }
        try:
            _write_json(Path(args.json_path).expanduser(), payload)
        except BenchError as exc:
            _log.error("bench.json.failed", error=str(exc))
            return 1
        print(f"\nfull results written to {args.json_path}")

    failed = [result for result in results if not result.ok]
    if failed:
        _log.warning("bench.runs_failed", count=len(failed), total=len(results))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
