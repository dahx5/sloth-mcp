from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from choto.config import Settings
from choto.log import get_logger
from choto.models import BBox, OcrLine
from choto.ocr.engine import OcrEngine, OcrRequestCounter, create_ocr_engine

_log = get_logger(__name__)

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "fixtures"
EXPECTED_SUFFIX = ".expected.json"
DEFAULT_RUNS = 5


class HarnessError(RuntimeError): ...


class MatchMode(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"


class ExpectedItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    group: str = Field(min_length=1)
    match: MatchMode
    box: BBox


class Expectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    fixture: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    glyph_regions: list[BBox] | None
    note: str
    items: list[ExpectedItem] = Field(min_length=1)


@dataclass(frozen=True)
class TruthItem:
    text: str
    group: str
    match: MatchMode = MatchMode.CONTAINS
    aliases: tuple[str, ...] = ()
    box: BBox | None = None

    def spellings(self) -> list[str]:
        return list(self.aliases) if self.aliases else [self.text]


@dataclass(frozen=True)
class FixtureTruth:
    name: str
    filename: str
    note: str
    glyph_regions: tuple[BBox, ...] | None
    items: tuple[TruthItem, ...]

    @property
    def path(self) -> Path:
        return FIXTURES / self.filename

    @property
    def expected_path(self) -> Path:
        return FIXTURES / f"{Path(self.filename).stem}{EXPECTED_SUFFIX}"


_KEY_TRUTH: tuple[tuple[str, tuple[str, ...], int, int], ...] = (
    ("AC", ("AC",), 1002, 446),
    ("%", ("%",), 1056, 446),
    ("÷", ("÷", ":"), 1110, 446),
    ("7", ("7",), 948, 500),
    ("8", ("8",), 1002, 500),
    ("9", ("9",), 1056, 500),
    ("×", ("×", "x", "X", "*"), 1110, 500),
    ("4", ("4",), 948, 555),
    ("5", ("5",), 1002, 555),
    ("6", ("6",), 1056, 555),
    ("−", ("−", "-", "–", "—", "_"), 1110, 555),
    ("1", ("1",), 948, 609),
    ("2", ("2",), 1002, 609),
    ("3", ("3",), 1056, 609),
    ("+", ("+",), 1110, 609),
    ("0", ("0",), 1002, 663),
    ("=", ("=", "≡"), 1110, 663),
)
_BUTTON_RADIUS = 27

_CALCULATOR_PROSE: tuple[str, ...] = (
    "Calculator",
    "Window",
    "General",
    "Accessibility",
    "Appearance",
    "Displays",
    "Notifications",
    "Screen Time",
    "Bluetooth",
    "Software Update",
    "AppleCare & Warranty",
    "AirDrop & Handoff",
    "AutoFill & Passwords",
    "Date & Time",
    "Language & Region",
    "Login Items & Extensions",
    "Manage your overall setup and preferences for Mac, such as software",
    "Video transcription extraction",
    "Bypass permissions",
    "Hyperion materials and correspondence",
    "Accessibility check",
)

CALCULATOR_TRUTH = FixtureTruth(
    name="calculator",
    filename="calculator_dark.png",
    note=(
        "1710x1107 dark-mode screen: Calculator over a chat window and System Settings. The "
        "glyph pass is scoped to the Calculator window, exactly as the executor scopes it to "
        "the window it is driving."
    ),
    glyph_regions=(BBox(x=915, y=296, w=233, h=406),),
    items=(
        *(
            TruthItem(
                text=key,
                group="keys",
                match=MatchMode.EXACT,
                aliases=aliases,
                box=BBox(
                    x=cx - _BUTTON_RADIUS,
                    y=cy - _BUTTON_RADIUS,
                    w=2 * _BUTTON_RADIUS,
                    h=2 * _BUTTON_RADIUS,
                ),
            )
            for key, aliases, cx, cy in _KEY_TRUTH
        ),
        *(TruthItem(text=text, group="prose") for text in _CALCULATOR_PROSE),
    ),
)

SETTINGS_TRUTH = FixtureTruth(
    name="settings",
    filename="settings_native.png",
    note=(
        "1446x1250: the System Settings window alone, captured at native (Retina) resolution. "
        "Read unscoped, which is what a window whose bounds fill the frame amounts to."
    ),
    glyph_regions=None,
    items=(
        *(
            TruthItem(text=text, group="sidebar")
            for text in (
                "Bluetooth",
                "Network",
                "Battery",
                "General",
                "Accessibility",
                "Appearance",
                "Apple Intelligence & Siri",
                "Desktop & Dock",
                "Displays",
                "Menu Bar",
                "Spotlight",
                "Wallpaper",
                "Notifications",
                "Sound",
                "Focus",
                "Screen Time",
            )
        ),
        *(
            TruthItem(text=text, group="pane")
            for text in (
                "Built-in Display",
                "Larger Text",
                "Default",
                "More Space",
                "Brightness",
                "Automatically adjust brightness",
                "True Tone",
                "Automatically adapt display to make colors appear consistent in different",
                "Color profile",
                "SRGB IEC61966-2.1",
                "When connected to TV",
                "Ask What to Show",
            )
        ),
    ),
)

TRUTHS: dict[str, FixtureTruth] = {
    truth.name: truth for truth in (CALCULATOR_TRUTH, SETTINGS_TRUTH)
}


def load_frame(path: Path) -> np.ndarray:
    if not path.is_file():
        raise HarnessError(
            f"Fixture {path} does not exist. The reference frames live in "
            f"{FIXTURES}; restore this one from the repository, or take a fresh "
            f"screenshot only if you are ready to recalibrate against it."
        )
    image = Image.open(path).convert("RGB")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def load_expectation(path: Path, *, width: int, height: int) -> Expectation:
    if not path.is_file():
        raise HarnessError(f"No expectation at {path}; run with --write-expected to record one.")
    try:
        expectation = Expectation.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise HarnessError(f"Expectation {path} is not usable: {exc}") from exc
    if (expectation.width, expectation.height) != (width, height):
        raise HarnessError(
            f"Expectation {path} was recorded on a {expectation.width}x{expectation.height} "
            f"frame but the fixture is {width}x{height}."
        )
    return expectation


def matching_lines(item: ExpectedItem, lines: Sequence[OcrLine]) -> list[OcrLine]:
    if item.match is MatchMode.EXACT:
        return [line for line in lines if line.text.strip() in item.aliases]
    return [line for line in lines if any(alias in line.text for alias in item.aliases)]


def box_contains_center(box: BBox, line: OcrLine) -> bool:
    cx, cy = line.bbox.center
    return box.x <= cx <= box.x + box.w and box.y <= cy <= box.y + box.h


@dataclass(frozen=True)
class ItemScore:
    item: ExpectedItem
    read: bool
    placed: bool


def score_item(item: ExpectedItem, lines: Sequence[OcrLine]) -> ItemScore:
    candidates = matching_lines(item, lines)
    return ItemScore(
        item=item,
        read=bool(candidates),
        placed=any(box_contains_center(item.box, line) for line in candidates),
    )


@dataclass(frozen=True)
class GroupScore:
    group: str
    total: int
    read: int
    placed: int

    @property
    def recall(self) -> float:
        return self.read / self.total

    @property
    def placement(self) -> float:
        return self.placed / self.read if self.read else 0.0

    def missed(self, scores: Sequence[ItemScore]) -> list[str]:
        return [s.item.text for s in scores if s.item.group == self.group and not s.read]


def group_scores(scores: Sequence[ItemScore]) -> list[GroupScore]:
    order: list[str] = []
    totals: dict[str, list[int]] = {}
    for score in scores:
        group = score.item.group
        if group not in totals:
            totals[group] = [0, 0, 0]
            order.append(group)
        bucket = totals[group]
        bucket[0] += 1
        bucket[1] += int(score.read)
        bucket[2] += int(score.placed)
    return [
        GroupScore(
            group=group, total=totals[group][0], read=totals[group][1], placed=totals[group][2]
        )
        for group in order
    ]


def overall_score(scores: Sequence[ItemScore]) -> GroupScore:
    return GroupScore(
        group="all",
        total=len(scores),
        read=sum(1 for s in scores if s.read),
        placed=sum(1 for s in scores if s.placed),
    )


@dataclass(frozen=True)
class Cost:
    runs: int
    median_ms: float
    worst_ms: float
    cpu_ms_median: float
    cpu_ratio_peak: float
    requests: int | None


def measure(
    engine: OcrEngine,
    frame: np.ndarray,
    glyph_regions: Sequence[BBox] | None,
    *,
    runs: int,
) -> tuple[list[OcrLine], Cost]:
    if runs < 1:
        raise HarnessError(f"runs must be at least 1, got {runs}.")

    regions = list(glyph_regions) if glyph_regions else None
    wall: list[float] = []
    cpu: list[float] = []
    readings: list[list[OcrLine]] = []
    for _ in range(runs):
        cpu_start = time.process_time()
        started = time.perf_counter()
        lines = engine.recognize(frame, glyph_regions=regions)
        elapsed = time.perf_counter() - started
        cpu_used = time.process_time() - cpu_start
        wall.append(elapsed)
        cpu.append(cpu_used)
        readings.append(lines)

    reading = readings[0]
    height, width = frame.shape[0], frame.shape[1]
    requests = (
        engine.request_count(width, height, regions)
        if isinstance(engine, OcrRequestCounter)
        else None
    )
    return reading, Cost(
        runs=runs,
        median_ms=statistics.median(wall) * 1000,
        worst_ms=max(wall) * 1000,
        cpu_ms_median=statistics.median(cpu) * 1000,
        cpu_ratio_peak=max(c / w for c, w in zip(cpu, wall, strict=True) if w > 0),
        requests=requests,
    )


@dataclass(frozen=True)
class FixtureReport:
    name: str
    width: int
    height: int
    note: str
    scores: list[ItemScore]
    groups: list[GroupScore]
    overall: GroupScore
    cost: Cost
    lines_read: int


def run_fixture(truth: FixtureTruth, engine: OcrEngine, *, runs: int) -> FixtureReport:
    frame = load_frame(truth.path)
    height, width = frame.shape[0], frame.shape[1]
    expectation = load_expectation(truth.expected_path, width=width, height=height)
    lines, cost = measure(engine, frame, expectation.glyph_regions, runs=runs)
    scores = [score_item(item, lines) for item in expectation.items]
    _log.info(
        "ocrharness.fixture",
        fixture=truth.name,
        read=sum(1 for s in scores if s.read),
        total=len(scores),
        median_ms=round(cost.median_ms, 1),
    )
    return FixtureReport(
        name=truth.name,
        width=width,
        height=height,
        note=expectation.note,
        scores=scores,
        groups=group_scores(scores),
        overall=overall_score(scores),
        cost=cost,
        lines_read=len(lines),
    )


def resolve_box(item: TruthItem, lines: Sequence[OcrLine]) -> BBox:
    if item.box is not None:
        return item.box
    spellings = item.spellings()
    for line in lines:
        hit = (
            line.text.strip() in spellings
            if item.match is MatchMode.EXACT
            else any(alias in line.text for alias in spellings)
        )
        if hit:
            return line.bbox
    raise HarnessError(
        f"The current engine does not read {item.text!r}, so no baseline box can be recorded "
        "for it. Either the fixture changed or the string does not belong in the truth table."
    )


def record_expectation(truth: FixtureTruth, engine: OcrEngine) -> Expectation:
    frame = load_frame(truth.path)
    height, width = frame.shape[0], frame.shape[1]
    regions = list(truth.glyph_regions) if truth.glyph_regions else None
    lines = engine.recognize(frame, glyph_regions=regions)
    return Expectation(
        fixture=truth.filename,
        width=width,
        height=height,
        glyph_regions=list(truth.glyph_regions) if truth.glyph_regions else None,
        note=truth.note,
        items=[
            ExpectedItem(
                text=item.text,
                aliases=item.spellings(),
                group=item.group,
                match=item.match,
                box=resolve_box(item, lines),
            )
            for item in truth.items
        ],
    )


def write_expectation(truth: FixtureTruth, expectation: Expectation) -> Path:
    path = truth.expected_path
    payload = expectation.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return (
        "| "
        + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        + " |"
    )


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) for i in range(len(headers))
    ]
    lines = [_row([str(h) for h in headers], widths)]
    lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    lines.extend(_row([str(cell) for cell in row], widths) for row in rows)
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_report(reports: Sequence[FixtureReport], *, engine_name: str) -> str:
    out: list[str] = [f"engine: {engine_name}", ""]

    out.append("## 1. Recall and placement")
    quality_rows: list[list[str]] = []
    for report in reports:
        for index, group in enumerate([*report.groups, report.overall]):
            quality_rows.append(
                [
                    report.name if index == 0 else "",
                    group.group,
                    str(group.total),
                    f"{group.read}/{group.total}",
                    _pct(group.recall),
                    f"{group.placed}/{group.read}",
                    _pct(group.placement),
                ]
            )
    out.append(
        render_table(
            ["fixture", "group", "items", "read", "recall", "placed", "placement"], quality_rows
        )
    )

    out.append("")
    out.append("## 2. What one parse costs")
    cost_rows = [
        [
            report.name,
            f"{report.width}x{report.height}",
            str(report.lines_read),
            "n/a" if report.cost.requests is None else str(report.cost.requests),
            f"{report.cost.median_ms:.0f}",
            f"{report.cost.worst_ms:.0f}",
            f"{report.cost.cpu_ms_median:.0f}",
            f"{report.cost.cpu_ratio_peak:.2f}x",
        ]
        for report in reports
    ]
    out.append(
        render_table(
            [
                "fixture",
                "frame",
                "lines",
                "requests",
                "median ms",
                "worst ms",
                "cpu ms",
                "peak cpu",
            ],
            cost_rows,
        )
    )

    misses = [
        (report.name, group.group, missed)
        for report in reports
        for group in report.groups
        for missed in [group.missed(report.scores)]
        if missed
    ]
    out.append("")
    out.append("## 3. Unread strings")
    if not misses:
        out.append("(none — every expected string was read)")
    else:
        for fixture, group, missed in misses:
            out.append(f"{fixture}/{group}: " + ", ".join(repr(text) for text in missed))

    out.append("")
    out.append(
        "Placement is measured over the strings that were read: the centre of the recognized "
        "line must fall inside the recorded rectangle."
    )
    out.append(
        "CPU is this process only (user + system). Recognition work Apple performs in another "
        "process is invisible here, so 'cpu ms' is a lower bound."
    )
    return "\n".join(out)


def report_payload(reports: Sequence[FixtureReport], *, engine_name: str) -> dict[str, object]:
    return {
        "engine": engine_name,
        "fixtures": [
            {
                "fixture": report.name,
                "width": report.width,
                "height": report.height,
                "lines_read": report.lines_read,
                "groups": [
                    {
                        "group": group.group,
                        "total": group.total,
                        "read": group.read,
                        "placed": group.placed,
                        "missed": group.missed(report.scores),
                    }
                    for group in [*report.groups, report.overall]
                ],
                "cost": {
                    "runs": report.cost.runs,
                    "median_ms": report.cost.median_ms,
                    "worst_ms": report.cost.worst_ms,
                    "cpu_ms_median": report.cost.cpu_ms_median,
                    "cpu_ratio_peak": report.cost.cpu_ratio_peak,
                    "requests": report.cost.requests,
                },
            }
            for report in reports
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure an OCR engine against the reading this project already depends on."
    )
    parser.add_argument(
        "--fixture",
        action="append",
        choices=sorted(TRUTHS),
        default=None,
        help="measure only this fixture (repeatable); default is all of them",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="timed recognitions per fixture"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write measurements here")
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="re-record the baselines from the current engine instead of measuring",
    )
    return parser.parse_args(argv)


def selected(names: Iterable[str] | None) -> list[FixtureTruth]:
    return [TRUTHS[name] for name in (names or sorted(TRUTHS))]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings()
    engine = create_ocr_engine(settings)
    truths = selected(args.fixture)

    try:
        if args.write_expected:
            for truth in truths:
                expectation = record_expectation(truth, engine)
                path = write_expectation(truth, expectation)
                print(f"wrote {path} ({len(expectation.items)} items)")
            return 0

        reports = [run_fixture(truth, engine, runs=args.runs) for truth in truths]
    except HarnessError as exc:
        print(f"ocrharness: {exc}")
        return 1

    print(render_report(reports, engine_name=settings.ocr_engine))
    if args.json is not None:
        args.json.write_text(
            json.dumps(report_payload(reports, engine_name=settings.ocr_engine), indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
