from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from choto.capture.monitor import MonitorInfo
from choto.capture.source import Frame
from choto.config import Settings
from choto.executor.applauncher import AppIdentity
from choto.executor.runner import Executor, _densest_cell_center
from choto.executor.screen_reader import ScreenReader
from choto.executor.scrolltrack import ScrollPass
from choto.executor.workarea import resolve_work_area
from choto.graph.engine import create_engine_from_settings, create_session_factory
from choto.graph.map_models import ScrollEdge
from choto.graph.repository import GraphRepository
from choto.matching.matcher import TargetMatcher
from choto.models import Action, BBox, OcrLine, Plan, Step, Target
from choto.ocr.engine import OcrPasses
from choto.vision.windowsource import Obstructions, Window, WindowList, WindowRect

APP = "GateBrowser"

TITLE = "Operating system"

SCREEN_W, SCREEN_H = 1200, 820

WINDOW = BBox(x=0, y=60, w=SCREEN_W, h=SCREEN_H - 60)

BAR_Y = 70

BODY_TOP = 130

LINE_H = 24

DOC_TOP, DOC_STEP = 140, 80

PX_PER_TICK = 100

MAX_OFFSET = 1400

TABS = (("Article", 600), ("Discussion", 760))

RAIL = (
    "Contents",
    "History",
    "Sources",
    "See also",
    "References",
    "Editions",
    "Languages",
    "Tools",
    "Printable",
    "Cite page",
)

RAIL_X, RAIL_TOP, RAIL_STEP, RAIL_H = 20, 150, 30, 18

# PR-025
RAIL_JUMP = 112

# PR-025
BODY_W = 640

RAIL_EDGE = 400

BODY_X = 620

ECHO_INDEX = 2

CASE_OFFSET = 100

WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]


def _line(text: str, y: int, x: int = 40, h: int = LINE_H, w: int | None = None) -> OcrLine:
    width = 10 * len(text) if w is None else w
    return OcrLine(text=text, bbox=BBox(x=x, y=y, w=width, h=h), confidence=0.95)


def doc_y_of(word: str) -> int:
    return DOC_TOP + WORDS.index(word) * DOC_STEP


@dataclass
class Desk:
    offset: int = 0
    panel: str = "Article"
    scrolls: int = 0
    clicks: list[str] = field(default_factory=list)
    echoes: str | None = None
    # PR-030
    rail: bool = False
    bare: bool = False
    wheel: tuple[int, int] | None = None
    # PR-031
    frozen: bool = False
    # PR-025
    sticky: bool = False

    def body(self) -> list[str]:
        if self.bare:
            return []
        words = list(WORDS) if self.panel == "Article" else [f"re {word}" for word in WORDS]
        if self.echoes is not None:
            words[ECHO_INDEX] = self.echoes
        return words

    def page(self) -> list[OcrLine]:
        lines = [_line(name, BAR_Y, x=x, h=18) for name, x in TABS]
        if self.rail:
            # PR-025
            jump = RAIL_JUMP if self.sticky and self.offset else 0
            lines.extend(
                _line(text, RAIL_TOP + index * RAIL_STEP - jump, x=RAIL_X, h=RAIL_H)
                for index, text in enumerate(RAIL)
            )
        for index, text in enumerate(self.body()):
            y = DOC_TOP + index * DOC_STEP - self.offset
            if y >= BODY_TOP and y + LINE_H <= SCREEN_H:
                lines.append(
                    _line(
                        text,
                        y,
                        x=BODY_X if self.rail else 40,
                        w=BODY_W if self.sticky else None,
                    )
                )
        return lines

    def image(self) -> np.ndarray:
        canvas = np.full((SCREEN_H, SCREEN_W, 3), 255, dtype=np.uint8)
        for line in self.page():
            box = line.bbox
            shade = (sum(ord(char) for char in line.text) * 17) % 200
            canvas[box.y : box.y + box.h, box.x : box.x + box.w] = shade
        return np.ascontiguousarray(canvas)

    # PR-030
    def over_rail(self) -> bool:
        return self.rail and self.wheel is not None and self.wheel[0] < RAIL_EDGE

    def scroll(self, amount: int) -> None:
        self.scrolls += 1
        # PR-030, PR-031
        if self.frozen or self.over_rail():
            return
        self.offset = max(0, min(MAX_OFFSET, self.offset + amount * PX_PER_TICK))

    def click(self, x: int, y: int) -> None:
        for line in self.page():
            box = line.bbox
            if box.x <= x < box.x + box.w and box.y <= y < box.y + box.h:
                where = "the bar" if box.y == BAR_Y else "the body"
                self.clicks.append(f'"{line.text}" in {where} at ({x},{y})')
                if box.y == BAR_Y:
                    self.panel = line.text
                return
        self.clicks.append(f"({x},{y})")


class DeskSource:
    def __init__(self, desk: Desk) -> None:
        self._desk = desk

    def latest(self) -> Frame:
        image = self._desk.image()
        return Frame(
            image=image,
            box=BBox(x=0, y=0, w=SCREEN_W, h=SCREEN_H),
            captured_at_ms=time.monotonic() * 1000.0,
            age_ms=0.0,
        )

    def region(self, box: BBox) -> Frame:
        whole = self._desk.image()
        return Frame(
            image=np.ascontiguousarray(whole[box.y : box.y + box.h, box.x : box.x + box.w]),
            box=box,
            captured_at_ms=time.monotonic() * 1000.0,
            age_ms=0.0,
        )

    def geometry(self) -> MonitorInfo:
        return MonitorInfo(
            width_px=SCREEN_W,
            height_px=SCREEN_H,
            width_pt=SCREEN_W,
            height_pt=SCREEN_H,
            scale=1.0,
        )

    def close(self) -> None:
        return None


class DeskOcr:
    def __init__(self, desk: Desk) -> None:
        self._desk = desk

    def recognize(self, image: np.ndarray, *, glyph_regions: Sequence[BBox] | None = None):
        return self._desk.page()

    def recognize_passes(
        self, image: np.ndarray, *, glyph_regions: Sequence[BBox] | None = None
    ) -> OcrPasses:
        page = self._desk.page()
        return OcrPasses(lines=page, text_lines=list(page))


class DeskInput:
    def __init__(self, desk: Desk) -> None:
        self._desk = desk

    def ensure_permission(self) -> bool:
        return True

    def request_permission(self) -> bool:
        return True

    def mouse_position(self) -> tuple[float, float]:
        return (600.0, 400.0)

    def main_display_size(self) -> tuple[float, float]:
        return (float(SCREEN_W), float(SCREEN_H))

    def read_clipboard(self) -> str | None:
        return None

    def move(self, x: float, y: float) -> None:
        # PR-030
        self._desk.wheel = (round(x), round(y))

    def click(self, x: float, y: float, **kwargs) -> None:
        self._desk.click(round(x), round(y))

    def right_click(self, x: float, y: float, **kwargs) -> None:
        self._desk.click(round(x), round(y))

    def drag(self, path, **kwargs) -> None:
        return None

    def scroll(self, amount: int) -> None:
        self._desk.scroll(amount)

    def type_text(self, text: str) -> None:
        return None

    def hotkey(self, keys: list[str]) -> None:
        return None

    def close(self) -> None:
        return None


class DeskWindows:
    def frontmost_windows(self) -> WindowList:
        return WindowList(
            app_name=APP,
            windows=(
                Window(
                    app_name=APP,
                    title=TITLE,
                    rect=WindowRect(
                        x=float(WINDOW.x), y=float(WINDOW.y), w=float(WINDOW.w), h=float(WINDOW.h)
                    ),
                ),
            ),
        )

    def obstructions_over(self, rect: WindowRect, *, owner: str) -> Obstructions:
        return Obstructions()

    def frontmost_app_name(self) -> str:
        return APP

    def close(self) -> None:
        return None


class DeskApps:
    def activate(self, app_name: str) -> str:
        return app_name

    def resolve(self, app_name: str) -> AppIdentity | None:
        return AppIdentity(display_name=APP)

    def same_app(self, observed: str, wanted: str) -> bool:
        return observed == wanted

    def app_version(self, app_name: str) -> str:
        return "0"

    def close(self) -> None:
        return None


def stand_settings(db: Path) -> Settings:
    return Settings(
        db_path=db,
        icon_detection_enabled=False,
        window_rect_refinement_enabled=False,
        motion_tracking_enabled=False,
        settle_quiet_grace_ms=0,
        settle_in_flight_ceiling_ms=0,
    )


@dataclass
class Stand:
    desk: Desk
    repo: GraphRepository
    reader: ScreenReader
    executor: Executor
    settings: Settings


def build(db: Path, desk: Desk | None = None) -> Stand:
    desk = desk if desk is not None else Desk()
    settings = stand_settings(db)
    repo = GraphRepository(create_session_factory(create_engine_from_settings(settings)))
    windows = DeskWindows()
    reader = ScreenReader(DeskSource(desk), DeskOcr(desk), repo, settings, windows)
    executor = Executor(
        reader=reader,
        matcher=TargetMatcher(settings),
        input_controller=DeskInput(desk),
        windows=windows,
        apps=DeskApps(),
        repo=repo,
        settings=settings,
    )
    return Stand(desk=desk, repo=repo, reader=reader, executor=executor, settings=settings)


def work_area(stand: Stand):
    return resolve_work_area(
        DeskWindows().frontmost_windows(),
        monitor=stand.reader.monitor_info(),
        target_app=APP,
        window_query=None,
        same_app=lambda observed, wanted: observed == wanted,
    )


def nodes(db: Path) -> list[tuple[int, str, int, int, int]]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT id, phash, coalesce(scroll_covered_top, -1),"
            " coalesce(scroll_covered_bottom, -1), coalesce(scroll_viewport_h, -1)"
            " FROM screens WHERE app_name = ? ORDER BY id",
            (APP,),
        ).fetchall()


def edges(db: Path) -> list[tuple[int, int, str]]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT e.from_screen, e.to_screen, e.action FROM edges e"
            " JOIN screens s ON s.id = e.from_screen WHERE s.app_name = ? ORDER BY e.id",
            (APP,),
        ).fetchall()


def placements(db: Path, screen_id: int) -> str:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT coalesce(scroll_role, 'unknown'), offscreen_side IS NOT NULL"
            " FROM elements WHERE screen_id = ?",
            (screen_id,),
        ).fetchall()
    seen: dict[str, int] = {}
    for role, gone in rows:
        key = f"{role}{' past the edge' if gone else ''}"
        seen[key] = seen.get(key, 0) + 1
    return ", ".join(f"{count} {key}" for key, count in sorted(seen.items()))


def show_nodes(db: Path, prefix: str = "  ") -> None:
    listed = nodes(db)
    print(f"{prefix}{len(listed)} node(s) for {APP}")
    for screen_id, phash, top, bottom, viewport in listed:
        covered = "-" if viewport < 0 else f"{top}..{bottom} viewport_h={viewport}"
        print(f"{prefix}#{screen_id} phash={phash} coverage={covered}")
        print(f"{prefix}   elements: {placements(db, screen_id)}")


def gate_unpinned(db: Path) -> None:
    print("\n=== PR-027 before: one read per scroll position, nothing pinned")
    stand = build(db)
    area = work_area(stand)
    walked = ScrollPass(WINDOW)
    last: int | None = None
    for offset in (0, 300, 600):
        stand.desk.offset = offset
        read = stand.reader.read(area, force_ocr=True)
        walked.page(area.in_window(read.state.lines))
        last = read.screen_db_id
        print(f"  offset {offset:>3}: node #{read.screen_db_id}")
    extent = walked.extent({ScrollEdge.BOTTOM})
    if extent is not None and last is not None:
        stand.repo.record_scroll_pass(last, extent, walked.placements())
    show_nodes(db)


def gate_pinned(db: Path) -> None:
    print("\n=== PR-027 after: one pass over one window, reads pinned to its node")
    stand = build(db)
    plan = Plan(
        steps=[
            Step(action=Action.CLICK, target=Target(text="papa"), timeout_ms=4000),
            Step(action=Action.CLICK, target=Target(text="Discussion"), timeout_ms=4000),
        ]
    )
    report = stand.executor.execute(plan)
    print(
        f"  status: {report.status.value}   scrolls: {stand.desk.scrolls}"
        f"   clicked: {stand.desk.clicks}"
    )
    for told in report.journey:
        print(f"  {told}")
    show_nodes(db)
    for from_id, to_id, action in edges(db):
        print(f"  edge #{from_id} -> #{to_id} {action}")


def read_case(
    db: Path,
    label: str,
    target: Target | None,
    from_text: str,
    to_text: str,
    desk: Desk | None = None,
) -> None:
    stand = build(db, desk)
    plan = Plan(
        steps=[
            Step(
                action=Action.READ,
                target=target,
                from_text=from_text,
                to_text=to_text,
                timeout_ms=20000,
            )
        ]
    )
    report = stand.executor.execute(plan)
    print(f"  {label}: {report.status.value}, {stand.desk.scrolls} scroll(s)")
    print(f"    {report.journey[0]}")
    if report.reason is not None:
        print(f"    reason: {report.reason}")
    for item in report.extracted:
        print(f"    read: {item.text.splitlines()}")


def gate_reading(db: Path) -> None:
    print("\n=== PR-028: a reading boundary below the fold")
    print(f'  "quebec" sits at doc y={doc_y_of("quebec")}, the fold is at y={SCREEN_H}')
    read_case(
        db,
        "scroll_to_find off",
        Target(text="", scroll_to_find=False),
        "quebec",
        "tango",
    )
    read_case(db, "scroll_to_find on (default)", None, "quebec", "tango")
    read_case(db, "a marker that is nowhere", None, "marker nobody wrote", "tango")


def click_case(db: Path, label: str, wanted: str, echoes: str | None) -> None:
    stand = build(db, Desk(echoes=echoes, offset=CASE_OFFSET))
    area = work_area(stand)
    known = stand.reader.read(area, force_ocr=True, persist=False).navigation
    plan = Plan(steps=[Step(action=Action.CLICK, target=Target(text=wanted), timeout_ms=4000)])
    stand.executor.execute(plan)
    print(f"  {label}: navigation knows {sorted(known.labels)}")
    print(f'    click "{wanted}" landed on {stand.desk.clicks}')


def plant_windows(db: Path, rail: bool = False) -> None:
    stand = build(db, Desk(rail=rail))
    area = work_area(stand)
    for offset in (0, 300, 600):
        stand.desk.offset = offset
        stand.reader.read(area, force_ocr=True)


def gate_navigation(db: Path) -> None:
    print("\n=== PR-029: the window's own content first, the application's navigation second")
    click_case(db, "no navigation derived yet", "Discussion", "Discussion")
    plant_windows(db)
    click_case(db, "the same caption in the bar and in the body", "Discussion", "Discussion")
    click_case(db, "the caption is only in the bar", "Article", None)
    read_case(
        db,
        "a reading boundary the bar also carries",
        None,
        "Discussion",
        "echo",
        Desk(echoes="Discussion", offset=CASE_OFFSET),
    )


# PR-030
def zone(point: tuple[int, int] | None) -> str:
    if point is None:
        return "nowhere: it refused to scroll"
    where = "the fixed rail" if point[0] < RAIL_EDGE else "the article body"
    return f"({point[0]},{point[1]}), {where}"


# PR-030
def anchor_case(db: Path, label: str, desk: Desk) -> None:
    stand = build(db, desk)
    area = work_area(stand)
    read = stand.reader.read(area, force_ocr=True, persist=False)
    dense = _densest_cell_center(WINDOW, area.in_window(read.state.lines))
    plan = Plan(
        steps=[
            Step(
                action=Action.READ,
                from_text=RAIL[0] if desk.bare else "alpha",
                to_text="marker nobody wrote",
                timeout_ms=20000,
            )
        ]
    )
    report = stand.executor.execute(plan)
    print(f"  {label}: navigation holds {len(read.navigation.slots)} label(s)")
    print(f"    densest text of the window: {zone(dense)}")
    print(f"    the wheel went to:          {zone(stand.desk.wheel)}")
    print(f"    offset after the pass: {stand.desk.offset} after {stand.desk.scrolls} scroll(s)")
    print(f"    {report.journey[0]}")


def gate_anchor(plain: Path, known: Path) -> None:
    print("\n=== PR-030: the wheel turns over the content, not over the navigation")
    print(f"  the rail is dense and fixed at x={RAIL_X}, the article is sparse at x={BODY_X}")
    anchor_case(plain, "no navigation derived yet", Desk(rail=True))
    plant_windows(known, rail=True)
    anchor_case(known, "the rail is known navigation", Desk(rail=True))
    anchor_case(known, "nothing on screen but navigation", Desk(rail=True, bare=True))


# PR-031
def ends_recorded(db: Path) -> str:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT id, scroll_ends FROM screens WHERE app_name = ? ORDER BY id",
            (APP,),
        ).fetchall()
    return ", ".join(f"#{screen_id} ends={ends}" for screen_id, ends in rows)


# PR-031
def gate_moving(db: Path) -> None:
    print("\n=== PR-031: a page that did not move is not a page that ended")
    nowhere = "marker nobody wrote"
    read_case(db, "the wheel has no effect", None, "alpha", nowhere, Desk(frozen=True))
    print(f"    written: {ends_recorded(db)}")
    read_case(db, "the document really ends", None, "alpha", nowhere)
    print(f"    written: {ends_recorded(db)}")


# PR-025, PR-027
def gate_sticky(db: Path) -> None:
    print("\n=== PR-025: a rail that jumps once must not outvote the article")
    print(
        f"  the rail is {len(RAIL)} short captions at x={RAIL_X}; it jumps {RAIL_JUMP}px once"
        f" and then stands still, while the article moves {PX_PER_TICK * 3}px per scroll"
    )
    # PR-030
    plant_windows(db, rail=True)
    planted = len(nodes(db))
    stand = build(db, Desk(rail=True, sticky=True))
    plan = Plan(
        steps=[
            Step(
                action=Action.READ,
                from_text="alpha",
                to_text="marker nobody wrote",
                timeout_ms=20000,
            )
        ]
    )
    stand.executor.execute(plan)
    walked = stand.executor._scroll_pass
    seen = walked.seen if walked is not None else 0
    pages = walked.pages if walked is not None else 0
    offset = walked.offset if walked is not None else 0
    print(f"  the pass was fed {seen} page(s), placed {pages}, last offset {offset}")
    print(f"  the desk stopped at offset {stand.desk.offset} after {stand.desk.scrolls} scroll(s)")
    print(f"  the pass added {len(nodes(db)) - planted} node(s) to the {planted} already known")
    show_nodes(db)


def main() -> None:
    parser = argparse.ArgumentParser(description="target-finding gates")
    parser.add_argument("--db", type=Path, default=Path("data/choto.db"))
    parser.add_argument("--out", type=Path, default=Path("tmp"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    before, after = args.out / "targetgate-before.db", args.out / "targetgate-after.db"
    read_db = args.out / "targetgate-read.db"
    nav_db = args.out / "targetgate-nav.db"
    plain_db = args.out / "targetgate-plain.db"
    rail_db = args.out / "targetgate-rail.db"
    move_db = args.out / "targetgate-move.db"
    sticky_db = args.out / "targetgate-sticky.db"
    for copy in (before, after, read_db, nav_db, plain_db, rail_db, move_db, sticky_db):
        shutil.copyfile(args.db, copy)

    gate_unpinned(before)
    gate_pinned(after)
    gate_reading(read_db)
    gate_navigation(nav_db)
    gate_anchor(plain_db, rail_db)
    gate_moving(move_db)
    gate_sticky(sticky_db)


if __name__ == "__main__":
    main()
