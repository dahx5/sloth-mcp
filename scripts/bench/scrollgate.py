from __future__ import annotations

import argparse
import shutil
import sqlite3
from collections.abc import Collection, Sequence
from pathlib import Path

from choto.config import Settings
from choto.executor.scrolltrack import ScrollPass
from choto.graph.engine import create_engine_from_settings, create_session_factory
from choto.graph.map_models import ScrollEdge
from choto.graph.repository import GraphRepository
from choto.matching.matcher import TargetMatcher
from choto.models import BBox, OcrLine, ScreenState
from choto.recall import RecallService

APP_NAME = "ScrollGate"

WINDOW_TITLE = "Long page"

FRAME = BBox(x=0, y=0, w=1200, h=820)

VIEWPORT = BBox(x=0, y=60, w=1200, h=740)

DOC_TOP = 100

DOC_BOTTOM = 2000

DOC_STEP = 80

LINE_H = 24

OFFSETS = (0, 300, 600, 0)

ROLE_FIELDS_BREACH = """
    NOT (
      (scroll_role IS NULL AND doc_x IS NULL AND doc_y IS NULL AND offscreen_side IS NULL)
      OR (scroll_role = 'pinned' AND doc_x IS NULL AND doc_y IS NULL AND offscreen_side IS NULL)
      OR (scroll_role = 'document' AND doc_x IS NOT NULL AND doc_y IS NOT NULL)
    )
"""

OFFSCREEN_BOX_BREACH = """
    NOT (
      (offscreen_side IS NULL AND offscreen_distance IS NULL
       AND x IS NOT NULL AND y IS NOT NULL)
      OR (offscreen_side IS NOT NULL AND offscreen_distance IS NOT NULL
          AND x IS NULL AND y IS NULL AND scroll_role = 'document')
    )
"""


def _line(text: str, y: int, x: int = 40, w: int = 320) -> OcrLine:
    return OcrLine(text=text, bbox=BBox(x=x, y=y, w=w, h=LINE_H), confidence=0.95)


def page_at(offset: int) -> list[OcrLine]:
    lines = [_line("scrollgate toolbar", 12), _line("search field", 12, x=820, w=280)]
    for doc_y in range(DOC_TOP, DOC_BOTTOM, DOC_STEP):
        y = doc_y - offset
        if y >= VIEWPORT.y and y + LINE_H <= VIEWPORT.y + VIEWPORT.h:
            lines.append(_line(f"paragraph {doc_y}", y))
    return lines


def open_repo(db_path: Path) -> GraphRepository:
    settings = Settings(db_path=db_path)
    return GraphRepository(create_session_factory(create_engine_from_settings(settings)))


def plant_window(repo: GraphRepository, phash: str, title: str) -> int:
    resting = page_at(0)
    state = ScreenState(
        app_name=APP_NAME, width=FRAME.w, height=FRAME.h, phash=phash, lines=resting
    )
    screen = repo.upsert_screen(state, [], glyph_scope="", window_title=title)
    repo.replace_elements(screen.id, resting)
    return screen.id


def walk(offsets: Sequence[int]) -> ScrollPass:
    walked = ScrollPass(VIEWPORT)
    for offset in offsets:
        walked.page(page_at(offset))
    return walked


def store(
    repo: GraphRepository, screen_id: int, walked: ScrollPass, ends: Collection[ScrollEdge]
) -> bool:
    extent = walked.extent(ends)
    if extent is None:
        return False
    repo.record_scroll_pass(screen_id, extent, walked.placements())
    return True


def screen_row(db: Path, screen_id: int) -> str:
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT scroll_covered_top, scroll_covered_bottom, scroll_viewport_h, scroll_ends"
            " FROM screens WHERE id = ?",
            (screen_id,),
        ).fetchone()
    return f"covered={row[0]}..{row[1]} viewport_h={row[2]} ends={row[3]}"


def combinations(db: Path, screen_id: int) -> dict[str, int]:
    clauses = {
        "document on screen": "scroll_role = 'document' AND offscreen_side IS NULL",
        "document past the edge": "scroll_role = 'document' AND offscreen_side IS NOT NULL",
        "pinned to the window": "scroll_role = 'pinned'",
        "role unknown": "scroll_role IS NULL",
    }
    with sqlite3.connect(db) as conn:
        return {
            name: conn.execute(
                f"SELECT count(*) FROM elements WHERE screen_id = ? AND ({clause})", (screen_id,)
            ).fetchone()[0]
            for name, clause in clauses.items()
        }


def breaches(db: Path) -> dict[str, int]:
    with sqlite3.connect(db) as conn:
        return {
            name: conn.execute(f"SELECT count(*) FROM elements WHERE {clause}").fetchone()[0]
            for name, clause in (
                ("ck_elements_scroll_role_fields", ROLE_FIELDS_BREACH),
                ("ck_elements_offscreen_has_no_screen_box", OFFSCREEN_BOX_BREACH),
            )
        }


def elements(db: Path, screen_id: int) -> list[str]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT text, scroll_role, x, y, doc_x, doc_y, offscreen_side, offscreen_distance"
            " FROM elements WHERE screen_id = ? ORDER BY doc_y IS NULL, doc_y, text",
            (screen_id,),
        ).fetchall()
    return [
        f"    {text:<22} role={role or '-':<9} screen=({x},{y}) doc=({doc_x},{doc_y})"
        f" offscreen={side or '-'}{f'+{distance}' if side else ''}"
        for text, role, x, y, doc_x, doc_y, side, distance in rows
    ]


def recall_text(repo: GraphRepository) -> str:
    service = RecallService(repo, TargetMatcher(Settings()))
    window = service.render(app_name=APP_NAME, window=WINDOW_TITLE)
    found = service.render(app_name=APP_NAME, query="paragraph 900")
    keep = ("paragraph 900", "coverage:", WINDOW_TITLE)
    return "\n".join(
        line for line in f"{window}\n{found}".splitlines() if any(mark in line for mark in keep)
    )


def gate_constraints(db: Path, screen_id: int) -> None:
    print("\n=== gate: database constraints, on a copy of the live map")
    print(f"  screen {screen_id}: {screen_row(db, screen_id)}")
    for name, count in combinations(db, screen_id).items():
        print(f"  {name:<24} {count} row(s)")
    for name, count in breaches(db).items():
        print(f"  {name}: {count} row(s) in breach across the whole table")
    print("\n".join(elements(db, screen_id)))


def gate_readers(before: str, after: str) -> None:
    print("\n=== gate: readers")
    print("--- before the writer ran")
    print(before)
    print("--- after the writer ran")
    print(after)


def gate_honesty(repo: GraphRepository, db: Path) -> None:
    print("\n=== gate: honesty")
    cases: list[tuple[str, ScrollPass]] = []

    still = walk((0,))
    cases.append(("never scrolled", still))

    lost = walk((0,))
    lost.page([_line("nothing in common here", 200)])
    cases.append(("offset unmeasurable", lost))

    for index, (label, walked) in enumerate(cases):
        screen_id = plant_window(repo, f"70f1a2b3c4d5e6{index:02d}", f"{WINDOW_TITLE} {label}")
        wrote = store(repo, screen_id, walked, {ScrollEdge.BOTTOM})
        print(f"  {label:<20} at_rest={walked.at_rest} written={wrote} {screen_row(db, screen_id)}")
        print(f"  {'':<20} {combinations(db, screen_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="scrolled-knowledge writer gates")
    parser.add_argument("--db", type=Path, default=Path("data/choto.db"))
    parser.add_argument("--copy", type=Path, default=Path("tmp/scrollgate.db"))
    args = parser.parse_args()

    args.copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.db, args.copy)
    repo = open_repo(args.copy)

    screen_id = plant_window(repo, "5c3a9d6e1f04b872", WINDOW_TITLE)
    before = recall_text(repo)
    walked = walk(OFFSETS)
    wrote = store(repo, screen_id, walked, {ScrollEdge.BOTTOM})
    print(f"pass: at_rest={walked.at_rest} resting offset={walked.offset} written={wrote}")
    gate_constraints(args.copy, screen_id)
    gate_readers(before, recall_text(repo))
    gate_honesty(repo, args.copy)


if __name__ == "__main__":
    main()
