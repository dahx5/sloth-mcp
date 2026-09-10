from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from choto.config import Settings
from choto.graph import normalize_text
from choto.graph.engine import create_engine_from_settings, create_session_factory
from choto.graph.repository import GraphRepository
from choto.models import ElementKind, OcrLine
from choto.vision.anchors import anchors_agree, anchors_confirm, select_anchors
from choto.vision.navigation import (
    NAV_MIN_SHARE,
    NAV_MIN_WINDOWS,
    NAV_PLACE_TOLERANCE_PX,
    derive_navigation,
    place_clusters,
)


def open_repo(db_path: Path) -> GraphRepository:
    settings = Settings(db_path=db_path)
    return GraphRepository(create_session_factory(create_engine_from_settings(settings)))


def screen_lines(repo: GraphRepository, screen_ids: list[int]) -> dict[int, list[OcrLine]]:
    labels = repo.window_labels(screen_ids)
    return {
        screen_id: [
            OcrLine(text=label.text, bbox=label.bbox, confidence=label.confidence)
            for label in labels.get(screen_id, [])
            if label.kind is ElementKind.TEXT and label.text.strip()
        ]
        for screen_id in screen_ids
    }


def occupancy(repo: GraphRepository, app_name: str) -> Counter[int]:
    places = repo.app_label_places(app_name)
    grouped: dict[str, list] = {}
    for place in places:
        text = normalize_text(place.text)
        if text:
            grouped.setdefault(text, []).append(place)
    counts: Counter[int] = Counter()
    for occurrences in grouped.values():
        for cluster in place_clusters(occurrences):
            counts[len({item.screen_id for item in cluster})] += 1
    return counts


def anchor_spread(anchors: dict[int, list[str]]) -> tuple[Counter[str], int, int]:
    served: Counter[str] = Counter()
    for picked in anchors.values():
        for text in {normalize_text(item) for item in picked}:
            served[text] += 1
    pairs = list(combinations(sorted(anchors), 2))
    same = sum(1 for left, right in pairs if anchors_agree(anchors[left], anchors[right]))
    confirmed = sum(
        1
        for left, right in pairs
        if anchors[left] and anchors[right] and anchors_confirm(anchors[left], anchors[right])
    )
    return served, same, confirmed


def report_app(repo: GraphRepository, app_name: str) -> None:
    ids = sorted(window.screen_id for window in repo.window_summaries(app_name=app_name))
    lines = screen_lines(repo, ids)
    navigation = derive_navigation(repo.app_label_places(app_name))

    print(f"\n=== {app_name}: {len(ids)} windows, quorum {navigation.quorum or '-'}")
    counts = occupancy(repo, app_name)
    print("  labels by how many windows share the same place:")
    for shared in sorted(counts, reverse=True):
        print(f"    {shared:>3}/{len(ids)}: {counts[shared]}")
    print(f"  navigation: {len(navigation.slots)} labels in {len(navigation.areas)} areas")
    for area in navigation.areas:
        print(f"    area x={area.x} y={area.y} w={area.w} h={area.h}")

    before = {screen_id: select_anchors(rows) for screen_id, rows in lines.items()}
    after = {
        screen_id: select_anchors([row for row in rows if not navigation.holds(row)])
        for screen_id, rows in lines.items()
    }
    for title, anchors in (("before", before), ("after", after)):
        served, same, confirmed = anchor_spread(anchors)
        shared = sum(1 for count in served.values() if count >= 2)
        print(
            f"  {title}: anchors serving 2+ windows: {shared}"
            f", widest: {served.most_common(3)}"
            f", pairs agreed: {same}, pairs confirmed: {confirmed}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure application navigation in the graph.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--app", action="append", default=[])
    args = parser.parse_args(argv)

    repo = open_repo(args.db)
    print(
        f"quorum rule: >= {NAV_MIN_SHARE:.0%} of windows and at least {NAV_MIN_WINDOWS}"
        f", same place within {NAV_PLACE_TOLERANCE_PX} px"
    )
    wanted = args.app or [summary.app_name for summary in repo.app_summaries()]
    for app_name in wanted:
        report_app(repo, app_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
