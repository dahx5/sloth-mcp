from __future__ import annotations

from collections.abc import Sequence

from choto.graph import normalize_text
from choto.models import OcrLine

_MIN_ANCHOR_LEN = 3

_MIN_SHARED_ANCHORS = 2

_AGREEMENT_NUMERATOR = 2
_AGREEMENT_DENOMINATOR = 3


def select_anchors(lines: list[OcrLine], n: int = 5) -> list[str]:
    if n <= 0:
        return []

    candidates = [line for line in lines if len(line.text.strip()) >= _MIN_ANCHOR_LEN]
    candidates.sort(key=lambda line: (-line.bbox.h, -line.confidence, line.text))

    anchors: list[str] = []
    seen: set[str] = set()
    for line in candidates:
        key = line.text.strip()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(key)
        if len(anchors) >= n:
            break
    return anchors


def _shared(stored: Sequence[str], fresh: Sequence[str]) -> tuple[set[str], set[str]]:
    return (
        {normalize_text(text) for text in stored if text.strip()},
        {normalize_text(text) for text in fresh if text.strip()},
    )


# PR-005
def anchors_confirm(stored: Sequence[str], fresh: Sequence[str]) -> bool:
    stored_keys, fresh_keys = _shared(stored, fresh)
    if not stored_keys or not fresh_keys:
        return True
    required = min(_MIN_SHARED_ANCHORS, len(stored_keys), len(fresh_keys))
    return len(stored_keys & fresh_keys) >= required


# PR-004
def anchors_agree(stored: Sequence[str], fresh: Sequence[str]) -> bool:
    stored_keys, fresh_keys = _shared(stored, fresh)
    shared = len(stored_keys & fresh_keys)
    union = len(stored_keys | fresh_keys)
    if shared < _MIN_SHARED_ANCHORS:
        return False
    return shared * _AGREEMENT_DENOMINATOR >= union * _AGREEMENT_NUMERATOR
