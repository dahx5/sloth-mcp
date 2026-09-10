from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from model2vec import StaticModel
from rapidfuzz import fuzz

from choto.config import Settings
from choto.graph.repository import normalize_text
from choto.log import get_logger
from choto.matching.embeddings import load_semantic_model
from choto.matching.polarity import OppositePoles, polarity_guard
from choto.models import BBox, OcrLine, Region, Target

_log = get_logger(__name__)

TIER_EXACT = "exact"
TIER_FUZZY = "fuzzy"
TIER_SEMANTIC = "semantic"

_VETO_LOG_LINES = 5

_CONTAINMENT_SCORE = 0.9

MIN_PARTIAL_MATCH_TARGET_CHARS = 3

MIN_APPROXIMATE_MATCH_LENGTH_RATIO = 0.6

MAX_CONTAINMENT_EXTRA_CHARS = 20

MIN_DIVERGENT_TOKEN_SIMILARITY = 0.6


@dataclass(frozen=True)
class MatchResult:
    line: OcrLine
    score: float
    tier: str
    opposed: OppositePoles | None = None


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def region_bounds(region: Region, screen_w: int, screen_h: int) -> BBox:
    if region is Region.TOP:
        return BBox(x=0, y=0, w=screen_w, h=_ceil_div(screen_h, 2))
    if region is Region.BOTTOM:
        top = _ceil_div(screen_h, 2)
        return BBox(x=0, y=top, w=screen_w, h=max(screen_h - top, 0))
    if region is Region.LEFT:
        return BBox(x=0, y=0, w=_ceil_div(screen_w, 2), h=screen_h)
    if region is Region.RIGHT:
        left = _ceil_div(screen_w, 2)
        return BBox(x=left, y=0, w=max(screen_w - left, 0), h=screen_h)
    x0 = _ceil_div(screen_w, 3)
    y0 = _ceil_div(screen_h, 3)
    return BBox(
        x=x0,
        y=y0,
        w=max(_ceil_div(2 * screen_w, 3) - x0, 0),
        h=max(_ceil_div(2 * screen_h, 3) - y0, 0),
    )


def _partial_match_allowed(norm_target: str) -> bool:
    return len(norm_target) >= MIN_PARTIAL_MATCH_TARGET_CHARS


def _candidate_long_enough(norm_target: str, norm_line: str) -> bool:
    if not norm_target or not norm_line:
        return False
    return len(norm_line) / len(norm_target) >= MIN_APPROXIMATE_MATCH_LENGTH_RATIO


def _contains_target(norm_target: str, norm_line: str) -> bool:
    if not norm_target or norm_target not in norm_line:
        return False
    return len(norm_line) - len(norm_target) <= MAX_CONTAINMENT_EXTRA_CHARS


def _fuzzy_score(norm_target: str, norm_line: str) -> float:
    return max(fuzz.ratio(norm_target, norm_line), fuzz.token_sort_ratio(norm_target, norm_line))


def _divergent_texts(target_tokens: list[str], line_tokens: list[str]) -> tuple[str, str] | None:
    shared = set(target_tokens) & set(line_tokens)
    if not shared:
        return None
    left = [token for token in target_tokens if token not in shared]
    right = [token for token in line_tokens if token not in shared]
    if not left or not right:
        return None
    return (" ".join(left), " ".join(right))


def _sort_key(scored: tuple[OcrLine, float, int]) -> tuple[float, int, int, int]:
    line, score, extra = scored
    return (-score, extra, line.bbox.y, line.bbox.x)


class TargetMatcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def resolve(
        self,
        target: Target,
        lines: list[OcrLine],
        screen_w: int,
        screen_h: int,
    ) -> MatchResult | None:
        candidates = self._filter_region(target.region, lines, screen_w, screen_h)
        if not candidates:
            return None

        norm_target = normalize_text(target.text)
        partial = _partial_match_allowed(norm_target)
        candidates = self._allowed(target.text, norm_target, candidates)
        if not candidates:
            return None

        result = self._match_exact(norm_target, candidates, allow_containment=partial)
        if result is not None or not partial:
            return result

        result = self._match_fuzzy(norm_target, candidates)
        if result is not None:
            return result

        return self._match_semantic(target.text, candidates)

    def rank(
        self,
        target: Target,
        lines: list[OcrLine],
        screen_w: int,
        screen_h: int,
        limit: int,
    ) -> list[MatchResult]:
        if limit <= 0:
            return []
        candidates = self._filter_region(target.region, lines, screen_w, screen_h)
        if not candidates:
            return []

        norm_target = normalize_text(target.text)
        scores = [self._best_lexical_score(norm_target, line) for line in candidates]

        if _partial_match_allowed(norm_target):
            for index, semantic in enumerate(self._semantic_scores(target.text, candidates)):
                if semantic > scores[index][0]:
                    scores[index] = (semantic, TIER_SEMANTIC, 0)

        guard = polarity_guard(norm_target)
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: _sort_key((pair[0], pair[1][0], pair[1][2])),
        )
        return [
            MatchResult(
                line=line,
                score=score,
                tier=tier,
                opposed=guard.opposes(normalize_text(line.text)),
            )
            for line, (score, tier, _) in ordered[:limit]
        ]

    def count(
        self,
        target: Target,
        lines: list[OcrLine],
        screen_w: int,
        screen_h: int,
    ) -> int:
        candidates = self._filter_region(target.region, lines, screen_w, screen_h)
        if not candidates:
            return 0

        norm_target = normalize_text(target.text)
        partial = _partial_match_allowed(norm_target)
        candidates = self._allowed(target.text, norm_target, candidates)
        if not candidates:
            return 0

        fuzzy_threshold = self._settings.fuzzy_match_threshold
        matched = 0
        undecided: list[OcrLine] = []
        for line in candidates:
            norm_line = normalize_text(line.text)
            if norm_line == norm_target or (partial and _contains_target(norm_target, norm_line)):
                matched += 1
                continue
            if not partial:
                continue
            if (
                _candidate_long_enough(norm_target, norm_line)
                and _fuzzy_score(norm_target, norm_line) >= fuzzy_threshold
            ):
                matched += 1
                continue
            undecided.append(line)

        if not undecided:
            return matched

        semantic_threshold = self._settings.semantic_match_threshold
        matched += sum(
            1
            for score in self._semantic_scores(target.text, undecided)
            if score > 0.0 and score >= semantic_threshold
        )
        return matched

    def _allowed(
        self, target_text: str, norm_target: str, candidates: list[OcrLine]
    ) -> list[OcrLine]:
        guard = polarity_guard(norm_target)
        if not guard.watching:
            return candidates
        kept: list[OcrLine] = []
        refused: list[str] = []
        for line in candidates:
            conflict = guard.opposes(normalize_text(line.text))
            if conflict is None:
                kept.append(line)
            else:
                refused.append(f"{line.text} ({conflict.candidate_word}/{conflict.target_word})")
        if refused:
            _log.info(
                "match.polarity_veto",
                target=target_text,
                refused=len(refused),
                lines=refused[:_VETO_LOG_LINES],
            )
        return kept

    @staticmethod
    def _best_lexical_score(norm_target: str, line: OcrLine) -> tuple[float, str, int]:
        norm_line = normalize_text(line.text)
        if norm_line == norm_target:
            return (1.0, TIER_EXACT, 0)
        if not _partial_match_allowed(norm_target):
            return (0.0, TIER_EXACT, 0)
        if _contains_target(norm_target, norm_line):
            return (_CONTAINMENT_SCORE, TIER_EXACT, len(norm_line) - len(norm_target))
        if not _candidate_long_enough(norm_target, norm_line):
            return (0.0, TIER_FUZZY, 0)
        return (_fuzzy_score(norm_target, norm_line) / 100.0, TIER_FUZZY, 0)

    def _semantic_scores(self, target_text: str, candidates: list[OcrLine]) -> list[float]:
        norm_target = normalize_text(target_text)
        scores = [0.0] * len(candidates)
        eligible = [
            index
            for index, line in enumerate(candidates)
            if _candidate_long_enough(norm_target, normalize_text(line.text))
        ]
        if not eligible:
            return scores

        model = self._get_model()
        if model is None:
            return scores

        target_tokens = norm_target.split()
        texts = [target_text] + [candidates[index].text for index in eligible]
        divergent_at: dict[int, int] = {}
        for index in eligible:
            divergent = _divergent_texts(
                target_tokens, normalize_text(candidates[index].text).split()
            )
            if divergent is not None:
                divergent_at[index] = len(texts)
                texts.extend(divergent)

        embeddings = np.asarray(model.encode(texts), dtype=np.float32)
        sims = _cosine_similarity(embeddings[0], embeddings[1 : 1 + len(eligible)])
        for index, sim in zip(eligible, sims, strict=True):
            start = divergent_at.get(index)
            if start is not None:
                divergence = float(
                    _cosine_similarity(embeddings[start], embeddings[start + 1 : start + 2])[0]
                )
                if divergence < MIN_DIVERGENT_TOKEN_SIMILARITY:
                    continue
            scores[index] = float(sim)
        return scores

    def _match_exact(
        self, norm_target: str, candidates: list[OcrLine], *, allow_containment: bool = True
    ) -> MatchResult | None:
        scored: list[tuple[OcrLine, float, int]] = []
        for line in candidates:
            norm_line = normalize_text(line.text)
            if norm_line == norm_target:
                scored.append((line, 1.0, 0))
            elif allow_containment and _contains_target(norm_target, norm_line):
                scored.append((line, _CONTAINMENT_SCORE, len(norm_line) - len(norm_target)))
        if not scored:
            return None
        line, score, _ = min(scored, key=_sort_key)
        return MatchResult(line=line, score=score, tier=TIER_EXACT)

    def _match_fuzzy(self, norm_target: str, candidates: list[OcrLine]) -> MatchResult | None:
        threshold = self._settings.fuzzy_match_threshold
        scored: list[tuple[OcrLine, float, int]] = []
        for line in candidates:
            norm_line = normalize_text(line.text)
            if not _candidate_long_enough(norm_target, norm_line):
                continue
            raw = _fuzzy_score(norm_target, norm_line)
            if raw >= threshold:
                scored.append((line, raw / 100.0, 0))
        if not scored:
            return None
        line, score, _ = min(scored, key=_sort_key)
        return MatchResult(line=line, score=score, tier=TIER_FUZZY)

    def _match_semantic(self, target_text: str, candidates: list[OcrLine]) -> MatchResult | None:
        scores = self._semantic_scores(target_text, candidates)
        threshold = self._settings.semantic_match_threshold
        scored: list[tuple[OcrLine, float, int]] = [
            (line, score, 0)
            for line, score in zip(candidates, scores, strict=True)
            if score > 0.0 and score >= threshold
        ]
        if not scored:
            return None
        line, score, _ = min(scored, key=_sort_key)
        return MatchResult(line=line, score=score, tier=TIER_SEMANTIC)

    @staticmethod
    def _filter_region(
        region: Region | None,
        lines: list[OcrLine],
        screen_w: int,
        screen_h: int,
    ) -> list[OcrLine]:
        if region is None:
            return list(lines)

        bounds = region_bounds(region, screen_w, screen_h)

        def keep(line: OcrLine) -> bool:
            cx, cy = line.bbox.center
            return bounds.x <= cx < bounds.x + bounds.w and bounds.y <= cy < bounds.y + bounds.h

        return [line for line in lines if keep(line)]

    def _get_model(self) -> StaticModel | None:
        return load_semantic_model()


def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = float(np.linalg.norm(vec))
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * vec_norm
    dots = matrix @ vec
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, dots / denom, 0.0)
    return sims
