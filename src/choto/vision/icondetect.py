from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from choto.graph import glyph_key, hamming_hex
from choto.log import get_logger
from choto.models import BBox
from choto.vision.glyphkey import background_colour, glyph_parts
from choto.vision.iconmodel import load_icon_model

_log = get_logger(__name__)

_LETTERBOX_GREY = 114

_MIN_ANALYSABLE_SIDE = 3

_MIN_KEYABLE_SIDE = 4


@dataclass(frozen=True)
class IconBox:
    bbox: BBox
    score: float


@dataclass(frozen=True, eq=False)
class IconCandidate:
    bbox: BBox
    phash: str
    crop: np.ndarray


@dataclass(frozen=True)
class IconDetectorParams:
    scale: int = 1
    confidence: float = 0.05
    nms_iou: float = 0.1
    max_side: int = 160
    text_cover: float = 0.40
    text_aspect: float = 2.0
    ink_threshold: float = 28.0
    phash_max_hamming: int = 4

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError(f"scale must be at least 1, got {self.scale}.")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}.")
        if not 0.0 < self.nms_iou <= 1.0:
            raise ValueError(f"nms_iou must be in (0, 1], got {self.nms_iou}.")
        if self.max_side < 1:
            raise ValueError(f"max_side must be at least 1 point, got {self.max_side}.")
        if not 0.0 < self.text_cover <= 1.0:
            raise ValueError(f"text_cover must be in (0, 1], got {self.text_cover}.")
        if self.text_aspect < 1.0:
            raise ValueError(
                f"text_aspect must be at least 1 (a box wider than it is tall), got "
                f"{self.text_aspect}."
            )
        if self.ink_threshold <= 0:
            raise ValueError(f"ink_threshold must be positive, got {self.ink_threshold}.")
        if self.phash_max_hamming < 0:
            raise ValueError(
                f"phash_max_hamming must not be negative, got {self.phash_max_hamming}."
            )

    @property
    def max_side_px(self) -> int:
        return self.max_side * self.scale


DEFAULT_PARAMS = IconDetectorParams()


def detect_icon_boxes(
    frame: np.ndarray, window: BBox, params: IconDetectorParams = DEFAULT_PARAMS
) -> list[IconBox]:
    _validate_frame(frame)
    view_box = _clip(window, frame.shape[1], frame.shape[0])
    if view_box.w < _MIN_ANALYSABLE_SIDE or view_box.h < _MIN_ANALYSABLE_SIDE:
        _log.debug(
            "icons.window_too_small",
            window=(view_box.x, view_box.y, view_box.w, view_box.h),
        )
        return []

    model = load_icon_model()
    if model is None:
        return []

    view = np.ascontiguousarray(
        frame[view_box.y : view_box.y + view_box.h, view_box.x : view_box.x + view_box.w]
    )
    letterboxed, ratio = _letterbox(view, model.side)
    head = model.predict(letterboxed)

    picked = head[4] >= params.confidence
    scores = head[4, picked]
    corners = _corners(head[:4, picked] / ratio)
    kept = _suppress_overlaps(corners, scores, params.nms_iou)

    boxes: list[IconBox] = []
    for index in kept:
        box = _clip(_to_bbox(corners[index]), view_box.w, view_box.h)
        if box.w <= 0 or box.h <= 0 or max(box.w, box.h) > params.max_side_px:
            continue
        if _is_window_corner(box, view_box):
            continue
        placed = BBox(x=view_box.x + box.x, y=view_box.y + box.y, w=box.w, h=box.h)
        boxes.append(IconBox(bbox=placed, score=float(scores[index])))
    boxes.sort(key=lambda item: (item.bbox.y, item.bbox.x))
    _log.debug(
        "icons.boxes",
        window=(view_box.x, view_box.y, view_box.w, view_box.h),
        proposed=int(picked.sum()),
        after_suppression=len(kept),
        kept=len(boxes),
    )
    return boxes


def icon_candidates(
    frame: np.ndarray,
    boxes: Sequence[IconBox],
    text_boxes: Sequence[BBox],
    params: IconDetectorParams = DEFAULT_PARAMS,
) -> list[IconCandidate]:
    _validate_frame(frame)
    lines = [box for box in text_boxes if box.w > 0 and box.h > 0]
    kept = [box for box in boxes if not _reads_as_text(box.bbox, lines, params)]

    clusters = _GlyphClusters(params)
    candidates: list[IconCandidate] = []
    for box in kept:
        refined = _refine_to_ink(frame, box.bbox, params)
        if refined is None:
            continue
        crop = frame[refined.y : refined.y + refined.h, refined.x : refined.x + refined.w].copy()
        digest, colour = clusters.of(crop)
        candidates.append(IconCandidate(bbox=refined, phash=glyph_key(digest, colour), crop=crop))

    _log.debug(
        "icons.candidates",
        proposed=len(boxes),
        text_duplicates=len(boxes) - len(kept),
        kept=len(candidates),
        distinct_phashes=len({candidate.phash for candidate in candidates}),
    )
    return candidates


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected a numpy array, got {type(frame).__name__}.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB frame, got shape {frame.shape}.")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 frame, got dtype {frame.dtype}.")


def _clip(box: BBox, width: int, height: int) -> BBox:
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    return BBox(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))


def _letterbox(view: np.ndarray, side: int) -> tuple[Image.Image, float]:
    height, width = view.shape[:2]
    ratio = min(side / width, side / height)
    scaled_w, scaled_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    canvas = np.full((side, side, 3), _LETTERBOX_GREY, dtype=np.uint8)
    canvas[:scaled_h, :scaled_w] = cv2.resize(
        view, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
    )
    return Image.fromarray(canvas), ratio


def _corners(centred: np.ndarray) -> np.ndarray:
    centre_x, centre_y, width, height = centred
    return np.stack(
        [
            centre_x - width / 2,
            centre_y - height / 2,
            centre_x + width / 2,
            centre_y + height / 2,
        ],
        axis=1,
    )


def _to_bbox(corners: np.ndarray) -> BBox:
    x0, y0, x1, y1 = (int(round(float(value))) for value in corners)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _suppress_overlaps(corners: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = np.argsort(-scores)
    x0, y0, x1, y1 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    kept: list[int] = []
    while order.size:
        best = int(order[0])
        kept.append(best)
        rest = order[1:]
        inter_w = np.clip(np.minimum(x1[best], x1[rest]) - np.maximum(x0[best], x0[rest]), 0, None)
        inter_h = np.clip(np.minimum(y1[best], y1[rest]) - np.maximum(y0[best], y0[rest]), 0, None)
        intersection = inter_w * inter_h
        union = areas[best] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlap <= iou]
    return kept


def _is_window_corner(box: BBox, view: BBox) -> bool:
    left, top = box.x <= 0, box.y <= 0
    right, bottom = box.x + box.w >= view.w, box.y + box.h >= view.h
    return (left or right) and (top or bottom)


def _reads_as_text(box: BBox, text_boxes: Sequence[BBox], params: IconDetectorParams) -> bool:
    if box.h <= 0 or box.w / box.h < params.text_aspect:
        return False
    covered = np.zeros((box.h, box.w), dtype=bool)
    for line in text_boxes:
        overlap = _clip(BBox(x=line.x - box.x, y=line.y - box.y, w=line.w, h=line.h), box.w, box.h)
        if overlap.w and overlap.h:
            covered[overlap.y : overlap.y + overlap.h, overlap.x : overlap.x + overlap.w] = True
    return float(covered.mean()) >= params.text_cover


def _refine_to_ink(frame: np.ndarray, box: BBox, params: IconDetectorParams) -> BBox | None:
    box = _clip(box, frame.shape[1], frame.shape[0])
    if box.w < _MIN_ANALYSABLE_SIDE or box.h < _MIN_ANALYSABLE_SIDE:
        return None
    patch = frame[box.y : box.y + box.h, box.x : box.x + box.w].astype(np.float32)
    ink = np.linalg.norm(patch - background_colour(patch), axis=2) > params.ink_threshold
    rows, columns = np.nonzero(ink)
    if columns.size == 0:
        return None
    x0, y0 = int(columns.min()), int(rows.min())
    x1, y1 = int(columns.max()), int(rows.max())
    refined = BBox(x=box.x + x0, y=box.y + y0, w=x1 - x0 + 1, h=y1 - y0 + 1)
    if min(refined.w, refined.h) < _MIN_KEYABLE_SIDE:
        return None
    return refined


class _GlyphClusters:
    def __init__(self, params: IconDetectorParams) -> None:
        self._params = params
        self._representatives: list[tuple[str, int]] = []

    def of(self, crop: np.ndarray) -> tuple[str, int]:
        digest, colour = glyph_parts(crop)
        shared = next(
            (
                rep
                for rep in self._representatives
                if rep[1] == colour
                and hamming_hex(digest, rep[0]) <= self._params.phash_max_hamming
            ),
            None,
        )
        if shared is None:
            shared = (digest, colour)
            self._representatives.append(shared)
        return shared
