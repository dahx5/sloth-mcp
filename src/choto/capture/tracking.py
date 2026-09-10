from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import imagehash
import numpy as np

from choto.log import get_logger
from choto.models import BBox
from choto.vision.imaging import phash_image

_log = get_logger(__name__)

_MATCH_SCORE_THRESHOLD = 0.75
_TEMPLATE_REFRESH_SCORE = 0.92
_TEMPLATE_REFRESH_INTERVAL = 4
_TEMPLATE_PHASH_MAX_DISTANCE = 20
_VELOCITY_SAMPLES = 5
_MIN_TEMPLATE_SIDE = 8
_MIN_TEMPLATE_CONTRAST = 3.0


class TrackStatus(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"


@dataclass(frozen=True)
class TrackedTarget:
    status: TrackStatus
    score: float
    bbox: BBox | None
    at_ms: float


@dataclass(frozen=True)
class Velocity:
    vx: float
    vy: float


def _require_rgb(image: np.ndarray, what: str) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{what} must be a numpy array, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{what} must be an (H, W, 3) RGB image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"{what} must be uint8, got dtype {image.dtype}.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{what} is empty (shape {image.shape}).")


def _phash_distance(left: str, right: str) -> int:
    return int(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right))


class TargetTracker:
    def __init__(self) -> None:
        self._template_gray: np.ndarray | None = None
        self._locked_phash: str | None = None
        self._size: tuple[int, int] | None = None
        self._history: deque[tuple[float, float, float]] = deque(maxlen=_VELOCITY_SAMPLES)
        self._last: TrackedTarget | None = None
        self._since_refresh = 0

    @property
    def last(self) -> TrackedTarget | None:
        return self._last

    def lock(self, frame: np.ndarray, bbox: BBox, *, at_ms: float) -> None:
        _require_rgb(frame, "frame")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if bbox.w < _MIN_TEMPLATE_SIDE or bbox.h < _MIN_TEMPLATE_SIDE:
            raise ValueError(
                f"Target {bbox.w}x{bbox.h} is smaller than the {_MIN_TEMPLATE_SIDE} px "
                "minimum template side."
            )
        if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.w > width or bbox.y + bbox.h > height:
            raise ValueError(
                f"Target {bbox.x},{bbox.y},{bbox.w},{bbox.h} is not fully inside the "
                f"{width}x{height} frame."
            )

        patch = np.ascontiguousarray(frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w])
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        contrast = float(gray.std())
        if contrast < _MIN_TEMPLATE_CONTRAST:
            raise ValueError(
                f"Target patch is too flat to track (grayscale stddev {contrast:.2f} < "
                f"{_MIN_TEMPLATE_CONTRAST})."
            )

        self._template_gray = gray
        self._locked_phash = phash_image(patch)
        self._size = (bbox.w, bbox.h)
        self._history = deque(maxlen=_VELOCITY_SAMPLES)
        self._history.append((float(at_ms), bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0))
        self._last = TrackedTarget(
            status=TrackStatus.TRACKED, score=1.0, bbox=bbox, at_ms=float(at_ms)
        )
        self._since_refresh = 0
        _log.debug(
            "tracking.locked",
            bbox=(bbox.x, bbox.y, bbox.w, bbox.h),
            contrast=round(contrast, 2),
        )

    def update(
        self,
        frame_region: np.ndarray,
        region_origin: tuple[int, int],
        *,
        at_ms: float,
    ) -> TrackedTarget:
        if self._template_gray is None or self._size is None or self._locked_phash is None:
            raise RuntimeError("Tracker has not been locked onto a target yet.")
        _require_rgb(frame_region, "frame_region")

        template_w, template_h = self._size
        region_h, region_w = int(frame_region.shape[0]), int(frame_region.shape[1])
        if region_w < template_w or region_h < template_h:
            raise ValueError(
                f"Search region {region_w}x{region_h} is smaller than the "
                f"{template_w}x{template_h} template."
            )
        origin_x, origin_y = int(region_origin[0]), int(region_origin[1])
        if origin_x < 0 or origin_y < 0:
            raise ValueError(f"Region origin {region_origin} must be non-negative.")
        sample_ms = float(at_ms)
        if self._history and sample_ms < self._history[-1][0]:
            raise ValueError(
                f"Timestamp {sample_ms} is before the previous sample {self._history[-1][0]}."
            )

        gray = cv2.cvtColor(frame_region, cv2.COLOR_RGB2GRAY)
        surface = cv2.matchTemplate(gray, self._template_gray, cv2.TM_CCOEFF_NORMED)
        np.nan_to_num(surface, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, peak = cv2.minMaxLoc(surface)
        score = float(score)

        if score < _MATCH_SCORE_THRESHOLD:
            self._last = TrackedTarget(
                status=TrackStatus.LOST, score=score, bbox=None, at_ms=sample_ms
            )
            _log.debug("tracking.lost", score=round(score, 3))
            return self._last

        bbox = BBox(
            x=origin_x + int(peak[0]),
            y=origin_y + int(peak[1]),
            w=template_w,
            h=template_h,
        )
        self._history.append((sample_ms, bbox.x + template_w / 2.0, bbox.y + template_h / 2.0))
        self._since_refresh += 1
        self._maybe_refresh_template(frame_region, peak, score)
        self._last = TrackedTarget(
            status=TrackStatus.TRACKED, score=score, bbox=bbox, at_ms=sample_ms
        )
        return self._last

    def _maybe_refresh_template(
        self, frame_region: np.ndarray, peak: tuple[int, int], score: float
    ) -> None:
        if score < _TEMPLATE_REFRESH_SCORE or self._since_refresh < _TEMPLATE_REFRESH_INTERVAL:
            return
        if self._size is None or self._locked_phash is None:
            return
        template_w, template_h = self._size
        self._since_refresh = 0
        patch = np.ascontiguousarray(
            frame_region[peak[1] : peak[1] + template_h, peak[0] : peak[0] + template_w]
        )
        candidate_phash = phash_image(patch)
        distance = _phash_distance(candidate_phash, self._locked_phash)
        if distance > _TEMPLATE_PHASH_MAX_DISTANCE:
            _log.debug(
                "tracking.template_refresh_refused",
                phash_distance=distance,
                score=round(score, 3),
            )
            return
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        if float(gray.std()) < _MIN_TEMPLATE_CONTRAST:
            _log.debug("tracking.template_refresh_refused", reason="flat")
            return
        self._template_gray = gray

    def velocity(self) -> Velocity | None:
        if len(self._history) < 2:
            return None
        times = np.fromiter((sample[0] for sample in self._history), dtype=np.float64)
        spread = float(((times - times.mean()) ** 2).sum())
        if spread <= 0.0:
            return None
        xs = np.fromiter((sample[1] for sample in self._history), dtype=np.float64)
        ys = np.fromiter((sample[2] for sample in self._history), dtype=np.float64)
        centered_t = times - times.mean()
        vx = float((centered_t * (xs - xs.mean())).sum() / spread)
        vy = float((centered_t * (ys - ys.mean())).sum() / spread)
        return Velocity(vx=vx, vy=vy)

    def predict(self, dt_ms: float) -> BBox | None:
        if dt_ms < 0:
            raise ValueError(f"dt_ms must not be negative, got {dt_ms}.")
        velocity = self.velocity()
        if velocity is None or self._size is None:
            return None
        _, cx, cy = self._history[-1]
        template_w, template_h = self._size
        return BBox(
            x=int(round(cx + velocity.vx * dt_ms - template_w / 2.0)),
            y=int(round(cy + velocity.vy * dt_ms - template_h / 2.0)),
            w=template_w,
            h=template_h,
        )
