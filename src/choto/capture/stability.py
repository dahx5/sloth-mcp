from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from choto.capture.framing import clamp_box
from choto.capture.overlaymask import aligned_images
from choto.capture.source import Frame, ScreenSource
from choto.config import Settings
from choto.log import get_logger
from choto.models import BBox

_log = get_logger(__name__)

_DIFF_WIDTH = 320
_DIFF_INTENSITY_THRESHOLD = 25

_CHANGE_MERGE_RADIUS = 2
_CHANGE_MIN_AREA = 4
_CHANGE_TOP_K = 4


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D frame, got shape {frame.shape}.")
    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Cannot diff an empty frame.")


def _to_small_gray(frame: np.ndarray) -> np.ndarray:
    _validate_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    target_h = max(1, round(h * _DIFF_WIDTH / w))
    return cv2.resize(gray, (_DIFF_WIDTH, target_h), interpolation=cv2.INTER_AREA)


def _to_change_gray(frame: np.ndarray) -> np.ndarray:
    _validate_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    while gray.shape[1] >= 2 * _DIFF_WIDTH:
        gray = cv2.pyrDown(gray)
    h, w = gray.shape[:2]
    target_h = max(1, round(h * _DIFF_WIDTH / w))
    return cv2.resize(gray, (_DIFF_WIDTH, target_h), interpolation=cv2.INTER_AREA)


def _diff_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ga = _to_small_gray(a)
    gb = _to_small_gray(b)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.absdiff(ga, gb) > _DIFF_INTENSITY_THRESHOLD


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = _diff_mask(a, b)
    return int(np.count_nonzero(mask)) / mask.size


def diff_frames(a: Frame, b: Frame) -> float:
    left, right = aligned_images(a, b)
    return frame_diff(left, right)


def changed_regions(a: np.ndarray, b: np.ndarray, *, top_k: int = _CHANGE_TOP_K) -> list[BBox]:
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    ga = _to_change_gray(a)
    gb = _to_change_gray(b)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    delta = cv2.absdiff(ga, gb)
    _, mask = cv2.threshold(delta, _DIFF_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
    side = 2 * _CHANGE_MERGE_RADIUS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (side, side))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= _CHANGE_MIN_AREA
    ]
    found.sort(key=lambda item: item[0], reverse=True)

    height, width = int(a.shape[0]), int(a.shape[1])
    scale_x = width / ga.shape[1]
    scale_y = height / ga.shape[0]
    boxes: list[BBox] = []
    for _, label in found[:top_k]:
        sx = int(stats[label, cv2.CC_STAT_LEFT])
        sy = int(stats[label, cv2.CC_STAT_TOP])
        sw = int(stats[label, cv2.CC_STAT_WIDTH])
        sh = int(stats[label, cv2.CC_STAT_HEIGHT])
        x0 = int(np.floor(sx * scale_x))
        y0 = int(np.floor(sy * scale_y))
        x1 = int(np.ceil((sx + sw) * scale_x))
        y1 = int(np.ceil((sy + sh) * scale_y))
        boxes.append(clamp_box(BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0), width, height))
    return boxes


def diff_split(a: Frame, b: Frame, column: BBox) -> tuple[float, float]:
    left, right = aligned_images(a, b)
    mask = _diff_mask(left, right)
    width = int(left.shape[1])
    columns = mask.shape[1]
    first = min(columns, max(0, int(np.floor(column.x * columns / width))))
    last = min(columns, max(first, int(np.ceil((column.x + column.w) * columns / width))))
    total = int(np.count_nonzero(mask))
    inside = int(np.count_nonzero(mask[:, first:last]))
    return total / mask.size, (total - inside) / mask.size


def acted_column(a: Frame, b: Frame, point: tuple[int, int]) -> BBox | None:
    left, right = aligned_images(a, b)
    height, width = int(left.shape[0]), int(left.shape[1])
    x = point[0] - a.box.x
    y = point[1] - a.box.y
    if not (0 <= x < width and 0 <= y < height):
        return None
    mask = _diff_mask(left, right)
    rows, columns = mask.shape
    gx = min(columns - 1, int(x * columns / width))
    gy = min(rows - 1, int(y * rows / height))
    if not mask[gy, gx]:
        return None
    occupied = mask.any(axis=0)
    first = gx
    while first > 0 and occupied[first - 1]:
        first -= 1
    last = gx
    while last + 1 < columns and occupied[last + 1]:
        last += 1
    x0 = int(np.floor(first * width / columns))
    x1 = int(np.ceil((last + 1) * width / columns))
    return clamp_box(BBox(x=x0, y=0, w=x1 - x0, h=height), width, height)


def acknowledgement(
    baseline: Frame,
    current: Frame,
    point: tuple[int, int],
    changed: float,
    settings: Settings,
) -> BBox | None:
    if changed < settings.stable_diff_threshold:
        return None
    column = acted_column(baseline, current, point)
    if column is None:
        return None
    if diff_split(baseline, current, column)[1] > settings.expect_screen_change_pixel_fraction:
        return None
    return column


def region_frame(frame: Frame, region: BBox | None) -> Frame:
    if region is None:
        return frame
    image = frame.image
    clamped = clamp_box(region, int(image.shape[1]), int(image.shape[0]))
    if clamped.w == 0 or clamped.h == 0:
        raise ValueError(
            f"Region {region.x},{region.y},{region.w},{region.h} lies outside the "
            f"{int(image.shape[1])}x{int(image.shape[0])} frame."
        )
    return Frame(
        image=np.ascontiguousarray(
            image[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w]
        ),
        box=BBox(x=frame.box.x + clamped.x, y=frame.box.y + clamped.y, w=clamped.w, h=clamped.h),
        captured_at_ms=frame.captured_at_ms,
        age_ms=frame.age_ms,
        overlay=frame.overlay,
    )


class SettleOutcome(str, Enum):
    SETTLED = "settled"
    QUIET = "quiet"
    ACKNOWLEDGED = "acknowledged"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class Settled:
    frame: Frame
    outcome: SettleOutcome
    elapsed_ms: int
    quiet_needed_ms: int = 0
    saw_motion: bool = False
    watched_whole_screen: bool = False

    def describe(self) -> str:
        if self.outcome is SettleOutcome.SETTLED:
            said = f"the transition finished after {self.elapsed_ms}ms"
        elif self.outcome is SettleOutcome.QUIET:
            said = f"nothing moved in the {self.elapsed_ms}ms after the action"
        elif self.outcome is SettleOutcome.ACKNOWLEDGED:
            said = (
                f"the control under the click answered but nothing outside it moved in "
                f"{self.elapsed_ms}ms"
            )
        elif self.saw_motion:
            said = (
                f"the screen was still moving {self.elapsed_ms}ms after the action, so the "
                "wait gave up on it settling"
            )
        else:
            said = (
                f"{self.elapsed_ms}ms after the action the screen had still not held "
                f"{self.quiet_needed_ms}ms of quiet, which is what it takes to call an "
                "unmoved screen settled"
            )
        if self.watched_whole_screen:
            said += "; watched the whole screen because the window lay outside the captured frame"
        return said


class _Verdict(str, Enum):
    MOVING = "moving"
    SETTLING = "settling"
    SETTLED = "settled"
    TRANSIENT = "transient"
    ACKNOWLEDGED = "acknowledged"
    ACK_UNDONE = "ack_undone"
    QUIET = "quiet"


@dataclass(frozen=True)
class _Poll:
    verdict: _Verdict
    previous: Frame
    motion_seen: bool
    consecutive_stable: int
    acked: BBox | None
    changed: float = 0.0


def _poll_verdict(
    baseline: Frame,
    previous: Frame,
    current: Frame,
    *,
    motion_seen: bool,
    consecutive_stable: int,
    acked: BBox | None,
    acted_at: tuple[int, int] | None,
    settings: Settings,
) -> _Poll:
    if motion_seen:
        stable = (
            consecutive_stable + 1
            if diff_frames(previous, current) < settings.stable_diff_threshold
            else 0
        )
        if stable < settings.stable_frames_required:
            return _Poll(_Verdict.SETTLING, current, True, stable, acked)
        changed = diff_frames(baseline, current)
        if changed > settings.expect_screen_change_pixel_fraction:
            # PR-017
            if acted_at is not None and acked is None:
                fresh = acknowledgement(baseline, current, acted_at, changed, settings)
                if fresh is not None:
                    return _Poll(_Verdict.ACKNOWLEDGED, current, False, 0, fresh, changed)
            return _Poll(_Verdict.SETTLED, current, True, stable, acked, changed)
        return _Poll(_Verdict.TRANSIENT, current, False, 0, acked, changed)

    if acked is None:
        since_baseline = outside = diff_frames(baseline, current)
    else:
        since_baseline, outside = diff_split(baseline, current, acked)
    if outside > settings.expect_screen_change_pixel_fraction:
        return _Poll(_Verdict.MOVING, current, True, consecutive_stable, acked)
    # PR-017
    if acked is not None and since_baseline <= settings.expect_screen_change_pixel_fraction:
        return _Poll(_Verdict.ACK_UNDONE, previous, False, consecutive_stable, None)
    return _Poll(_Verdict.QUIET, previous, False, consecutive_stable, acked)


@dataclass(frozen=True)
class _Watch:
    region: BBox | None
    off_frame: bool = False


def _watchable(frame: Frame, region: BBox | None) -> _Watch:
    if region is None:
        return _Watch(None)
    image = frame.image
    clamped = clamp_box(region, int(image.shape[1]), int(image.shape[0]))
    if clamped.w and clamped.h:
        return _Watch(region)
    _log.warning(
        "stability.region_off_frame",
        region=(region.x, region.y, region.w, region.h),
        frame=(int(image.shape[1]), int(image.shape[0])),
    )
    return _Watch(None, off_frame=True)


def wait_until_stable(
    source: ScreenSource,
    *,
    timeout_ms: int,
    settings: Settings,
    quiet_grace_ms: int,
    region: BBox | None = None,
    acted_at: tuple[int, int] | None = None,
    baseline: Frame | None = None,
    since_ms: int = 0,
) -> Settled:
    interval_s = settings.stable_poll_interval_ms / 1000.0
    started = time.monotonic()
    deadline = started + timeout_ms / 1000.0
    grace_left_s = max(0, quiet_grace_ms - since_ms) / 1000.0
    quiet_deadline = min(started + grace_left_s, deadline)
    ceiling_left_s = max(0, settings.settle_in_flight_ceiling_ms - since_ms) / 1000.0
    in_flight_deadline = deadline

    def elapsed() -> int:
        return since_ms + round((time.monotonic() - started) * 1000)

    last = source.latest()
    seen_at = last.captured_at_ms
    newest_age_ms = last.age_ms
    watch = _watchable(last, region)
    # PR-017
    baseline = region_frame(last if baseline is None else baseline, watch.region)
    previous = baseline
    motion_seen = False
    consecutive_stable = 0
    polls = 0
    repeats = 0
    transients = 0
    acked: BBox | None = None
    next_poll_at = started + interval_s

    while True:
        now = time.monotonic()
        sleep_until = min(next_poll_at, in_flight_deadline)
        if now < sleep_until:
            time.sleep(sleep_until - now)
        poll_started = time.monotonic()
        if poll_started >= in_flight_deadline:
            break
        next_poll_at = poll_started + interval_s
        polls += 1
        frame = source.latest()
        newest_age_ms = frame.age_ms
        if frame.captured_at_ms == seen_at:
            repeats += 1
            continue
        last = frame
        seen_at = frame.captured_at_ms
        current = region_frame(last, watch.region)

        poll = _poll_verdict(
            baseline,
            previous,
            current,
            motion_seen=motion_seen,
            consecutive_stable=consecutive_stable,
            acked=acked,
            acted_at=acted_at,
            settings=settings,
        )
        previous = poll.previous
        motion_seen = poll.motion_seen
        consecutive_stable = poll.consecutive_stable
        acked = poll.acked

        if poll.verdict is _Verdict.SETTLING or poll.verdict is _Verdict.MOVING:
            in_flight_deadline = min(started + ceiling_left_s, deadline)  # PR-017
            continue
        if acked is not None and poll.verdict is _Verdict.ACKNOWLEDGED:
            quiet_deadline = min(started + ceiling_left_s, deadline)  # PR-017
            in_flight_deadline = quiet_deadline
            _log.debug(
                "stability.acknowledged",
                elapsed_ms=elapsed(),
                polls=polls,
                changed=round(poll.changed, 6),
                column=(acked.x, acked.w),
            )
            continue
        if poll.verdict is _Verdict.SETTLED:
            elapsed_ms = elapsed()
            _log.debug(
                "stability.settled",
                elapsed_ms=elapsed_ms,
                polls=polls,
                repeats=repeats,
                transients=transients,
                age_ms=round(newest_age_ms),
            )
            return Settled(
                frame=last,
                outcome=SettleOutcome.SETTLED,
                elapsed_ms=elapsed_ms,
                watched_whole_screen=watch.off_frame,
            )
        if poll.verdict is _Verdict.TRANSIENT:
            transients += 1
            _log.debug(
                "stability.transient",
                elapsed_ms=elapsed(),
                polls=polls,
                transients=transients,
            )
            continue
        if poll.verdict is _Verdict.ACK_UNDONE:
            transients += 1
            quiet_deadline = min(started + grace_left_s, deadline)
            _log.debug(
                "stability.acknowledgement_undone",
                elapsed_ms=elapsed(),
                polls=polls,
                transients=transients,
            )
        if time.monotonic() >= quiet_deadline:
            elapsed_ms = elapsed()
            _log.debug(
                "stability.quiet",
                elapsed_ms=elapsed_ms,
                polls=polls,
                repeats=repeats,
                transients=transients,
                grace_ms=quiet_grace_ms,
                age_ms=round(newest_age_ms),
            )
            return Settled(
                frame=last,
                outcome=(SettleOutcome.ACKNOWLEDGED if acked is not None else SettleOutcome.QUIET),
                elapsed_ms=elapsed_ms,
                watched_whole_screen=watch.off_frame,
            )

    # PR-017
    elapsed_ms = elapsed()
    told = {
        "timeout_ms": timeout_ms,
        "elapsed_ms": elapsed_ms,
        "polls": polls,
        "repeats": repeats,
        "transients": transients,
        "motion_seen": motion_seen,
        "consecutive_stable": consecutive_stable,
        "required": settings.stable_frames_required,
        "age_ms": round(newest_age_ms),
    }
    if acked is None:
        outcome = SettleOutcome.TIMED_OUT
        _log.warning("stability.timed_out", **told)
    else:
        outcome = SettleOutcome.ACKNOWLEDGED
        _log.debug("stability.acknowledged_only", **told)
    return Settled(
        frame=last,
        outcome=outcome,
        elapsed_ms=elapsed_ms,
        quiet_needed_ms=quiet_grace_ms,
        saw_motion=motion_seen or transients > 0,
        watched_whole_screen=watch.off_frame,
    )
