from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

BUTTON_RADII_POINTS: tuple[int, ...] = (5, 6, 7, 8, 9)

INTERIOR_CONTRAST_MIN = 18.0
INTERIOR_SPREAD_MAX = 16.0

BUTTON_CONTRAST_MIN = 24.0
BUTTON_SPREAD_MAX = 12.0
SURROUND_SPREAD_MAX = 22.0

RAY_COUNT = 16
RAY_STEP_PIXELS = 0.5

EDGE_JITTER_FLOOR_PIXELS = 0.45
EDGE_JITTER_FRACTION = 0.06
EDGE_BIAS_MAX_POINTS = 2.5
DISC_INSET_POINTS = 1.5
SURROUND_INNER_POINTS = 2.5
SURROUND_OUTER_POINTS = 5.0
DISC_MERGE_POINTS = 4.0

BUTTON_PITCH_MIN_POINTS = 15.0
BUTTON_PITCH_MAX_POINTS = 30.0
BUTTON_PITCH_TOLERANCE = 0.22
BUTTON_ROW_TOLERANCE_POINTS = 3.0
BUTTON_RADIUS_AGREEMENT_POINTS = 1.5
ANCHOR_MERGE_POINTS = 8.0

BACKGROUND_TOLERANCE = 26.0

COLOURED_SPREAD_MIN = 60.0


SIDE_STEP_SPAN_POINTS = 2
SIDE_STEP_RGB = 20.0
TOP_STEP_SPAN_POINTS = 4
TOP_STEP_RGB = 6.0

EDGE_GAP_POINTS = 6.0

SIDE_SUPPORT_MIN = 0.5
TOP_SUPPORT_MIN = 0.5

TITLEBAR_HEIGHT_MAX_POINTS = 64.0
TITLEBAR_TOP_CLEARANCE_POINTS = 4.0

BUTTONS_INSET_MAX_POINTS = 80.0

MAX_REFINEMENT_POINTS = 16.0
SEARCH_MARGIN_POINTS = 48.0


def _to_pixels(points: float, scale: float) -> int:
    return int(round(points * scale))


def _span_pixels(points: float, scale: float) -> int:
    return max(1, _to_pixels(points, scale))


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}.")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(
                f"A rectangle's edges must be ordered, got "
                f"({self.left}, {self.top}, {self.right}, {self.bottom})."
            )

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def edges(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class TitlebarAnchor:
    buttons: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    radius: float
    pitch: float
    background: tuple[float, float, float]
    coloured: bool
    contrast: float

    @property
    def leftmost(self) -> tuple[int, int]:
        return self.buttons[0]


@dataclass(frozen=True)
class RefinedRect:
    rect: Rect
    anchor: TitlebarAnchor
    shifts: tuple[int, int, int, int]
    side_support: tuple[float, float]
    top_support: float


class RefusalCode(str, Enum):
    NO_ANCHOR = "no_anchor"
    ANCHOR_OUTSIDE = "anchor_outside"
    NO_SIDE_EDGE = "no_side_edge"
    NO_TOP_EDGE = "no_top_edge"
    NO_BOTTOM_EDGE = "no_bottom_edge"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class Refusal:
    code: RefusalCode
    detail: str


@dataclass(frozen=True)
class _Gauge:
    radius: int
    scale: float

    @property
    def reach(self) -> int:
        return self.radius + math.ceil(SURROUND_OUTER_POINTS * self.scale)


@dataclass(frozen=True)
class _Peaks:
    centres: np.ndarray
    interior: np.ndarray
    contrast: np.ndarray
    offset: int

    def at(self, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.clip(centres[:, 1] - self.offset, 0, self.interior.shape[0] - 1)
        cols = np.clip(centres[:, 0] - self.offset, 0, self.interior.shape[1] - 1)
        return self.interior[rows, cols], self.contrast[rows, cols]


_NO_PEAKS = _Peaks(
    centres=np.zeros((0, 2), np.int32),
    interior=np.zeros((0, 0), np.float32),
    contrast=np.zeros((0, 0), np.float32),
    offset=0,
)

_RAY_ANGLES = np.arange(RAY_COUNT) * (2 * np.pi / RAY_COUNT)


@dataclass(frozen=True)
class _Disc:
    x: int
    y: int
    radius: float
    contrast: float
    colour: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class _Edges:
    rect: Rect
    side_support: tuple[float, float]
    top_support: float


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be a height x width x 3 RGB array, got shape {frame.shape}.")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(f"frame must have a positive size, got shape {frame.shape}.")


def _validate_scale(scale: float) -> None:
    if not isinstance(scale, int | float) or isinstance(scale, bool):
        raise TypeError(f"scale must be a number, got {type(scale).__name__}.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}.")


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)))


def _box_sums(table: np.ndarray, side: int) -> np.ndarray:
    return table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]


def _disc_peaks(grey: np.ndarray, radius: int) -> _Peaks:
    interior_side = max(3, int(round(1.2 * radius)) | 1)
    surround_side = 2 * radius + 9
    if min(grey.shape) < surround_side + 2:
        return _NO_PEAKS

    table, table_squared = _integral(grey), _integral(grey * grey)
    interior_sum = _box_sums(table, interior_side)
    interior_squared = _box_sums(table_squared, interior_side)
    surround_sum = _box_sums(table, surround_side)
    interior_count = interior_side * interior_side
    surround_count = surround_side * surround_side

    interior_mean = interior_sum / interior_count
    interior_spread = np.sqrt(
        np.maximum(0.0, interior_squared / interior_count - interior_mean * interior_mean)
    )
    pad = (surround_side - interior_side) // 2
    rows = min(interior_mean.shape[0] - pad, surround_sum.shape[0])
    cols = min(interior_mean.shape[1] - pad, surround_sum.shape[1])
    if rows < 1 or cols < 1:
        return _NO_PEAKS
    interior = interior_mean[pad : pad + rows, pad : pad + cols].astype(np.float32)
    spread = interior_spread[pad : pad + rows, pad : pad + cols]
    surround = (
        (surround_sum[:rows, :cols] - interior_sum[pad : pad + rows, pad : pad + cols])
        / (surround_count - interior_count)
    ).astype(np.float32)
    contrast = np.abs(interior - surround)

    accepted = (contrast >= INTERIOR_CONTRAST_MIN) & (spread <= INTERIOR_SPREAD_MAX)
    window = max(3, radius | 1)
    accepted &= contrast >= cv2.dilate(contrast, np.ones((window, window), np.uint8))
    ys, xs = np.nonzero(accepted)
    offset = pad + interior_side // 2
    centres = np.stack([xs + offset, ys + offset], axis=1).astype(np.int32)
    return _Peaks(centres=centres, interior=interior, contrast=contrast, offset=offset)


def _ray_radii(
    grey: np.ndarray,
    centres: np.ndarray,
    interior: np.ndarray,
    contrast: np.ndarray,
    reach: int,
) -> np.ndarray:
    steps = np.arange(1.0, reach + 0.01, RAY_STEP_PIXELS)
    offsets_y = np.rint(np.sin(_RAY_ANGLES)[:, None] * steps[None, :]).astype(int)
    offsets_x = np.rint(np.cos(_RAY_ANGLES)[:, None] * steps[None, :]).astype(int)
    samples = grey[
        centres[:, 1][:, None, None] + offsets_y[None],
        centres[:, 0][:, None, None] + offsets_x[None],
    ]
    left_interior = np.abs(samples - interior[:, None, None]) > 0.5 * contrast[:, None, None]
    first = np.where(left_interior.any(axis=2), left_interior.argmax(axis=2), len(steps) - 1)
    return steps[first]


def _inside_frame(centres: np.ndarray, shape: tuple[int, int], margin: int) -> np.ndarray:
    height, width = shape
    return (
        (centres[:, 0] >= margin)
        & (centres[:, 0] < width - margin)
        & (centres[:, 1] >= margin)
        & (centres[:, 1] < height - margin)
    )


def _recentred(grey: np.ndarray, peaks: _Peaks, centres: np.ndarray, gauge: _Gauge) -> np.ndarray:
    interior, contrast = peaks.at(centres)
    radii = _ray_radii(grey, centres, interior, contrast, gauge.reach)
    shift_x = 2.0 / RAY_COUNT * (radii * np.cos(_RAY_ANGLES)[None, :]).sum(axis=1)
    shift_y = 2.0 / RAY_COUNT * (radii * np.sin(_RAY_ANGLES)[None, :]).sum(axis=1)
    moved = centres + np.stack([np.rint(shift_x), np.rint(shift_y)], axis=1).astype(np.int32)
    too_far = (np.abs(shift_x) > gauge.radius) | (np.abs(shift_y) > gauge.radius)
    return moved[_inside_frame(moved, grey.shape, gauge.reach + 2) & ~too_far]


def _round_enough(
    grey: np.ndarray, peaks: _Peaks, centres: np.ndarray, gauge: _Gauge
) -> tuple[np.ndarray, np.ndarray]:
    interior, contrast = peaks.at(centres)
    keep = contrast >= INTERIOR_CONTRAST_MIN
    centres, interior, contrast = centres[keep], interior[keep], contrast[keep]
    if len(centres) == 0:
        return centres, np.zeros(0, dtype=np.float32)
    radii = _ray_radii(grey, centres, interior, contrast, gauge.reach)
    measured = radii.mean(axis=1)
    allowed_jitter = EDGE_JITTER_FLOOR_PIXELS + EDGE_JITTER_FRACTION * measured
    circular = (radii.std(axis=1) <= allowed_jitter) & (
        np.abs(measured - gauge.radius) <= EDGE_BIAS_MAX_POINTS * gauge.scale
    )
    return centres[circular], measured[circular]


def _solid_discs(
    frame: np.ndarray, centres: np.ndarray, measured: np.ndarray, gauge: _Gauge
) -> list[_Disc]:
    reach = gauge.reach
    grid_y, grid_x = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    distance = np.hypot(grid_y, grid_x)
    disc = distance <= gauge.radius - DISC_INSET_POINTS * gauge.scale
    annulus = (distance >= gauge.radius + SURROUND_INNER_POINTS * gauge.scale) & (
        distance <= gauge.radius + SURROUND_OUTER_POINTS * gauge.scale
    )
    if not disc.any() or not annulus.any():
        return []
    patches = frame[
        centres[:, 1][:, None, None] + grid_y[None],
        centres[:, 0][:, None, None] + grid_x[None],
    ].astype(np.float32)
    disc_colour = patches[:, disc].mean(axis=1)
    disc_spread = patches[:, disc].std(axis=1)
    ring_colour = patches[:, annulus].mean(axis=1)
    ring_spread = patches[:, annulus].std(axis=1)
    colour_contrast = np.linalg.norm(disc_colour - ring_colour, axis=1)
    solid = (
        (colour_contrast >= BUTTON_CONTRAST_MIN)
        & (disc_spread.max(axis=1) <= BUTTON_SPREAD_MAX)
        & (ring_spread.max(axis=1) <= SURROUND_SPREAD_MAX)
    )
    return [
        _Disc(
            x=int(centres[i, 0]),
            y=int(centres[i, 1]),
            radius=float(measured[i]),
            contrast=float(colour_contrast[i]),
            colour=disc_colour[i],
            background=ring_colour[i],
        )
        for i in np.nonzero(solid)[0]
    ]


def _verify_discs(frame: np.ndarray, grey: np.ndarray, gauge: _Gauge) -> list[_Disc]:
    peaks = _disc_peaks(grey, gauge.radius)
    centres = peaks.centres[_inside_frame(peaks.centres, grey.shape, gauge.reach + 2)]
    if len(centres) == 0:
        return []
    centres = _recentred(grey, peaks, centres, gauge)
    if len(centres) == 0:
        return []
    centres, measured = _round_enough(grey, peaks, centres, gauge)
    if len(centres) == 0:
        return []
    return _solid_discs(frame, centres, measured, gauge)


def _find_discs(frame: np.ndarray, scale: float) -> list[_Disc]:
    grey = frame.mean(axis=2).astype(np.float32)
    found: list[_Disc] = []
    for points in BUTTON_RADII_POINTS:
        found += _verify_discs(
            frame, grey, _Gauge(radius=max(3, _to_pixels(points, scale)), scale=scale)
        )
    found.sort(key=lambda disc: -disc.contrast)
    merge = DISC_MERGE_POINTS * scale
    kept: list[_Disc] = []
    for disc in found:
        if all(abs(disc.x - k.x) > merge or abs(disc.y - k.y) > merge for k in kept):
            kept.append(disc)
    return kept


def _triples(discs: Sequence[_Disc], scale: float) -> list[TitlebarAnchor]:
    pitch_min = BUTTON_PITCH_MIN_POINTS * scale
    pitch_max = BUTTON_PITCH_MAX_POINTS * scale
    row_tolerance = BUTTON_ROW_TOLERANCE_POINTS * scale
    ordered = sorted(discs, key=lambda disc: disc.x)
    anchors: list[TitlebarAnchor] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            first_gap = ordered[j].x - ordered[i].x
            if first_gap < pitch_min:
                continue
            if first_gap > pitch_max:
                break
            if abs(ordered[j].y - ordered[i].y) > row_tolerance:
                continue
            for k in range(j + 1, len(ordered)):
                second_gap = ordered[k].x - ordered[j].x
                if second_gap > pitch_max:
                    break
                if second_gap < pitch_min:
                    continue
                if abs(second_gap - first_gap) > BUTTON_PITCH_TOLERANCE * first_gap:
                    continue
                trio = (ordered[i], ordered[j], ordered[k])
                if max(abs(disc.y - trio[0].y) for disc in trio) > row_tolerance:
                    continue
                radii = np.array([disc.radius for disc in trio])
                if radii.max() - radii.min() > BUTTON_RADIUS_AGREEMENT_POINTS * scale:
                    continue
                backgrounds = np.array([disc.background for disc in trio])
                mean_background = backgrounds.mean(axis=0)
                spread_of_background = np.linalg.norm(backgrounds - mean_background, axis=1).max()
                if spread_of_background > BACKGROUND_TOLERANCE:
                    continue
                colours = np.array([disc.colour for disc in trio])
                spread = max(
                    float(np.linalg.norm(colours[a] - colours[b]))
                    for a in range(3)
                    for b in range(a + 1, 3)
                )
                anchors.append(
                    TitlebarAnchor(
                        buttons=(
                            (trio[0].x, trio[0].y),
                            (trio[1].x, trio[1].y),
                            (trio[2].x, trio[2].y),
                        ),
                        radius=float(radii.mean()),
                        pitch=(first_gap + second_gap) / 2.0,
                        background=(
                            float(mean_background[0]),
                            float(mean_background[1]),
                            float(mean_background[2]),
                        ),
                        coloured=spread > COLOURED_SPREAD_MIN,
                        contrast=sum(disc.contrast for disc in trio),
                    )
                )
    anchors.sort(key=lambda anchor: -anchor.contrast)
    merge = ANCHOR_MERGE_POINTS * scale
    kept: list[TitlebarAnchor] = []
    for anchor in anchors:
        x, y = anchor.leftmost
        if all(
            abs(x - other.leftmost[0]) > merge or abs(y - other.leftmost[1]) > merge
            for other in kept
        ):
            kept.append(anchor)
    return kept


def find_titlebar_anchors(
    frame: np.ndarray, scale: float, within: Rect | None = None
) -> list[TitlebarAnchor]:
    _validate_frame(frame)
    _validate_scale(scale)
    origin_x, origin_y = 0, 0
    region = frame
    if within is not None:
        height, width = frame.shape[:2]
        if within.left < 0 or within.top < 0 or within.right >= width or within.bottom >= height:
            raise ValueError(f"within {within.edges} does not fit a {width}x{height} frame.")
        origin_x, origin_y = within.left, within.top
        region = frame[within.top : within.bottom + 1, within.left : within.right + 1]
    anchors = _triples(_find_discs(np.ascontiguousarray(region), scale), scale)
    if origin_x == 0 and origin_y == 0:
        return anchors
    return [_moved_by(anchor, origin_x, origin_y) for anchor in anchors]


def _moved_by(anchor: TitlebarAnchor, dx: int, dy: int) -> TitlebarAnchor:
    first, second, third = anchor.buttons
    return TitlebarAnchor(
        buttons=(
            (first[0] + dx, first[1] + dy),
            (second[0] + dx, second[1] + dy),
            (third[0] + dx, third[1] + dy),
        ),
        radius=anchor.radius,
        pitch=anchor.pitch,
        background=anchor.background,
        coloured=anchor.coloured,
        contrast=anchor.contrast,
    )


def _step_maps(frame: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    values = frame.astype(np.float32)
    side_span = _span_pixels(SIDE_STEP_SPAN_POINTS, scale)
    vertical = np.zeros(frame.shape[:2], dtype=bool)
    if frame.shape[1] > 2 * side_span:
        distance = np.linalg.norm(values[:, 2 * side_span :] - values[:, : -2 * side_span], axis=2)
        vertical[:, side_span:-side_span] = distance >= SIDE_STEP_RGB
    top_span = _span_pixels(TOP_STEP_SPAN_POINTS, scale)
    horizontal = np.zeros(frame.shape[:2], dtype=bool)
    if frame.shape[0] > 2 * top_span:
        distance = np.linalg.norm(values[2 * top_span :, :] - values[: -2 * top_span, :], axis=2)
        horizontal[top_span:-top_span, :] = distance >= TOP_STEP_RGB
    return vertical, horizontal


def _forgive_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    kernel = np.ones((2 * gap + 1, 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _column_runs(vertical: np.ndarray, row: int, gap: int) -> np.ndarray:
    bridged = _forgive_gaps(vertical, gap)
    upwards = np.logical_and.accumulate(bridged[row::-1, :], axis=0).sum(axis=0)
    downwards = np.logical_and.accumulate(bridged[row:, :], axis=0).sum(axis=0)
    return upwards + downwards - bridged[row, :]


def _column_end(vertical: np.ndarray, column: int, row: int, gap: int) -> int | None:
    strip = vertical[:, column]
    bridged = _forgive_gaps(strip[:, None], gap)[:, 0]
    end = row + int(np.logical_and.accumulate(bridged[row:]).sum()) - 1
    if end >= len(strip) - 1:
        return None
    real = np.flatnonzero(strip[: end + 1])
    return int(real[-1]) if len(real) else end


def _best_position(support: np.ndarray, low: int, high: int, prefer: int) -> tuple[int, float]:
    segment = support[low : high + 1]
    best = float(segment.max())
    tied = np.flatnonzero(segment == best) + low
    breaks = np.flatnonzero(np.diff(tied) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(tied) - 1]))
    centres = [(tied[start] + tied[end]) // 2 for start, end in zip(starts, ends, strict=True)]
    return int(min(centres, key=lambda centre: (abs(centre - prefer), centre))), best


def _anchor_zone(approximate: Rect, scale: float) -> tuple[int, int, int, int]:
    margin = _to_pixels(SEARCH_MARGIN_POINTS, scale)
    return (
        approximate.left - margin,
        approximate.left + _to_pixels(BUTTONS_INSET_MAX_POINTS, scale),
        approximate.top - margin,
        approximate.top + _to_pixels(TITLEBAR_HEIGHT_MAX_POINTS, scale),
    )


def _search_band(frame: np.ndarray, approximate: Rect, scale: float) -> Rect:
    height, width = frame.shape[:2]
    margin = _to_pixels(SEARCH_MARGIN_POINTS, scale)
    inset = _to_pixels(BUTTONS_INSET_MAX_POINTS, scale)
    bar = _to_pixels(TITLEBAR_HEIGHT_MAX_POINTS, scale)
    return Rect(
        left=max(0, approximate.left - margin),
        top=max(0, approximate.top - margin),
        right=min(width - 1, min(approximate.right, approximate.left + inset) + margin),
        bottom=min(height - 1, approximate.top + bar + margin),
    )


def _band_around(position: int, margin: int, limit: int) -> tuple[int, int]:
    low = min(max(0, position - margin), limit)
    high = max(min(limit, position + margin), 0)
    return (min(low, high), max(low, high))


def _read_edges(
    vertical: np.ndarray,
    horizontal: np.ndarray,
    anchor: tuple[int, int],
    approximate: Rect,
    scale: float,
    clipped_by_frame: tuple[bool, bool, bool, bool],
) -> _Edges | Refusal:
    height, width = horizontal.shape
    left, top, right, bottom = approximate.edges
    margin = _to_pixels(SEARCH_MARGIN_POINTS, scale)
    gap = _span_pixels(EDGE_GAP_POINTS, scale)
    anchor_y = anchor[1]

    runs = _column_runs(vertical, anchor_y, gap).astype(np.float32)
    support = np.minimum(1.0, runs / max(1, approximate.height))
    if clipped_by_frame[0]:
        support[0] = 1.0
    if clipped_by_frame[2]:
        support[width - 1] = 1.0
    found_left, left_support = _best_position(support, *_band_around(left, margin, width - 1), left)
    found_right, right_support = _best_position(
        support, *_band_around(right, margin, width - 1), right
    )
    if left_support < SIDE_SUPPORT_MIN or right_support < SIDE_SUPPORT_MIN:
        return Refusal(
            code=RefusalCode.NO_SIDE_EDGE,
            detail=(
                f"the best vertical step within {margin} px of the left side runs for "
                f"{left_support:.2f} of the window's height and the one near the right side for "
                f"{right_support:.2f}, under the {SIDE_SUPPORT_MIN:.2f} a side must carry; the "
                f"window's sides are not visible where this rectangle puts them"
            ),
        )
    if found_right <= found_left:
        return Refusal(
            code=RefusalCode.NO_SIDE_EDGE,
            detail=(
                f"the two sides were both read at columns {found_left} and {found_right}, which do "
                f"not bound a window; this rectangle is {approximate.width} px wide, against the "
                f"{2 * margin} px band each side is looked for in"
            ),
        )

    row_sums = np.cumsum(horizontal.astype(np.int32), axis=1)
    coverage = (
        row_sums[:, found_right] - row_sums[:, found_left] + horizontal[:, found_left]
    ).astype(np.float32) / max(1, found_right - found_left + 1)
    if clipped_by_frame[1]:
        coverage[0] = 1.0
    top_high = min(
        height - 1, top + margin, anchor_y - _span_pixels(TITLEBAR_TOP_CLEARANCE_POINTS, scale)
    )
    top_low = max(0, top - margin)
    if top_high < top_low:
        return Refusal(
            code=RefusalCode.NO_TOP_EDGE,
            detail=(
                f"the buttons sit at row {anchor_y}, above the {top_low}..{top + margin} px band "
                f"where this rectangle's top edge would have to be"
            ),
        )
    found_top, top_support = _best_position(coverage, top_low, top_high, top)
    if top_support < TOP_SUPPORT_MIN:
        return Refusal(
            code=RefusalCode.NO_TOP_EDGE,
            detail=(
                f"the best horizontal step above the buttons runs across {top_support:.2f} of the "
                f"window's width, under the {TOP_SUPPORT_MIN:.2f} a top edge must carry"
            ),
        )

    ends = []
    for column in (found_left, found_right):
        for delta in (-1, 0, 1):
            end = _column_end(vertical, min(width - 1, max(0, column + delta)), anchor_y, gap)
            if end is not None:
                ends.append(end)
    if clipped_by_frame[3]:
        ends.append(height - 1)
    wanted_low, wanted_high = bottom - margin, bottom + margin
    within = [end for end in ends if wanted_low <= end <= wanted_high]
    if not within:
        return Refusal(
            code=RefusalCode.NO_BOTTOM_EDGE,
            detail=(
                f"the sides' vertical steps end at rows {sorted(set(ends))}, none of them in the "
                f"{wanted_low}..{wanted_high} px band around the bottom this rectangle gives; the "
                f"bottom edge is the one side a title bar cannot supply"
            ),
        )
    return _Edges(
        rect=Rect(left=found_left, top=found_top, right=found_right, bottom=max(within)),
        side_support=(left_support, right_support),
        top_support=top_support,
    )


def refine_window_rect(frame: np.ndarray, scale: float, approximate: Rect) -> RefinedRect | Refusal:
    _validate_frame(frame)
    _validate_scale(scale)
    if not isinstance(approximate, Rect):
        raise TypeError(f"approximate must be a Rect, got {type(approximate).__name__}.")
    height, width = frame.shape[:2]
    if (
        approximate.right < 0
        or approximate.bottom < 0
        or approximate.left >= width
        or approximate.top >= height
    ):
        raise ValueError(
            f"approximate {approximate.edges} lies outside the {width}x{height} frame."
        )

    anchors = find_titlebar_anchors(frame, scale, _search_band(frame, approximate, scale))
    if not anchors:
        return Refusal(
            code=RefusalCode.NO_ANCHOR,
            detail=(
                f"no three title-bar buttons within {SEARCH_MARGIN_POINTS:.0f} pt of the top-left "
                f"corner of {approximate.edges}; this window is drawn without them (full "
                f"screen, an undecorated tiling layout, or a toolkit that draws no traffic "
                f"lights)"
            ),
        )
    x_low, x_high, y_low, y_high = _anchor_zone(approximate, scale)
    accepted = [
        anchor
        for anchor in anchors
        if x_low <= anchor.leftmost[0] <= x_high and y_low <= anchor.leftmost[1] <= y_high
    ]
    if not accepted:
        return Refusal(
            code=RefusalCode.ANCHOR_OUTSIDE,
            detail=(
                f"the {len(anchors)} title bar(s) near this rectangle have their buttons at "
                f"{[anchor.leftmost for anchor in anchors]}, none inside the "
                f"x {x_low}..{x_high}, y {y_low}..{y_high} px zone where the buttons of "
                f"{approximate.edges} would be; the rectangle and the pixels describe different "
                f"windows"
            ),
        )
    anchor = accepted[0]

    margin = _to_pixels(SEARCH_MARGIN_POINTS, scale)
    crop_left = max(0, approximate.left - margin)
    crop_top = max(0, approximate.top - margin)
    crop_right = min(width - 1, approximate.right + margin)
    crop_bottom = min(height - 1, approximate.bottom + margin)
    crop = frame[crop_top : crop_bottom + 1, crop_left : crop_right + 1]
    vertical, horizontal = _step_maps(crop, scale)
    read = _read_edges(
        vertical,
        horizontal,
        (anchor.leftmost[0] - crop_left, anchor.leftmost[1] - crop_top),
        Rect(
            left=approximate.left - crop_left,
            top=approximate.top - crop_top,
            right=approximate.right - crop_left,
            bottom=approximate.bottom - crop_top,
        ),
        scale,
        (
            approximate.left <= 0,
            approximate.top <= 0,
            approximate.right >= width - 1,
            approximate.bottom >= height - 1,
        ),
    )
    if isinstance(read, Refusal):
        return read

    refined = Rect(
        left=read.rect.left + crop_left,
        top=read.rect.top + crop_top,
        right=read.rect.right + crop_left,
        bottom=read.rect.bottom + crop_top,
    )
    shifts = (
        refined.left - approximate.left,
        refined.top - approximate.top,
        refined.right - approximate.right,
        refined.bottom - approximate.bottom,
    )
    allowed = _to_pixels(MAX_REFINEMENT_POINTS, scale)
    if max(abs(shift) for shift in shifts) > allowed:
        return Refusal(
            code=RefusalCode.DIVERGED,
            detail=(
                f"the edges read off the pixels are {refined.edges} against the "
                f"{approximate.edges} given — sides moved by {shifts} px, over the {allowed} px "
                f"a refinement may move "
                f"one; the rectangle is stale or belongs to another window, and the pixels are not "
                f"trusted over it"
            ),
        )
    return RefinedRect(
        rect=refined,
        anchor=anchor,
        shifts=shifts,
        side_support=read.side_support,
        top_support=read.top_support,
    )


__all__ = [
    "BUTTONS_INSET_MAX_POINTS",
    "BUTTON_RADII_POINTS",
    "MAX_REFINEMENT_POINTS",
    "SEARCH_MARGIN_POINTS",
    "Rect",
    "RefinedRect",
    "Refusal",
    "RefusalCode",
    "TitlebarAnchor",
    "find_titlebar_anchors",
    "refine_window_rect",
]
