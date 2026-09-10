from __future__ import annotations

import math
from collections.abc import Sequence

FAILSAFE_CORNER_SIZE = 10

DRAG_POINT_SPACING_PT = 40.0
DRAG_MIN_STEPS_PER_SEGMENT = 8
DRAG_MAX_STEPS_PER_SEGMENT = 32

DRAG_STEP_DELAY_S = 0.006
DRAG_PRESS_SETTLE_S = 0.05
DRAG_RELEASE_SETTLE_S = 0.05


def in_failsafe_corner(pos: tuple[float, float]) -> bool:
    x, y = pos
    return 0.0 <= x <= FAILSAFE_CORNER_SIZE and 0.0 <= y <= FAILSAFE_CORNER_SIZE


def segment_steps(start: tuple[float, float], end: tuple[float, float]) -> int:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = math.ceil(distance / DRAG_POINT_SPACING_PT)
    return max(DRAG_MIN_STEPS_PER_SEGMENT, min(DRAG_MAX_STEPS_PER_SEGMENT, steps))


def validate_point(x: float, y: float, size: tuple[float, float]) -> tuple[float, float]:
    try:
        point = (float(x), float(y))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Coordinates must be numbers, got ({x!r}, {y!r}).") from exc
    width, height = size
    if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
        raise ValueError(
            f"Point ({point[0]:g}, {point[1]:g}) is outside the main display "
            f"({width:g}x{height:g} points)."
        )
    return point


def validate_drag_point(x: float, y: float, size: tuple[float, float]) -> tuple[float, float]:
    point = validate_point(x, y, size)
    if in_failsafe_corner(point):
        raise ValueError(
            f"drag point ({point[0]:g}, {point[1]:g}) is inside the failsafe corner "
            f"(0..{FAILSAFE_CORNER_SIZE} points); a drag must not cross the kill-switch."
        )
    return point


def validate_path(
    path: Sequence[tuple[float, float]], size: tuple[float, float]
) -> list[tuple[float, float]]:
    waypoints = list(path)
    if len(waypoints) < 2:
        raise ValueError(
            f"drag needs at least two path points (press and release), got {len(waypoints)}."
        )
    validated: list[tuple[float, float]] = []
    for index, waypoint in enumerate(waypoints):
        try:
            x, y = waypoint
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"drag path point {index} must be an (x, y) pair, got {waypoint!r}."
            ) from exc
        point = validate_point(x, y, size)
        if in_failsafe_corner(point):
            raise ValueError(
                f"drag path point {index} ({point[0]:g}, {point[1]:g}) is inside the "
                f"failsafe corner (0..{FAILSAFE_CORNER_SIZE} points); a drag must not "
                "cross the kill-switch."
            )
        validated.append(point)
    return validated


def interpolate_path(waypoints: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    motion: list[tuple[float, float]] = []
    for start, end in zip(waypoints, waypoints[1:], strict=False):
        steps = segment_steps(start, end)
        for step in range(1, steps + 1):
            progress = step / steps
            point = (
                start[0] + (end[0] - start[0]) * progress,
                start[1] + (end[1] - start[1]) * progress,
            )
            if in_failsafe_corner(point):
                raise ValueError(
                    f"drag path passes through the failsafe corner at "
                    f"({point[0]:g}, {point[1]:g}); a drag must not cross the kill-switch."
                )
            motion.append(point)
    return motion
