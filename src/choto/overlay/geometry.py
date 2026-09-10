from __future__ import annotations

from dataclasses import dataclass

FRAME_THICKNESS_PT = 4.0

STOP_BUTTON_WIDTH_PT = 78.0
STOP_BUTTON_HEIGHT_PT = 24.0

STOP_BUTTON_GAP_PT = 4.0


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def empty(self) -> bool:
        return self.w <= 0.0 or self.h <= 0.0


def intersect(first: Rect, second: Rect) -> Rect:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.right, second.right)
    y1 = min(first.bottom, second.bottom)
    return Rect(x=x0, y=y0, w=max(x1 - x0, 0.0), h=max(y1 - y0, 0.0))


def clamp_into(rect: Rect, bounds: Rect) -> Rect:
    x = rect.x
    y = rect.y
    if rect.w <= bounds.w:
        x = min(max(x, bounds.x), bounds.right - rect.w)
    if rect.h <= bounds.h:
        y = min(max(y, bounds.y), bounds.bottom - rect.h)
    return intersect(Rect(x=x, y=y, w=rect.w, h=rect.h), bounds)


def frame_bands(
    target: Rect, screen: Rect, thickness: float = FRAME_THICKNESS_PT
) -> tuple[Rect, ...]:
    if target.empty or screen.empty or thickness <= 0.0:
        return ()
    if intersect(target, screen).empty:
        return ()
    outer_w = target.w + 2 * thickness
    bands = (
        Rect(x=target.x - thickness, y=target.y - thickness, w=outer_w, h=thickness),
        Rect(x=target.x - thickness, y=target.bottom, w=outer_w, h=thickness),
        Rect(x=target.x - thickness, y=target.y, w=thickness, h=target.h),
        Rect(x=target.right, y=target.y, w=thickness, h=target.h),
    )
    placed = (clamp_into(band, screen) for band in bands)
    return tuple(band for band in placed if not band.empty)


def stop_button_rect(
    target: Rect,
    screen: Rect,
    thickness: float = FRAME_THICKNESS_PT,
    width: float = STOP_BUTTON_WIDTH_PT,
    height: float = STOP_BUTTON_HEIGHT_PT,
    gap: float = STOP_BUTTON_GAP_PT,
) -> Rect:
    rect = Rect(
        x=target.right + thickness - width,
        y=target.y - thickness - gap - height,
        w=width,
        h=height,
    )
    return clamp_into(rect, screen)


def to_cocoa(rect: Rect, primary_height: float) -> Rect:
    return Rect(x=rect.x, y=primary_height - rect.bottom, w=rect.w, h=rect.h)
