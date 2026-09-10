from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from choto.models import BBox


class WindowRole(str, Enum):
    NORMAL = "normal"
    CHROME = "chrome"
    TRANSIENT = "transient"


@dataclass(frozen=True)
class WindowRect:
    x: float
    y: float
    w: float
    h: float


def window_rect_to_pixels(rect: WindowRect, scale: float) -> BBox:
    if scale <= 0:
        raise ValueError(f"Display scale must be positive, got {scale}.")
    return BBox(
        x=round(rect.x * scale),
        y=round(rect.y * scale),
        w=round(rect.w * scale),
        h=round(rect.h * scale),
    )


@dataclass(frozen=True)
class Window:
    app_name: str
    title: str
    rect: WindowRect
    is_chrome: bool = False


@dataclass(frozen=True)
class WindowList:
    app_name: str
    windows: tuple[Window, ...]
    reason: str | None = None

    @property
    def app_windows(self) -> tuple[Window, ...]:
        return tuple(window for window in self.windows if not window.is_chrome)

    @property
    def chrome(self) -> tuple[Window, ...]:
        return tuple(window for window in self.windows if window.is_chrome)


def rects_overlap(left: WindowRect, right: WindowRect) -> bool:
    return (
        left.x < right.x + right.w
        and right.x < left.x + left.w
        and left.y < right.y + right.h
        and right.y < left.y + left.h
    )


@dataclass(frozen=True)
class Obstructions:
    windows: tuple[Window, ...] = ()
    reason: str | None = None


class WindowSource(Protocol):
    def frontmost_windows(self) -> WindowList: ...

    def obstructions_over(self, rect: WindowRect, *, owner: str) -> Obstructions: ...

    def frontmost_app_name(self) -> str: ...

    def close(self) -> None: ...
