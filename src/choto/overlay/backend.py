from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from choto.overlay.geometry import Rect

__all__ = [
    "NULL_OVERLAY_CAPABILITIES",
    "NullOverlayBackend",
    "OverlayBackend",
    "OverlayCapabilities",
]


@dataclass(frozen=True, slots=True)
class OverlayCapabilities:
    draws_frame: bool
    stop_button: bool
    excluded_from_capture: bool


NULL_OVERLAY_CAPABILITIES = OverlayCapabilities(
    draws_frame=False,
    stop_button=False,
    excluded_from_capture=False,
)
"""What a backend that draws nothing declares: all three, ``False``."""


@runtime_checkable
class OverlayBackend(Protocol):
    @property
    def capabilities(self) -> OverlayCapabilities: ...

    def begin(self) -> None: ...

    def stop_requested(self) -> bool: ...

    def show(self, rect: Rect) -> None: ...

    def hide(self) -> None: ...

    def drawn_rects(self, screen: Rect) -> tuple[Rect, ...] | None: ...


class NullOverlayBackend:
    __slots__ = ()

    @property
    def capabilities(self) -> OverlayCapabilities:
        return NULL_OVERLAY_CAPABILITIES

    def begin(self) -> None: ...

    def stop_requested(self) -> bool:
        return False

    def show(self, rect: Rect) -> None: ...

    def hide(self) -> None: ...

    def drawn_rects(self, screen: Rect) -> tuple[Rect, ...] | None:
        return None
