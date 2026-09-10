from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TypedText:
    skipped: tuple[str, ...]


@runtime_checkable
class InputBackend(Protocol):
    def mouse_position(self) -> tuple[float, float]: ...

    def read_clipboard(self) -> str | None: ...

    def main_display_size(self) -> tuple[float, float]: ...

    def ensure_permission(self) -> bool: ...

    def request_permission(self) -> bool: ...

    def move(self, x: float, y: float) -> None: ...

    def click(
        self,
        x: float,
        y: float,
        *,
        button: str = "left",
        count: int = 1,
        modifiers: Sequence[str] | None = None,
    ) -> None: ...

    def right_click(
        self, x: float, y: float, *, modifiers: Sequence[str] | None = None
    ) -> None: ...

    def drag(
        self,
        path: Sequence[tuple[float, float]],
        *,
        button: str = "left",
        modifiers: Sequence[str] | None = None,
    ) -> None: ...

    def drag_hold(
        self,
        x: float,
        y: float,
        *,
        button: str = "left",
        modifiers: Sequence[str] | None = None,
    ) -> AbstractContextManager[Callable[[float, float], None]]: ...

    def scroll(self, amount: int) -> None: ...

    def type_text(self, text: str) -> TypedText: ...

    def hotkey(self, keys: list[str]) -> None: ...

    def close(self) -> None: ...
