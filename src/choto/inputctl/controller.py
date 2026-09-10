from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from choto.inputctl import eventtap
from choto.inputctl.clipboard import read_clipboard_text
from choto.inputctl.errors import FailsafeTriggeredError
from choto.inputctl.eventtap import ButtonEvents
from choto.inputctl.gestures import (
    DRAG_PRESS_SETTLE_S,
    DRAG_RELEASE_SETTLE_S,
    DRAG_STEP_DELAY_S,
    in_failsafe_corner,
    interpolate_path,
    validate_drag_point,
    validate_path,
    validate_point,
)
from choto.inputctl.keycodes import (
    SHIFT_FLAG_MASK,
    SHIFT_KEYCODE,
    Keystroke,
    plan_typing,
    resolve_chord,
    resolve_modifiers,
)
from choto.inputctl.protocol import TypedText
from choto.log import get_logger

_log = get_logger(__name__)

_DRAG_CLICK_STATE = 1


class MacInputBackend:
    def __init__(self, stop_requested: Callable[[], bool] | None = None) -> None:
        self._stop_requested = stop_requested

    def mouse_position(self) -> tuple[float, float]:
        return eventtap.cursor_location()

    def read_clipboard(self) -> str | None:
        return read_clipboard_text()

    def main_display_size(self) -> tuple[float, float]:
        return eventtap.main_display_size()

    def ensure_permission(self) -> bool:
        return eventtap.post_access_granted()

    def request_permission(self) -> bool:
        return eventtap.request_post_access()

    def move(self, x: float, y: float) -> None:
        point = validate_point(x, y, eventtap.main_display_size())
        eventtap.require_post_access()
        eventtap.post_mouse(eventtap.MOUSE_MOVED, point, eventtap.LEFT_BUTTON)

    def click(
        self,
        x: float,
        y: float,
        *,
        button: str = "left",
        count: int = 1,
        modifiers: Sequence[str] | None = None,
    ) -> None:
        if count < 1:
            raise ValueError(f"click count must be >= 1, got {count}.")
        events = eventtap.button_events(button)
        point = validate_point(x, y, eventtap.main_display_size())
        held = resolve_modifiers(modifiers)

        eventtap.require_post_access()
        with self._modifiers_held(held.codes):
            for click_state in range(1, count + 1):
                eventtap.post_mouse(
                    events.down, point, events.button, flags=held.flags, click_state=click_state
                )
                eventtap.post_mouse(
                    events.up, point, events.button, flags=held.flags, click_state=click_state
                )

    def right_click(self, x: float, y: float, *, modifiers: Sequence[str] | None = None) -> None:
        self.click(x, y, button="right", count=1, modifiers=modifiers)

    def drag(
        self,
        path: Sequence[tuple[float, float]],
        *,
        button: str = "left",
        modifiers: Sequence[str] | None = None,
    ) -> None:
        waypoints = validate_path(path, eventtap.main_display_size())
        interpolate_path(waypoints)
        with self.drag_hold(*waypoints[0], button=button, modifiers=modifiers) as steer:
            for waypoint in waypoints[1:]:
                steer(*waypoint)

    @contextmanager
    def drag_hold(
        self,
        x: float,
        y: float,
        *,
        button: str = "left",
        modifiers: Sequence[str] | None = None,
    ) -> Iterator[Callable[[float, float], None]]:
        events = eventtap.button_events(button)
        origin = validate_drag_point(x, y, eventtap.main_display_size())
        held = resolve_modifiers(modifiers)

        eventtap.require_post_access()
        self._raise_if_failsafe("before the drag started")

        with self._modifiers_held(held.codes):
            eventtap.post_mouse(
                events.down,
                origin,
                events.button,
                flags=held.flags,
                click_state=_DRAG_CLICK_STATE,
            )
            reached = origin

            def steer(to_x: float, to_y: float) -> None:
                nonlocal reached
                destination = validate_drag_point(to_x, to_y, eventtap.main_display_size())
                for point in interpolate_path((reached, destination)):
                    self._raise_if_failsafe("mid-drag")
                    eventtap.post_mouse(events.dragged, point, events.button, flags=held.flags)
                    reached = point
                    time.sleep(DRAG_STEP_DELAY_S)

            try:
                time.sleep(DRAG_PRESS_SETTLE_S)
                yield steer
                time.sleep(DRAG_RELEASE_SETTLE_S)
            finally:
                self._release(events, reached, held.flags)

    def scroll(self, amount: int) -> None:
        eventtap.require_post_access()
        eventtap.post_scroll(amount)

    def type_text(self, text: str) -> TypedText:
        if not text:
            return TypedText(())
        eventtap.require_post_access()
        typing = plan_typing(text)

        shift_held = False
        try:
            for piece in typing.pieces:
                if isinstance(piece, Keystroke):
                    shift_held = self._apply_shift(shift_held, wanted=piece.shift)
                    eventtap.post_keystroke(
                        piece.keycode,
                        flags=SHIFT_FLAG_MASK if piece.shift else 0,
                        char=piece.char,
                    )
                else:
                    shift_held = self._apply_shift(shift_held, wanted=False)
                    eventtap.post_unicode(piece.text)
            shift_held = self._apply_shift(shift_held, wanted=False)
        finally:
            if shift_held:
                self._apply_shift(True, wanted=False)

        if typing.skipped:
            _log.warning(
                "type_text skipped untypeable characters",
                count=len(typing.skipped),
                codepoints=[f"U+{ord(char):04X}" for char in typing.skipped],
            )
        return TypedText(typing.skipped)

    def hotkey(self, keys: list[str]) -> None:
        chord = resolve_chord(keys)

        eventtap.require_post_access()
        with self._modifiers_held(chord.modifiers.codes):
            eventtap.post_keystroke(chord.keycode, flags=chord.modifiers.flags)

    def close(self) -> None: ...

    def _raise_if_failsafe(self, stage: str) -> None:
        position = self.mouse_position()
        if in_failsafe_corner(position):
            raise FailsafeTriggeredError(
                f"Kill-switch: the cursor entered the failsafe corner "
                f"({position[0]:g}, {position[1]:g}) {stage}; the gesture was stopped."
            )
        if self._stop_requested is not None and self._stop_requested():
            raise FailsafeTriggeredError(
                f"Kill-switch: the stop button was pressed {stage}; the gesture was stopped."
            )

    @staticmethod
    def _release(events: ButtonEvents, point: tuple[float, float], flags: int) -> None:
        try:
            eventtap.post_mouse(
                events.up, point, events.button, flags=flags, click_state=_DRAG_CLICK_STATE
            )
        except Exception as exc:
            _log.error(
                "inputctl.button_release_failed",
                x=point[0],
                y=point[1],
                button=events.button,
                error=str(exc),
            )
            raise

    @staticmethod
    @contextmanager
    def _modifiers_held(keycodes: Sequence[int]) -> Iterator[None]:
        pressed: list[int] = []
        try:
            for keycode in keycodes:
                eventtap.post_modifier(keycode, down=True)
                pressed.append(keycode)
            yield
        finally:
            for keycode in reversed(pressed):
                try:
                    eventtap.post_modifier(keycode, down=False)
                except Exception as exc:
                    _log.error("inputctl.modifier_release_failed", keycode=keycode, error=str(exc))
                    raise

    @staticmethod
    def _apply_shift(held: bool, *, wanted: bool) -> bool:
        if held == wanted:
            return held
        eventtap.post_modifier(SHIFT_KEYCODE, down=wanted)
        return wanted
