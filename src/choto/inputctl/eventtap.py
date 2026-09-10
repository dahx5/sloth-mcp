from __future__ import annotations

import time
from typing import NamedTuple

import Quartz

from choto.inputctl.errors import InputControlError, InputPermissionError

MOUSE_MOVED = Quartz.kCGEventMouseMoved
LEFT_BUTTON = Quartz.kCGMouseButtonLeft

UNICODE_PATH_KEYCODE = 0
UNICODE_CHUNK_SIZE = 20

_INTER_EVENT_DELAY_S = 0.004
_UNICODE_CHUNK_DELAY_S = 0.008


class ButtonEvents(NamedTuple):
    down: int
    up: int
    dragged: int
    button: int


_BUTTON_EVENTS: dict[str, ButtonEvents] = {
    "left": ButtonEvents(
        Quartz.kCGEventLeftMouseDown,
        Quartz.kCGEventLeftMouseUp,
        Quartz.kCGEventLeftMouseDragged,
        Quartz.kCGMouseButtonLeft,
    ),
    "right": ButtonEvents(
        Quartz.kCGEventRightMouseDown,
        Quartz.kCGEventRightMouseUp,
        Quartz.kCGEventRightMouseDragged,
        Quartz.kCGMouseButtonRight,
    ),
    "center": ButtonEvents(
        Quartz.kCGEventOtherMouseDown,
        Quartz.kCGEventOtherMouseUp,
        Quartz.kCGEventOtherMouseDragged,
        Quartz.kCGMouseButtonCenter,
    ),
    "middle": ButtonEvents(
        Quartz.kCGEventOtherMouseDown,
        Quartz.kCGEventOtherMouseUp,
        Quartz.kCGEventOtherMouseDragged,
        Quartz.kCGMouseButtonCenter,
    ),
}


def button_events(button: str) -> ButtonEvents:
    try:
        return _BUTTON_EVENTS[button]
    except KeyError as exc:
        raise ValueError(
            f"Unknown mouse button {button!r}; expected one of {', '.join(sorted(_BUTTON_EVENTS))}."
        ) from exc


def main_display_size() -> tuple[float, float]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return float(bounds.size.width), float(bounds.size.height)


def cursor_location() -> tuple[float, float]:
    event = Quartz.CGEventCreate(None)
    location = Quartz.CGEventGetLocation(event)
    return (float(location.x), float(location.y))


def post_access_granted() -> bool:
    return bool(Quartz.CGPreflightPostEventAccess())


def request_post_access() -> bool:
    return bool(Quartz.CGRequestPostEventAccess())


def require_post_access() -> None:
    if not post_access_granted():
        raise InputPermissionError(
            "Not permitted to post input events. Grant this process "
            "Accessibility access in System Settings -> Privacy & Security "
            "-> Accessibility, then retry."
        )


def post_mouse(
    event_type: int,
    point: tuple[float, float],
    button: int,
    *,
    flags: int = 0,
    click_state: int | None = None,
) -> None:
    event = Quartz.CGEventCreateMouseEvent(
        None,
        event_type,
        Quartz.CGPointMake(point[0], point[1]),
        button,
    )
    if event is not None:
        if click_state is not None:
            Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_state)
        if flags:
            Quartz.CGEventSetFlags(event, flags)
    _post(event)


def post_keystroke(keycode: int, *, flags: int, char: str | None = None) -> None:
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        if event is not None:
            Quartz.CGEventSetFlags(event, flags)
            if char is not None:
                Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        _post(event)


def post_modifier(keycode: int, *, down: bool) -> None:
    _post(Quartz.CGEventCreateKeyboardEvent(None, keycode, down))


def post_unicode(text: str) -> None:
    for start in range(0, len(text), UNICODE_CHUNK_SIZE):
        chunk = text[start : start + UNICODE_CHUNK_SIZE]
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, UNICODE_PATH_KEYCODE, down)
            Quartz.CGEventKeyboardSetUnicodeString(event, len(chunk), chunk)
            _post(event)
        time.sleep(_UNICODE_CHUNK_DELAY_S)


def post_scroll(lines: int) -> None:
    _post(
        Quartz.CGEventCreateScrollWheelEvent(
            None,
            Quartz.kCGScrollEventUnitLine,
            1,
            -int(lines),
        )
    )


def _post(event: object) -> None:
    if event is None:
        raise InputControlError("Quartz returned a null event; the event could not be constructed.")
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    time.sleep(_INTER_EVENT_DELAY_S)
