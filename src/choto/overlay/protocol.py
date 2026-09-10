from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator


class OverlayProtocolError(ValueError): ...


class _Message(BaseModel):
    """Shared strictness for everything crossing the pipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShowCommand(_Message):
    """Outline this rectangle — the window Choto is currently occupied with.

    Coordinates are display points with the origin at the top-left of the main
    display: the space ``CGWindowListCopyWindowInfo`` reports window bounds in,
    so the daemon passes on what it already has instead of converting into a
    space the helper would have to convert back out of.
    """

    cmd: Literal["show"] = "show"
    x: float
    y: float
    w: float = Field(gt=0.0)
    h: float = Field(gt=0.0)


class HideCommand(_Message):
    """Take the frame and the button off the screen; keep the helper alive."""

    cmd: Literal["hide"] = "hide"


class QuitCommand(_Message):
    """Shut the helper down (it also exits when its stdin reaches EOF)."""

    cmd: Literal["quit"] = "quit"


type Command = Annotated[ShowCommand | HideCommand | QuitCommand, Field(discriminator="cmd")]


class ReadyEvent(_Message):
    """The helper process started and is reading its input.

    Sent before the run loop is turning, so it says nothing about anything being
    on screen — the only proof of that is :class:`FrameEvent`, which is measured
    rather than assumed.
    """

    event: Literal["ready"] = "ready"


class StopEvent(_Message):
    """The user clicked the stop button: the run must abort."""

    event: Literal["stop"] = "stop"


class FrameEvent(_Message):
    """What the window server says about the panels the helper has placed.

    The overlay's one measured fact. ``orderFrontRegardless`` returns nothing
    and creating a panel cannot fail visibly, so without this the daemon would
    only ever know that a command was *written to a pipe* — and Choto has seen
    that be false: panels alive, ``kCGWindowIsOnscreen`` back to 0, nothing in
    the log (see :func:`choto.overlay.app._application`). A frame that is not on
    screen while the machine acts in the window under it is the failure the
    indicator exists to prevent, so it is counted rather than trusted.

    ``applied``
        How many commands the helper has taken off its queue when this
        measurement was emitted. The daemon compares it with what it has sent:
        a report from before the command it is currently drawing describes a
        screen that has already been replaced, and is dropped.
    ``expected``
        How many panels the helper ordered front and believes should be on
        screen. Zero after a hide — including the hide the stop button performs
        by itself, which is the only way the daemon learns of it.
    ``onscreen``
        How many of those the window server actually reports as on screen.
    ``button_onscreen``
        Whether the stop pill is one of them. Separate because it is a
        kill-switch and the bands are not: a frame without its button leaves the
        user watching something they cannot stop.
    """

    event: Literal["frame"] = "frame"
    applied: int = Field(ge=0)
    expected: int = Field(ge=0)
    onscreen: int = Field(ge=0)
    button_onscreen: bool = False

    @model_validator(mode="after")
    def _no_more_than_placed(self) -> FrameEvent:
        if self.onscreen > self.expected:
            raise ValueError(f"{self.onscreen} panels on screen but only {self.expected} placed")
        if self.button_onscreen and self.onscreen < 1:
            raise ValueError("the stop button is on screen but no panel is")
        return self


type Event = Annotated[ReadyEvent | StopEvent | FrameEvent, Field(discriminator="event")]

_COMMANDS: TypeAdapter[Command] = TypeAdapter(Command)
_EVENTS: TypeAdapter[Event] = TypeAdapter(Event)


def encode(message: _Message) -> str:
    return message.model_dump_json() + "\n"


def parse_command(line: str) -> Command:
    return _parse(_COMMANDS, line, "command")


def parse_event(line: str) -> Event:
    return _parse(_EVENTS, line, "event")


def _parse(adapter: TypeAdapter, line: str, kind: str):
    text = line.strip()
    if not text:
        raise OverlayProtocolError(f"empty overlay {kind} line")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OverlayProtocolError(f"overlay {kind} is not JSON: {exc}") from exc
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        raise OverlayProtocolError(f"unusable overlay {kind}: {exc}") from exc
