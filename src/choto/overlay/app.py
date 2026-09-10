from __future__ import annotations

import sys
import threading
from collections import deque

import objc
import Quartz
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSButton,
    NSCenterTextAlignment,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMutableParagraphStyle,
    NSPanel,
    NSParagraphStyleAttributeName,
    NSScreen,
    NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSAttributedString, NSMakeRect, NSObject

from choto.log import get_logger, setup_logging
from choto.overlay.geometry import (
    FRAME_THICKNESS_PT,
    STOP_BUTTON_HEIGHT_PT,
    STOP_BUTTON_WIDTH_PT,
    Rect,
    frame_bands,
    stop_button_rect,
    to_cocoa,
)
from choto.overlay.protocol import (
    FrameEvent,
    HideCommand,
    OverlayProtocolError,
    QuitCommand,
    ReadyEvent,
    ShowCommand,
    StopEvent,
    encode,
    parse_command,
)

_log = get_logger(__name__)

_FRAME_RGBA = (1.0, 0.45, 0.0, 0.7)
_BUTTON_RGBA = (0.85, 0.32, 0.0, 0.92)

_BUTTON_TITLE = "■ Stop"
_BUTTON_FONT_SIZE = 12.0

_MAX_BANDS = 4

_LOGGED_LINE_CHARS = 200

_STOP_SELECTOR = b"stopPressed:"
_APPLY_SELECTOR = "applyPending:"
_QUIT_SELECTOR = "quitNow:"
_REPORT_SELECTOR = "reportFrame:"

_FRAME_REPORT_DELAY_S = 0.15

_COLLECTION_BEHAVIOR = (
    NSWindowCollectionBehaviorCanJoinAllSpaces
    | NSWindowCollectionBehaviorStationary
    | NSWindowCollectionBehaviorIgnoresCycle
)

_PANEL_STYLE = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel

_stdout_lock = threading.Lock()


def emit(event: ReadyEvent | StopEvent) -> bool:
    try:
        with _stdout_lock:
            sys.stdout.write(encode(event))
            sys.stdout.flush()
    except (OSError, ValueError) as exc:
        _log.warning("overlay.emit_failed", event=event.event, error=str(exc))
        return False
    return True


def _color(rgba: tuple[float, float, float, float]):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(*rgba)


def _application():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    return app


def _panel(rect: Rect, *, clickable: bool):
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(rect.x, rect.y, rect.w, rect.h),
        _PANEL_STYLE,
        NSBackingStoreBuffered,
        False,
    )
    panel.setOpaque_(False)
    panel.setHasShadow_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setLevel_(NSStatusWindowLevel)
    panel.setSharingType_(NSWindowSharingNone)
    panel.setCollectionBehavior_(_COLLECTION_BEHAVIOR)
    panel.setIgnoresMouseEvents_(not clickable)
    panel.setHidesOnDeactivate_(False)
    panel.setCanHide_(False)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setReleasedWhenClosed_(False)
    return panel


def _onscreen_windows(numbers: list[int]) -> frozenset[int]:
    asked = [number for number in numbers if number > 0]
    if not asked:
        return frozenset()
    try:
        entries = Quartz.CGWindowListCreateDescriptionFromArray(asked)
    except Exception as exc:  # noqa: BLE001 - a count is never worth the helper
        _log.warning("overlay.window_list_failed", error=str(exc))
        return frozenset()
    if entries is None:
        return frozenset()
    return frozenset(
        int(entry[Quartz.kCGWindowNumber])
        for entry in entries
        if Quartz.kCGWindowNumber in entry and entry.get(Quartz.kCGWindowIsOnscreen, False)
    )


def _stop_title(title: str):
    paragraph = NSMutableParagraphStyle.alloc().init()
    paragraph.setAlignment_(NSCenterTextAlignment)
    attributes = {
        NSForegroundColorAttributeName: NSColor.whiteColor(),
        NSFontAttributeName: NSFont.boldSystemFontOfSize_(_BUTTON_FONT_SIZE),
        NSParagraphStyleAttributeName: paragraph,
    }
    return NSAttributedString.alloc().initWithString_attributes_(title, attributes)


class OverlayController(NSObject):
    def init(self):
        self = objc.super(OverlayController, self).init()
        if self is None:  # pragma: no cover - allocation failure is not testable
            return None
        self._pending = deque()
        self._pending_lock = threading.Lock()
        self._bands = []
        self._button_panel = None
        self._button = None
        self._applied = 0
        self._placed = []
        self._button_placed = False
        self._report_scheduled = False
        return self

    @objc.python_method
    def enqueue(self, command) -> None:
        with self._pending_lock:
            self._pending.append(command)
        self.performSelectorOnMainThread_withObject_waitUntilDone_(_APPLY_SELECTOR, None, False)

    @objc.python_method
    def request_quit(self) -> None:
        self.performSelectorOnMainThread_withObject_waitUntilDone_(_QUIT_SELECTOR, None, False)

    def applyPending_(self, _sender) -> None:
        applied = 0
        while True:
            with self._pending_lock:
                if not self._pending:
                    break
                command = self._pending.popleft()
            self._applied += 1
            applied += 1
            try:
                self._apply(command)
            except Exception:  # noqa: BLE001 - one bad frame must not kill the helper
                _log.exception("overlay.apply_failed", command=command.cmd)
        if applied:
            self._schedule_report()

    def quitNow_(self, _sender) -> None:
        _log.info("overlay.quitting")
        NSApplication.sharedApplication().terminate_(None)

    def stopPressed_(self, _sender) -> None:
        _log.warning("overlay.stop_clicked")
        self._hide()
        self._report()
        if not emit(StopEvent()):
            self.quitNow_(None)

    def reportFrame_(self, _sender) -> None:
        self._report_scheduled = False
        self._report()

    @objc.python_method
    def _apply(self, command) -> None:
        if isinstance(command, ShowCommand):
            self._show(Rect(x=command.x, y=command.y, w=command.w, h=command.h))
        elif isinstance(command, HideCommand):
            self._hide()
        elif isinstance(command, QuitCommand):
            self.quitNow_(None)
        else:  # pragma: no cover - the protocol union is exhaustive above
            raise TypeError(f"unhandled overlay command: {command!r}")

    @objc.python_method
    def _screen(self) -> Rect | None:
        screens = NSScreen.screens()
        if not screens:
            _log.warning("overlay.no_screen")
            return None
        frame = screens[0].frame()
        return Rect(x=0.0, y=0.0, w=float(frame.size.width), h=float(frame.size.height))

    @objc.python_method
    def _show(self, target: Rect) -> None:
        screen = self._screen()
        if screen is None:
            return
        bands = frame_bands(target, screen, FRAME_THICKNESS_PT)
        if not bands:
            _log.info("overlay.off_screen", x=target.x, y=target.y, w=target.w, h=target.h)
            self._hide()
            return
        placed = []
        for index, band in enumerate(bands[:_MAX_BANDS]):
            panel = self._band(index)
            panel.setBackgroundColor_(_color(_FRAME_RGBA))
            self._place(panel, band, screen)
            placed.append(panel)
        for index in range(len(bands), len(self._bands)):
            self._bands[index].orderOut_(None)
        self._placed = placed
        self._place_button(
            stop_button_rect(target, screen, FRAME_THICKNESS_PT, STOP_BUTTON_WIDTH_PT), screen
        )
        _log.debug("overlay.shown", x=target.x, y=target.y, w=target.w, h=target.h)

    @objc.python_method
    def _hide(self) -> None:
        for panel in self._bands:
            panel.orderOut_(None)
        if self._button_panel is not None:
            self._button_panel.orderOut_(None)
        self._placed = []
        self._button_placed = False

    @objc.python_method
    def _place(self, panel, rect: Rect, screen: Rect) -> None:
        cocoa = to_cocoa(rect, screen.h)
        panel.setFrame_display_(NSMakeRect(cocoa.x, cocoa.y, cocoa.w, cocoa.h), True)
        panel.orderFrontRegardless()

    @objc.python_method
    def _band(self, index: int):
        while len(self._bands) <= index:
            self._bands.append(_panel(Rect(x=0.0, y=0.0, w=1.0, h=1.0), clickable=False))
        return self._bands[index]

    @objc.python_method
    def _place_button(self, rect: Rect, screen: Rect) -> None:
        if rect.empty:
            if self._button_panel is not None:
                self._button_panel.orderOut_(None)
            self._button_placed = False
            _log.info("overlay.no_room_for_stop_button", w=rect.w, h=rect.h)
            return
        if self._button_panel is None:
            self._button_panel = _panel(rect, clickable=True)
            button = NSButton.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, rect.w, rect.h))
            button.setBordered_(False)
            button.setWantsLayer_(True)
            button.layer().setCornerRadius_(STOP_BUTTON_HEIGHT_PT / 2.0)
            button.setTarget_(self)
            button.setAction_(_STOP_SELECTOR)
            self._button_panel.setContentView_(button)
            self._button = button
        self._button.layer().setBackgroundColor_(_color(_BUTTON_RGBA).CGColor())
        self._button.setAttributedTitle_(_stop_title(_BUTTON_TITLE))
        self._button.setFrame_(NSMakeRect(0.0, 0.0, rect.w, rect.h))
        self._place(self._button_panel, rect, screen)
        self._button_placed = True

    @objc.python_method
    def _schedule_report(self) -> None:
        if self._report_scheduled:
            return
        self._report_scheduled = True
        self.performSelector_withObject_afterDelay_(_REPORT_SELECTOR, None, _FRAME_REPORT_DELAY_S)

    @objc.python_method
    def _report(self) -> None:
        panels = list(self._placed)
        button = self._button_panel if self._button_placed else None
        if button is not None:
            panels.append(button)
        numbers = [int(panel.windowNumber()) for panel in panels]
        onscreen = _onscreen_windows(numbers)
        button_number = numbers[-1] if button is not None else None
        event = FrameEvent(
            applied=self._applied,
            expected=len(numbers),
            onscreen=sum(1 for number in numbers if number in onscreen),
            button_onscreen=button_number is not None and button_number in onscreen,
        )
        if event.expected and event.onscreen < event.expected:
            _log.warning(
                "overlay.panels_not_on_screen",
                expected=event.expected,
                onscreen=event.onscreen,
            )
        if not emit(event):
            self.quitNow_(None)


def _read_commands(controller: OverlayController, stream) -> None:
    try:
        for line in stream:
            try:
                command = parse_command(line)
            except OverlayProtocolError as exc:
                _log.warning(
                    "overlay.unreadable_command",
                    line=line.strip()[:_LOGGED_LINE_CHARS],
                    error=str(exc),
                )
                continue
            controller.enqueue(command)
    except (OSError, ValueError) as exc:
        _log.warning("overlay.command_stream_failed", error=str(exc))
    finally:
        _log.info("overlay.command_stream_closed")
        controller.request_quit()


def main() -> int:
    setup_logging()
    app = _application()

    controller = OverlayController.alloc().init()
    reader = threading.Thread(
        target=_read_commands,
        args=(controller, sys.stdin),
        name="choto-overlay-commands",
        daemon=True,
    )
    reader.start()

    if not emit(ReadyEvent()):
        return 1
    _log.info("overlay.started")
    app.run()
    return 0
