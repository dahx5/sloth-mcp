from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import Quartz
from AppKit import NSWorkspace

from choto.log import get_logger
from choto.overlay.identity import is_overlay_helper
from choto.vision.windowsource import (
    Obstructions,
    Window,
    WindowList,
    WindowRect,
    WindowRole,
    rects_overlap,
    window_rect_to_pixels,
)

__all__ = [
    "FrontmostApp",
    "MacWindowSource",
    "Obstructions",
    "Window",
    "WindowList",
    "WindowRect",
    "WindowRole",
    "frontmost_app_name",
    "frontmost_app_owner",
    "frontmost_windows",
    "obstructions_over",
    "window_rect_to_pixels",
]

_log = get_logger(__name__)

_UNKNOWN_APP = "unknown"

_WINDOW_LIST_OPTIONS = (
    Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
)
_RELATIVE_TO_NONE = Quartz.kCGNullWindowID

_NORMAL_WINDOW_LAYER = 0

_OVERLAY_LAYER_MIN = 100

_MENU_BAR_WINDOW = ("Window Server", "Menubar")

_INVISIBLE_ALPHA = 0.0

_OPAQUE_ALPHA = 1.0

_LAYER_ABSENT = -1

_PID_ABSENT = -1

_CONTAINER_SLACK_PT = 1.0

_TRANSIENT_MAX_AREA_PT2 = 30_000.0


@dataclass(frozen=True)
class FrontmostApp:
    name: str
    pid: int | None


@dataclass(frozen=True)
class _Listed:
    owner: str
    title: str
    layer: int
    pid: int
    rect: WindowRect | None

    @property
    def role(self) -> WindowRole:
        if self.layer >= _OVERLAY_LAYER_MIN:
            return WindowRole.CHROME
        if (self.owner, self.title) == _MENU_BAR_WINDOW:
            return WindowRole.CHROME
        if self.title:
            return WindowRole.NORMAL
        if self.rect is not None and self.rect.w * self.rect.h < _TRANSIENT_MAX_AREA_PT2:
            return WindowRole.TRANSIENT
        return WindowRole.NORMAL


@dataclass(frozen=True)
class _Listing:
    windows: WindowList
    plaques: tuple[str, ...]


def _copy_window_info() -> Any:
    return Quartz.CGWindowListCopyWindowInfo(_WINDOW_LIST_OPTIONS, _RELATIVE_TO_NONE)


def _text(window: Any, key: Any) -> str:
    return str(window.get(key) or "")


def _layer(window: Any) -> int:
    layer = window.get(Quartz.kCGWindowLayer)
    return _LAYER_ABSENT if layer is None else int(layer)


def _pid(window: Any) -> int:
    pid = window.get(Quartz.kCGWindowOwnerPID)
    return _PID_ABSENT if pid is None else int(pid)


def _alpha(window: Any) -> float:
    alpha = window.get(Quartz.kCGWindowAlpha)
    return _OPAQUE_ALPHA if alpha is None else float(alpha)


def _window_rect(window: Any) -> WindowRect | None:
    if _alpha(window) <= _INVISIBLE_ALPHA:
        return None
    bounds = window.get(Quartz.kCGWindowBounds)
    if bounds is None:
        return None
    ok, rect = Quartz.CGRectMakeWithDictionaryRepresentation(bounds, None)
    if not ok or min(rect.size.width, rect.size.height) <= 0:
        return None
    return WindowRect(
        x=float(rect.origin.x),
        y=float(rect.origin.y),
        w=float(rect.size.width),
        h=float(rect.size.height),
    )


def _read(window: Any) -> _Listed:
    return _Listed(
        owner=_text(window, Quartz.kCGWindowOwnerName),
        title=_text(window, Quartz.kCGWindowName),
        layer=_layer(window),
        pid=_pid(window),
        rect=_window_rect(window),
    )


def _frontmost(listed: Sequence[_Listed], own_pid: int) -> _Listed | None:
    for window in listed:
        if window.layer != _NORMAL_WINDOW_LAYER:
            continue
        if window.pid in (_PID_ABSENT, own_pid):
            continue
        if window.role is WindowRole.TRANSIENT:
            continue
        if window.owner:
            return window
    return None


def _listing(listed: Sequence[_Listed], own_pid: int) -> _Listing:
    front = _frontmost(listed, own_pid)
    if front is None:
        return _Listing(
            windows=WindowList(
                app_name="",
                windows=(),
                reason=("no ordinary window is on screen, so no frontmost app could be identified"),
            ),
            plaques=(),
        )

    app_windows: list[Window] = []
    chrome: list[Window] = []
    plaques: list[str] = []
    for window in listed:
        if window.rect is None:
            continue
        role = window.role
        if role is WindowRole.TRANSIENT:
            plaques.append(f"{window.owner} {window.rect.w:.0f}x{window.rect.h:.0f}")
            continue
        if role is WindowRole.CHROME:
            chrome.append(
                Window(
                    app_name=window.owner,
                    title=window.title,
                    rect=window.rect,
                    is_chrome=True,
                )
            )
            continue
        if window.pid == front.pid:
            app_windows.append(Window(app_name=front.owner, title=window.title, rect=window.rect))

    if not app_windows:
        return _Listing(
            windows=WindowList(
                app_name=front.owner,
                windows=(),
                reason=(
                    f'the frontmost app "{front.owner}" has no on-screen window the window '
                    "server can locate"
                ),
            ),
            plaques=tuple(plaques),
        )
    return _Listing(
        windows=WindowList(app_name=front.owner, windows=tuple(app_windows + chrome)),
        plaques=tuple(plaques),
    )


def frontmost_windows() -> WindowList:
    try:
        windows = _copy_window_info()
    except Exception as exc:  # noqa: BLE001 - report the failure, never raise here
        _log.exception("windows.query_failed")
        return WindowList(app_name="", windows=(), reason=f"window-server query failed: {exc}")

    if not windows:
        return WindowList(
            app_name="", windows=(), reason="the window server reported no on-screen windows"
        )

    listing = _listing([_read(window) for window in windows], os.getpid())
    if listing.plaques:
        _log.info("windows.plaques_ignored", plaques=list(listing.plaques))
    resolved = listing.windows
    if resolved.reason is None:
        _log.debug(
            "windows.resolved",
            app_name=resolved.app_name,
            app_windows=len(resolved.app_windows),
            chrome_windows=len(resolved.chrome),
        )
    return resolved


def _main_display_size() -> tuple[float, float]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return float(bounds.size.width), float(bounds.size.height)


def _is_container_rect(rect: WindowRect, layer: int, display: tuple[float, float]) -> bool:
    if layer <= _NORMAL_WINDOW_LAYER:
        return False
    width, height = display
    return rect.w >= width - _CONTAINER_SLACK_PT and rect.h >= height - _CONTAINER_SLACK_PT


def _obstructions(
    listed: Sequence[_Listed],
    rect: WindowRect,
    *,
    owner: str,
    display: tuple[float, float],
) -> Obstructions:
    over: list[Window] = []
    for window in listed:
        if window.rect is None:
            continue
        if _is_container_rect(window.rect, window.layer, display):
            continue
        if not rects_overlap(window.rect, rect):
            continue
        role = window.role
        if role is WindowRole.TRANSIENT:
            continue
        is_chrome = role is WindowRole.CHROME
        if window.owner == owner and not is_chrome:
            return Obstructions(windows=tuple(over))
        over.append(
            Window(
                app_name=window.owner,
                title=window.title,
                rect=window.rect,
                is_chrome=is_chrome,
            )
        )
    return Obstructions(
        reason=(
            f'no window of "{owner}" is on screen under that rectangle any more, so what is '
            "drawn there now belongs to something else"
        )
    )


def obstructions_over(rect: WindowRect, *, owner: str) -> Obstructions:
    try:
        windows = _copy_window_info()
    except Exception as exc:  # noqa: BLE001 - a failure is a value on this seam
        _log.exception("obstructions.query_failed")
        return Obstructions(reason=f"window-server query failed: {exc}")
    if not windows:
        return Obstructions(reason="the window server reported no on-screen windows")

    own_pid = os.getpid()
    listed = [
        window
        for window in map(_read, windows)
        if window.pid != own_pid and not is_overlay_helper(window.pid)
    ]
    return _obstructions(listed, rect, owner=owner, display=_main_display_size())


def _frontmost_from_window_list() -> FrontmostApp | None:
    windows = _copy_window_info()
    if not windows:
        return None
    front = _frontmost([_read(window) for window in windows], os.getpid())
    return None if front is None else FrontmostApp(name=front.owner, pid=front.pid)


def _frontmost_from_workspace() -> FrontmostApp | None:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None
    pid = int(app.processIdentifier())
    name = app.localizedName()
    if name:
        return FrontmostApp(name=str(name), pid=pid)
    bundle_id = app.bundleIdentifier()
    return FrontmostApp(name=str(bundle_id), pid=pid) if bundle_id else None


_UNKNOWN_OWNER = FrontmostApp(name=_UNKNOWN_APP, pid=None)

_FRONTMOST_PROBES: tuple[tuple[Callable[[], FrontmostApp | None], str], ...] = (
    (_frontmost_from_window_list, "frontmost.window_list_failed"),
    (_frontmost_from_workspace, "frontmost.workspace_failed"),
)


def frontmost_app_owner() -> FrontmostApp:
    for probe, failure in _FRONTMOST_PROBES:
        try:
            front = probe()
        except Exception:
            _log.exception(failure)
            continue
        if front is not None:
            return front if front.name else _UNKNOWN_OWNER
    return _UNKNOWN_OWNER


def frontmost_app_name() -> str:
    return frontmost_app_owner().name


class MacWindowSource:
    def frontmost_windows(self) -> WindowList:
        return frontmost_windows()

    def obstructions_over(self, rect: WindowRect, *, owner: str) -> Obstructions:
        return obstructions_over(rect, owner=owner)

    def frontmost_app_name(self) -> str:
        return frontmost_app_name()

    def close(self) -> None: ...
