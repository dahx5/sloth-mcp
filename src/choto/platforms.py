from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from choto.config import PLATFORM_AUTO, PLATFORM_MACOS, Settings

if TYPE_CHECKING:  # pragma: no cover - for type checkers, never at runtime
    from choto.capture.grabber import PullGrabber
    from choto.capture.source import ScreenSource
    from choto.executor.applauncher import AppLauncher
    from choto.inputctl.protocol import InputBackend
    from choto.overlay.backend import OverlayBackend
    from choto.vision.windowsource import WindowSource

__all__ = [
    "PLATFORMS",
    "PlatformBackends",
    "PlatformError",
    "create_overlay_backend",
    "current_platform",
    "resolve_platform",
]


class PlatformError(RuntimeError): ...


_SYS_PLATFORMS = {
    "darwin": PLATFORM_MACOS,
}


@dataclass(frozen=True)
class PlatformBackends:
    window_source: Callable[[], WindowSource]
    input_backend: Callable[[Callable[[], bool] | None], InputBackend]
    app_launcher: Callable[[], AppLauncher]
    overlay_backend: Callable[[], OverlayBackend]
    screen_source: Callable[[], ScreenSource]
    screen_grabber: Callable[[], PullGrabber]
    close: Callable[[], None]


def _macos_backends() -> PlatformBackends:
    from choto.capture.sckshared import SharedStream, open_shared_sck_source

    stream = SharedStream()

    def window_source() -> WindowSource:
        from choto.vision.parser import MacWindowSource

        return MacWindowSource()

    def input_backend(stop_requested: Callable[[], bool] | None) -> InputBackend:
        from choto.inputctl.controller import MacInputBackend

        return MacInputBackend(stop_requested=stop_requested)

    def app_launcher() -> AppLauncher:
        from choto.executor.apps import MacAppLauncher

        return MacAppLauncher()

    def overlay_backend() -> OverlayBackend:
        from choto.overlay.supervisor import MacOverlayBackend

        return MacOverlayBackend()

    def screen_grabber() -> PullGrabber:
        return open_shared_sck_source(stream)

    def screen_source() -> ScreenSource:
        return open_shared_sck_source(stream)

    return PlatformBackends(
        window_source=window_source,
        input_backend=input_backend,
        app_launcher=app_launcher,
        overlay_backend=overlay_backend,
        screen_source=screen_source,
        screen_grabber=screen_grabber,
        close=stream.close,
    )


PLATFORMS: dict[str, Callable[[], PlatformBackends]] = {
    PLATFORM_MACOS: _macos_backends,
}
"""The name → implementation-set table ``platform_backend`` selects from.

This is the one place a platform is wired in: adding a port means adding a row
here, and no consumer names an implementation directly.
"""


def current_platform() -> str:
    for prefix, name in _SYS_PLATFORMS.items():
        if sys.platform.startswith(prefix):
            return name
    raise PlatformError(
        f"Choto has no implementation for sys.platform {sys.platform!r}; "
        f"supported platforms are {sorted(PLATFORMS)}."
    )


def resolve_platform(name: str | None = None) -> PlatformBackends:
    resolved = current_platform() if not name or name == PLATFORM_AUTO else name
    builder = PLATFORMS.get(resolved)
    if builder is None:
        raise PlatformError(
            f"Unknown platform {resolved!r} (setting platform_backend); "
            f"available platforms are {sorted(PLATFORMS)}."
        )
    return builder()


def create_overlay_backend(backends: PlatformBackends, settings: Settings) -> OverlayBackend:
    if not settings.overlay_enabled:
        from choto.overlay.backend import NullOverlayBackend

        return NullOverlayBackend()
    return backends.overlay_backend()
