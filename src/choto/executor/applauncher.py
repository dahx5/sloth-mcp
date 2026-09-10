from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AppActivationError(RuntimeError): ...


@dataclass(frozen=True)
class AppIdentity:
    display_name: str


class AppLauncher(Protocol):
    def activate(self, app_name: str) -> str: ...

    def resolve(self, app_name: str) -> AppIdentity | None: ...

    def same_app(self, observed: str, wanted: str) -> bool: ...

    def app_version(self, app_name: str) -> str: ...

    def close(self) -> None: ...
