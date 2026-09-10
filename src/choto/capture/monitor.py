from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorInfo:
    width_px: int
    height_px: int
    width_pt: int
    height_pt: int
    scale: float
