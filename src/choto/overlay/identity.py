from __future__ import annotations

import threading

_lock = threading.Lock()
_helper_pids: set[int] = set()


def register_helper(pid: int) -> None:
    with _lock:
        _helper_pids.add(pid)


def forget_helper(pid: int) -> None:
    with _lock:
        _helper_pids.discard(pid)


def is_overlay_helper(pid: int) -> bool:
    with _lock:
        return pid in _helper_pids
