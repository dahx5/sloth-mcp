from __future__ import annotations

from typing import Any

from AppKit import NSPasteboard, NSPasteboardTypeString

from choto.log import get_logger

_log = get_logger(__name__)


def general_pasteboard() -> Any:
    return NSPasteboard.generalPasteboard()


def read_clipboard_text() -> str | None:
    pasteboard = general_pasteboard()
    if pasteboard is None:
        _log.warning("clipboard.unavailable")
        return None
    value = pasteboard.stringForType_(NSPasteboardTypeString)
    if value is None:
        return None
    return str(value)
