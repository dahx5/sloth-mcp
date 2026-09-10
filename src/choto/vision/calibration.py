from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

_MAX_MINUTES_WITH_HOURS = 60
_MAX_SECONDS = 60
_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 3600.0
_TIMECODE_HMS = re.compile(r"^(?P<sign>[-+])?(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2}(?:[.,]\d+)?)$")
_TIMECODE_MS = re.compile(r"^(?P<sign>[-+])?(?P<m>\d{1,3}):(?P<s>\d{2}(?:[.,]\d+)?)$")
_MINUS_CHARS = "-−–"
_THOUSANDS_SPACES = "   "
_NUMBER_CHARS = frozenset("0123456789 ,.")
_GROUP_SIZE = 3
_DECIBEL_SUFFIX = "db"


def _parse_timecode(text: str) -> float | None:
    stripped = text.strip()
    match = _TIMECODE_HMS.match(stripped)
    if match is not None:
        minutes = int(match.group("m"))
        seconds = float(match.group("s").replace(",", "."))
        if minutes >= _MAX_MINUTES_WITH_HOURS or seconds >= _MAX_SECONDS:
            return None
        total = int(match.group("h")) * _SECONDS_PER_HOUR + minutes * _SECONDS_PER_MINUTE + seconds
        return -total if match.group("sign") == "-" else total

    match = _TIMECODE_MS.match(stripped)
    if match is None:
        return None
    seconds = float(match.group("s").replace(",", "."))
    if seconds >= _MAX_SECONDS:
        return None
    total = int(match.group("m")) * _SECONDS_PER_MINUTE + seconds
    return -total if match.group("sign") == "-" else total


def _split_sign(text: str) -> tuple[bool, str]:
    if text[:1] and text[0] in _MINUS_CHARS:
        return True, text[1:]
    if text.startswith("+"):
        return False, text[1:]
    return False, text


def _looks_like_group(digits: str) -> bool:
    return 1 <= len(digits) <= _GROUP_SIZE and not digits.startswith("0")


def _ungroup(digits: str, separator: str) -> str | None:
    groups = digits.split(separator)
    if not _looks_like_group(groups[0]):
        return None
    if any(len(group) != _GROUP_SIZE for group in groups[1:]):
        return None
    return "".join(groups)


def _decimal_separator(body: str, separators: Sequence[str]) -> str:
    last = separators[-1]
    if last == " ":
        return ""
    non_space = [char for char in separators if char != " "]
    if non_space.count(last) > 1:
        return ""
    if len(set(non_space)) > 1 or " " in separators:
        return last
    before, _, after = body.rpartition(last)
    if len(after) == _GROUP_SIZE and _looks_like_group(before):
        return ""
    return last


def _parse_number(text: str) -> float | None:
    negative, body = _split_sign(text.strip())
    for space in _THOUSANDS_SPACES:
        body = body.replace(space, " ")
    if not body or not body[0].isdigit() or not body[-1].isdigit():
        return None
    if any(char not in _NUMBER_CHARS for char in body):
        return None
    if any(not first.isdigit() and not second.isdigit() for first, second in pairwise(body)):
        return None

    separators = [char for char in body if not char.isdigit()]
    if separators:
        separator = _decimal_separator(body, separators)
        integer_part, _, fraction = body.rpartition(separator) if separator else (body, "", "")
    else:
        integer_part, fraction = body, ""

    group_kinds = {char for char in integer_part if not char.isdigit()}
    if len(group_kinds) > 1:
        return None
    if group_kinds:
        ungrouped = _ungroup(integer_part, group_kinds.pop())
        if ungrouped is None:
            return None
        integer_part = ungrouped

    value = float(f"{integer_part}.{fraction}") if fraction else float(integer_part)
    return -value if negative else value


def _parse_percent(text: str) -> float | None:
    stripped = text.strip()
    if not stripped.endswith("%"):
        return None
    return _parse_number(stripped[:-1])


def _parse_decibel(text: str) -> float | None:
    stripped = text.strip()
    if stripped[-2:].lower() != _DECIBEL_SUFFIX:
        return None
    return _parse_number(stripped[:-2])


@dataclass(frozen=True)
class TickParser:
    name: str
    units: str
    parse: Callable[[str], float | None]


DEFAULT_PARSERS: tuple[TickParser, ...] = (
    TickParser(name="timecode", units="s", parse=_parse_timecode),
    TickParser(name="percent", units="%", parse=_parse_percent),
    TickParser(name="decibel", units="dB", parse=_parse_decibel),
    TickParser(name="number", units="", parse=_parse_number),
)


__all__ = [
    "DEFAULT_PARSERS",
    "TickParser",
]
