from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from string import ascii_lowercase, ascii_uppercase, digits
from typing import NamedTuple

import Quartz

from choto.inputctl.errors import UnknownKeyError

KEYCODES: dict[str, int] = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "o": 0x1F,
    "u": 0x20,
    "i": 0x22,
    "p": 0x23,
    "l": 0x25,
    "j": 0x26,
    "k": 0x28,
    "n": 0x2D,
    "m": 0x2E,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "5": 0x17,
    "6": 0x16,
    "7": 0x1A,
    "8": 0x1C,
    "9": 0x19,
    "0": 0x1D,
    "=": 0x18,
    "-": 0x1B,
    "]": 0x1E,
    "[": 0x21,
    "'": 0x27,
    ";": 0x29,
    "\\": 0x2A,
    ",": 0x2B,
    "/": 0x2C,
    ".": 0x2F,
    "`": 0x32,
    "minus": 0x1B,
    "hyphen": 0x1B,
    "dash": 0x1B,
    "equal": 0x18,
    "equals": 0x18,
    "bracketright": 0x1E,
    "rightbracket": 0x1E,
    "bracketleft": 0x21,
    "leftbracket": 0x21,
    "quote": 0x27,
    "apostrophe": 0x27,
    "semicolon": 0x29,
    "backslash": 0x2A,
    "comma": 0x2B,
    "slash": 0x2C,
    "period": 0x2F,
    "dot": 0x2F,
    "grave": 0x32,
    "backtick": 0x32,
    "backquote": 0x32,
    "return": 0x24,
    "enter": 0x24,
    "tab": 0x30,
    "space": 0x31,
    "delete": 0x33,
    "backspace": 0x33,
    "escape": 0x35,
    "esc": 0x35,
    "forwarddelete": 0x75,
    "home": 0x73,
    "end": 0x77,
    "pageup": 0x74,
    "pagedown": 0x79,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
    "f1": 0x7A,
    "f2": 0x78,
    "f3": 0x63,
    "f4": 0x76,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f8": 0x64,
    "f9": 0x65,
    "f10": 0x6D,
    "f11": 0x67,
    "f12": 0x6F,
}

MODIFIERS: dict[str, tuple[int, int]] = {
    "cmd": (0x37, Quartz.kCGEventFlagMaskCommand),
    "command": (0x37, Quartz.kCGEventFlagMaskCommand),
    "shift": (0x38, Quartz.kCGEventFlagMaskShift),
    "alt": (0x3A, Quartz.kCGEventFlagMaskAlternate),
    "opt": (0x3A, Quartz.kCGEventFlagMaskAlternate),
    "option": (0x3A, Quartz.kCGEventFlagMaskAlternate),
    "ctrl": (0x3B, Quartz.kCGEventFlagMaskControl),
    "control": (0x3B, Quartz.kCGEventFlagMaskControl),
    "fn": (0x3F, Quartz.kCGEventFlagMaskSecondaryFn),
}

SHIFT_KEYCODE, SHIFT_FLAG_MASK = MODIFIERS["shift"]

UNTYPEABLE_CATEGORIES = frozenset({"Cc", "Cs"})


UNSHIFTED_PUNCTUATION = "-=[]\\;',./`"

SHIFT_PAIRS: dict[str, str] = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "<": ",",
    ">": ".",
    "?": "/",
    "~": "`",
}

SHIFTED_KEY_NAMES: dict[str, str] = {
    "exclam": "!",
    "exclamation": "!",
    "at": "@",
    "numbersign": "#",
    "hash": "#",
    "dollar": "$",
    "percent": "%",
    "asciicircum": "^",
    "caret": "^",
    "ampersand": "&",
    "asterisk": "*",
    "star": "*",
    "parenleft": "(",
    "parenright": ")",
    "underscore": "_",
    "plus": "+",
    "braceleft": "{",
    "braceright": "}",
    "bar": "|",
    "pipe": "|",
    "colon": ":",
    "quotedbl": '"',
    "doublequote": '"',
    "less": "<",
    "greater": ">",
    "question": "?",
    "asciitilde": "~",
    "tilde": "~",
}

WHITESPACE_KEY_NAMES: dict[str, str] = {
    " ": "space",
    "\t": "tab",
    "\n": "return",
    "\r": "return",
}

RETURN_EVENT_CHAR = "\r"


def _build_char_keycodes() -> dict[str, tuple[int, bool]]:
    table: dict[str, tuple[int, bool]] = {}
    for char in ascii_lowercase:
        table[char] = (KEYCODES[char], False)
    for char in ascii_uppercase:
        table[char] = (KEYCODES[char.lower()], True)
    for char in digits:
        table[char] = (KEYCODES[char], False)
    for char in UNSHIFTED_PUNCTUATION:
        table[char] = (KEYCODES[char], False)
    for shifted, base in SHIFT_PAIRS.items():
        table[shifted] = (KEYCODES[base], True)
    for char, key_name in WHITESPACE_KEY_NAMES.items():
        table[char] = (KEYCODES[key_name], False)
    return table


CHAR_KEYCODES: dict[str, tuple[int, bool]] = _build_char_keycodes()


def resolve_char(char: str) -> tuple[int, bool] | None:
    if len(char) != 1:
        raise ValueError(f"resolve_char expects a single character, got {char!r}.")
    return CHAR_KEYCODES.get(char)


def event_char(char: str) -> str:
    return RETURN_EVENT_CHAR if char == "\n" else char


def is_modifier(key: str) -> bool:
    return key.strip().lower() in MODIFIERS


def resolve_keycode(key: str) -> int:
    normalized = key.strip().lower()
    code = KEYCODES.get(normalized)
    if code is not None:
        return code
    glyph = SHIFTED_KEY_NAMES.get(normalized)
    if glyph is not None:
        base = SHIFT_PAIRS[glyph]
        raise UnknownKeyError(
            f"The key {key!r} means {glyph!r}, which the US layout types as Shift and "
            f"{base!r} together, so it is not a key of its own: write it as the chord "
            f'["shift", {base!r}] (adding whatever else the shortcut holds).'
        )
    raise UnknownKeyError(
        f"Unknown key {key!r}: no virtual keycode is defined for it. "
        f"Known keys: {', '.join(sorted(KEYCODES))}."
    )


class Modifiers(NamedTuple):
    codes: tuple[int, ...]
    flags: int


class Chord(NamedTuple):
    keycode: int
    modifiers: Modifiers


class Keystroke(NamedTuple):
    keycode: int
    char: str
    shift: bool


class UnicodeRun(NamedTuple):
    text: str


class Typing(NamedTuple):
    pieces: tuple[Keystroke | UnicodeRun, ...]
    skipped: tuple[str, ...]


def resolve_modifiers(names: Sequence[str] | None) -> Modifiers:
    codes: list[int] = []
    flags = 0
    for name in names or ():
        normalized = name.strip().lower()
        try:
            code, mask = MODIFIERS[normalized]
        except KeyError as exc:
            raise UnknownKeyError(
                f"Unknown modifier {name!r}; expected one of {', '.join(sorted(MODIFIERS))}."
            ) from exc
        flags |= mask
        if code not in codes:
            codes.append(code)
    return Modifiers(tuple(codes), flags)


def resolve_chord(keys: Sequence[str]) -> Chord:
    if not keys:
        raise ValueError("hotkey requires a non-empty list of keys.")
    held = [key for key in keys if is_modifier(key)]
    pressed = [key for key in keys if not is_modifier(key)]
    if len(pressed) != 1:
        raise ValueError(
            f"hotkey requires exactly one non-modifier key, got {pressed!r} from {list(keys)!r}."
        )
    return Chord(resolve_keycode(pressed[0]), resolve_modifiers(held))


def plan_typing(text: str) -> Typing:
    pieces: list[Keystroke | UnicodeRun] = []
    unmapped: list[str] = []
    skipped: list[str] = []

    def close_run() -> None:
        if unmapped:
            pieces.append(UnicodeRun("".join(unmapped)))
            unmapped.clear()

    for char in text:
        resolved = resolve_char(char)
        if resolved is None:
            if unicodedata.category(char) in UNTYPEABLE_CATEGORIES:
                skipped.append(char)
            else:
                unmapped.append(char)
            continue
        close_run()
        keycode, needs_shift = resolved
        pieces.append(Keystroke(keycode, event_char(char), needs_shift))
    close_run()
    return Typing(tuple(pieces), tuple(skipped))
