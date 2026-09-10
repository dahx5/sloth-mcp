from __future__ import annotations

import re
from dataclasses import dataclass

_STEM_SUFFIX = "-"

_MIN_STEM_CHARS = 4

_WORD_RE = re.compile(r"\w+")

POLARITY_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("previous", "prev", "earlier"), ("next", "later")),
    (("back", "backward", "backwards"), ("forward", "forwards")),
    (("open-",), ("close-",)),
    (("on",), ("off",)),
    (("enable-",), ("disable-",)),
    (("show-", "shown", "reveal-"), ("hide", "hides", "hiding", "hidden", "conceal-")),
    (
        ("add", "adds", "added", "adding"),
        ("remove", "removes", "removed", "removing", "delete", "deletes", "deleted", "deleting"),
    ),
    (("create-",), ("delete-", "destroy-")),
    (("install-",), ("uninstall-",)),
    (
        ("start-", "begin", "begins", "resume-", "play"),
        ("stop-", "end", "ends", "paus-", "finish-"),
    ),
    (("connect-",), ("disconnect-",)),
    (("attach-",), ("detach-",)),
    (("up",), ("down",)),
    (("left",), ("right",)),
    (("in", "signin", "login"), ("out", "signout", "logout")),
    (("expand-",), ("collaps-",)),
    (("maximi-",), ("minimi-",)),
    (("more", "increase-", "increment-", "louder"), ("less", "fewer", "decrease-", "quieter")),
    (
        ("accept-", "approve-", "agree", "agrees", "allow-", "confirm-"),
        ("decline-", "reject-", "deny", "denies", "denied", "block-", "cancel-", "dismiss-"),
    ),
    (("yes",), ("no",)),
    (("mute", "muted", "mutes"), ("unmute", "unmuted", "unmutes")),
    (("lock", "locks", "locked", "locking"), ("unlock", "unlocks", "unlocked", "unlocking")),
    (("check-",), ("uncheck-",)),
    (("select-",), ("deselect-", "unselect-")),
    (("read", "reads"), ("unread",)),
    (("like", "likes", "liked"), ("dislike", "dislikes", "disliked", "unlike")),
    (("follow", "follows", "followed", "following"), ("unfollow", "unfollows", "unfollowed")),
    (("subscrib-",), ("unsubscrib-",)),
    (("undo",), ("redo",)),
    (("save-",), ("discard-",)),
    (("import-",), ("export-",)),
)


@dataclass(frozen=True)
class OppositePoles:
    target_word: str
    candidate_word: str


@dataclass(frozen=True)
class _Watch:
    word: str
    same: tuple[str, ...]
    opposite: tuple[str, ...]


@dataclass(frozen=True)
class PolarityGuard:
    watches: tuple[_Watch, ...]

    @property
    def watching(self) -> bool:
        return bool(self.watches)

    def opposes(self, text: str) -> OppositePoles | None:
        if not self.watches:
            return None
        tokens = _tokens(text)
        for watch in self.watches:
            candidate_word = _carried(tokens, watch.opposite)
            if candidate_word is not None and _carried(tokens, watch.same) is None:
                return OppositePoles(target_word=watch.word, candidate_word=candidate_word)
        return None


def polarity_guard(text: str) -> PolarityGuard:
    tokens = _tokens(text)
    watches: list[_Watch] = []
    for left, right in POLARITY_PAIRS:
        on_left = _carried(tokens, left)
        on_right = _carried(tokens, right)
        if on_left is not None and on_right is None:
            watches.append(_Watch(word=on_left, same=left, opposite=right))
        elif on_right is not None and on_left is None:
            watches.append(_Watch(word=on_right, same=right, opposite=left))
    return PolarityGuard(watches=tuple(watches))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _matches(token: str, marker: str) -> bool:
    if marker.endswith(_STEM_SUFFIX):
        return token.startswith(marker[: -len(_STEM_SUFFIX)])
    return token == marker


def _carried(tokens: list[str], markers: tuple[str, ...]) -> str | None:
    for token in tokens:
        for marker in markers:
            if _matches(token, marker):
                return token
    return None


def _body(marker: str) -> str:
    return marker[: -len(_STEM_SUFFIX)] if marker.endswith(_STEM_SUFFIX) else marker


def _collide(left: str, right: str) -> bool:
    left_body, right_body = _body(left), _body(right)
    if left.endswith(_STEM_SUFFIX) and right.endswith(_STEM_SUFFIX):
        return left_body.startswith(right_body) or right_body.startswith(left_body)
    if left.endswith(_STEM_SUFFIX):
        return right_body.startswith(left_body)
    if right.endswith(_STEM_SUFFIX):
        return left_body.startswith(right_body)
    return left_body == right_body


def _validate(pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]) -> None:
    for left, right in pairs:
        if not left or not right:
            raise ValueError(f"Polarity pair with an empty pole: {left!r}/{right!r}")
        for markers in (left, right):
            for marker in markers:
                body = _body(marker)
                if not _WORD_RE.fullmatch(body) or body != body.casefold():
                    raise ValueError(
                        f"Polarity marker {marker!r} must be one casefolded word "
                        "of letters or digits"
                    )
                if marker.endswith(_STEM_SUFFIX) and len(body) < _MIN_STEM_CHARS:
                    raise ValueError(
                        f"Polarity stem {marker!r} is shorter than {_MIN_STEM_CHARS} "
                        "characters and would claim unrelated words"
                    )
        for marker in left:
            for other in right:
                if _collide(marker, other):
                    raise ValueError(
                        f"Polarity markers {marker!r} and {other!r} are on opposite poles "
                        "of the same pair but match the same token, which disables the pair"
                    )


_validate(POLARITY_PAIRS)
