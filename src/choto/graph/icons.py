from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from choto.models import BBox, IconLabelSource

ICON_ELEMENT_CONFIDENCE = 1.0

BORROWED_NAME_MARKER = " (borrowed name)"


def icon_element_text(label: str, *, at_home: bool) -> str:
    if not label:
        return ""
    return label if at_home else f"{label}{BORROWED_NAME_MARKER}"


GLYPH_COLOUR_SEPARATOR = "-c"

GLYPH_COLOUR_CLASSES = 7


def glyph_key(phash: str, colour_class: int) -> str:
    _validate_digest(phash)
    _validate_colour_class(colour_class, phash)
    return f"{phash}{GLYPH_COLOUR_SEPARATOR}{colour_class}"


def parse_glyph_key(key: str) -> tuple[str, int]:
    digest, separator, colour = key.rpartition(GLYPH_COLOUR_SEPARATOR)
    if not separator:
        raise ValueError(
            f"A glyph key must be a hexadecimal pHash followed by its colour class, "
            f"e.g. 'b1c3e0f2a4d68590{GLYPH_COLOUR_SEPARATOR}3'; got {key!r}."
        )
    _validate_digest(digest, key)
    try:
        colour_class = int(colour)
    except ValueError as exc:
        raise ValueError(
            f"A glyph key's colour class must be a number, got {colour!r} in {key!r}."
        ) from exc
    _validate_colour_class(colour_class, key)
    return digest, colour_class


def _validate_digest(digest: str, key: str | None = None) -> None:
    subject = key if key is not None else digest
    if not digest:
        raise ValueError(f"A glyph key must not have an empty pHash, got {subject!r}.")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(
            f"A glyph key must be a hexadecimal pHash followed by its colour class, "
            f"e.g. 'b1c3e0f2a4d68590{GLYPH_COLOUR_SEPARATOR}3'; got {subject!r}."
        ) from exc


def _validate_colour_class(colour_class: int, subject: str) -> None:
    if not 0 <= colour_class < GLYPH_COLOUR_CLASSES:
        raise ValueError(
            f"A glyph key's colour class must be in 0..{GLYPH_COLOUR_CLASSES - 1}, "
            f"got {colour_class} in {subject!r}."
        )


def _validate_phash(phash: str) -> None:
    parse_glyph_key(phash)


@dataclass(frozen=True)
class IconRecord:
    bbox: BBox
    phash: str
    crop_png: bytes

    def __post_init__(self) -> None:
        if self.bbox.w <= 0 or self.bbox.h <= 0:
            raise ValueError(f"An icon box must have area, got {self.bbox.w}x{self.bbox.h}.")
        _validate_phash(self.phash)
        if not self.crop_png:
            raise ValueError("An icon crop must not be empty: the sheet is built from it.")


@dataclass(frozen=True)
class GlyphPosition:
    screen_id: int
    window_title: str
    bbox: BBox


@dataclass(frozen=True)
class UnlabeledGlyph:
    glyph_id: int
    app_name: str
    phash: str
    crop_png: bytes
    positions: tuple[GlyphPosition, ...]


@dataclass(frozen=True)
class GlyphLabel:
    app_name: str
    phash: str
    label: str
    source: IconLabelSource

    def __post_init__(self) -> None:
        if not self.app_name.strip():
            raise ValueError("A glyph label must name its application.")
        _validate_phash(self.phash)
        if not self.label.strip():
            raise ValueError(
                f"A glyph label must not be blank (app {self.app_name!r}, glyph {self.phash})."
            )
        if self.source is IconLabelSource.UNLABELED:
            raise ValueError("A glyph label must say where it came from: 'llm' or 'tooltip'.")

    @property
    def text(self) -> str:
        return self.label.strip()


@dataclass(frozen=True)
class IconSaveResult:
    elements_created: int
    glyphs_created: int
    elements_named: int


@dataclass(frozen=True)
class GlyphLabelResult:
    glyphs_labeled: int
    elements_updated: int


@dataclass(frozen=True)
class IconCounts:
    glyphs_total: int
    glyphs_labeled: int
    icon_elements: int


def encode_icon_png(crop: np.ndarray) -> bytes:
    if not isinstance(crop, np.ndarray):
        raise TypeError(f"Expected a numpy array, got {type(crop).__name__}.")
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB crop, got shape {crop.shape}.")
    if crop.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 crop, got dtype {crop.dtype}.")
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError(f"Expected a crop with area, got shape {crop.shape}.")

    buffer = io.BytesIO()
    Image.fromarray(crop, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def decode_icon_png(crop_png: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(crop_png)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"An icon crop is not a readable image: {exc}") from exc


def icon_record(bbox: BBox, phash: str, crop: np.ndarray) -> IconRecord:
    return IconRecord(bbox=bbox, phash=phash, crop_png=encode_icon_png(crop))
