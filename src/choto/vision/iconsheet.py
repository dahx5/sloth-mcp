from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from choto.log import get_logger

_log = get_logger(__name__)

_PIXELS_PER_IMAGE_TOKEN = 750.0

_CHECK_LIGHT = (232, 232, 232)
_CHECK_DARK = (206, 206, 206)
_BORDER_COLOUR = (64, 64, 70)
_CAPTION_BACKGROUND = (250, 250, 250)
_CAPTION_INK = (0, 0, 0)

_CAPTION_SIDE_PADDING = 2

_MONOSPACE_FONT_PATHS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
)


@dataclass(frozen=True, eq=False)
class SheetEntry:
    crop: np.ndarray
    key: str
    context_lines: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.crop, np.ndarray):
            raise TypeError(f"crop must be a numpy array, got {type(self.crop).__name__}.")
        if self.crop.ndim != 3 or self.crop.shape[2] != 3:
            raise ValueError(f"crop must be an (H, W, 3) RGB array, got shape {self.crop.shape}.")
        if self.crop.dtype != np.uint8:
            raise ValueError(f"crop must be uint8, got dtype {self.crop.dtype}.")
        if self.crop.shape[0] < 1 or self.crop.shape[1] < 1:
            raise ValueError(f"crop must have pixels, got shape {self.crop.shape}.")
        if not isinstance(self.key, str):
            raise TypeError(f"key must be a string, got {type(self.key).__name__}.")
        if not self.key.strip():
            raise ValueError("key must not be blank.")
        if isinstance(self.context_lines, str):
            raise TypeError("context_lines must be a sequence of strings, not one string.")
        cleaned = tuple(
            " ".join(str(line).split()) for line in self.context_lines if str(line).strip()
        )
        if not cleaned:
            raise ValueError(
                "context_lines must carry at least one non-blank line: an icon with no "
                "context cannot be named."
            )
        object.__setattr__(self, "context_lines", cleaned)


@dataclass(frozen=True)
class SheetParams:
    max_width_px: int = 1024
    max_tokens_per_sheet: int = 600
    margin: int = 4
    cell_gap: int = 4
    border: int = 1
    caption_height: int = 18
    caption_font_size: int = 14
    check_size: int = 8

    def __post_init__(self) -> None:
        if self.max_width_px < 1:
            raise ValueError(f"max_width_px must be positive, got {self.max_width_px}.")
        if self.max_tokens_per_sheet < 1:
            raise ValueError(
                f"max_tokens_per_sheet must be positive, got {self.max_tokens_per_sheet}."
            )
        if self.margin < 0 or self.cell_gap < 0:
            raise ValueError(
                f"margin and cell_gap must not be negative, got margin={self.margin} "
                f"cell_gap={self.cell_gap}."
            )
        if self.border < 1:
            raise ValueError(f"border must be at least 1 pixel, got {self.border}.")
        if self.caption_font_size < 6:
            raise ValueError(f"caption_font_size must be at least 6, got {self.caption_font_size}.")
        if self.caption_height < self.caption_font_size:
            raise ValueError(
                f"caption_height ({self.caption_height}) must fit caption_font_size "
                f"({self.caption_font_size})."
            )
        if self.check_size < 1:
            raise ValueError(f"check_size must be positive, got {self.check_size}.")

    @property
    def max_pixels_per_sheet(self) -> float:
        return self.max_tokens_per_sheet * _PIXELS_PER_IMAGE_TOKEN


DEFAULT_SHEET_PARAMS = SheetParams()


@dataclass(frozen=True)
class SheetResult:
    png: bytes
    legend: str
    cell_keys: dict[int, str]


@dataclass(frozen=True)
class _Cell:
    number: int
    entry: SheetEntry
    width: int
    height: int

    @property
    def crop_offset_x(self) -> int:
        return (self.width - self.entry.crop.shape[1]) // 2


@dataclass(frozen=True)
class _Row:
    cells: tuple[_Cell, ...]
    width: int
    height: int


@dataclass(frozen=True)
class _Sheet:
    rows: tuple[_Row, ...]
    width: int
    height: int
    cells: tuple[_Cell, ...]


def build_icon_sheet(
    entries: Sequence[SheetEntry], params: SheetParams = DEFAULT_SHEET_PARAMS
) -> list[SheetResult]:
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise TypeError(f"entries must be a sequence, got {type(entries).__name__}.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, SheetEntry):
            raise TypeError(f"entries[{index}] must be a SheetEntry, got {type(entry).__name__}.")
    if not entries:
        return []

    keys = [entry.key for entry in entries]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"entries must carry distinct keys; repeated: {', '.join(duplicates)}.")

    font = _caption_font(params.caption_font_size)
    cells = [
        _build_cell(number, entry, font, params) for number, entry in enumerate(entries, start=1)
    ]
    sheets = _pack_rows(_wrap_cells(cells, params), params)
    results = [_render(sheet, font, params) for sheet in sheets]

    _log.debug(
        "iconsheet.built",
        entries=len(entries),
        sheets=len(results),
        sizes=[(sheet.width, sheet.height) for sheet in sheets],
        image_tokens=[
            round(sheet.width * sheet.height / _PIXELS_PER_IMAGE_TOKEN) for sheet in sheets
        ],
    )
    return results


def _caption_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _MONOSPACE_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    _log.debug("iconsheet.font_fallback", tried=_MONOSPACE_FONT_PATHS, size=size)
    return ImageFont.load_default(size=size)


def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return math.ceil(right - left), math.ceil(bottom - top)


def _build_cell(
    number: int, entry: SheetEntry, font: ImageFont.FreeTypeFont, params: SheetParams
) -> _Cell:
    crop_h, crop_w = entry.crop.shape[:2]
    caption_w = _text_size(str(number), font)[0] + 2 * _CAPTION_SIDE_PADDING
    width = max(crop_w + 2 * params.border, caption_w)
    height = crop_h + 2 * params.border + params.caption_height
    return _Cell(number=number, entry=entry, width=width, height=height)


def _wrap_cells(cells: Sequence[_Cell], params: SheetParams) -> list[_Row]:
    limit = max(1, params.max_width_px - 2 * params.margin)
    rows: list[_Row] = []
    current: list[_Cell] = []
    width = 0
    for cell in cells:
        extra = cell.width if not current else params.cell_gap + cell.width
        if current and width + extra > limit:
            rows.append(_close_row(current, width))
            current, width = [cell], cell.width
            continue
        current.append(cell)
        width += extra
    if current:
        rows.append(_close_row(current, width))
    return rows


def _close_row(cells: Sequence[_Cell], width: int) -> _Row:
    return _Row(cells=tuple(cells), width=width, height=max(cell.height for cell in cells))


def _pack_rows(rows: Sequence[_Row], params: SheetParams) -> list[_Sheet]:
    sheets: list[_Sheet] = []
    current: list[_Row] = []
    for row in rows:
        candidate = [*current, row]
        width, height = _sheet_size(candidate, params)
        if current and width * height > params.max_pixels_per_sheet:
            sheets.append(_close_sheet(current, params))
            current = [row]
            continue
        current = candidate
    if current:
        sheets.append(_close_sheet(current, params))
    return sheets


def _sheet_size(rows: Sequence[_Row], params: SheetParams) -> tuple[int, int]:
    width = max(row.width for row in rows) + 2 * params.margin
    height = sum(row.height for row in rows) + params.cell_gap * (len(rows) - 1) + 2 * params.margin
    return width, height


def _close_sheet(rows: Sequence[_Row], params: SheetParams) -> _Sheet:
    width, height = _sheet_size(rows, params)
    cells = tuple(cell for row in rows for cell in row.cells)
    return _Sheet(rows=tuple(rows), width=width, height=height, cells=cells)


def _checkerboard(width: int, height: int, params: SheetParams) -> Image.Image:
    rows = np.arange(height)[:, None] // params.check_size
    columns = np.arange(width)[None, :] // params.check_size
    light = (rows + columns) % 2 == 0
    board = np.where(
        light[:, :, None],
        np.array(_CHECK_LIGHT, dtype=np.uint8),
        np.array(_CHECK_DARK, dtype=np.uint8),
    ).astype(np.uint8)
    return Image.fromarray(board, mode="RGB")


def _render(sheet: _Sheet, font: ImageFont.FreeTypeFont, params: SheetParams) -> SheetResult:
    canvas = _checkerboard(sheet.width, sheet.height, params)
    draw = ImageDraw.Draw(canvas)
    placements: list[tuple[_Cell, int, int]] = []

    y = params.margin
    for row in sheet.rows:
        x = params.margin
        for cell in row.cells:
            crop_h, crop_w = cell.entry.crop.shape[:2]
            crop_x = x + cell.crop_offset_x
            crop_y = y + params.border
            draw.rectangle(
                (
                    crop_x - params.border,
                    crop_y - params.border,
                    crop_x + crop_w + params.border - 1,
                    crop_y + crop_h + params.border - 1,
                ),
                outline=_BORDER_COLOUR,
                width=params.border,
            )
            _draw_caption(draw, cell, x, y + crop_h + 2 * params.border, font, params)
            placements.append((cell, crop_x, crop_y))
            x += cell.width + params.cell_gap
        y += row.height + params.cell_gap

    for cell, crop_x, crop_y in placements:
        canvas.paste(
            Image.fromarray(np.ascontiguousarray(cell.entry.crop), mode="RGB"), (crop_x, crop_y)
        )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return SheetResult(
        png=buffer.getvalue(),
        legend="\n".join(
            f"#{cell.number}: {'; '.join(cell.entry.context_lines)}" for cell in sheet.cells
        ),
        cell_keys={cell.number: cell.entry.key for cell in sheet.cells},
    )


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    cell: _Cell,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    params: SheetParams,
) -> None:
    draw.rectangle(
        (x, y, x + cell.width - 1, y + params.caption_height - 1), fill=_CAPTION_BACKGROUND
    )
    text = str(cell.number)
    left, top, right, bottom = font.getbbox(text)
    text_x = x + (cell.width - (right - left)) / 2 - left
    text_y = y + (params.caption_height - (bottom - top)) / 2 - top
    draw.text((text_x, text_y), text, font=font, fill=_CAPTION_INK)
