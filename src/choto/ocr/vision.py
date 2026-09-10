from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import Quartz
import Vision
from PIL import Image

from choto.config import Settings
from choto.log import get_logger
from choto.models import BBox, OcrLine
from choto.ocr.engine import OcrError, OcrPasses, bbox_overlap_fraction, too_small_to_recognize

_log = get_logger(__name__)


@dataclass(frozen=True)
class GlyphCrop:
    x0: int
    y0: int
    x1: int
    y1: int
    factor: int = 1

    def to_frame(self, box: BBox) -> BBox:
        return BBox(
            x=self.x0 + round(box.x / self.factor),
            y=self.y0 + round(box.y / self.factor),
            w=round(box.w / self.factor),
            h=round(box.h / self.factor),
        )


@dataclass(frozen=True)
class GlyphPlan:
    crops: tuple[GlyphCrop, ...]
    unreadable: tuple[str, ...]
    scoped: bool


def vision_bbox_to_pixels(
    norm_box: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> BBox:
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_w}x{img_h}.")

    nx, ny, nw, nh = norm_box
    x = round(nx * img_w)
    y = round((1.0 - (ny + nh)) * img_h)
    w = round(nw * img_w)
    h = round(nh * img_h)

    x = min(max(x, 0), img_w)
    y = min(max(y, 0), img_h)
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    return BBox(x=x, y=y, w=w, h=h)


def _cgimage_from_rgb(image: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype {image.dtype}.")

    h, w, _ = image.shape
    if w == 0 or h == 0:
        raise ValueError("Cannot OCR an empty image.")

    rgb = np.ascontiguousarray(image)
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    rgba = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))
    data = rgba.tobytes()

    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    bitmap_info = Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrderDefault
    cg_image = Quartz.CGImageCreate(
        w,
        h,
        8,
        32,
        w * 4,
        color_space,
        bitmap_info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    if cg_image is None:
        raise OcrError("Failed to construct a CGImage from the input frame.")
    return cg_image


def _lines_from_observations(observations: Sequence[Any], width: int, height: int) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for observation in observations:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        top = candidates[0]
        text = top.string()
        if not text:
            continue
        box = observation.boundingBox()
        norm_box = (
            float(box.origin.x),
            float(box.origin.y),
            float(box.size.width),
            float(box.size.height),
        )
        lines.append(
            OcrLine(
                text=text,
                bbox=vision_bbox_to_pixels(norm_box, width, height),
                confidence=min(max(float(top.confidence()), 0.0), 1.0),
            )
        )
    return lines


def _grid_tiles(
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    *,
    cols: int,
    rows: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    step_x = width / cols
    step_y = height / rows

    tiles: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x0 = max(0, round(col * step_x) - overlap)
            y0 = max(0, round(row * step_y) - overlap)
            x1 = min(width, round((col + 1) * step_x) + overlap)
            y1 = min(height, round((row + 1) * step_y) + overlap)
            tiles.append((origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1))
    return tiles


def tile_limit(frame_size: int, divisor: int) -> int:
    if frame_size <= 0:
        raise ValueError(f"Frame size must be positive, got {frame_size}.")
    if divisor < 1:
        raise ValueError(f"Tile divisor must be at least 1, got {divisor}.")
    return max(1, -(-frame_size // divisor))


def region_tiles(
    region: BBox,
    *,
    max_tile_w: int,
    max_tile_h: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    if region.w <= 0 or region.h <= 0:
        raise ValueError(f"Region must have a positive area, got {region.w}x{region.h}.")
    if max_tile_w < 1 or max_tile_h < 1:
        raise ValueError(f"Tile limits must be positive, got {max_tile_w}x{max_tile_h}.")
    if overlap < 0:
        raise ValueError(f"Tile overlap must not be negative, got {overlap}.")

    return _grid_tiles(
        region.x,
        region.y,
        region.w,
        region.h,
        cols=-(-region.w // max_tile_w),
        rows=-(-region.h // max_tile_h),
        overlap=overlap,
    )


def clamp_regions(regions: Sequence[BBox], width: int, height: int) -> list[BBox]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {width}x{height}.")

    clamped: list[BBox] = []
    for region in regions:
        x0 = min(max(region.x, 0), width)
        y0 = min(max(region.y, 0), height)
        x1 = min(max(region.x + region.w, 0), width)
        y1 = min(max(region.y + region.h, 0), height)
        if x1 <= x0 or y1 <= y0:
            continue
        clamped.append(BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))
    return clamped


def merge_ocr_lines(
    primary: list[OcrLine],
    glyphs: list[OcrLine],
    *,
    max_chars: int,
    min_overlap_fraction: float,
) -> list[OcrLine]:
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}.")
    if not 0.0 < min_overlap_fraction <= 1.0:
        raise ValueError(f"min_overlap_fraction must be in (0, 1], got {min_overlap_fraction}.")

    merged = list(primary)
    for line in glyphs:
        text = line.text.strip()
        if not text or len(text) > max_chars:
            continue
        if any(
            bbox_overlap_fraction(line.bbox, kept.bbox) >= min_overlap_fraction for kept in merged
        ):
            continue
        merged.append(line.model_copy(update={"text": text}))
    return merged


def _magnify(image: np.ndarray, factor: int) -> np.ndarray:
    height, width = image.shape[0], image.shape[1]
    enlarged = Image.fromarray(image, mode="RGB").resize(
        (width * factor, height * factor), Image.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(enlarged, dtype=np.uint8))


def _supported_revisions() -> list[int]:
    index_set = Vision.VNRecognizeTextRequest.supportedRevisions()
    if index_set is None or index_set.count() == 0:
        return []
    return [
        revision
        for revision in range(int(index_set.firstIndex()), int(index_set.lastIndex()) + 1)
        if index_set.containsIndex_(revision)
    ]


def resolve_glyph_revision(configured: int, supported: Sequence[int]) -> int:
    if not supported or configured in supported:
        return configured
    return min(supported, key=lambda revision: (abs(revision - configured), revision))


_reported_revision_fallbacks: set[tuple[int, int]] = set()
_revision_fallback_lock = threading.Lock()


def _report_revision_fallback(configured: int, chosen: int, supported: Sequence[int]) -> None:
    key = (configured, chosen)
    with _revision_fallback_lock:
        if key in _reported_revision_fallbacks:
            return
        _reported_revision_fallbacks.add(key)
    _log.warning(
        "ocr.glyph_revision_unavailable",
        configured=configured,
        using=chosen,
        supported=list(supported),
        detail=(
            "the revision the glyph pass is pinned to is gone from this system; "
            "single characters may go unread"
        ),
    )


class AppleVisionOcr:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def recognize(
        self,
        image: np.ndarray,
        *,
        glyph_regions: Sequence[BBox] | None = None,
    ) -> list[OcrLine]:
        return self.recognize_passes(image, glyph_regions=glyph_regions).lines

    def recognize_passes(
        self,
        image: np.ndarray,
        *,
        glyph_regions: Sequence[BBox] | None = None,
    ) -> OcrPasses:
        settings = self._settings
        primary = self._recognize_frame(image, revision=settings.ocr_text_revision)
        if not settings.ocr_glyph_pass_enabled:
            return OcrPasses(lines=primary, text_lines=primary)

        glyphs = self._recognize_glyphs(image, glyph_regions)
        merged = merge_ocr_lines(
            primary,
            glyphs,
            max_chars=settings.ocr_glyph_max_chars,
            min_overlap_fraction=settings.ocr_duplicate_overlap_fraction,
        )
        _log.debug(
            "ocr.passes_merged",
            primary_lines=len(primary),
            glyph_candidates=len(glyphs),
            glyphs_added=len(merged) - len(primary),
        )
        return OcrPasses(lines=merged, text_lines=primary)

    def glyph_plan(self, width: int, height: int, regions: Sequence[BBox] | None) -> GlyphPlan:
        settings = self._settings
        max_tile_w = tile_limit(width, settings.ocr_glyph_tile_divisor)
        max_tile_h = tile_limit(height, settings.ocr_glyph_tile_divisor)
        factor = settings.ocr_glyph_magnify_factor
        budget = settings.ocr_glyph_magnify_max_pixels

        clamped = clamp_regions(regions, width, height) if regions else []
        usable: list[BBox] = []
        unreadable: list[str] = []
        for region in clamped:
            refusal = too_small_to_recognize(region.w, region.h)
            if refusal is None:
                usable.append(region)
            else:
                unreadable.append(f"{region.w}x{region.h}")
        scoped = bool(usable)
        if not scoped:
            usable = [BBox(x=0, y=0, w=width, h=height)]

        crops: list[GlyphCrop] = []
        for region in usable:
            crops.extend(
                GlyphCrop(*tile)
                for tile in region_tiles(
                    region,
                    max_tile_w=max_tile_w,
                    max_tile_h=max_tile_h,
                    overlap=settings.ocr_glyph_tile_overlap_px,
                )
            )
            if scoped and factor > 1 and region.w * region.h * factor * factor <= budget:
                crops.append(
                    GlyphCrop(region.x, region.y, region.x + region.w, region.y + region.h, factor)
                )
        return GlyphPlan(
            crops=tuple(dict.fromkeys(crops)),
            unreadable=tuple(unreadable),
            scoped=scoped,
        )

    def request_count(
        self, width: int, height: int, glyph_regions: Sequence[BBox] | None = None
    ) -> int:
        if not self._settings.ocr_glyph_pass_enabled:
            return 1
        return 1 + len(self.glyph_plan(width, height, glyph_regions).crops)

    def _recognize_glyphs(self, image: np.ndarray, regions: Sequence[BBox] | None) -> list[OcrLine]:
        height, width = image.shape[0], image.shape[1]
        plan = self.glyph_plan(width, height, regions)
        if plan.unreadable:
            _log.info("ocr.glyph_regions_too_small", sizes=list(plan.unreadable))
        if regions and not plan.scoped:
            _log.warning(
                "ocr.glyph_regions_unusable",
                regions=len(regions),
                width=width,
                height=height,
            )
        crops = plan.crops
        revision = self._glyph_revision()
        inverted = np.uint8(255) - image

        glyphs: list[OcrLine] = []
        for crop in crops:
            tile = np.ascontiguousarray(inverted[crop.y0 : crop.y1, crop.x0 : crop.x1])
            if crop.factor > 1:
                tile = _magnify(tile, crop.factor)
            for line in self._recognize_frame(tile, revision=revision):
                glyphs.append(line.model_copy(update={"bbox": crop.to_frame(line.bbox)}))
        _log.debug("ocr.glyph_pass", crops=len(crops), scoped=bool(regions), candidates=len(glyphs))
        return glyphs

    def _glyph_revision(self) -> int:
        configured = self._settings.ocr_glyph_revision
        supported = _supported_revisions()
        chosen = resolve_glyph_revision(configured, supported)
        if chosen != configured:
            _report_revision_fallback(configured, chosen, supported)
        return chosen

    def _recognize_frame(self, image: np.ndarray, *, revision: int) -> list[OcrLine]:
        if image.ndim >= 2:
            refusal = too_small_to_recognize(int(image.shape[1]), int(image.shape[0]))
            if refusal is not None:
                raise OcrError(refusal)

        supported = _supported_revisions()
        if revision not in supported:
            raise OcrError(
                f"Vision text recognition revision {revision} is not supported on this "
                f"system; supported revisions are {supported}."
            )

        cg_image = _cgimage_from_rgb(image)
        h, w = image.shape[0], image.shape[1]

        request = self._text_request(revision)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        try:
            success, error = handler.performRequests_error_([request], None)
        except Exception as exc:  # noqa: BLE001 - normalize any ObjC error to OcrError.
            raise OcrError(f"Vision request raised: {exc}") from exc
        if not success:
            raise OcrError(f"Vision request failed: {error}")
        return _lines_from_observations(request.results() or [], w, h)

    def _text_request(self, revision: int) -> Any:
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(self._settings.ocr_uses_language_correction)
        request.setRecognitionLanguages_(list(self._settings.ocr_languages))
        request.setRevision_(revision)
        return request
