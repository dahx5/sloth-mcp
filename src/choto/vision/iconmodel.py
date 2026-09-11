from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from choto import userpaths
from choto.log import get_logger

if TYPE_CHECKING:
    import coremltools as ct

_log = get_logger(__name__)

ICON_MODEL_FILENAME = "icon_detect.mlpackage"

_PACKAGE_MANIFEST = "Manifest.json"

MODEL_RETRY_AFTER_SECONDS = 60.0

_BOX_FIELDS = 5


class IconModelError(RuntimeError): ...


@dataclass(frozen=True)
class IconModelPresence:
    path: Path
    installed: bool


@dataclass(frozen=True)
class IconModel:
    model: ct.models.MLModel
    input_name: str
    output_name: str
    side: int

    def predict(self, image: Image.Image) -> np.ndarray:
        if image.size != (self.side, self.side):
            raise ValueError(
                f"The detector takes a {self.side}x{self.side} image, got {image.size[0]}x"
                f"{image.size[1]}."
            )
        raw = self.model.predict({self.input_name: image})[self.output_name]
        head = np.asarray(raw, dtype=np.float32)
        if head.ndim != 3 or head.shape[0] != 1 or head.shape[1] != _BOX_FIELDS:
            raise IconModelError(
                f"The detector answered with shape {head.shape}, expected "
                f"(1, {_BOX_FIELDS}, boxes)."
            )
        return head[0]


@dataclass(frozen=True)
class _Attempt:
    model: IconModel | None
    at: float


class _ModelCache:
    def __init__(
        self,
        load: Callable[[], IconModel | None],
        retry_after_s: float = MODEL_RETRY_AFTER_SECONDS,
    ) -> None:
        self._load = load
        self._retry_after_s = retry_after_s
        self._lock = threading.Lock()
        self._attempt: _Attempt | None = None

    def get(self) -> IconModel | None:
        with self._lock:
            attempt = self._attempt
            if attempt is not None and (
                attempt.model is not None or time.monotonic() - attempt.at < self._retry_after_s
            ):
                return attempt.model
            model = self._load()
            self._attempt = _Attempt(model=model, at=time.monotonic())
            return model

    def reset(self) -> None:
        with self._lock:
            self._attempt = None


def icon_model_path() -> Path:
    return userpaths.model_dir() / ICON_MODEL_FILENAME


def installed_icon_model() -> IconModelPresence:
    path = icon_model_path()
    return IconModelPresence(path=path, installed=path.is_dir())


def install_icon_model(source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.is_dir() or not (resolved / _PACKAGE_MANIFEST).is_file():
        raise IconModelError(
            f"{resolved} is not a CoreML package: expected a directory containing "
            f"{_PACKAGE_MANIFEST}."
        )
    _read_package(resolved)

    destination = icon_model_path()
    if destination.exists() and destination.resolve() == resolved:
        _log.info("icon_model.install.in_place", path=str(destination))
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".incoming")
    replaced = destination.with_name(destination.name + ".replaced")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(replaced, ignore_errors=True)
        shutil.copytree(resolved, staging)
        if destination.exists():
            destination.rename(replaced)
        staging.rename(destination)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if replaced.exists() and not destination.exists():
            replaced.rename(destination)
        raise IconModelError(f"Could not install {resolved} into {destination}: {exc}") from exc
    finally:
        shutil.rmtree(replaced, ignore_errors=True)

    reset_icon_model_cache()
    _log.info("icon_model.installed", source=str(resolved), path=str(destination))
    return destination


def load_icon_model() -> IconModel | None:
    return _CACHE.get()


def reset_icon_model_cache() -> None:
    _CACHE.reset()


def _load() -> IconModel | None:
    presence = installed_icon_model()
    if not presence.installed:
        _log.warning(
            "icon_detection_disabled",
            reason="no icon detector installed on this machine",
            remedy="choto model install --icon-detector PATH",
            path=str(presence.path),
        )
        return None
    try:
        model = _read_package(presence.path)
    except IconModelError as exc:
        _log.warning("icon_model_load_failed", path=str(presence.path), error=str(exc))
        return None
    _log.info("icon_model_loaded", path=str(presence.path), side=model.side)
    return model


def _read_package(path: Path) -> IconModel:
    import coremltools as ct

    try:
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        spec = model.get_spec()
    except Exception as exc:  # noqa: BLE001 - coremltools raises bare exceptions
        raise IconModelError(f"Could not read the CoreML package at {path}: {exc}") from exc

    inputs = list(spec.description.input)
    outputs = list(spec.description.output)
    if len(inputs) != 1 or not inputs[0].type.HasField("imageType"):
        raise IconModelError(
            f"The detector at {path} must take exactly one image input, got "
            f"{[item.name for item in inputs]}."
        )
    if len(outputs) != 1:
        raise IconModelError(
            f"The detector at {path} must have exactly one output, got "
            f"{[item.name for item in outputs]}."
        )
    image = inputs[0].type.imageType
    if image.width != image.height:
        raise IconModelError(
            f"The detector at {path} takes a {image.width}x{image.height} image; this "
            "reader letterboxes into a square one."
        )
    shape = list(outputs[0].type.multiArrayType.shape)
    if shape[:2] != [1, _BOX_FIELDS]:
        raise IconModelError(
            f"The detector at {path} answers with shape {shape}, expected "
            f"[1, {_BOX_FIELDS}, boxes] — four box numbers and one class score."
        )
    return IconModel(
        model=model,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
        side=int(image.width),
    )


_CACHE = _ModelCache(_load)
