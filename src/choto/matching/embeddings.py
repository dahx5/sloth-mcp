from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from model2vec import StaticModel

from choto.log import get_logger

_log = get_logger(__name__)

SEMANTIC_MODEL = "minishlab/potion-multilingual-128M"
SEMANTIC_MODEL_FALLBACK = "minishlab/potion-base-8M"

MODEL_RETRY_AFTER_SECONDS = 60.0

_ETAG_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ModelPresence:
    model_id: str
    folder: Path | None


@dataclass(frozen=True)
class _Attempt:
    model: StaticModel | None
    at: float


class _ModelCache:
    def __init__(
        self,
        load: Callable[[], StaticModel | None],
        retry_after_s: float = MODEL_RETRY_AFTER_SECONDS,
    ) -> None:
        self._load = load
        self._retry_after_s = retry_after_s
        self._lock = threading.Lock()
        self._attempt: _Attempt | None = None

    def get(self) -> StaticModel | None:
        with self._lock:
            attempt = self._attempt
            if attempt is not None and (
                attempt.model is not None or time.monotonic() - attempt.at < self._retry_after_s
            ):
                return attempt.model
            model = self._load()
            self._attempt = _Attempt(model=model, at=time.monotonic())
            return model


def cached_snapshot(model_id: str) -> Path | None:
    try:
        return Path(snapshot_download(model_id, repo_type="model", local_files_only=True))
    except OSError as exc:
        _log.debug("semantic_model_not_cached", model=model_id, error=str(exc))
        return None


def installed_models() -> list[ModelPresence]:
    return [
        ModelPresence(model_id=model_id, folder=cached_snapshot(model_id))
        for model_id in (SEMANTIC_MODEL, SEMANTIC_MODEL_FALLBACK)
    ]


def install_semantic_model(model_id: str) -> Path:
    folder = Path(
        snapshot_download(model_id, repo_type="model", etag_timeout=_ETAG_TIMEOUT_SECONDS)
    )
    _log.info("semantic_model_installed", model=model_id, folder=str(folder))
    return folder


def load_semantic_model() -> StaticModel | None:
    return _CACHE.get()


def _load() -> StaticModel | None:
    for model_id in (SEMANTIC_MODEL, SEMANTIC_MODEL_FALLBACK):
        folder = cached_snapshot(model_id)
        if folder is None:
            continue
        try:
            model = StaticModel.from_pretrained(folder)
        except Exception as exc:  # noqa: BLE001 - degrade on any load failure
            _log.warning(
                "semantic_model_load_failed", model=model_id, folder=str(folder), error=str(exc)
            )
            continue
        _log.info("semantic_model_loaded", model=model_id, folder=str(folder))
        return model

    _log.warning(
        "semantic_tier_disabled",
        reason="no embedding model in the local cache; using exact+fuzzy only",
        remedy="choto model install",
        models=f"{SEMANTIC_MODEL}, {SEMANTIC_MODEL_FALLBACK}",
    )
    return None


_CACHE = _ModelCache(_load)
