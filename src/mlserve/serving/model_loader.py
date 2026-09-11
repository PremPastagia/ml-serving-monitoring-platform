"""Model loading, hot reload and served-version identity.

The loader is the component rollback actually runs through. Promotion and rollback
repoint a registry alias; the API only follows that repoint when it reloads, so this
class is where "the service is now serving version N-1" becomes true.

Design decisions worth defending:

* **The old model keeps serving until the new one is fully loaded.** ``reload()``
  builds the replacement into a local variable and swaps a single reference under a
  lock only on success. A failed reload therefore leaves the previous model serving
  rather than leaving the service with no model at all -- which is the failure mode
  that turns a bad promotion into an outage.
* **Reference swap, not mutation.** Readers take the reference once and use it for the
  whole request, so a reload mid-request cannot make one batch score against two
  different models.
* **Load failure is a state, not a crash.** The process starts even when the registry
  is empty or unreachable; ``/health`` reports ``degraded`` and ``/predict`` returns
  503. A serving process that refuses to boot cannot tell anyone why it is unhappy.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlserve.config import Config, load_config
from mlserve.features.pipeline import MODEL_FEATURES, expected_input_columns


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted with no usable model."""


@dataclass(frozen=True)
class LoadedModel:
    """An immutable bundle of a fitted pipeline and everything known about it."""

    pipeline: Any
    model_name: str
    model_version: str
    model_alias: str | None
    source: str
    run_id: str | None = None
    dataset_version: str | None = None
    code_version: str | None = None
    git_commit: str | None = None
    training_fingerprint: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    loaded_at: str = ""
    load_seconds: float = 0.0

    def predict_proba(self, frame: pd.DataFrame):
        return self.pipeline.predict_proba(frame)[:, 1]

    def to_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "model_source": self.source,
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "training_fingerprint": self.training_fingerprint,
            "loaded_at": self.loaded_at,
            "load_seconds": round(self.load_seconds, 6),
            "input_columns": expected_input_columns(),
            "model_features": list(MODEL_FEATURES),
            "metrics": self.metrics,
        }


def _metrics_from_tags(tags: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in tags.items():
        if key.endswith(("roc_auc", "pr_auc", "accuracy", "f1")):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


class ModelLoader:
    """Owns the currently-served model and the rules for replacing it."""

    #: Environment overrides, so a container can switch source without a new config
    #: file. Precedence: explicit argument > environment > configuration.
    SOURCE_ENV = "MLSERVE_MODEL_SOURCE"
    ALIAS_ENV = "MLSERVE_MODEL_ALIAS"

    def __init__(self, config: Config | None = None, *, source: str | None = None,
                 alias: str | None = None, bundle_dir: str | Path | None = None):
        self.config = config or load_config()
        self.source = (
            source
            or os.environ.get(self.SOURCE_ENV)
            or str(self.config.require("serving.model_source"))
        )
        if self.source not in {"registry", "file"}:
            raise ValueError(
                f"model source must be 'registry' or 'file', got {self.source!r}"
            )
        self.alias = (
            alias
            or os.environ.get(self.ALIAS_ENV)
            or str(self.config.require("serving.model_alias"))
        )
        self.bundle_dir = Path(bundle_dir) if bundle_dir else self.config.path("paths.artifact_dir") / "current"
        self._lock = threading.Lock()
        self._model: LoadedModel | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------------- accessors

    @property
    def model(self) -> LoadedModel | None:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def require(self) -> LoadedModel:
        model = self._model
        if model is None:
            raise ModelNotLoadedError(self._last_error or "no model has been loaded")
        return model

    # --------------------------------------------------------------------- loading

    def _load_from_registry(self) -> LoadedModel:
        # Imported lazily so that a file-source deployment (the container) does not
        # need MLflow present at all.
        from mlserve.models.registry import ModelRegistry

        registry = ModelRegistry(self.config)
        ref = registry.resolve_alias(self.alias)
        if ref is None:
            raise RuntimeError(
                f"registry alias {self.alias!r} is not set on model "
                f"{registry.model_name!r}; run scripts/train.py first"
            )
        start = time.perf_counter()
        pipeline = registry.load(ref)
        elapsed = time.perf_counter() - start
        return LoadedModel(
            pipeline=pipeline,
            model_name=ref.name,
            model_version=ref.version,
            model_alias=self.alias,
            source="registry",
            run_id=ref.run_id,
            dataset_version=ref.tags.get("dataset_version"),
            code_version=ref.tags.get("code_version"),
            git_commit=ref.tags.get("git_commit"),
            training_fingerprint=ref.tags.get("training_fingerprint"),
            metrics=_metrics_from_tags(ref.tags),
            loaded_at=datetime.now(UTC).isoformat(),
            load_seconds=elapsed,
        )

    def _load_from_file(self) -> LoadedModel:
        model_path = self.bundle_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"no model bundle at {model_path}")
        start = time.perf_counter()
        pipeline = joblib.load(model_path)
        elapsed = time.perf_counter() - start

        meta: dict = {}
        meta_path = self.bundle_dir / "run.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        metrics = {
            k: v for k, v in (meta.get("metrics") or {}).items()
            if k.endswith(("roc_auc", "pr_auc", "accuracy", "f1"))
        }
        return LoadedModel(
            pipeline=pipeline,
            model_name=str(self.config.require("mlflow.registered_model_name")),
            model_version=f"file:{meta.get('training_fingerprint', 'unknown')[:12]}",
            model_alias=None,
            source="file",
            run_id=None,
            dataset_version=meta.get("dataset_version"),
            code_version=meta.get("code_version"),
            git_commit=meta.get("git_commit"),
            training_fingerprint=meta.get("training_fingerprint"),
            metrics=metrics,
            loaded_at=datetime.now(UTC).isoformat(),
            load_seconds=elapsed,
        )

    def load(self) -> LoadedModel:
        """Load per the configured source, replacing the current model on success."""
        loader = self._load_from_registry if self.source == "registry" else self._load_from_file
        try:
            candidate = loader()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            self._model = candidate
            self._last_error = None
        return candidate

    def try_load(self) -> LoadedModel | None:
        """Load without raising. Used at startup so a bad registry cannot block boot."""
        try:
            return self.load()
        except Exception:
            return None

    def reload(self) -> tuple[LoadedModel | None, dict]:
        """Re-resolve the alias and swap in the new model if the pointer moved.

        Returns ``(model, details)`` where ``details`` records the version before and
        after and the wall-clock cost -- which is what the rollback benchmark measures.
        """
        before = self._model.model_version if self._model else None
        start = time.perf_counter()
        try:
            model = self.load()
        except Exception as exc:
            return self._model, {
                "ok": False,
                "from_version": before,
                "to_version": before,
                "changed": False,
                "seconds": round(time.perf_counter() - start, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        elapsed = time.perf_counter() - start
        return model, {
            "ok": True,
            "from_version": before,
            "to_version": model.model_version,
            "changed": before != model.model_version,
            "seconds": round(elapsed, 6),
            "error": None,
        }

    def unload(self) -> None:
        """Drop the current model. Used by the failure-injection tests."""
        with self._lock:
            self._model = None
            self._last_error = "model was explicitly unloaded"
