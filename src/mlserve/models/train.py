"""Reproducible training with complete provenance capture.

Every run records, in MLflow and in a local JSON sidecar:

    dataset version | split id | code version | git commit | config digest |
    random seed | estimator params | ordered feature list | metrics | wall-clock time |
    Python/OS/library versions

Reproducibility is enforced, not hoped for: `training_fingerprint` hashes the fitted
model's predictions on a fixed probe set, so "same inputs produce the same model" is a
checkable claim rather than an assertion. `scripts/verify/verify_training_repro.py`
compares two independent runs against that fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlserve.config import Config, code_version, environment_info, git_commit, load_config
from mlserve.data.ingest import load_raw
from mlserve.data.split import Split, make_split, overlap_report
from mlserve.data.validate import DataValidationError, validate_frame
from mlserve.features.pipeline import MODEL_FEATURES, build_pipeline, expected_input_columns
from mlserve.models.evaluate import EvaluationResult, evaluate

#: Number of rows from the validation split hashed into the training fingerprint.
PROBE_ROWS = 256


def set_global_seeds(seed: int) -> None:
    """Pin every RNG that can influence the fit.

    scikit-learn estimators take ``random_state`` explicitly, which is the binding
    control; these three cover library code that reaches for a global RNG, and
    PYTHONHASHSEED covers set/dict iteration order in any code that builds a
    vocabulary from an unordered container.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TrainingRun:
    """The complete, serialisable record of one training run."""

    run_name: str
    estimator: str
    params: dict
    seed: int
    dataset_version: str
    dataset_files: dict[str, str]
    split_id: str
    split_sizes: dict[str, int]
    code_version: str
    git_commit: str
    config_digest: str
    feature_list: list[str]
    input_columns: list[str]
    metrics: dict[str, float]
    evaluations: dict[str, dict] = field(default_factory=dict)
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    environment: dict = field(default_factory=dict)
    validation_stats: dict = field(default_factory=dict)
    leakage_report: dict = field(default_factory=dict)
    training_fingerprint: str = ""
    model_path: str = ""

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "estimator": self.estimator,
            "params": self.params,
            "seed": self.seed,
            "dataset_version": self.dataset_version,
            "dataset_files": self.dataset_files,
            "split_id": self.split_id,
            "split_sizes": self.split_sizes,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "config_digest": self.config_digest,
            "feature_list": self.feature_list,
            "input_columns": self.input_columns,
            "metrics": self.metrics,
            "evaluations": self.evaluations,
            "training_seconds": round(self.training_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "environment": self.environment,
            "validation_stats": self.validation_stats,
            "leakage_report": self.leakage_report,
            "training_fingerprint": self.training_fingerprint,
            "model_path": self.model_path,
        }

    @property
    def primary_metric(self) -> float:
        return self.metrics["validation_roc_auc"]


def training_fingerprint(pipeline, probe: pd.DataFrame) -> str:
    """Hash the fitted model's behaviour, not its bytes.

    Pickle bytes differ between runs for reasons that have nothing to do with the
    model (memory addresses, dict ordering, joblib framing). Predictions on a fixed
    probe set are the thing that actually has to be identical, so that is what is
    hashed -- at full float64 precision, so a genuinely different fit cannot hide
    behind rounding.
    """
    proba = pipeline.predict_proba(probe)[:, 1].astype(np.float64)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(proba).tobytes())
    h.update(f"|{len(probe)}|{'|'.join(MODEL_FEATURES)}".encode())
    return h.hexdigest()


def load_and_split(config: Config, *, strict_checksum: bool = True) -> Split:
    development, test, version = load_raw(config.path("paths.raw_dir"), strict=strict_checksum)
    return make_split(
        development,
        test,
        version,
        seed=config.seed,
        validation_fraction=float(config.require("data.validation_fraction")),
        stratify=bool(config.require("data.stratify")),
    )


def train_model(
    config: Config | None = None,
    *,
    split: Split | None = None,
    estimator: str | None = None,
    params: dict | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    validate: bool = True,
    evaluate_test: bool = True,
) -> tuple[object, TrainingRun]:
    """Fit one model end to end and return it with its provenance record."""
    config = config or load_config()
    t_total = time.perf_counter()

    seed = config.seed if seed is None else int(seed)
    set_global_seeds(seed)

    estimator = estimator or str(config.require("model.estimator"))
    params = dict(params if params is not None else config.require("model.params"))
    threshold = float(config.require("evaluation.decision_threshold"))

    if split is None:
        split = load_and_split(config)

    validation_stats: dict = {}
    if validate:
        report = validate_frame(split.train, require_target=True, check_leakage=True)
        if not report.ok:
            raise DataValidationError(report)
        validation_stats = report.stats

    X_train, y_train = split.xy("train")
    X_val, y_val = split.xy("validation")

    pipeline = build_pipeline(estimator, params, seed=seed)

    t_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t_fit

    evaluations: dict[str, EvaluationResult] = {
        "train": evaluate(pipeline, X_train, y_train, split="train", threshold=threshold),
        "validation": evaluate(pipeline, X_val, y_val, split="validation", threshold=threshold),
    }
    if evaluate_test:
        X_test, y_test = split.xy("test")
        evaluations["test"] = evaluate(pipeline, X_test, y_test, split="test", threshold=threshold)

    metrics: dict[str, float] = {}
    for name, result in evaluations.items():
        metrics.update(result.flat_metrics(name))

    probe = X_val.head(PROBE_ROWS)
    run = TrainingRun(
        run_name=run_name or f"{estimator}-seed{seed}",
        estimator=estimator,
        params=params,
        seed=seed,
        dataset_version=split.dataset_version.dataset_id,
        dataset_files=split.dataset_version.files,
        split_id=split.split_id,
        split_sizes=split.sizes,
        code_version=code_version(),
        git_commit=git_commit(),
        config_digest=config.digest,
        feature_list=list(MODEL_FEATURES),
        input_columns=expected_input_columns(),
        metrics=metrics,
        evaluations={k: v.to_dict() for k, v in evaluations.items()},
        training_seconds=fit_seconds,
        environment=environment_info().to_dict(),
        validation_stats=validation_stats,
        leakage_report=overlap_report(split),
        training_fingerprint=training_fingerprint(pipeline, probe),
    )
    run.total_seconds = time.perf_counter() - t_total
    return pipeline, run


def save_bundle(pipeline, run: TrainingRun, out_dir: str | Path) -> Path:
    """Persist model + metadata together so a bundle is self-describing."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(pipeline, model_path)
    run.model_path = str(model_path)
    (out / "run.json").write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    (out / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n"
    )
    return model_path
