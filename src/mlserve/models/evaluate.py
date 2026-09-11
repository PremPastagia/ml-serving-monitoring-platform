"""Evaluation metrics shared by training, retraining acceptance and drift analysis.

One function computes every metric so the retraining gate, the MLflow run and the
drift report can never disagree about what "ROC-AUC" meant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    split: str
    n_rows: int
    positive_rate: float
    threshold: float
    metrics: dict[str, float]
    confusion: dict[str, int] = field(default_factory=dict)
    scoring_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "n_rows": self.n_rows,
            "positive_rate": self.positive_rate,
            "threshold": self.threshold,
            "scoring_seconds": round(self.scoring_seconds, 6),
            **{k: v for k, v in self.metrics.items()},
            **{f"cm_{k}": v for k, v in self.confusion.items()},
        }

    def flat_metrics(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in self.metrics.items()}


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, *, threshold: float) -> dict[str, float]:
    """All headline metrics from labels and positive-class probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    out: dict[str, float] = {
        # Ranking quality, threshold-free. Primary because the API returns a probability.
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_proba)),
        # Precision-recall AUC: the metric that actually moves when the positive class
        # becomes rarer, which is exactly what a prior-shift drift scenario does.
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Calibration: a model can rank well and still return unusable probabilities.
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float("nan") if single_class else float(log_loss(y_true, y_proba, labels=[0, 1])),
    }
    return {k: (round(v, 6) if v == v else v) for k, v in out.items()}


def evaluate(
    pipeline,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    *,
    split: str,
    threshold: float = 0.5,
) -> EvaluationResult:
    start = time.perf_counter()
    proba = pipeline.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - start

    y_arr = np.asarray(y).astype(int)
    metrics = compute_metrics(y_arr, proba, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(
        y_arr, (proba >= threshold).astype(int), labels=[0, 1]
    ).ravel()
    return EvaluationResult(
        split=split,
        n_rows=int(len(X)),
        positive_rate=round(float(y_arr.mean()), 6),
        threshold=threshold,
        metrics=metrics,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        scoring_seconds=elapsed,
    )
