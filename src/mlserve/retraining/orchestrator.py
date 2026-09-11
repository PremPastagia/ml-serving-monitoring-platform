"""The drift-triggered retraining workflow.

    current window
      -> drift detection
      -> (if drift) validate the proposed training data
      -> retrain a candidate
      -> evaluate candidate AND incumbent on the SAME holdout
      -> measure the candidate's serving latency
      -> apply the acceptance criteria
      -> promote (registry alias moves) or reject (candidate kept, not aliased)

Two decisions worth defending:

**The incumbent is re-scored, not remembered.** The comparison uses the incumbent's
score on the *same* holdout as the candidate, computed now. Comparing against the
number recorded at the incumbent's training time would compare two different test
sets, which is the most common way an automated promotion gate silently promotes a
worse model.

**A rejected candidate is still registered.** It gets a registry version and a
``rejected`` tag but no alias. Discarding it would throw away the evidence for why the
loop did nothing, and a rejection is exactly the event someone will want to inspect.

This workflow is *triggered explicitly* -- by `scripts/retrain.py` or by a test. There
is no scheduler and no automatic production trigger, and RETRAINING.md says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlserve.config import Config, load_config
from mlserve.data.schema import FEATURE_NAMES, TARGET
from mlserve.data.validate import validate_frame
from mlserve.models.evaluate import compute_metrics
from mlserve.models.registry import ModelRegistry
from mlserve.models.tracking import log_training_run
from mlserve.models.train import train_model
from mlserve.monitoring.drift import DriftDetector, DriftReport
from mlserve.retraining.decide import (
    AcceptanceCriteria,
    CandidateEvidence,
    PromotionDecision,
    decide,
)

#: Batch size used for the candidate's latency probe. 1 is the pessimistic case that
#: matters for an online API: per-record overhead dominates and cannot be amortised.
LATENCY_PROBE_BATCH = 1
LATENCY_PROBE_REQUESTS = 200


@dataclass
class RetrainingOutcome:
    """The complete, serialisable record of one retraining cycle."""

    triggered: bool
    trigger_reason: str
    drift: dict = field(default_factory=dict)
    decision: dict | None = None
    candidate_version: str | None = None
    incumbent_version: str | None = None
    promoted_version: str | None = None
    promotion: dict | None = None
    retraining_seconds: float = 0.0
    total_seconds: float = 0.0
    candidate_run: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "drift": self.drift,
            "decision": self.decision,
            "candidate_version": self.candidate_version,
            "incumbent_version": self.incumbent_version,
            "promoted_version": self.promoted_version,
            "promotion": self.promotion,
            "retraining_seconds": round(self.retraining_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "error": self.error,
        }


def measure_p95_latency(pipeline, sample: pd.DataFrame, *,
                        n_requests: int = LATENCY_PROBE_REQUESTS,
                        batch_size: int = LATENCY_PROBE_BATCH) -> dict:
    """In-process latency probe for the acceptance gate.

    Deliberately in-process: this measures the *model*, so that a candidate rejected
    for latency is rejected because of the model rather than because of HTTP framing
    or a busy event loop. The end-to-end HTTP figures come from the load test.
    """
    if sample.empty:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "n": 0}
    rows = sample[FEATURE_NAMES]
    timings: list[float] = []
    # One untimed call first: the first predict_proba pays lazy allocation costs that
    # would otherwise land entirely in the p99.
    pipeline.predict_proba(rows.head(batch_size))
    for i in range(n_requests):
        start = i * batch_size % max(len(rows) - batch_size, 1)
        batch = rows.iloc[start:start + batch_size]
        t0 = time.perf_counter()
        pipeline.predict_proba(batch)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "mean_ms": round(float(arr.mean()), 4),
        "n": len(arr),
        "batch_size": batch_size,
    }


def measure_latency_ab(candidate, incumbent, sample: pd.DataFrame, *,
                       n_requests: int = LATENCY_PROBE_REQUESTS,
                       batch_size: int = LATENCY_PROBE_BATCH) -> dict:
    """Time two models against each other by interleaving their requests.

    Measuring one model fully and then the other does not cancel host contention: the
    load can change between the two windows, and it does. During development that
    produced a measured 10.5x latency ratio (216ms vs 21ms) between two models whose
    real cost differs by well under 2x, which rejected a genuinely better candidate.

    Interleaving -- candidate, incumbent, candidate, incumbent, on the same rows --
    puts both models in the same load conditions sample for sample, so a scheduling
    spike lands on whichever model happens to be next rather than systematically on
    one of them.

    The ratio is taken on the **median**, not p95. On a shared host the tail is
    dominated by scheduler outliers that belong to neither model; the median isolates
    the model's own cost, which is what the acceptance gate is trying to compare. Both
    p95 figures are still reported, because the tail is what a latency budget is about
    and a reviewer must be able to see it.
    """
    if sample.empty or incumbent is None:
        return {"candidate": measure_p95_latency(candidate, sample, n_requests=n_requests,
                                                 batch_size=batch_size),
                "incumbent": {"p50_ms": None, "p95_ms": None}, "ratio_p50": None,
                "ratio_p95": None, "interleaved": False}

    rows = sample[FEATURE_NAMES]
    # Untimed warm-up for both: the first call pays lazy allocation inside scikit-learn.
    candidate.predict_proba(rows.head(batch_size))
    incumbent.predict_proba(rows.head(batch_size))

    timings: dict[str, list[float]] = {"candidate": [], "incumbent": []}
    span = max(len(rows) - batch_size, 1)
    for i in range(n_requests):
        start = (i * batch_size) % span
        batch = rows.iloc[start:start + batch_size]
        for name, model in (("candidate", candidate), ("incumbent", incumbent)):
            t0 = time.perf_counter()
            model.predict_proba(batch)
            timings[name].append((time.perf_counter() - t0) * 1000.0)

    def percentiles(values: list[float]) -> dict:
        arr = np.array(values)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "mean_ms": round(float(arr.mean()), 4),
            "n": len(arr),
            "batch_size": batch_size,
        }

    candidate_stats = percentiles(timings["candidate"])
    incumbent_stats = percentiles(timings["incumbent"])
    return {
        "candidate": candidate_stats,
        "incumbent": incumbent_stats,
        "ratio_p50": round(candidate_stats["p50_ms"] / incumbent_stats["p50_ms"], 4)
        if incumbent_stats["p50_ms"] else None,
        "ratio_p95": round(candidate_stats["p95_ms"] / incumbent_stats["p95_ms"], 4)
        if incumbent_stats["p95_ms"] else None,
        "interleaved": True,
    }


def score_on_holdout(pipeline, holdout: pd.DataFrame, *, threshold: float = 0.5) -> dict:
    """Score any pipeline on a labelled holdout, for a like-for-like comparison."""
    X = holdout[FEATURE_NAMES]
    y = (holdout[TARGET].astype(str) == ">50K").astype(int).to_numpy()
    proba = pipeline.predict_proba(X)[:, 1]
    return compute_metrics(y, proba, threshold=threshold)


class RetrainingOrchestrator:
    """Runs one drift-check-and-maybe-retrain cycle."""

    def __init__(self, config: Config | None = None, *, registry: ModelRegistry | None = None):
        self.config = config or load_config()
        self.registry = registry or ModelRegistry(self.config)
        self.criteria = AcceptanceCriteria.from_config(self.config)
        self.metric_name = str(self.config.require("evaluation.primary_metric"))
        self.threshold = float(self.config.require("evaluation.decision_threshold"))

    # ------------------------------------------------------------------- trigger

    def check_drift(self, reference: pd.DataFrame, current: pd.DataFrame,
                    *, scenario: str | None = None) -> DriftReport:
        detector = DriftDetector.from_config(reference[FEATURE_NAMES], self.config)
        return detector.detect(current, scenario=scenario)

    # -------------------------------------------------------------------- cycle

    def run(
        self,
        *,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        training_data,
        holdout: pd.DataFrame,
        scenario: str | None = None,
        force: bool = False,
        register: bool = True,
    ) -> RetrainingOutcome:
        """Execute one cycle.

        ``training_data`` is the :class:`~mlserve.data.split.Split` the candidate is
        fitted on. ``holdout`` is the labelled frame both models are scored against.
        ``force=True`` retrains regardless of drift, which is how the "no drift, no
        retraining" branch is distinguished from a broken trigger in tests.
        """
        t_total = time.perf_counter()
        report = self.check_drift(reference, current, scenario=scenario)

        if not report.drift_detected and not force:
            return RetrainingOutcome(
                triggered=False,
                trigger_reason="no drift detected; retraining not required",
                drift=report.to_dict(),
                total_seconds=time.perf_counter() - t_total,
            )

        trigger_reason = (
            "forced by caller" if (force and not report.drift_detected)
            else f"drift detected: {report.n_drifted} feature(s) alerted, "
                 f"{len(report.schema_failures)} schema failure(s)"
        )

        incumbent_ref = self.registry.production()
        outcome = RetrainingOutcome(
            triggered=True,
            trigger_reason=trigger_reason,
            drift=report.to_dict(),
            incumbent_version=incumbent_ref.version if incumbent_ref else None,
        )

        # --- validate the proposed training data before fitting anything on it ----
        validation = validate_frame(training_data.train, require_target=True, check_leakage=True)

        # --- retrain ---------------------------------------------------------------
        t_fit = time.perf_counter()
        try:
            candidate, candidate_run = train_model(
                self.config, split=training_data, run_name=f"retrain-{scenario or 'manual'}",
                validate=False,  # already validated above; the report drives the gate
            )
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.retraining_seconds = time.perf_counter() - t_fit
            outcome.total_seconds = time.perf_counter() - t_total
            outcome.decision = {
                "decision": "reject",
                "reasons": [f"retraining raised {type(exc).__name__}: {exc}"],
                "failed_criteria": ["retraining_failed"],
                "evidence": {},
                "criteria": self.criteria.to_dict(),
            }
            return outcome
        outcome.retraining_seconds = time.perf_counter() - t_fit
        outcome.candidate_run = candidate_run.to_dict()

        # --- score both models on the SAME holdout --------------------------------
        candidate_metrics = score_on_holdout(candidate, holdout, threshold=self.threshold)
        incumbent_metrics = None
        incumbent_pipeline = None
        if incumbent_ref is not None:
            incumbent_pipeline = self.registry.load(incumbent_ref)
            incumbent_metrics = score_on_holdout(incumbent_pipeline, holdout, threshold=self.threshold)

        # Time both models by interleaving their requests on the same rows, so host
        # contention lands on both equally and cancels in the ratio the gate uses.
        probe_rows = holdout.head(512)
        latency_ab = measure_latency_ab(candidate, incumbent_pipeline, probe_rows)
        latency = latency_ab["candidate"]
        incumbent_latency = latency_ab["incumbent"]

        evidence = CandidateEvidence(
            candidate_metric=float(candidate_metrics[self.metric_name]),
            incumbent_metric=(float(incumbent_metrics[self.metric_name])
                              if incumbent_metrics else None),
            metric_name=self.metric_name,
            candidate_p95_latency_ms=latency["p95_ms"],
            incumbent_p95_latency_ms=incumbent_latency["p95_ms"],
            candidate_p50_latency_ms=latency.get("p50_ms"),
            incumbent_p50_latency_ms=incumbent_latency.get("p50_ms"),
            training_rows=len(training_data.train),
            validation_errors=len(validation.errors),
            training_seconds=candidate_run.training_seconds,
        )
        decision: PromotionDecision = decide(evidence, self.criteria)
        outcome.decision = decision.to_dict()
        outcome.decision["candidate_holdout_metrics"] = candidate_metrics
        outcome.decision["incumbent_holdout_metrics"] = incumbent_metrics
        outcome.decision["candidate_latency"] = latency
        outcome.decision["incumbent_latency"] = incumbent_latency
        outcome.decision["latency_comparison"] = {
            "ratio_p50": latency_ab["ratio_p50"],
            "ratio_p95": latency_ab["ratio_p95"],
            "interleaved": latency_ab["interleaved"],
        }

        # --- register, then promote only if the gate passed ------------------------
        if register:
            X_val, _ = training_data.xy("validation")
            logged = log_training_run(
                candidate, candidate_run, self.config, input_example=X_val,
                tags={"lifecycle": "retraining", "scenario": scenario or "manual",
                      "decision": decision.decision.value},
            )
            ref = self.registry.register(
                logged.model_uri,
                tags={
                    "dataset_version": candidate_run.dataset_version,
                    "code_version": candidate_run.code_version,
                    "git_commit": candidate_run.git_commit,
                    "training_fingerprint": candidate_run.training_fingerprint,
                    "lifecycle": "retraining",
                    "decision": decision.decision.value,
                    f"holdout_{self.metric_name}": f"{evidence.candidate_metric:.6f}",
                    "validation_roc_auc": f"{candidate_run.metrics['validation_roc_auc']:.6f}",
                },
            )
            outcome.candidate_version = ref.version
            if decision.promote:
                outcome.promotion = self.registry.promote(ref.version)
                outcome.promoted_version = ref.version
            else:
                self.registry.set_version_tags(ref.version, {"promoted": "false"})

        outcome.total_seconds = time.perf_counter() - t_total
        return outcome

    # ------------------------------------------------------------------ rollback

    def rollback(self, *, to_version: str | None = None) -> dict:
        """Move the production alias back. Timing is returned for the benchmark."""
        return self.registry.rollback(to_version=to_version)
