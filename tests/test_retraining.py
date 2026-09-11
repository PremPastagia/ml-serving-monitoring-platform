"""Retraining workflow: trigger, acceptance decision, promotion, rejection, rollback.

The decision logic is tested exhaustively as a pure function; the orchestrator is then
tested end to end against a real registry for the paths that matter -- no drift, drift
with a better candidate, drift with a worse candidate, and a training failure.
"""

from __future__ import annotations

import pytest

from mlserve.data.schema import FEATURE_NAMES
from mlserve.models.registry import PREVIOUS, ModelRegistry
from mlserve.models.tracking import log_training_run
from mlserve.models.train import train_model
from mlserve.monitoring.scenarios import build_scenario
from mlserve.retraining.decide import (
    AcceptanceCriteria,
    CandidateEvidence,
    Decision,
    decide,
)
from mlserve.retraining.orchestrator import (
    RetrainingOrchestrator,
    measure_p95_latency,
    score_on_holdout,
)


@pytest.fixture
def criteria(config):
    return AcceptanceCriteria.from_config(config)


def evidence(**kwargs):
    base = dict(candidate_metric=0.94, incumbent_metric=0.92,
                candidate_p95_latency_ms=3.0, incumbent_p95_latency_ms=3.0,
                training_rows=26048, validation_errors=0)
    base.update(kwargs)
    return CandidateEvidence(**base)


# ------------------------------------------------------------------ decision logic


def test_a_clearly_better_candidate_is_promoted(criteria):
    assert decide(evidence(), criteria).decision is Decision.PROMOTE


def test_a_worse_candidate_is_rejected(criteria):
    result = decide(evidence(candidate_metric=0.90), criteria)
    assert result.decision is Decision.REJECT
    assert "max_allowed_degradation" in result.failed_criteria


def test_a_marginally_better_candidate_is_rejected(criteria):
    """Improvement below the margin is noise, and promoting on noise causes churn."""
    result = decide(evidence(candidate_metric=0.9205, incumbent_metric=0.92), criteria)
    assert result.decision is Decision.REJECT
    assert "min_absolute_improvement" in result.failed_criteria


def test_improvement_exactly_at_the_margin_is_promoted(criteria):
    at_margin = 0.92 + criteria.min_absolute_improvement
    assert decide(evidence(candidate_metric=at_margin), criteria).promote


def test_a_candidate_much_slower_than_the_incumbent_is_rejected(criteria):
    result = decide(
        evidence(candidate_p95_latency_ms=3.0 * criteria.max_latency_ratio + 0.5,
                 incumbent_p95_latency_ms=3.0),
        criteria,
    )
    assert "max_latency_ratio" in result.failed_criteria


def test_latency_exactly_at_the_permitted_ratio_is_accepted(criteria):
    assert decide(
        evidence(candidate_p95_latency_ms=3.0 * criteria.max_latency_ratio,
                 incumbent_p95_latency_ms=3.0),
        criteria,
    ).promote


def test_a_slow_host_does_not_reject_a_candidate_that_matches_the_incumbent(criteria):
    """The reason the gate is a ratio.

    Both models measured at 400ms because the host was saturated. An absolute 50ms
    budget would reject a candidate that is exactly as fast as the model it replaces;
    the ratio correctly does not.
    """
    result = decide(
        evidence(candidate_p95_latency_ms=400.0, incumbent_p95_latency_ms=395.0),
        criteria,
    )
    assert result.promote
    assert "max_latency_ratio" not in result.failed_criteria
    assert result.evidence["latency_ratio"] == pytest.approx(400.0 / 395.0, rel=1e-3)


def test_the_absolute_ceiling_applies_only_without_an_incumbent(criteria):
    """With no baseline the ratio is undefined, so some bound is better than none."""
    result = decide(
        evidence(incumbent_metric=None, incumbent_p95_latency_ms=None,
                 candidate_p95_latency_ms=criteria.max_p95_latency_ms + 1),
        criteria,
    )
    assert "max_p95_latency_ms" in result.failed_criteria


def test_dirty_training_data_blocks_promotion(criteria):
    result = decide(evidence(validation_errors=1), criteria)
    assert "require_clean_validation" in result.failed_criteria


def test_too_little_training_data_blocks_promotion(criteria):
    result = decide(evidence(training_rows=criteria.min_training_rows - 1), criteria)
    assert "min_training_rows" in result.failed_criteria


def test_the_absolute_floor_applies_even_when_beating_the_incumbent(criteria):
    """Without a floor, a decayed incumbent lets the loop ratchet downwards."""
    result = decide(evidence(candidate_metric=0.80, incumbent_metric=0.60), criteria)
    assert "min_candidate_roc_auc" in result.failed_criteria


def test_the_first_model_is_promoted_without_an_incumbent(criteria):
    result = decide(evidence(incumbent_metric=None), criteria)
    assert result.promote
    assert result.evidence["delta"] is None


def test_every_failed_criterion_is_reported(criteria):
    """A rejection must name every problem, not just the first."""
    result = decide(
        evidence(candidate_metric=0.10, candidate_p95_latency_ms=9999,
                 incumbent_p95_latency_ms=3.0, training_rows=1, validation_errors=5),
        criteria,
    )
    assert set(result.failed_criteria) >= {
        "require_clean_validation", "min_training_rows",
        "min_candidate_roc_auc", "max_latency_ratio",
    }
    assert len(result.reasons) >= 4


def test_decision_serialises_with_its_evidence_and_criteria(criteria):
    import json

    payload = decide(evidence(), criteria).to_dict()
    json.dumps(payload)
    assert payload["criteria"]["min_absolute_improvement"] == criteria.min_absolute_improvement
    assert payload["evidence"]["delta"] == pytest.approx(0.02)


def test_criteria_come_from_config(config, criteria):
    assert criteria.min_absolute_improvement == float(config.require("retraining.min_absolute_improvement"))
    assert criteria.min_candidate_roc_auc == float(config.require("retraining.min_candidate_roc_auc"))
    assert criteria.max_latency_ratio == float(config.require("retraining.max_latency_ratio"))


# ------------------------------------------------------------------- latency probe


def test_latency_probe_returns_ordered_percentiles(pipeline, small_split):
    result = measure_p95_latency(pipeline, small_split.test.head(100), n_requests=30)
    assert result["n"] == 30
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]


def test_latency_probe_on_an_empty_sample_is_safe(pipeline):
    import pandas as pd

    result = measure_p95_latency(pipeline, pd.DataFrame(columns=FEATURE_NAMES))
    assert result["n"] == 0
    assert result["p95_ms"] is None


def test_score_on_holdout_matches_a_direct_evaluation(pipeline, small_split):
    from mlserve.models.evaluate import evaluate

    holdout = small_split.test.head(800)
    direct = evaluate(pipeline, holdout[FEATURE_NAMES],
                      (holdout["income"] == ">50K").astype(int), split="test")
    assert score_on_holdout(pipeline, holdout)["roc_auc"] == pytest.approx(direct.metrics["roc_auc"])


# -------------------------------------------------------------- orchestrator paths


@pytest.fixture
def retrain_config(tmp_path):
    """Isolated config with the row floor scaled to the test split.

    `min_training_rows` is 5000 in production config and the test split has 4499 train
    rows, so the real floor would reject every candidate here for the wrong reason.
    Lowering the knob keeps the *promotion* path under test; the floor itself is
    covered by `test_too_little_training_data_blocks_promotion`.
    """
    from tests.conftest import clone_config

    return clone_config(tmp_path, {"retraining.min_training_rows": 1000})


@pytest.fixture
def orchestrator(retrain_config, small_split):
    """An orchestrator whose registry already holds a production model."""
    registry = ModelRegistry(retrain_config)
    pipeline, run = train_model(retrain_config, split=small_split, evaluate_test=False)
    X_val, _ = small_split.xy("validation")
    logged = log_training_run(pipeline, run, retrain_config, input_example=X_val)
    ref = registry.register(logged.model_uri, tags={"role": "incumbent"})
    registry.promote(ref.version)
    return RetrainingOrchestrator(retrain_config, registry=registry), registry, ref


def test_no_drift_does_not_trigger_retraining(orchestrator, small_split):
    orch, registry, ref = orchestrator
    reference = small_split.train[FEATURE_NAMES]
    current = build_scenario("no_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=reference, current=current,
                       training_data=small_split, holdout=small_split.test,
                       scenario="no_drift")
    assert outcome.triggered is False
    assert outcome.decision is None
    assert registry.production().version == ref.version


def test_drift_triggers_retraining_and_a_decision(orchestrator, small_split):
    orch, registry, ref = orchestrator
    reference = small_split.train[FEATURE_NAMES]
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=reference, current=current,
                       training_data=small_split, holdout=small_split.test,
                       scenario="large_drift")
    assert outcome.triggered is True
    assert "drift detected" in outcome.trigger_reason
    assert outcome.decision is not None
    assert outcome.candidate_version is not None
    assert outcome.retraining_seconds > 0


def test_an_identical_candidate_is_rejected_not_promoted(orchestrator, small_split):
    """Retraining on the same data cannot beat the incumbent by the required margin."""
    orch, registry, ref = orchestrator
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test,
                       scenario="large_drift")
    assert outcome.decision["decision"] == "reject"
    assert "min_absolute_improvement" in outcome.decision["failed_criteria"]
    assert outcome.promoted_version is None
    assert registry.production().version == ref.version, "a rejected candidate must not be serving"


def test_a_rejected_candidate_is_still_registered_for_inspection(orchestrator, small_split):
    orch, registry, ref = orchestrator
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test)
    assert outcome.candidate_version is not None
    candidate = registry.get_version(outcome.candidate_version)
    assert candidate.tags["decision"] == "reject"
    assert candidate.alias is None


def test_a_better_candidate_is_promoted_and_the_previous_is_kept(retrain_config, small_split):
    """The promotion path, with a genuinely better candidate.

    The incumbent is weakened by *learning rate and tree depth*, not by tree count.
    That distinction is load-bearing: an earlier version used `max_iter: 5`, which made
    the incumbent both less accurate and 4.5x cheaper per prediction, so the latency
    gate correctly rejected the more accurate candidate for being 7x slower. Holding
    the tree count fixed keeps inference cost constant and isolates accuracy, which is
    what this test is actually about. The accuracy/latency trade-off itself is covered
    by `test_a_candidate_much_slower_than_the_incumbent_is_rejected`.
    """
    registry = ModelRegistry(retrain_config)
    weak_params = {**dict(retrain_config.require("model.params")),
                   "learning_rate": 0.01, "max_leaf_nodes": 3}
    weak, weak_run = train_model(retrain_config, split=small_split, params=weak_params,
                                 evaluate_test=False, run_name="weak-incumbent")
    X_val, _ = small_split.xy("validation")
    logged = log_training_run(weak, weak_run, retrain_config, input_example=X_val)
    incumbent = registry.register(logged.model_uri, tags={"role": "weak-incumbent"})
    registry.promote(incumbent.version)

    orch = RetrainingOrchestrator(retrain_config, registry=registry)
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test,
                       scenario="large_drift")

    assert outcome.decision["decision"] == "promote", outcome.decision["reasons"]
    assert outcome.promoted_version == outcome.candidate_version
    assert registry.production().version == outcome.candidate_version
    assert registry.resolve_alias(PREVIOUS).version == incumbent.version
    candidate_auc = outcome.decision["candidate_holdout_metrics"]["roc_auc"]
    incumbent_auc = outcome.decision["incumbent_holdout_metrics"]["roc_auc"]
    assert candidate_auc > incumbent_auc


def test_rollback_after_a_promotion_restores_the_incumbent(retrain_config, small_split):
    registry = ModelRegistry(retrain_config)
    # Weakened by learning rate, not tree count, so inference cost is unchanged --
    # see test_a_better_candidate_is_promoted_and_the_previous_is_kept.
    weak_params = {**dict(retrain_config.require("model.params")),
                   "learning_rate": 0.01, "max_leaf_nodes": 3}
    X_val, _ = small_split.xy("validation")
    weak, weak_run = train_model(retrain_config, split=small_split, params=weak_params,
                                 evaluate_test=False, run_name="weak")
    incumbent = registry.register(
        log_training_run(weak, weak_run, retrain_config, input_example=X_val).model_uri)
    registry.promote(incumbent.version)

    orch = RetrainingOrchestrator(retrain_config, registry=registry)
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test)
    assert outcome.promoted_version is not None

    result = orch.rollback()
    assert result["to"] == incumbent.version
    assert registry.production().version == incumbent.version
    assert result["seconds"] >= 0.0


def test_a_failed_retraining_is_reported_and_promotes_nothing(orchestrator, small_split, monkeypatch):
    """Training that raises must be caught, recorded, and leave production alone."""
    orch, registry, ref = orchestrator

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("mlserve.retraining.orchestrator.train_model", explode)
    current = build_scenario("large_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test)

    assert outcome.triggered is True
    assert outcome.error is not None
    assert "synthetic training failure" in outcome.error
    assert outcome.decision["failed_criteria"] == ["retraining_failed"]
    assert outcome.promoted_version is None
    assert registry.production().version == ref.version


def test_schema_break_triggers_retraining(orchestrator, small_split):
    """A missing column must trigger the workflow even with no statistical drift."""
    orch, registry, ref = orchestrator
    current = build_scenario("missing_feature", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test,
                       scenario="missing_feature")
    assert outcome.triggered is True
    assert outcome.drift["schema_failures"]


def test_force_retrains_without_drift(orchestrator, small_split):
    orch, _, _ = orchestrator
    current = build_scenario("no_drift", small_split.test, seed=5, n_rows=1500)
    outcome = orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                       training_data=small_split, holdout=small_split.test,
                       force=True, register=False)
    assert outcome.triggered is True
    assert outcome.trigger_reason == "forced by caller"
    assert outcome.decision is not None


def test_outcome_serialises(orchestrator, small_split):
    import json

    orch, _, _ = orchestrator
    current = build_scenario("no_drift", small_split.test, seed=5, n_rows=1500)
    json.dumps(orch.run(reference=small_split.train[FEATURE_NAMES], current=current,
                        training_data=small_split, holdout=small_split.test).to_dict())
