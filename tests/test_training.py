"""Training: reproducibility, provenance completeness and sanity of the fit."""

from __future__ import annotations

import json

import pytest

from mlserve.data.validate import DataValidationError
from mlserve.models.evaluate import compute_metrics, evaluate
from mlserve.models.train import (
    PROBE_ROWS,
    save_bundle,
    set_global_seeds,
    train_model,
    training_fingerprint,
)

# ------------------------------------------------------------------ reproducibility


def test_repeated_training_is_bit_identical(config, small_split):
    """Same seed, same data, same code -> the same model, checked by fingerprint."""
    _, first = train_model(config, split=small_split, evaluate_test=False)
    _, second = train_model(config, split=small_split, evaluate_test=False)
    assert first.training_fingerprint == second.training_fingerprint
    assert first.metrics["validation_roc_auc"] == second.metrics["validation_roc_auc"]


def test_estimator_seed_does_not_change_this_model_and_that_is_intended(config, small_split):
    """`random_state` is inert for this configuration, which is a stronger guarantee.

    HistGradientBoostingClassifier consumes `random_state` in exactly two places: the
    internal train/validation split used for early stopping, and the subsample drawn to
    compute bin thresholds when there are more than 200k rows. This project sets
    `early_stopping: false` and trains on 26k rows, so neither applies and the fit is
    deterministic regardless of the seed.

    That is worth asserting rather than assuming. If a future change enables early
    stopping or crosses the subsampling threshold, the fit silently becomes
    seed-dependent, and this test is what catches it -- at which point the seed must be
    recorded as load-bearing rather than merely logged.
    """
    _, a = train_model(config, split=small_split, seed=1, evaluate_test=False)
    _, b = train_model(config, split=small_split, seed=999, evaluate_test=False)
    assert a.training_fingerprint == b.training_fingerprint
    assert config.require("model.params")["early_stopping"] is False


def test_a_different_hyperparameter_produces_a_different_model(config, small_split):
    """Negative control for the fingerprint: it must be able to tell models apart."""
    base = dict(config.require("model.params"))
    _, a = train_model(config, split=small_split, params={**base, "max_iter": 20},
                       evaluate_test=False)
    _, b = train_model(config, split=small_split, params={**base, "max_iter": 60},
                       evaluate_test=False)
    assert a.training_fingerprint != b.training_fingerprint


def test_a_different_data_split_produces_a_different_model(raw_data, config):
    """The seed that *is* load-bearing is the split seed, not the estimator seed."""
    from mlserve.data.split import make_split

    development, test, version = raw_data
    development = development.head(6000)
    test = test.head(1000)
    fast = {**dict(config.require("model.params")), "max_iter": 20}
    a = make_split(development, test, version, seed=1, validation_fraction=0.25)
    b = make_split(development, test, version, seed=2, validation_fraction=0.25)
    _, run_a = train_model(config, split=a, params=fast, evaluate_test=False)
    _, run_b = train_model(config, split=b, params=fast, evaluate_test=False)
    assert a.split_id != b.split_id
    assert run_a.training_fingerprint != run_b.training_fingerprint


def test_fingerprint_is_sensitive_to_the_probe_predictions(config, small_split, pipeline):
    """Negative control for the fingerprint itself."""
    X_val, _ = small_split.xy("validation")
    probe = X_val.head(PROBE_ROWS)
    baseline = training_fingerprint(pipeline, probe)
    tweaked = probe.copy()
    tweaked["age"] = 90
    assert training_fingerprint(pipeline, tweaked) != baseline


def test_set_global_seeds_makes_numpy_deterministic():
    import numpy as np

    set_global_seeds(123)
    first = np.random.rand(5)
    set_global_seeds(123)
    assert (np.random.rand(5) == first).all()


# ---------------------------------------------------------------------- provenance


def test_run_records_every_required_provenance_field(trained):
    _, run = trained
    payload = run.to_dict()
    required = [
        "dataset_version", "dataset_files", "split_id", "code_version", "git_commit",
        "config_digest", "seed", "params", "feature_list", "input_columns", "metrics",
        "training_seconds", "environment", "training_fingerprint",
    ]
    for key in required:
        assert key in payload, f"{key} is not recorded"
        assert payload[key] not in (None, "", [], {}), f"{key} is empty"


def test_environment_records_library_versions(trained):
    _, run = trained
    assert run.environment["python_version"]
    assert run.environment["pkg.scikit-learn"] != "absent"
    assert run.environment["pkg.numpy"] != "absent"


def test_dataset_files_are_checksummed(trained):
    _, run = trained
    assert set(run.dataset_files) >= {"adult.data", "adult.test"}
    for digest in run.dataset_files.values():
        assert len(digest) == 64


def test_leakage_report_is_attached(trained):
    _, run = trained
    assert run.leakage_report["index_overlap_train_validation"] == 0


def test_run_serialises_to_json(trained):
    _, run = trained
    json.dumps(run.to_dict())


# ---------------------------------------------------------------------- model fit


def test_model_beats_the_majority_class_baseline(trained):
    _, run = trained
    assert run.metrics["validation_roc_auc"] > 0.85
    assert run.metrics["validation_accuracy"] > 0.80


def test_generalisation_gap_is_not_catastrophic(trained):
    """A large train/validation gap would mean the fit is memorising."""
    _, run = trained
    gap = run.metrics["train_roc_auc"] - run.metrics["validation_roc_auc"]
    assert gap < 0.10, f"train-validation ROC-AUC gap of {gap:.4f} indicates overfitting"


def test_boosted_model_beats_the_linear_baseline(config, small_split):
    """A sanity floor: if this fails, the pipeline is broken, not the data."""
    _, tree = train_model(config, split=small_split, evaluate_test=False)
    _, linear = train_model(
        config, split=small_split, estimator="logistic_regression",
        params={"C": 1.0, "max_iter": 1000}, evaluate_test=False,
    )
    assert tree.metrics["validation_roc_auc"] > linear.metrics["validation_roc_auc"]


def test_evaluation_covers_every_configured_metric(config, trained, small_split):
    pipeline, _ = trained
    X, y = small_split.xy("validation")
    result = evaluate(pipeline, X, y, split="validation", threshold=0.5)
    expected = {config.require("evaluation.primary_metric")} | set(config.require("evaluation.secondary_metrics"))
    assert expected.issubset(set(result.metrics))
    assert result.confusion["tp"] + result.confusion["fn"] == int(y.sum())


def test_metrics_are_degenerate_safe_on_single_class():
    import numpy as np

    metrics = compute_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10), threshold=0.5)
    assert metrics["roc_auc"] != metrics["roc_auc"]  # NaN, not a crash or a fake 0.5


# ----------------------------------------------------------------- failure surface


def test_training_refuses_data_that_fails_validation(config, small_split):
    """A failed training run must fail loudly rather than produce a model."""
    import copy

    broken = copy.copy(small_split)
    corrupted = small_split.train.copy()
    corrupted.loc[corrupted.index[0], "age"] = 999
    object.__setattr__(broken, "train", corrupted)
    with pytest.raises(DataValidationError):
        train_model(config, split=broken, validate=True, evaluate_test=False)


def test_bundle_round_trips(trained, tmp_path):
    import joblib

    pipeline, run = trained
    path = save_bundle(pipeline, run, tmp_path)
    assert path.exists()
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["training_fingerprint"] == run.training_fingerprint
    reloaded = joblib.load(path)
    assert hasattr(reloaded, "predict_proba")
