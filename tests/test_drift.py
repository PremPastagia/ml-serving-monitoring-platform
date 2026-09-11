"""Drift detection: statistical correctness, and negative controls.

The hard part of testing a detector is proving it can be *quiet*. Every "this fires"
test below is paired with a "this does not fire" test on data drawn from the same
distribution, because a detector that always alarms would otherwise pass the entire
positive half of this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlserve.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
from mlserve.monitoring.drift import (
    DriftDetector,
    _is_effectively_discrete,
    categorical_feature_drift,
    numeric_feature_drift,
    population_stability_index,
    prediction_drift,
)
from mlserve.monitoring.scenarios import (
    REQUIRED_SCENARIOS,
    SCENARIOS,
    build_scenario,
    scenario_table,
)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20240101)


# ---------------------------------------------------------------------------- PSI


def test_psi_of_a_distribution_against_itself_is_zero(rng):
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_near_zero_for_two_draws_from_one_distribution(rng):
    """The negative control for PSI: sampling noise alone must stay well under warn."""
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    assert population_stability_index(a, b) < 0.01


def test_psi_grows_with_the_size_of_the_shift(rng):
    reference = rng.normal(size=20000)
    scores = [population_stability_index(reference, rng.normal(loc=shift, size=20000))
              for shift in (0.1, 0.5, 1.0, 2.0)]
    assert scores == sorted(scores)
    assert scores[0] < 0.1 < scores[-1]


def test_psi_is_only_approximately_symmetric(rng):
    """Bin edges come from the reference, so swapping the arguments moves the result.

    The PSI formula is symmetric for a fixed binning; this implementation derives the
    edges from `reference`, which is deliberate (see `_numeric_bin_edges`). The
    consequence is a small asymmetry, pinned here so it stays small and stays known.
    """
    a, b = rng.normal(size=8000), rng.normal(loc=0.8, size=8000)
    forward = population_stability_index(a, b)
    backward = population_stability_index(b, a)
    assert forward == pytest.approx(backward, rel=0.05)
    assert forward != backward


def test_a_tied_numeric_column_is_recognised_as_discrete(raw_data):
    """`hours_per_week` and `capital_gain` are discrete variables in a numeric dtype."""
    development, _, _ = raw_data
    assert (development["hours_per_week"] == 40).mean() > 0.40
    assert _is_effectively_discrete(development["hours_per_week"].to_numpy(float), 10)
    assert _is_effectively_discrete(development["capital_gain"].to_numpy(float), 10)
    assert not _is_effectively_discrete(development["age"].to_numpy(float), 10)


def test_psi_on_a_tied_feature_is_stable_across_reference_subsamples(raw_data):
    """Regression test for a real defect found while building this.

    With quantile binning, PSI for the *same* +2-hour shift scored 0.002 against the
    full reference and 2.26 against a 4.5k-row subsample of it -- because the smaller
    sample happened to place a bin edge at 42, splitting the 40-hour mode, while the
    full sample placed it at 48 and hid the move entirely. A detector whose verdict
    depends on that is not usable, so tied columns now use value-based bins.

    This test fails if quantile binning is ever reinstated for tied columns.
    """
    development, _, _ = raw_data
    hours = development["hours_per_week"]
    shifted = (hours + 2).clip(1, 99)
    full_reference = population_stability_index(hours, shifted)
    subsample_reference = population_stability_index(hours.sample(4500, random_state=0), shifted)
    assert full_reference > 1.0, "relocating a 47% mode must not read as 'no drift'"
    assert subsample_reference > 1.0
    assert abs(full_reference - subsample_reference) < 1.0


def test_discrete_psi_saturates_so_wasserstein_carries_the_magnitude(raw_data):
    """The residual limitation, pinned.

    On a tied column, value-based PSI answers "did the distribution move?" and
    saturates: a +2 and a +12 shift score almost the same. Magnitude therefore has to
    come from the Wasserstein distance, which is why the report carries both.
    """
    development, _, _ = raw_data
    hours = development["hours_per_week"]
    small = population_stability_index(hours, (hours + 2).clip(1, 99))
    large = population_stability_index(hours, (hours + 12).clip(1, 99))
    assert small > 1.0 and large > 1.0
    assert abs(large - small) < 2.0, "PSI is expected to saturate here"

    from scipy.stats import wasserstein_distance

    w_small = wasserstein_distance(hours, (hours + 2).clip(1, 99))
    w_large = wasserstein_distance(hours, (hours + 12).clip(1, 99))
    assert w_large > 4 * w_small, "Wasserstein must still order the two shifts"


def test_rare_discrete_values_are_pooled(raw_data):
    """A long tail of singleton values must not dominate a discrete PSI."""
    development, _, _ = raw_data
    reference = development["capital_gain"]
    current = reference.sample(frac=0.5, random_state=1)
    assert population_stability_index(reference, current) < 0.05


def test_psi_handles_an_absent_category_without_exploding():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 1000)
    value = population_stability_index(reference, current, categorical=True)
    assert np.isfinite(value)
    assert value > 1.0


def test_psi_of_a_constant_reference_does_not_crash():
    assert np.isfinite(population_stability_index(np.ones(100), np.ones(100)))


def test_psi_of_an_empty_sample_is_infinite():
    assert population_stability_index(np.array([]), np.ones(10)) == float("inf")


# ---------------------------------------------------------------- numeric feature


def test_numeric_no_drift_is_not_flagged(rng):
    a = pd.Series(rng.normal(50, 10, 10000))
    b = pd.Series(rng.normal(50, 10, 10000))
    result = numeric_feature_drift("age", a, b, psi_warn=0.1, psi_alert=0.25, ks_alpha=0.01, n_bins=10)
    assert result.drifted is False
    assert result.severity == "none"


def test_numeric_large_shift_is_flagged(rng):
    a = pd.Series(rng.normal(50, 10, 10000))
    b = pd.Series(rng.normal(70, 10, 10000))
    result = numeric_feature_drift("age", a, b, psi_warn=0.1, psi_alert=0.25, ks_alpha=0.01, n_bins=10)
    assert result.drifted is True
    assert result.severity == "alert"
    assert result.psi > 0.25
    assert result.p_value < 0.01


def test_numeric_variance_only_change_is_detected(rng):
    """A change in spread with the same mean must still be visible."""
    a = pd.Series(rng.normal(50, 5, 20000))
    b = pd.Series(rng.normal(50, 20, 20000))
    result = numeric_feature_drift("age", a, b, psi_warn=0.1, psi_alert=0.25, ks_alpha=0.01, n_bins=10)
    assert result.drifted is True
    assert abs(result.reference_mean - result.current_mean) < 1.0


def test_wasserstein_is_reported_in_feature_units(rng):
    a = pd.Series(rng.normal(50, 5, 20000))
    b = pd.Series(rng.normal(62, 5, 20000))
    result = numeric_feature_drift("age", a, b, psi_warn=0.1, psi_alert=0.25, ks_alpha=0.01, n_bins=10)
    assert result.wasserstein == pytest.approx(12.0, abs=0.5)


def test_numeric_drift_on_an_empty_current_window_is_flagged():
    result = numeric_feature_drift("age", pd.Series([30, 40, 50]), pd.Series([], dtype=float),
                                   psi_warn=0.1, psi_alert=0.25, ks_alpha=0.01, n_bins=10)
    assert result.drifted is True
    assert "no usable" in result.note


# ------------------------------------------------------------ categorical feature


def test_categorical_no_drift_is_not_flagged(rng):
    levels = ["Private", "Local-gov", "State-gov", "Self-emp-not-inc"]
    probabilities = [0.7, 0.1, 0.1, 0.1]
    a = pd.Series(rng.choice(levels, 10000, p=probabilities))
    b = pd.Series(rng.choice(levels, 10000, p=probabilities))
    result = categorical_feature_drift("workclass", a, b, psi_warn=0.1, psi_alert=0.25, chi2_alpha=0.01)
    assert result.drifted is False


def test_categorical_mix_change_is_flagged(rng):
    levels = ["Private", "Local-gov", "State-gov", "Self-emp-not-inc"]
    a = pd.Series(rng.choice(levels, 10000, p=[0.7, 0.1, 0.1, 0.1]))
    b = pd.Series(rng.choice(levels, 10000, p=[0.25, 0.25, 0.25, 0.25]))
    result = categorical_feature_drift("workclass", a, b, psi_warn=0.1, psi_alert=0.25, chi2_alpha=0.01)
    assert result.drifted is True
    assert result.psi > 0.25


def test_rare_levels_are_pooled_for_chi_square(rng):
    common = ["United-States"] * 5000
    rare = [f"Country-{i}" for i in range(30)]
    a = pd.Series(common + rare)
    b = pd.Series(common + rare)
    result = categorical_feature_drift("native_country", a, b, psi_warn=0.1, psi_alert=0.25, chi2_alpha=0.01)
    assert "pooled" in result.note
    assert result.drifted is False


def test_a_brand_new_level_moves_psi():
    a = pd.Series(["Male"] * 500 + ["Female"] * 500)
    b = pd.Series(["Male"] * 400 + ["Female"] * 400 + ["Unspecified"] * 200)
    result = categorical_feature_drift("sex", a, b, psi_warn=0.1, psi_alert=0.25, chi2_alpha=0.01)
    assert result.psi > 0.25


def test_single_level_on_both_sides_is_not_drift():
    a = pd.Series(["Male"] * 1000)
    b = pd.Series(["Male"] * 1000)
    result = categorical_feature_drift("sex", a, b, psi_warn=0.1, psi_alert=0.25, chi2_alpha=0.01)
    assert result.drifted is False


# -------------------------------------------------------------------- detector


def test_detector_requires_the_full_contract(feature_frame):
    with pytest.raises(ValueError, match="missing contract feature"):
        DriftDetector(feature_frame.drop(columns=["age"]))


def test_detector_reports_every_feature(feature_frame):
    report = DriftDetector(feature_frame).detect(feature_frame)
    assert len(report.features) == len(FEATURE_NAMES)
    assert {f.feature for f in report.features} == set(FEATURE_NAMES)


def test_identical_frames_produce_no_drift(feature_frame):
    report = DriftDetector(feature_frame).detect(feature_frame)
    assert report.drift_detected is False
    assert report.n_drifted == 0
    assert report.max_psi == pytest.approx(0.0, abs=1e-6)


def test_detector_uses_the_right_test_per_feature_kind(feature_frame):
    report = DriftDetector(feature_frame).detect(feature_frame)
    by_name = {f.feature: f for f in report.features}
    for name in NUMERIC_FEATURES:
        assert by_name[name].test == "ks_2samp"
    for name in CATEGORICAL_FEATURES:
        assert by_name[name].test == "chi2_homogeneity"


def test_missing_column_is_a_schema_failure_not_a_statistic(feature_frame):
    report = DriftDetector(feature_frame).detect(feature_frame.drop(columns=["occupation"]))
    assert report.drift_detected is True
    failures = {f["feature"]: f for f in report.schema_failures}
    assert failures["occupation"]["failure"] == "missing_column"
    assert "occupation" not in {f.feature for f in report.features}


def test_null_values_are_a_schema_failure(feature_frame):
    current = feature_frame.copy()
    current["hours_per_week"] = current["hours_per_week"].astype(float)
    current.iloc[:100, current.columns.get_loc("hours_per_week")] = np.nan
    report = DriftDetector(feature_frame).detect(current)
    failures = {f["feature"]: f for f in report.schema_failures}
    assert failures["hours_per_week"]["failure"] == "null_values"
    assert failures["hours_per_week"]["null_rate"] > 0


def test_min_drifted_features_raises_the_alert_bar(feature_frame, rng):
    current = feature_frame.copy()
    current["age"] = (current["age"] + 25).clip(17, 90)
    lenient = DriftDetector(feature_frame, min_drifted_features=1).detect(current)
    strict = DriftDetector(feature_frame, min_drifted_features=5).detect(current)
    assert lenient.drift_detected is True
    assert strict.drift_detected is False
    assert lenient.n_drifted == strict.n_drifted


def test_report_serialises_completely(feature_frame):
    import json

    payload = DriftDetector(feature_frame).detect(feature_frame, scenario="x").to_dict()
    json.dumps(payload)
    assert payload["scenario"] == "x"
    assert payload["n_features"] == len(FEATURE_NAMES)
    assert set(payload["thresholds"]) >= {"psi_warn", "psi_alert", "ks_alpha", "chi2_alpha"}


def test_report_frame_has_one_row_per_feature(feature_frame):
    frame = DriftDetector(feature_frame).detect(feature_frame).to_frame()
    assert len(frame) == len(FEATURE_NAMES)


def test_detector_is_built_from_config(feature_frame, config):
    detector = DriftDetector.from_config(feature_frame, config)
    assert detector.psi_alert == float(config.require("drift.psi_alert"))
    assert detector.ks_alpha == float(config.require("drift.ks_alpha"))


def test_detection_is_deterministic(feature_frame):
    detector = DriftDetector(feature_frame)
    first = detector.detect(feature_frame.head(500)).to_dict()
    second = detector.detect(feature_frame.head(500)).to_dict()
    assert first["features"] == second["features"]


# -------------------------------------------------------------- prediction drift


def test_prediction_drift_is_quiet_on_identical_scores(rng):
    scores = rng.beta(2, 5, 10000)
    assert prediction_drift(scores, scores)["drifted"] is False


def test_prediction_drift_detects_an_output_shift(rng):
    reference = rng.beta(2, 5, 10000)
    current = rng.beta(5, 2, 10000)
    result = prediction_drift(reference, current)
    assert result["drifted"] is True
    assert result["current_mean"] > result["reference_mean"]
    assert result["current_positive_rate"] > result["reference_positive_rate"]


def test_prediction_drift_on_an_empty_window_is_flagged():
    assert prediction_drift(np.array([0.1, 0.2]), np.array([]))["drifted"] is True


# ----------------------------------------------------------------- scenarios


def test_all_required_scenarios_exist():
    assert set(REQUIRED_SCENARIOS).issubset(set(SCENARIOS))


def test_scenarios_are_deterministic(small_split):
    base = small_split.test
    for name in SCENARIOS:
        a = build_scenario(name, base, seed=3, n_rows=500)
        b = build_scenario(name, base, seed=3, n_rows=500)
        pd.testing.assert_frame_equal(a, b)


def test_a_different_scenario_seed_gives_a_different_window(small_split):
    a = build_scenario("large_drift", small_split.test, seed=1, n_rows=500)
    b = build_scenario("large_drift", small_split.test, seed=2, n_rows=500)
    assert not a.equals(b)


def test_no_drift_scenario_preserves_the_columns(small_split):
    window = build_scenario("no_drift", small_split.test, seed=1, n_rows=500)
    assert list(window.columns) == list(small_split.test.columns)


def test_missing_feature_scenario_actually_breaks_the_schema(small_split):
    window = build_scenario("missing_feature", small_split.test, seed=1, n_rows=500)
    assert "occupation" not in window.columns
    assert window["hours_per_week"].isna().mean() == pytest.approx(0.3, abs=0.01)


def test_performance_drift_changes_labels_not_features(small_split):
    """Concept drift by construction: the inputs are untouched."""
    base = small_split.test.head(500).reset_index(drop=True)
    window = build_scenario("performance_drift", base, seed=1)
    pd.testing.assert_frame_equal(window[FEATURE_NAMES], base[FEATURE_NAMES])
    assert not window["income"].equals(base["income"])


def test_prior_shift_moves_the_class_balance(small_split):
    base = small_split.test
    window = build_scenario("prior_shift", base, seed=1, n_rows=2000)
    assert (window["income"] == ">50K").mean() == pytest.approx(0.5, abs=0.02)


def test_unknown_scenario_raises(small_split):
    with pytest.raises(KeyError, match="unknown scenario"):
        build_scenario("teleportation", small_split.test, seed=1)


def test_scenario_table_documents_every_scenario():
    table = scenario_table()
    assert len(table) == len(SCENARIOS)
    assert table["description"].str.len().min() > 20


# ------------------------------------------------- end-to-end scenario behaviour


@pytest.mark.parametrize("scenario,should_alert", [
    ("no_drift", False),
    ("small_drift", False),
    ("large_drift", True),
    ("missing_feature", True),
    ("covariate_shift", True),
])
def test_scenario_detection_matches_expectation(small_split, config, scenario, should_alert):
    reference = small_split.train[FEATURE_NAMES]
    detector = DriftDetector.from_config(reference, config)
    window = build_scenario(scenario, small_split.test, seed=11, n_rows=2000)
    assert detector.detect(window, scenario=scenario).drift_detected is should_alert


def test_input_drift_does_not_see_concept_drift(small_split, config):
    """The honest limitation, asserted rather than buried in prose.

    `performance_drift` changes only the label-generating process. A detector that
    watches input distributions therefore cannot see it, and this test exists so that
    the claim in DRIFT_DETECTION.md stays true if the detector changes.
    """
    reference = small_split.train[FEATURE_NAMES]
    detector = DriftDetector.from_config(reference, config)
    window = build_scenario("performance_drift", small_split.test, seed=11, n_rows=2000)
    assert detector.detect(window, scenario="performance_drift").drift_detected is False


# ------------------------------------------ prediction drift: shape and level


def test_prediction_level_drift_catches_a_mean_shift_that_psi_misses():
    """The complementary signal, justified by the gap it was added to close.

    A prior shift moves the mean score sharply while leaving the score histogram's
    shape similar enough that PSI stays below any threshold that would not also fire
    on a harmless small covariate shift. The level test is what makes that case
    detectable.
    """
    rng = np.random.default_rng(7)
    reference = rng.beta(2, 6, 20000)          # mean ~0.25
    current = np.clip(reference * 1.6, 0, 1)   # same shape, higher level
    result = prediction_drift(reference, current, psi_alert=0.25, ks_alpha=0.01)
    assert result["level_drift"] is True
    assert result["drifted"] is True
    assert result["mean_shift"] > 0.25


def test_prediction_drift_stays_quiet_on_a_small_level_change():
    """Negative control for the level test: a few percent must not alarm."""
    rng = np.random.default_rng(8)
    reference = rng.beta(2, 6, 20000)
    current = np.clip(reference * 1.03, 0, 1)
    result = prediction_drift(reference, current, psi_alert=0.25, ks_alpha=0.01)
    assert result["level_drift"] is False
    assert result["drifted"] is False


def test_prediction_shape_and_level_are_reported_separately():
    rng = np.random.default_rng(9)
    reference = rng.beta(2, 6, 10000)
    result = prediction_drift(reference, reference)
    assert result["shape_drift"] is False
    assert result["level_drift"] is False
    assert result["mean_shift"] == pytest.approx(0.0, abs=1e-9)


def test_prediction_mean_shift_alert_is_configurable():
    rng = np.random.default_rng(10)
    reference = rng.beta(2, 6, 10000)
    current = np.clip(reference * 1.3, 0, 1)
    assert prediction_drift(reference, current, mean_shift_alert=0.9)["level_drift"] is False
    assert prediction_drift(reference, current, mean_shift_alert=0.05)["level_drift"] is True
