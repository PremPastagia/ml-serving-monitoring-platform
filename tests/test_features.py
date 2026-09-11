"""Feature engineering: correctness of the derived columns and train/serve identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlserve.data.schema import FEATURE_NAMES
from mlserve.features.pipeline import (
    ALL_CATEGORICAL,
    ALL_NUMERIC,
    MODEL_FEATURES,
    DerivedFeatures,
    build_pipeline,
    expected_input_columns,
)


def test_derived_columns_are_added(feature_frame):
    out = DerivedFeatures().fit_transform(feature_frame)
    for column in DerivedFeatures.OUTPUT_COLUMNS:
        assert column in out.columns


def test_net_capital_is_gain_minus_loss(feature_frame):
    out = DerivedFeatures().fit_transform(feature_frame)
    expected = feature_frame["capital_gain"].astype(float) - feature_frame["capital_loss"].astype(float)
    pd.testing.assert_series_equal(out["net_capital"], expected, check_names=False)


def test_has_capital_flow_flags_any_activity(feature_frame):
    out = DerivedFeatures().fit_transform(feature_frame)
    expected = ((feature_frame["capital_gain"] > 0) | (feature_frame["capital_loss"] > 0)).astype(int)
    pd.testing.assert_series_equal(out["has_capital_flow"], expected, check_names=False)


def test_log_capital_gain_is_monotonic_and_finite(feature_frame):
    out = DerivedFeatures().fit_transform(feature_frame)
    assert np.isfinite(out["log_capital_gain"]).all()
    assert (out["log_capital_gain"] >= 0).all()
    pairs = out[["capital_gain", "log_capital_gain"]].sort_values("capital_gain")
    assert pairs["log_capital_gain"].is_monotonic_increasing


@pytest.mark.parametrize("hours,band", [(1, "part_time"), (34, "part_time"), (35, "full_time"),
                                        (40, "full_time"), (44, "full_time"), (45, "overtime"),
                                        (99, "overtime")])
def test_hours_band_boundaries(feature_frame, hours, band):
    frame = feature_frame.head(1).copy()
    frame["hours_per_week"] = hours
    assert DerivedFeatures().fit_transform(frame)["hours_band"].iloc[0] == band


def test_derived_features_are_stateless(feature_frame):
    """A stateless transform cannot leak: fitting on any subset gives the same output."""
    a = DerivedFeatures().fit(feature_frame.head(10)).transform(feature_frame.head(100))
    b = DerivedFeatures().fit(feature_frame.tail(10)).transform(feature_frame.head(100))
    pd.testing.assert_frame_equal(a, b)


def test_row_order_does_not_change_a_row_result(feature_frame):
    """Every derived column is a pure function of one row."""
    head = feature_frame.head(50)
    forward = DerivedFeatures().fit_transform(head)
    reversed_ = DerivedFeatures().fit_transform(head.iloc[::-1]).iloc[::-1]
    pd.testing.assert_frame_equal(forward, reversed_)


def test_model_features_partition_into_numeric_and_categorical():
    assert set(ALL_NUMERIC) | set(ALL_CATEGORICAL) == set(MODEL_FEATURES)
    assert not set(ALL_NUMERIC) & set(ALL_CATEGORICAL)


def test_expected_input_columns_is_the_contract():
    assert expected_input_columns() == FEATURE_NAMES


@pytest.mark.parametrize("estimator,params", [
    ("hist_gradient_boosting", {"max_iter": 10}),
    ("logistic_regression", {"max_iter": 100}),
])
def test_both_estimators_build_and_fit(small_split, estimator, params):
    X, y = small_split.xy("train")
    pipeline = build_pipeline(estimator, params, seed=0)
    assert isinstance(pipeline, Pipeline)
    pipeline.fit(X.head(500), y.head(500))
    proba = pipeline.predict_proba(X.head(10))[:, 1]
    assert proba.shape == (10,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        build_pipeline("random_forest_of_dreams", {}, seed=0)


def test_preprocessing_statistics_come_from_train_only(small_split):
    """Fitting on train then transforming test must not refit anything.

    The check is behavioural: transforming the test frame twice, with an intervening
    transform of a wildly different frame, must give identical output. A transformer
    that adapted to the data it sees would fail this.
    """
    X, y = small_split.xy("train")
    pipeline = build_pipeline("hist_gradient_boosting", {"max_iter": 10}, seed=0)
    pipeline.fit(X, y)
    X_test, _ = small_split.xy("test")
    first = pipeline.predict_proba(X_test.head(200))[:, 1]
    shifted = X_test.head(200).copy()
    shifted["age"] = 90
    pipeline.predict_proba(shifted)
    second = pipeline.predict_proba(X_test.head(200))[:, 1]
    np.testing.assert_array_equal(first, second)


def test_unseen_category_does_not_raise_at_predict_time(small_split):
    """Defence in depth behind the validator: an unknown level degrades one row."""
    X, y = small_split.xy("train")
    pipeline = build_pipeline("hist_gradient_boosting", {"max_iter": 10}, seed=0)
    pipeline.fit(X, y)
    row = X.head(1).copy()
    row["native_country"] = "Atlantis"
    proba = pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_column_order_does_not_change_predictions(small_split):
    """The ColumnTransformer selects by name, so caller column order is irrelevant."""
    X, y = small_split.xy("train")
    pipeline = build_pipeline("hist_gradient_boosting", {"max_iter": 10}, seed=0)
    pipeline.fit(X, y)
    head = X.head(20)
    shuffled = head[list(reversed(head.columns))]
    np.testing.assert_allclose(
        pipeline.predict_proba(head)[:, 1], pipeline.predict_proba(shuffled)[:, 1]
    )
