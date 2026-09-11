"""Feature engineering and preprocessing, defined once and shared by train and serve.

The whole pipeline is a single scikit-learn estimator. That is the point: the object
persisted to the model registry *contains* the preprocessing, so the API cannot apply
a different transformation from the one training used. There is no second code path
to keep in sync, which removes the most common cause of train/serve skew.

All statistics (bin edges, one-hot vocabularies, scaler means) are learned inside
``fit`` and therefore only ever see the training split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from mlserve.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Adds the four engineered columns. Stateless, so it cannot leak.

    Each one encodes a hypothesis that the raw columns state awkwardly:

    * ``net_capital``      -- gains minus losses. The two raw columns are ~92% zero;
                              a tree needs several splits to express "net position",
                              which this gives it in one.
    * ``has_capital_flow`` -- whether the respondent reported any capital activity at
                              all. Separates the zero-inflation mass from the
                              magnitude, which is the part that actually carries signal.
    * ``log_capital_gain`` -- log1p of gains. The raw column spans 0..99999 with a
                              spike at the 99999 top-code; compressing it keeps the
                              top-code from dominating the histogram binning.
    * ``hours_band``       -- part-time / full-time / overtime. Hours has a strong
                              mode at exactly 40, and the band makes the "unusual
                              hours" contrast explicit rather than implicit in splits.

    All four are pure functions of a single row, so they compute identically in
    training and inside a request, with no fitted state to drift.
    """

    OUTPUT_COLUMNS = ["net_capital", "has_capital_flow", "log_capital_gain", "hours_band"]

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        gain = out["capital_gain"].astype(float)
        loss = out["capital_loss"].astype(float)
        out["net_capital"] = gain - loss
        out["has_capital_flow"] = ((gain > 0) | (loss > 0)).astype(int)
        out["log_capital_gain"] = np.log1p(gain.clip(lower=0))
        hours = out["hours_per_week"].astype(float)
        out["hours_band"] = pd.cut(
            hours, bins=[-np.inf, 34, 44, np.inf], labels=["part_time", "full_time", "overtime"]
        ).astype(str)
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + self.OUTPUT_COLUMNS, dtype=object)


DERIVED_NUMERIC = ["net_capital", "has_capital_flow", "log_capital_gain"]
DERIVED_CATEGORICAL = ["hours_band"]

ALL_NUMERIC = NUMERIC_FEATURES + DERIVED_NUMERIC
ALL_CATEGORICAL = CATEGORICAL_FEATURES + DERIVED_CATEGORICAL

#: The exact ordered feature list the model consumes after engineering.
MODEL_FEATURES = ALL_NUMERIC + ALL_CATEGORICAL


def _tree_preprocessor() -> ColumnTransformer:
    """Ordinal encoding for the gradient-boosted trees.

    Trees need no scaling and no one-hot: HistGradientBoosting can treat an
    ordinally-coded column as genuinely categorical, which avoids the sparse
    high-dimensional blow-up one-hot would cause on `native_country` (41 levels).
    ``handle_unknown='use_encoded_value'`` maps an unseen level to -1 rather than
    raising -- defence in depth behind the validator, so an unexpected category
    degrades a single prediction instead of taking the service down.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", ALL_NUMERIC),
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1),
                ALL_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _linear_preprocessor() -> ColumnTransformer:
    """Scale + one-hot for the linear baseline, which does need both."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ALL_NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                ALL_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: str, params: dict, *, seed: int) -> Pipeline:
    """Assemble the end-to-end estimator named in the configuration."""
    if estimator == "hist_gradient_boosting":
        categorical_mask = [name in ALL_CATEGORICAL for name in MODEL_FEATURES]
        model = HistGradientBoostingClassifier(
            random_state=seed,
            categorical_features=categorical_mask,
            **params,
        )
        pre = _tree_preprocessor()
    elif estimator == "logistic_regression":
        model = LogisticRegression(random_state=seed, **params)
        pre = _linear_preprocessor()
    else:
        raise ValueError(f"unknown estimator {estimator!r}")

    return Pipeline(
        steps=[
            ("derive", DerivedFeatures()),
            ("preprocess", pre),
            ("model", model),
        ]
    )


def expected_input_columns() -> list[str]:
    """Columns a caller must supply. Derived columns are produced internally."""
    return list(FEATURE_NAMES)
