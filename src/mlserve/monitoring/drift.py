"""Statistical data-drift detection.

Choosing a detector per feature type
------------------------------------
There is no single test that is correct for both a continuous column and a 41-level
categorical, so this module applies a different primary test to each and reports them
in one frame.

**Numeric -> two-sample Kolmogorov-Smirnov.** KS is distribution-free (no normality
assumption, which matters because ``capital_gain`` is ~92% zeros with a long tail), it
needs no binning choice, and it is sensitive to any difference in the CDF rather than
only to a shift in the mean. Its weakness is the one every large-sample test has: with
n in the thousands it rejects on differences too small to matter, so the p-value alone
is not used as the decision.

**Categorical -> chi-square test of homogeneity.** The natural test for comparing two
multinomials. Levels with an expected count below 5 are pooled into an ``__rare__``
bucket first, because the chi-square approximation is unreliable below that and
``native_country`` has a long tail of countries with single-digit counts.

**Both -> Population Stability Index, as the magnitude.** PSI answers the question the
p-value does not: *how big* is the change? It is the symmetric discrete
Kullback-Leibler divergence between the two distributions, is independent of sample
size, and has conventional operating points (0.1 / 0.25) that are widely used in
credit risk. This module treats PSI as the decision variable and the p-value as
corroboration, which is why a 200k-row no-drift comparison does not alarm.

**Wasserstein distance** is reported for numeric features as an interpretable
magnitude in the feature's own units ("the average age moved by 4.2 years"), which is
what makes a drift report actionable rather than merely true.

Why not Evidently
-----------------
Evidently would provide all of this. It was rejected for three reasons: it selects the
test for you based on cardinality and row count, so the most important interview
question about this project ("why this test?") would have no answer; its per-feature
defaults change between versions, which undermines the reproducibility claim; and it
adds a large dependency to the container for functionality that is ~200 lines here.
The cost is that this implementation covers fewer test types, which is recorded in
DRIFT_DETECTION.md.

Missing-feature drift is handled explicitly rather than statistically: a feature that
disappears or turns null is a schema failure, not a distribution shift, and is
reported as its own category so it cannot be diluted by averaging over the features
that are still fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from mlserve.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

#: Floor applied to a bin's proportion before the PSI log. Without it an empty bin in
#: either sample makes the term infinite, which would let one absent category
#: dominate the whole score.
PSI_EPSILON = 1e-6

#: Expected count below which chi-square levels are pooled.
MIN_EXPECTED_COUNT = 5.0

RARE_LEVEL = "__rare__"


@dataclass
class FeatureDrift:
    """The drift verdict and every statistic behind it, for one feature."""

    feature: str
    kind: str
    psi: float
    statistic: float
    p_value: float
    test: str
    drifted: bool
    severity: str
    wasserstein: float | None = None
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_n: int = 0
    current_n: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "psi": self.psi,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "test": self.test,
            "drifted": self.drifted,
            "severity": self.severity,
            "wasserstein": self.wasserstein,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "note": self.note,
        }


@dataclass
class DriftReport:
    """Aggregate verdict over every feature in one comparison."""

    features: list[FeatureDrift] = field(default_factory=list)
    schema_failures: list[dict] = field(default_factory=list)
    reference_rows: int = 0
    current_rows: int = 0
    elapsed_seconds: float = 0.0
    scenario: str | None = None
    thresholds: dict = field(default_factory=dict)

    @property
    def drifted_features(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.drifted]

    @property
    def n_drifted(self) -> int:
        return len(self.drifted_features)

    @property
    def drift_detected(self) -> bool:
        """Either a statistical alert or a structural failure counts as drift."""
        minimum = int(self.thresholds.get("min_drifted_features", 1))
        return bool(self.schema_failures) or self.n_drifted >= minimum

    @property
    def max_psi(self) -> float:
        return max((f.psi for f in self.features), default=0.0)

    @property
    def mean_psi(self) -> float:
        return float(np.mean([f.psi for f in self.features])) if self.features else 0.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "n_features": len(self.features),
            "n_drifted": self.n_drifted,
            "drift_detected": self.drift_detected,
            "max_psi": round(self.max_psi, 6),
            "mean_psi": round(self.mean_psi, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "schema_failures": self.schema_failures,
            "drifted_features": [f.feature for f in self.drifted_features],
            "thresholds": self.thresholds,
            "features": [f.to_dict() for f in self.features],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.features])


# --------------------------------------------------------------------------- PSI


def _is_effectively_discrete(reference: np.ndarray, n_bins: int) -> bool:
    """True when quantile binning cannot produce ``n_bins`` distinct edges.

    Collapsed edges mean the column has heavy ties -- it is a discrete variable wearing
    a numeric dtype. That is the condition under which quantile PSI becomes unstable
    (see :func:`_numeric_bin_edges`), so it is also the condition for switching to
    value-based bins.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    return len(np.unique(np.quantile(reference, quantiles))) < n_bins + 1


def _numeric_bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile edges from the REFERENCE sample only.

    Binning on the reference alone is what makes PSI a comparison against a fixed
    baseline. Recomputing edges from the pooled data would let the current window move
    the yardstick it is being measured against and hide exactly the shift being looked
    for.

    Quantile edges are used rather than fixed-width because the numeric features here
    are strongly skewed -- ``capital_gain`` is ~92% zeros with a tail to 99999 -- and
    fixed-width bins would put almost all mass in one bin and measure nothing.

    Callers must first check :func:`_is_effectively_discrete`: on a heavily tied column
    the quantile edges collapse and this binning becomes unreliable. See the module
    docstring and DRIFT_DETECTION.md for the measurement that motivated that check.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:  # a constant reference column
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


#: Reference proportion below which a discrete numeric value is pooled into one
#: "rare" bucket, so that a long tail of singleton values cannot dominate the score.
DISCRETE_RARE_PROPORTION = 0.005


def _discrete_proportions(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Value-based proportions for a tied numeric column, with rare values pooled."""
    ref_counts = reference.value_counts(normalize=True)
    keep = set(ref_counts[ref_counts >= DISCRETE_RARE_PROPORTION].index)
    if not keep:
        keep = set(ref_counts.index)

    def collapse(series: pd.Series) -> pd.Series:
        return series.where(series.isin(keep), other=np.nan)

    ref_collapsed, cur_collapsed = collapse(reference), collapse(current)
    levels = sorted(keep)
    ref_p = ref_collapsed.value_counts(normalize=False).reindex(levels, fill_value=0).to_numpy(dtype=float)
    cur_p = cur_collapsed.value_counts(normalize=False).reindex(levels, fill_value=0).to_numpy(dtype=float)
    # One extra cell for everything that was pooled, on each side.
    ref_p = np.append(ref_p, ref_collapsed.isna().sum())
    cur_p = np.append(cur_p, cur_collapsed.isna().sum())
    return ref_p / max(ref_p.sum(), 1.0), cur_p / max(cur_p.sum(), 1.0)


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    categorical: bool = False,
) -> float:
    """PSI between a reference and a current sample.

    PSI = sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref). Unlike a KS p-value it
    does not grow with sample size, which is why it is the decision variable here.

    On symmetry: the *formula* is symmetric for a fixed binning, but this function is
    only approximately symmetric in its arguments, because the bin edges are quantiles
    of ``reference``. Swapping the two samples re-derives the edges from the other
    sample and shifts the result slightly (measured at under 2% relative difference for
    a one-sigma shift). Reference-derived binning is the intended behaviour -- see
    :func:`_numeric_bin_edges` -- so the asymmetry is accepted, not a defect.

    Heavily tied numeric columns are routed to value-based bins instead of quantile
    bins -- see :func:`_is_effectively_discrete` for why.
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return float("inf")

    if categorical:
        levels = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        ref_p = ref.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
        cur_p = cur.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
    elif _is_effectively_discrete(ref.to_numpy(dtype=float), n_bins):
        # Heavy ties: quantile edges collapse and become unstable under resampling of
        # the reference, so compare the value distributions directly instead.
        ref_p, cur_p = _discrete_proportions(ref.astype(float), cur.astype(float))
    else:
        edges = _numeric_bin_edges(ref.to_numpy(dtype=float), n_bins)
        ref_counts, _ = np.histogram(ref.to_numpy(dtype=float), bins=edges)
        cur_counts, _ = np.histogram(cur.to_numpy(dtype=float), bins=edges)
        ref_p = ref_counts / max(ref_counts.sum(), 1)
        cur_p = cur_counts / max(cur_counts.sum(), 1)

    ref_p = np.clip(ref_p, PSI_EPSILON, None)
    cur_p = np.clip(cur_p, PSI_EPSILON, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


# ------------------------------------------------------------------- per feature


def _severity(psi: float, warn: float, alert: float) -> str:
    if psi >= alert:
        return "alert"
    if psi >= warn:
        return "warn"
    return "none"


def numeric_feature_drift(
    name: str, reference: pd.Series, current: pd.Series, *,
    psi_warn: float, psi_alert: float, ks_alpha: float, n_bins: int,
) -> FeatureDrift:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return FeatureDrift(name, "numeric", float("inf"), float("nan"), float("nan"),
                            "ks_2samp", True, "alert", reference_n=len(ref), current_n=len(cur),
                            note="one side had no usable numeric values")

    ks = stats.ks_2samp(ref.to_numpy(), cur.to_numpy(), method="asymp")
    psi = population_stability_index(ref, cur, n_bins=n_bins, categorical=False)
    wd = float(stats.wasserstein_distance(ref.to_numpy(), cur.to_numpy()))
    severity = _severity(psi, psi_warn, psi_alert)
    # AND, not OR: the p-value alone rejects on trivial differences at these sample
    # sizes, and PSI alone has no notion of sampling noise. Requiring both keeps the
    # measured no-drift false-positive rate low, which DRIFT_DETECTION.md reports.
    drifted = bool(psi >= psi_alert and ks.pvalue < ks_alpha)
    return FeatureDrift(
        feature=name, kind="numeric", psi=round(psi, 6),
        statistic=round(float(ks.statistic), 6), p_value=float(ks.pvalue),
        test="ks_2samp", drifted=drifted, severity=severity,
        wasserstein=round(wd, 6),
        reference_mean=round(float(ref.mean()), 6), current_mean=round(float(cur.mean()), 6),
        reference_n=int(len(ref)), current_n=int(len(cur)),
    )


def categorical_feature_drift(
    name: str, reference: pd.Series, current: pd.Series, *,
    psi_warn: float, psi_alert: float, chi2_alpha: float,
) -> FeatureDrift:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    if ref.empty or cur.empty:
        return FeatureDrift(name, "categorical", float("inf"), float("nan"), float("nan"),
                            "chi2_homogeneity", True, "alert",
                            reference_n=len(ref), current_n=len(cur),
                            note="one side had no usable values")

    levels = sorted(set(ref) | set(cur))
    ref_counts = ref.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)

    # Pool levels whose expected count is too small for the chi-square approximation.
    total = ref_counts.sum() + cur_counts.sum()
    row_totals = np.array([ref_counts.sum(), cur_counts.sum()])
    col_totals = ref_counts + cur_counts
    expected_min = np.outer(row_totals, col_totals).min(axis=0) / max(total, 1.0)
    keep = expected_min >= MIN_EXPECTED_COUNT
    if not keep.all():
        pooled_ref = np.append(ref_counts[keep], ref_counts[~keep].sum())
        pooled_cur = np.append(cur_counts[keep], cur_counts[~keep].sum())
        ref_counts, cur_counts = pooled_ref, pooled_cur
        note = f"{int((~keep).sum())} rare level(s) pooled into {RARE_LEVEL}"
    else:
        note = ""

    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value, _, _ = stats.chi2_contingency(table, correction=False)

    psi = population_stability_index(ref, cur, categorical=True)
    severity = _severity(psi, psi_warn, psi_alert)
    drifted = bool(psi >= psi_alert and p_value < chi2_alpha)
    return FeatureDrift(
        feature=name, kind="categorical", psi=round(psi, 6),
        statistic=round(float(statistic), 6), p_value=float(p_value),
        test="chi2_homogeneity", drifted=drifted, severity=severity,
        reference_n=int(len(ref)), current_n=int(len(cur)), note=note,
    )


# ----------------------------------------------------------------------- detector


class DriftDetector:
    """Compares a current window against a fixed reference sample."""

    def __init__(
        self,
        reference: pd.DataFrame,
        *,
        psi_warn: float = 0.1,
        psi_alert: float = 0.25,
        ks_alpha: float = 0.01,
        chi2_alpha: float = 0.01,
        n_bins: int = 10,
        min_drifted_features: int = 1,
        features: list[str] | None = None,
    ):
        self.features = features or list(FEATURE_NAMES)
        missing = [f for f in self.features if f not in reference.columns]
        if missing:
            raise ValueError(f"reference frame is missing contract feature(s): {missing}")
        self.reference = reference[self.features].copy()
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.ks_alpha = ks_alpha
        self.chi2_alpha = chi2_alpha
        self.n_bins = n_bins
        self.min_drifted_features = min_drifted_features

    @classmethod
    def from_config(cls, reference: pd.DataFrame, config) -> DriftDetector:
        return cls(
            reference,
            psi_warn=float(config.require("drift.psi_warn")),
            psi_alert=float(config.require("drift.psi_alert")),
            ks_alpha=float(config.require("drift.ks_alpha")),
            chi2_alpha=float(config.require("drift.chi2_alpha")),
            n_bins=int(config.require("drift.n_bins")),
            min_drifted_features=int(config.require("drift.min_drifted_features")),
        )

    @property
    def thresholds(self) -> dict:
        return {
            "psi_warn": self.psi_warn,
            "psi_alert": self.psi_alert,
            "ks_alpha": self.ks_alpha,
            "chi2_alpha": self.chi2_alpha,
            "n_bins": self.n_bins,
            "min_drifted_features": self.min_drifted_features,
        }

    def detect(self, current: pd.DataFrame, *, scenario: str | None = None) -> DriftReport:
        start = time.perf_counter()
        report = DriftReport(
            reference_rows=len(self.reference), current_rows=len(current),
            scenario=scenario, thresholds=self.thresholds,
        )

        for name in self.features:
            # Structural problems are reported as schema failures and excluded from the
            # statistical pass: running KS against an absent column would either crash
            # or, worse, silently return "no drift".
            if name not in current.columns:
                report.schema_failures.append(
                    {"feature": name, "failure": "missing_column",
                     "message": f"{name} is absent from the current window"}
                )
                continue
            column = current[name]
            null_rate = float(column.isna().mean()) if len(column) else 1.0
            if null_rate > 0.0:
                report.schema_failures.append(
                    {"feature": name, "failure": "null_values",
                     "null_rate": round(null_rate, 6),
                     "message": f"{name} is {null_rate:.1%} null; the contract permits none"}
                )
                if null_rate == 1.0:
                    continue

            if name in NUMERIC_FEATURES:
                report.features.append(numeric_feature_drift(
                    name, self.reference[name], column,
                    psi_warn=self.psi_warn, psi_alert=self.psi_alert,
                    ks_alpha=self.ks_alpha, n_bins=self.n_bins,
                ))
            elif name in CATEGORICAL_FEATURES:
                report.features.append(categorical_feature_drift(
                    name, self.reference[name], column,
                    psi_warn=self.psi_warn, psi_alert=self.psi_alert,
                    chi2_alpha=self.chi2_alpha,
                ))

        report.elapsed_seconds = time.perf_counter() - start
        return report


#: Relative change in the mean predicted probability that counts as output drift,
#: independently of PSI. Justified by measurement, not convention: on this dataset a
#: prior shift moves the mean score from 0.233 to 0.371 (+59%) while the score PSI
#: only reaches 0.16 -- below any PSI threshold that does not also fire on a harmless
#: small covariate shift. The two signals are therefore complementary, and 0.25 sits
#: in the measured gap between the no-drift/small-drift cases (0-3%) and the real
#: ones (35-63%). See DRIFT_DETECTION.md.
PREDICTION_MEAN_SHIFT_ALERT = 0.25


def prediction_drift(
    reference_scores: np.ndarray | pd.Series,
    current_scores: np.ndarray | pd.Series,
    *,
    psi_alert: float = 0.25,
    ks_alpha: float = 0.01,
    n_bins: int = 10,
    mean_shift_alert: float = PREDICTION_MEAN_SHIFT_ALERT,
) -> dict:
    """Drift in the model's *output* distribution.

    This is the only drift signal available when labels are absent, which in this
    problem is effectively always: income is observed months after the prediction, if
    ever. A shift here means the population reaching the model changed enough to move
    its scores.

    Two independent alert conditions, because one is not enough:

    * **Shape** -- PSI over the score histogram plus a KS test. Catches a change in the
      *form* of the score distribution.
    * **Level** -- relative change in the mean score. Catches a change in the *base
      rate* that leaves the shape broadly intact. A prior shift is exactly that case,
      and the shape test measurably misses it on this dataset.

    Either condition alone raises the alert. Neither can see concept drift, where the
    inputs and therefore the scores are unchanged and only the labels moved; that
    requires ground truth and is recorded as a limitation rather than papered over.
    """
    ref = pd.Series(reference_scores).dropna().to_numpy(dtype=float)
    cur = pd.Series(current_scores).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return {"psi": float("inf"), "p_value": float("nan"), "drifted": True,
                "reference_mean": None, "current_mean": None,
                "mean_shift": None, "shape_drift": True, "level_drift": True}

    ks = stats.ks_2samp(ref, cur, method="asymp")
    psi = population_stability_index(ref, cur, n_bins=n_bins)
    reference_mean, current_mean = float(ref.mean()), float(cur.mean())
    mean_shift = (
        abs(current_mean - reference_mean) / reference_mean if reference_mean > 0 else float("inf")
    )

    shape_drift = bool(psi >= psi_alert and ks.pvalue < ks_alpha)
    level_drift = bool(mean_shift >= mean_shift_alert)
    return {
        "psi": round(psi, 6),
        "ks_statistic": round(float(ks.statistic), 6),
        "p_value": float(ks.pvalue),
        "shape_drift": shape_drift,
        "level_drift": level_drift,
        "drifted": bool(shape_drift or level_drift),
        "reference_mean": round(reference_mean, 6),
        "current_mean": round(current_mean, 6),
        "mean_shift": round(mean_shift, 6),
        "reference_positive_rate": round(float((ref >= 0.5).mean()), 6),
        "current_positive_rate": round(float((cur >= 0.5).mean()), 6),
    }
