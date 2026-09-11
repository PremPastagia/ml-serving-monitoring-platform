"""Controlled drift scenarios.

Each scenario is a deterministic, seeded transformation of a held-out sample, so the
same scenario id always produces the same frame and a drift result is reproducible.

The five required scenarios, and what each one is actually testing:

``no_drift``          A disjoint resample from the same population. Nothing changed,
                      so every alert here is a false positive. This is the scenario
                      that produces the false-positive rate; without it a detector
                      that always alarms would look perfect.
``small_drift``       A shift deliberately sized below the alert threshold (+2 years
                      of age, +2 hours per week). Tests that the detector is not
                      hair-triggered on a change too small to act on.
``large_drift``       A shift that any usable detector must catch (+12 years, +12
                      hours, capital gains scaled 3x, occupation mix reweighted).
``missing_feature``   A column is dropped and another is made null. This is a schema
                      failure, not a distribution shift, and is reported separately.
``performance_drift`` A *concept* shift: the feature-to-label relationship changes
                      while the marginal feature distributions barely move. This is
                      the one that hurts, because input-distribution monitoring can
                      miss it entirely -- which is the honest limitation this project
                      has to demonstrate rather than hide.

``covariate_shift`` and ``prior_shift`` are included as extra sub-cases so that the
detection-rate figures are not computed from a single example per class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlserve.data.schema import TARGET, TARGET_CLASSES


@dataclass(frozen=True)
class Scenario:
    """One named, reproducible transformation of a base sample."""

    name: str
    description: str
    expect_drift: bool
    kind: str  # none | covariate | prior | schema | concept

    def apply(self, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        raise NotImplementedError


def _shift_numeric(frame: pd.DataFrame, column: str, delta: float,
                   lo: float, hi: float) -> pd.DataFrame:
    frame[column] = (frame[column].astype(float) + delta).clip(lo, hi).round().astype(int)
    return frame


class NoDrift(Scenario):
    def __init__(self):
        super().__init__("no_drift", "Disjoint resample of the same population.", False, "none")

    def apply(self, frame, rng):
        return frame.copy()


#: Fraction of records whose hours are perturbed in the small-drift scenario.
SMALL_DRIFT_HOURS_FRACTION = 0.12


class SmallDrift(Scenario):
    def __init__(self):
        super().__init__(
            "small_drift",
            "Age +2 years for everyone, and hours/week +3 for a random 12% of records "
            "-- a change too small to be worth acting on.",
            False, "covariate",
        )

    def apply(self, frame, rng):
        out = frame.copy()
        out = _shift_numeric(out, "age", 2, 17, 90)
        # Hours is perturbed for a *subset* rather than shifted wholesale. 46.7% of
        # respondents report exactly 40 hours, so adding a constant to every row moves
        # that entire mode across a PSI bin boundary and scores as large drift -- which
        # is correct, but makes it the wrong construction for a "small" scenario.
        # Perturbing a minority of rows leaves the mode in place and produces the
        # gradual shift this scenario is meant to represent.
        n = int(SMALL_DRIFT_HOURS_FRACTION * len(out))
        if n:
            idx = rng.choice(len(out), size=n, replace=False)
            column = out.columns.get_loc("hours_per_week")
            out.iloc[idx, column] = (
                out.iloc[idx, column].astype(float) + 3
            ).clip(1, 99).round().astype(int)
        return out


class LargeDrift(Scenario):
    def __init__(self):
        super().__init__(
            "large_drift",
            "Age +12, hours/week +12, capital gains x3, occupation mix reweighted "
            "towards Exec-managerial and Prof-specialty.",
            True, "covariate",
        )

    def apply(self, frame, rng):
        out = frame.copy()
        out = _shift_numeric(out, "age", 12, 17, 90)
        out = _shift_numeric(out, "hours_per_week", 12, 1, 99)
        out["capital_gain"] = (out["capital_gain"].astype(float) * 3).clip(0, 99999).round().astype(int)
        # Resample rows so that high-earning occupations are over-represented: this
        # moves a categorical distribution without inventing impossible records.
        favoured = out["occupation"].isin(["Exec-managerial", "Prof-specialty"])
        weights = np.where(favoured, 4.0, 1.0)
        weights = weights / weights.sum()
        picks = rng.choice(len(out), size=len(out), replace=True, p=weights)
        return out.iloc[picks].reset_index(drop=True)


class MissingFeatureDrift(Scenario):
    def __init__(self):
        super().__init__(
            "missing_feature",
            "The 'occupation' column is dropped and 'hours_per_week' is 30% null -- "
            "an upstream schema break rather than a distribution shift.",
            True, "schema",
        )

    def apply(self, frame, rng):
        out = frame.copy().drop(columns=["occupation"])
        n_null = int(0.3 * len(out))
        idx = rng.choice(len(out), size=n_null, replace=False)
        out["hours_per_week"] = out["hours_per_week"].astype("float64")
        out.iloc[idx, out.columns.get_loc("hours_per_week")] = np.nan
        return out


class PerformanceDrift(Scenario):
    def __init__(self):
        super().__init__(
            "performance_drift",
            "Concept shift: labels are re-drawn so education matters far less and "
            "hours worked far more, while the marginal feature distributions are "
            "left almost untouched.",
            True, "concept",
        )

    def apply(self, frame, rng):
        out = frame.copy()
        if TARGET not in out.columns:
            return out
        # A new, deliberately different label-generating process. Features are not
        # modified, so a detector that only watches inputs should mostly miss this --
        # which is exactly the point the experiment is making.
        z = (
            -4.0
            + 0.02 * out["age"].astype(float)
            + 0.01 * out["education_num"].astype(float)      # was the dominant driver
            + 0.075 * out["hours_per_week"].astype(float)    # now dominant
            + 0.00004 * out["capital_gain"].astype(float)
        )
        probability = 1.0 / (1.0 + np.exp(-z))
        draw = rng.random(len(out))
        out[TARGET] = np.where(draw < probability, TARGET_CLASSES[1], TARGET_CLASSES[0])
        return out


class CovariateShift(Scenario):
    def __init__(self):
        super().__init__(
            "covariate_shift",
            "Education distribution moved up by 2 levels and workclass reweighted "
            "towards self-employment.",
            True, "covariate",
        )

    def apply(self, frame, rng):
        out = frame.copy()
        out = _shift_numeric(out, "education_num", 2, 1, 16)
        selfemp = out["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"])
        weights = np.where(selfemp, 6.0, 1.0)
        weights = weights / weights.sum()
        picks = rng.choice(len(out), size=len(out), replace=True, p=weights)
        return out.iloc[picks].reset_index(drop=True)


class PriorShift(Scenario):
    def __init__(self):
        super().__init__(
            "prior_shift",
            "Class balance changed by over-sampling the positive class to ~50%, "
            "with the per-class feature distributions unchanged.",
            True, "prior",
        )

    def apply(self, frame, rng):
        out = frame.copy()
        if TARGET not in out.columns:
            return out
        positives = out[out[TARGET] == TARGET_CLASSES[1]]
        negatives = out[out[TARGET] == TARGET_CLASSES[0]]
        if positives.empty or negatives.empty:
            return out
        half = len(out) // 2
        pos_idx = rng.choice(len(positives), size=half, replace=True)
        neg_idx = rng.choice(len(negatives), size=len(out) - half, replace=True)
        combined = pd.concat(
            [positives.iloc[pos_idx], negatives.iloc[neg_idx]], ignore_index=True
        )
        return combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        NoDrift(), SmallDrift(), LargeDrift(), MissingFeatureDrift(),
        PerformanceDrift(), CovariateShift(), PriorShift(),
    ]
}

#: The five the brief requires, in order. The other two are supporting cases.
REQUIRED_SCENARIOS = ["no_drift", "small_drift", "large_drift", "missing_feature", "performance_drift"]


def build_scenario(name: str, base: pd.DataFrame, *, seed: int, n_rows: int | None = None) -> pd.DataFrame:
    """Materialise one scenario window from ``base``.

    ``seed`` fully determines both the row sample and any randomness in the
    transformation, so the same call always returns the same frame.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    sample = base
    if n_rows is not None and n_rows < len(base):
        idx = rng.choice(len(base), size=n_rows, replace=False)
        sample = base.iloc[idx].reset_index(drop=True)
    return SCENARIOS[name].apply(sample, rng)


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": s.name, "kind": s.kind, "expect_drift": s.expect_drift,
             "description": s.description}
            for s in SCENARIOS.values()
        ]
    )
