"""Data validation: one test per corruption class, plus the positive control.

Every negative test here is paired with :func:`test_clean_frame_passes`, which is the
positive control. Without it, a validator that rejected *everything* would score 100%
on the negative cases -- which is precisely the failure mode a validation suite is
supposed to rule out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlserve.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    MISSING_CATEGORY,
    NUMERIC_FEATURES,
    TARGET,
    TARGET_CLASSES,
)
from mlserve.data.validate import (
    DUPLICATE_WARN_RATE,
    LEAKAGE_AUC_THRESHOLD,
    DataValidationError,
    Severity,
    validate_frame,
    validate_or_raise,
)

# ------------------------------------------------------------------ positive control


def test_clean_frame_passes(clean_frame):
    report = validate_frame(clean_frame)
    assert report.ok, report.summary()
    assert report.errors == []


def test_clean_features_only_frame_passes_in_serving_mode(feature_frame):
    report = validate_frame(feature_frame, require_target=False)
    assert report.ok, report.summary()


def test_validate_or_raise_returns_report_for_clean_data(clean_frame):
    assert validate_or_raise(clean_frame).ok


# ----------------------------------------------------------------- missing columns


@pytest.mark.parametrize("column", FEATURE_NAMES)
def test_any_missing_feature_column_is_rejected(clean_frame, column):
    report = validate_frame(clean_frame.drop(columns=[column]))
    assert not report.ok
    assert "schema.missing_columns" in report.failed_checks
    assert column in report.errors[0].detail["missing"]


def test_missing_target_is_rejected_in_training_mode(clean_frame):
    report = validate_frame(clean_frame.drop(columns=[TARGET]))
    assert not report.ok
    assert "schema.missing_columns" in report.failed_checks


def test_unexpected_column_is_rejected(clean_frame):
    frame = clean_frame.copy()
    frame["leaked_customer_id"] = range(len(frame))
    report = validate_frame(frame)
    assert not report.ok
    assert "schema.unexpected_columns" in report.failed_checks


# ---------------------------------------------------------------------- data types


@pytest.mark.parametrize("column", NUMERIC_FEATURES)
def test_numeric_column_of_strings_is_rejected(clean_frame, column):
    frame = clean_frame.copy()
    frame[column] = frame[column].astype(str)
    report = validate_frame(frame)
    assert not report.ok
    assert "types.numeric" in report.failed_checks


def test_numeric_column_with_uncoercible_text_reports_the_count(clean_frame):
    frame = clean_frame.copy()
    frame["age"] = frame["age"].astype(object)
    frame.loc[frame.index[:7], "age"] = "not-a-number"
    frame["age"] = frame["age"].astype(str)
    report = validate_frame(frame)
    assert "types.numeric" in report.failed_checks
    finding = next(f for f in report.errors if f.check == "types.numeric")
    assert finding.detail["n_uncoercible"] == 7


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_categorical_column_of_numbers_is_rejected(clean_frame, column):
    frame = clean_frame.copy()
    frame[column] = np.arange(len(frame))
    report = validate_frame(frame)
    assert not report.ok
    assert "types.categorical" in report.failed_checks


def test_boolean_masquerading_as_numeric_is_rejected(clean_frame):
    frame = clean_frame.copy()
    frame["age"] = frame["age"] > 40
    report = validate_frame(frame)
    assert "types.numeric" in report.failed_checks


# -------------------------------------------------------------------------- ranges


@pytest.mark.parametrize("column,bad_value,check", [
    ("age", 5, "range.below_minimum"),
    ("age", 200, "range.above_maximum"),
    ("education_num", 0, "range.below_minimum"),
    ("education_num", 99, "range.above_maximum"),
    ("hours_per_week", 0, "range.below_minimum"),
    ("hours_per_week", 500, "range.above_maximum"),
    ("capital_gain", -1, "range.below_minimum"),
    ("capital_loss", -50, "range.below_minimum"),
])
def test_out_of_range_values_are_rejected(clean_frame, column, bad_value, check):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], column] = bad_value
    report = validate_frame(frame)
    assert not report.ok, f"{column}={bad_value} was accepted"
    assert check in report.failed_checks


def test_range_finding_reports_the_observed_extreme(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[:3], "age"] = 150
    report = validate_frame(frame)
    finding = next(f for f in report.errors if f.check == "range.above_maximum")
    assert finding.detail["observed_max"] == 150.0
    assert finding.detail["n_violations"] == 3


def test_boundary_values_are_accepted(clean_frame):
    """A value exactly on the limit is valid; an off-by-one check would reject it."""
    frame = clean_frame.copy()
    frame.loc[frame.index[0], "age"] = 17
    frame.loc[frame.index[1], "age"] = 90
    frame.loc[frame.index[2], "education_num"] = 1
    frame.loc[frame.index[3], "education_num"] = 16
    assert validate_frame(frame).ok


# ------------------------------------------------------------------ missing values


@pytest.mark.parametrize("column", FEATURE_NAMES)
def test_null_values_are_rejected(clean_frame, column):
    frame = clean_frame.copy()
    frame[column] = frame[column].astype(object)
    frame.loc[frame.index[0], column] = None
    report = validate_frame(frame)
    assert not report.ok
    assert "missing.nulls" in report.failed_checks


def test_survey_missing_marker_is_accepted_on_nullable_columns(clean_frame):
    """'?' becomes an explicit category at ingest; it must remain valid."""
    frame = clean_frame.copy()
    frame.loc[frame.index[:10], "occupation"] = MISSING_CATEGORY
    frame.loc[frame.index[:10], "workclass"] = MISSING_CATEGORY
    assert validate_frame(frame).ok


def test_missing_marker_is_rejected_on_a_non_nullable_column(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], "sex"] = MISSING_CATEGORY
    report = validate_frame(frame)
    assert not report.ok
    assert "categories.unknown_level" in report.failed_checks


# ---------------------------------------------------------------------- categories


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_unknown_category_level_is_rejected(clean_frame, column):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], column] = "Totally-New-Level"
    report = validate_frame(frame)
    assert not report.ok
    assert "categories.unknown_level" in report.failed_checks
    finding = next(f for f in report.errors if f.check == "categories.unknown_level")
    assert "Totally-New-Level" in finding.detail["unknown"]


# ----------------------------------------------------------------------- emptiness


def test_empty_frame_is_rejected():
    report = validate_frame(pd.DataFrame(columns=FEATURE_NAMES + [TARGET]))
    assert not report.ok
    assert "data.empty" in report.failed_checks


def test_empty_frame_short_circuits_without_crashing():
    """An empty frame must not produce a cascade of downstream exceptions."""
    report = validate_frame(pd.DataFrame())
    assert not report.ok
    assert len(report.errors) == 1


# ---------------------------------------------------------------------- duplicates


def test_duplicate_rate_is_always_measured(clean_frame):
    report = validate_frame(clean_frame)
    assert "duplicate_rate" in report.stats
    assert 0.0 <= report.stats["duplicate_rate"] <= 1.0


def test_excessive_duplication_warns_without_failing(clean_frame):
    """A doubled extract is suspicious, not invalid: it warns and stays usable."""
    doubled = pd.concat([clean_frame, clean_frame], ignore_index=True)
    report = validate_frame(doubled)
    assert report.ok, "duplication must not be a hard error"
    assert "duplicates.rate" in {f.check for f in report.warnings}
    assert report.stats["duplicate_rate"] > DUPLICATE_WARN_RATE


# -------------------------------------------------------------------------- target


def test_unknown_target_class_is_rejected(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], TARGET] = "MAYBE"
    report = validate_frame(frame)
    assert not report.ok
    assert "target.unknown_class" in report.failed_checks


def test_single_class_target_is_rejected(clean_frame):
    frame = clean_frame.copy()
    frame[TARGET] = TARGET_CLASSES[0]
    report = validate_frame(frame)
    assert not report.ok
    assert "target.single_class" in report.failed_checks


def test_null_target_is_rejected(clean_frame):
    frame = clean_frame.copy()
    frame[TARGET] = frame[TARGET].astype(object)
    frame.loc[frame.index[0], TARGET] = None
    report = validate_frame(frame)
    assert not report.ok
    assert "target.nulls" in report.failed_checks


def test_extreme_prevalence_warns(clean_frame):
    frame = clean_frame.copy()
    positives = frame[frame[TARGET] == TARGET_CLASSES[1]]
    negatives = frame[frame[TARGET] == TARGET_CLASSES[0]]
    skewed = pd.concat([positives.head(5), negatives], ignore_index=True)
    report = validate_frame(skewed)
    assert "target.prevalence" in {f.check for f in report.warnings}


# ------------------------------------------------------------------------ leakage


def test_perfect_leak_is_detected(clean_frame):
    """A numeric copy of the label must be caught."""
    frame = clean_frame.copy()
    frame["capital_gain"] = (frame[TARGET] == TARGET_CLASSES[1]).astype(int) * 1000
    report = validate_frame(frame)
    assert not report.ok
    assert "leakage.univariate_auc" in report.failed_checks
    finding = next(f for f in report.errors if f.check == "leakage.univariate_auc")
    assert "capital_gain" in finding.detail["offenders"]


def test_categorical_leak_is_detected(clean_frame):
    """A categorical whose levels separate the target perfectly is also a leak."""
    frame = clean_frame.copy()
    frame["occupation"] = np.where(
        frame[TARGET] == TARGET_CLASSES[1], "Exec-managerial", "Other-service"
    )
    report = validate_frame(frame)
    assert "leakage.univariate_auc" in report.failed_checks


def test_inverted_leak_is_detected(clean_frame):
    """AUC 0.0 is as much of a leak as AUC 1.0; the check must be direction-agnostic."""
    frame = clean_frame.copy()
    frame["capital_loss"] = (frame[TARGET] == TARGET_CLASSES[0]).astype(int) * 500
    report = validate_frame(frame)
    assert "leakage.univariate_auc" in report.failed_checks


def test_genuine_features_are_far_below_the_leakage_threshold(clean_frame):
    """The threshold must not be so tight that real signal trips it.

    This is what justifies LEAKAGE_AUC_THRESHOLD empirically rather than by assertion:
    the strongest honest predictor in this dataset is measured, and the margin to the
    threshold is checked.
    """
    report = validate_frame(clean_frame)
    aucs = report.stats["univariate_auc"]
    strongest = max(aucs.values())
    assert strongest < LEAKAGE_AUC_THRESHOLD
    assert strongest < 0.90, f"unexpectedly strong single feature: {aucs}"


def test_leakage_check_is_skipped_in_serving_mode(feature_frame):
    report = validate_frame(feature_frame, require_target=False)
    assert "univariate_auc" not in report.stats


# ------------------------------------------------------------------- error surface


def test_validate_or_raise_raises_with_a_readable_report(clean_frame):
    frame = clean_frame.drop(columns=["age"])
    with pytest.raises(DataValidationError) as excinfo:
        validate_or_raise(frame)
    assert "schema.missing_columns" in str(excinfo.value)
    assert excinfo.value.report.errors


def test_report_serialises_to_json_safe_types(clean_frame):
    import json

    frame = clean_frame.copy()
    frame.loc[frame.index[0], "age"] = 999
    payload = validate_frame(frame).to_dict()
    json.dumps(payload)  # must not raise
    assert payload["ok"] is False
    assert payload["n_errors"] >= 1


def test_all_findings_carry_a_severity(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], "age"] = 999
    for finding in validate_frame(frame).findings:
        assert finding.severity in (Severity.ERROR, Severity.WARNING)
        assert finding.message.strip()
