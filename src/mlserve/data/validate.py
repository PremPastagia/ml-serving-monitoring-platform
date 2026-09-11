"""Schema, quality and leakage validation for training and serving frames.

Design note: this is a hand-written validator rather than Great Expectations or
pandera. The reasons are (1) it has to run identically inside the FastAPI request
path and in CI, so a heavyweight dependency with its own execution engine is a poor
fit; (2) every rule here is derived from `mlserve.data.schema`, so a contract change
cannot leave the validator behind; and (3) every check is small enough to explain
and unit-test individually, which is the property this project is being judged on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlserve.data.schema import (
    BY_NAME,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    MISSING_CATEGORY,
    NUMERIC_FEATURES,
    TARGET,
    TARGET_CLASSES,
)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, check: str, severity: Severity, message: str, **detail) -> None:
        self.findings.append(Finding(check, severity, message, detail))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def failed_checks(self) -> set[str]:
        return {f.check for f in self.errors}

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
            "stats": self.stats,
        }

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "validation passed with no findings"
        lines = [f"validation {'passed' if self.ok else 'FAILED'}: "
                 f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for f in self.findings:
            lines.append(f"  [{f.severity.value}] {f.check}: {f.message}")
        return "\n".join(lines)


class DataValidationError(RuntimeError):
    """Raised by :func:`validate_or_raise` when a frame violates the contract."""

    def __init__(self, report: ValidationReport):
        super().__init__(report.summary())
        self.report = report


#: A feature whose single-variable ROC-AUC against the target exceeds this is almost
#: certainly a copy or a transform of the label. 0.99 is deliberately just below 1.0
#: so that a perfectly-encoded leak and a slightly-noised leak are both caught, while
#: the strongest legitimate Adult feature (marital_status / relationship, AUC ~0.75)
#: is nowhere near it. The threshold is asserted against real data in the tests.
LEAKAGE_AUC_THRESHOLD = 0.99

#: Above this fraction of exactly-repeated rows we warn: in a survey some repetition
#: is expected (measured at ~10.6% on Adult), but a sudden jump usually means the
#: extract was accidentally unioned with itself.
DUPLICATE_WARN_RATE = 0.25


def _column_checks(frame: pd.DataFrame, report: ValidationReport, *, require_target: bool) -> bool:
    """Structural checks. Returns False when the frame is too broken to check further."""
    expected = set(FEATURE_NAMES) | ({TARGET} if require_target else set())
    present = set(frame.columns)

    missing = sorted(expected - present)
    if missing:
        report.add("schema.missing_columns", Severity.ERROR,
                   f"required column(s) absent: {missing}", missing=missing)

    unexpected = sorted(present - set(FEATURE_NAMES) - {TARGET})
    if unexpected:
        # An unknown column is an error, not a warning: at serving time it usually
        # means the caller is on a different contract version, and silently ignoring
        # it is how a client ships a renamed field straight into production.
        report.add("schema.unexpected_columns", Severity.ERROR,
                   f"column(s) not in the contract: {unexpected}", unexpected=unexpected)

    return not missing


def _emptiness_check(frame: pd.DataFrame, report: ValidationReport) -> bool:
    if len(frame) == 0:
        report.add("data.empty", Severity.ERROR, "frame contains zero rows")
        return False
    return True


def _type_checks(frame: pd.DataFrame, report: ValidationReport) -> None:
    for name in NUMERIC_FEATURES:
        if name not in frame.columns:
            continue
        col = frame[name]
        if not pd.api.types.is_numeric_dtype(col):
            coerced = pd.to_numeric(col, errors="coerce")
            n_bad = int((coerced.isna() & col.notna()).sum())
            report.add("types.numeric", Severity.ERROR,
                       f"{name} has dtype {col.dtype}, expected a numeric dtype",
                       column=name, dtype=str(col.dtype), n_uncoercible=n_bad)
        elif pd.api.types.is_bool_dtype(col):
            report.add("types.numeric", Severity.ERROR,
                       f"{name} is boolean, expected a real numeric dtype", column=name)

    for name in CATEGORICAL_FEATURES:
        if name not in frame.columns:
            continue
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            report.add("types.categorical", Severity.ERROR,
                       f"{name} has dtype {col.dtype}, expected string labels",
                       column=name, dtype=str(col.dtype))


def _missing_checks(frame: pd.DataFrame, report: ValidationReport) -> None:
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            continue
        n_null = int(frame[name].isna().sum())
        if n_null:
            # Nullability in this contract is about the survey's '?' token, which
            # ingestion maps to an explicit category. A true NaN always means the
            # value was lost in transit and is always an error.
            report.add("missing.nulls", Severity.ERROR,
                       f"{name} has {n_null} null value(s); the contract has no null encoding",
                       column=name, n_null=n_null)


def _range_checks(frame: pd.DataFrame, report: ValidationReport) -> None:
    for name in NUMERIC_FEATURES:
        if name not in frame.columns or not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        spec = BY_NAME[name]
        col = frame[name].dropna()
        if col.empty:
            continue
        if spec.minimum is not None:
            below = col[col < spec.minimum]
            if len(below):
                report.add("range.below_minimum", Severity.ERROR,
                           f"{name} has {len(below)} value(s) below the contract minimum {spec.minimum}"
                           f" (observed min {col.min()})",
                           column=name, minimum=spec.minimum, observed_min=float(col.min()),
                           n_violations=int(len(below)))
        if spec.maximum is not None:
            above = col[col > spec.maximum]
            if len(above):
                report.add("range.above_maximum", Severity.ERROR,
                           f"{name} has {len(above)} value(s) above the contract maximum {spec.maximum}"
                           f" (observed max {col.max()})",
                           column=name, maximum=spec.maximum, observed_max=float(col.max()),
                           n_violations=int(len(above)))


def _category_checks(frame: pd.DataFrame, report: ValidationReport) -> None:
    for name in CATEGORICAL_FEATURES:
        if name not in frame.columns:
            continue
        spec = BY_NAME[name]
        allowed = set(spec.allowed) | ({MISSING_CATEGORY} if spec.nullable else set())
        observed = set(frame[name].dropna().astype(str).unique())
        unknown = sorted(observed - allowed)
        if unknown:
            report.add("categories.unknown_level", Severity.ERROR,
                       f"{name} contains {len(unknown)} level(s) outside the contract: {unknown[:5]}",
                       column=name, unknown=unknown[:20], n_unknown=len(unknown))


def _duplicate_check(frame: pd.DataFrame, report: ValidationReport) -> None:
    n_dup = int(frame.duplicated().sum())
    rate = n_dup / len(frame)
    report.stats["duplicate_rows"] = n_dup
    report.stats["duplicate_rate"] = round(rate, 6)
    if rate > DUPLICATE_WARN_RATE:
        report.add("duplicates.rate", Severity.WARNING,
                   f"{n_dup} duplicate row(s) = {rate:.1%} of the frame, above the "
                   f"{DUPLICATE_WARN_RATE:.0%} alert level",
                   n_duplicates=n_dup, rate=round(rate, 6))


def _target_checks(frame: pd.DataFrame, report: ValidationReport) -> None:
    if TARGET not in frame.columns:
        return
    col = frame[TARGET]
    n_null = int(col.isna().sum())
    if n_null:
        report.add("target.nulls", Severity.ERROR, f"target has {n_null} null value(s)", n_null=n_null)

    observed = set(col.dropna().astype(str).unique())
    unknown = sorted(observed - set(TARGET_CLASSES))
    if unknown:
        report.add("target.unknown_class", Severity.ERROR,
                   f"target contains value(s) outside {list(TARGET_CLASSES)}: {unknown[:5]}",
                   unknown=unknown[:20])

    present = observed & set(TARGET_CLASSES)
    if len(present) < 2:
        report.add("target.single_class", Severity.ERROR,
                   f"target has only {sorted(present)}; a classifier cannot be fitted or scored",
                   present=sorted(present))
    else:
        rate = float((col == TARGET_CLASSES[1]).mean())
        report.stats["positive_rate"] = round(rate, 6)
        if not 0.05 <= rate <= 0.60:
            report.add("target.prevalence", Severity.WARNING,
                       f"positive rate {rate:.3f} is outside the expected 0.05-0.60 band",
                       positive_rate=round(rate, 6))


def _leakage_check(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Flag any single feature that predicts the target almost perfectly.

    A column with AUC >= 0.99 on its own is, in practice, the label in disguise: a
    post-outcome field, a join artefact, or an accidentally copied target. Rank-based
    AUC is used because it needs no scaling and works for an ordinal categorical too.
    """
    if TARGET not in frame.columns:
        return
    y = (frame[TARGET].astype(str) == TARGET_CLASSES[1]).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return

    scores: dict[str, float] = {}
    for name in frame.columns:
        if name == TARGET:
            continue
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col):
            values = col.astype(float).to_numpy()
            if np.isnan(values).any() or np.all(values == values[0]):
                continue
        else:
            # Map each level to its observed positive rate; a level that separates the
            # target perfectly produces a perfect ranking and so an AUC of 1.
            rates = frame.groupby(name, observed=True)[TARGET].apply(
                lambda s: float((s.astype(str) == TARGET_CLASSES[1]).mean())
            )
            values = col.map(rates).astype(float).to_numpy()
            if np.isnan(values).any() or np.all(values == values[0]):
                continue
        auc = float(roc_auc_score(y, values))
        scores[name] = round(max(auc, 1.0 - auc), 6)

    report.stats["univariate_auc"] = scores
    leaks = {k: v for k, v in scores.items() if v >= LEAKAGE_AUC_THRESHOLD}
    if leaks:
        report.add("leakage.univariate_auc", Severity.ERROR,
                   f"feature(s) predict the target almost perfectly and are treated as leakage: "
                   f"{sorted(leaks)}",
                   threshold=LEAKAGE_AUC_THRESHOLD, offenders=leaks)


def validate_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    check_leakage: bool = True,
) -> ValidationReport:
    """Run the full contract over ``frame`` and return a report.

    ``require_target=False`` is the serving mode: the caller supplies features only.
    """
    report = ValidationReport()
    report.stats["n_rows"] = int(len(frame))
    report.stats["n_columns"] = int(frame.shape[1])

    if not _emptiness_check(frame, report):
        return report
    if not _column_checks(frame, report, require_target=require_target):
        return report

    _type_checks(frame, report)
    _missing_checks(frame, report)
    _range_checks(frame, report)
    _category_checks(frame, report)
    _duplicate_check(frame, report)
    if require_target:
        _target_checks(frame, report)
        if check_leakage and not report.errors:
            _leakage_check(frame, report)
    return report


def validate_or_raise(frame: pd.DataFrame, **kwargs) -> ValidationReport:
    report = validate_frame(frame, **kwargs)
    if not report.ok:
        raise DataValidationError(report)
    return report
