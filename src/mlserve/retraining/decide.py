"""Acceptance criteria for replacing the production model.

The decision is a pure function of measured numbers, separated from the orchestration
that produces them, so it can be unit-tested against every branch -- better candidate,
worse candidate, too-slow candidate, dirty data -- without training anything.

Every criterion exists because of a specific way an automated retraining loop goes
wrong:

``min_absolute_improvement``  Retraining on a new window almost always moves the score
                             a little. Promoting on any improvement means promoting on
                             noise, and the registry fills with churn. The candidate
                             must beat the incumbent by a margin.
``max_allowed_degradation``   An explicit ceiling on how much worse a candidate may be
                             and still be promoted. Set to 0.0 here, i.e. never; it is
                             a separate knob because a team that must ship a model
                             trained on fresher data may accept a small, bounded loss.
``min_candidate_roc_auc``     An absolute floor. If the incumbent has already decayed
                             badly, "better than the incumbent" is a very low bar, and
                             without a floor the loop would ratchet downwards.
``max_latency_ratio``         A model that is accurate and far slower than the one it
                             replaces is not deployable. This is expressed as a *ratio*
                             against the incumbent measured back to back in the same
                             process, not as an absolute millisecond budget: an absolute
                             threshold is not portable between machines and is not even
                             stable on one machine, because a busy host inflates every
                             measurement. During development on a loaded laptop an
                             absolute 50ms budget rejected a candidate that was 2.7%
                             *better* than the incumbent and no slower -- the host was
                             simply oversubscribed. A ratio cancels that out, because
                             both models pay the same scheduling tax.
``max_p95_latency_ms``        An absolute ceiling, applied only when there is no
                             incumbent to compare against. With no baseline a ratio is
                             undefined, and some bound is better than none.
``require_clean_validation``  Never train on data that failed validation. This is the
                             criterion that stops a corrupt upstream feed from being
                             laundered into a promoted model.
``min_training_rows``         A drift window can be small; fitting on it and promoting
                             is how a loop destroys a good model with 200 rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):  # noqa: UP042 - str+Enum keeps .value JSON-serialisable on 3.11
    PROMOTE = "promote"
    REJECT = "reject"


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_absolute_improvement: float = 0.002
    max_allowed_degradation: float = 0.0
    min_candidate_roc_auc: float = 0.85
    max_latency_ratio: float = 1.5
    max_p95_latency_ms: float = 50.0
    require_clean_validation: bool = True
    min_training_rows: int = 5000

    @classmethod
    def from_config(cls, config) -> AcceptanceCriteria:
        return cls(
            min_absolute_improvement=float(config.require("retraining.min_absolute_improvement")),
            max_allowed_degradation=float(config.require("retraining.max_allowed_degradation")),
            min_candidate_roc_auc=float(config.require("retraining.min_candidate_roc_auc")),
            max_latency_ratio=float(config.require("retraining.max_latency_ratio")),
            max_p95_latency_ms=float(config.require("retraining.max_p95_latency_ms")),
            require_clean_validation=bool(config.require("retraining.require_clean_validation")),
            min_training_rows=int(config.require("retraining.min_training_rows")),
        )

    def to_dict(self) -> dict:
        return {
            "min_absolute_improvement": self.min_absolute_improvement,
            "max_allowed_degradation": self.max_allowed_degradation,
            "min_candidate_roc_auc": self.min_candidate_roc_auc,
            "max_latency_ratio": self.max_latency_ratio,
            "max_p95_latency_ms": self.max_p95_latency_ms,
            "require_clean_validation": self.require_clean_validation,
            "min_training_rows": self.min_training_rows,
        }


@dataclass
class CandidateEvidence:
    """Everything measured about a candidate, gathered before any decision is made."""

    candidate_metric: float
    incumbent_metric: float | None
    metric_name: str = "roc_auc"
    candidate_p95_latency_ms: float | None = None
    #: The incumbent's p95 measured in the same process, immediately after the
    #: candidate's. Both therefore pay the same host contention, which is what makes
    #: the ratio meaningful where an absolute figure is not.
    incumbent_p95_latency_ms: float | None = None
    #: Medians from the same interleaved run. The ratio gate uses these, because on a
    #: shared host the tail is dominated by scheduler outliers belonging to neither
    #: model. The p95 figures above are reported but not gated on.
    candidate_p50_latency_ms: float | None = None
    incumbent_p50_latency_ms: float | None = None
    training_rows: int = 0
    validation_errors: int = 0
    training_seconds: float = 0.0

    @property
    def delta(self) -> float | None:
        if self.incumbent_metric is None:
            return None
        return self.candidate_metric - self.incumbent_metric

    @property
    def latency_ratio(self) -> float | None:
        """Candidate median latency as a multiple of the incumbent's.

        Falls back to the p95 pair when medians are unavailable, so evidence built by
        older callers still produces a comparison rather than silently skipping the
        latency gate entirely.
        """
        if self.candidate_p50_latency_ms and self.incumbent_p50_latency_ms:
            return self.candidate_p50_latency_ms / self.incumbent_p50_latency_ms
        if self.candidate_p95_latency_ms and self.incumbent_p95_latency_ms:
            return self.candidate_p95_latency_ms / self.incumbent_p95_latency_ms
        return None

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "candidate_metric": round(self.candidate_metric, 6),
            "incumbent_metric": round(self.incumbent_metric, 6) if self.incumbent_metric is not None else None,
            "delta": round(self.delta, 6) if self.delta is not None else None,
            "candidate_p95_latency_ms": self.candidate_p95_latency_ms,
            "incumbent_p95_latency_ms": self.incumbent_p95_latency_ms,
            "candidate_p50_latency_ms": self.candidate_p50_latency_ms,
            "incumbent_p50_latency_ms": self.incumbent_p50_latency_ms,
            "latency_ratio": round(self.latency_ratio, 4) if self.latency_ratio else None,
            "training_rows": self.training_rows,
            "validation_errors": self.validation_errors,
            "training_seconds": round(self.training_seconds, 4),
        }


@dataclass
class PromotionDecision:
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)

    @property
    def promote(self) -> bool:
        return self.decision is Decision.PROMOTE

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "failed_criteria": self.failed_criteria,
            "evidence": self.evidence,
            "criteria": self.criteria,
        }


def decide(evidence: CandidateEvidence, criteria: AcceptanceCriteria) -> PromotionDecision:
    """Apply every criterion and return the decision with its full reasoning.

    All criteria are evaluated even after the first failure, so a rejection report
    names every problem rather than only the first one found.
    """
    failed: list[str] = []
    reasons: list[str] = []

    if criteria.require_clean_validation and evidence.validation_errors > 0:
        failed.append("require_clean_validation")
        reasons.append(
            f"training data failed validation with {evidence.validation_errors} error(s)"
        )

    if evidence.training_rows < criteria.min_training_rows:
        failed.append("min_training_rows")
        reasons.append(
            f"only {evidence.training_rows} training rows, below the minimum "
            f"{criteria.min_training_rows}"
        )

    if evidence.candidate_metric < criteria.min_candidate_roc_auc:
        failed.append("min_candidate_roc_auc")
        reasons.append(
            f"candidate {evidence.metric_name} {evidence.candidate_metric:.6f} is below the "
            f"absolute floor {criteria.min_candidate_roc_auc}"
        )

    ratio = evidence.latency_ratio
    if ratio is not None:
        # Relative check: both models were timed back to back on the same host, so any
        # contention is common to both and cancels.
        if ratio > criteria.max_latency_ratio:
            failed.append("max_latency_ratio")
            reasons.append(
                f"candidate median latency is {ratio:.2f}x the incumbent's "
                f"(p95 {evidence.candidate_p95_latency_ms:.3f}ms vs "
                f"{evidence.incumbent_p95_latency_ms:.3f}ms), above the permitted "
                f"{criteria.max_latency_ratio}x"
            )
        else:
            reasons.append(
                f"candidate median latency is {ratio:.2f}x the incumbent's, within the "
                f"permitted {criteria.max_latency_ratio}x"
            )
    elif (
        evidence.candidate_p95_latency_ms is not None
        and evidence.candidate_p95_latency_ms > criteria.max_p95_latency_ms
    ):
        # No incumbent to compare against, so fall back to the absolute ceiling.
        failed.append("max_p95_latency_ms")
        reasons.append(
            f"no incumbent to compare latency against, and candidate p95 "
            f"{evidence.candidate_p95_latency_ms:.3f}ms exceeds the absolute budget "
            f"{criteria.max_p95_latency_ms}ms"
        )

    delta = evidence.delta
    if delta is None:
        # No incumbent: this is the first model, so there is nothing to improve on and
        # the absolute floor is the only accuracy bar that applies.
        reasons.append("no incumbent model; the absolute floor is the only accuracy bar")
    elif delta < -criteria.max_allowed_degradation:
        failed.append("max_allowed_degradation")
        reasons.append(
            f"candidate is {abs(delta):.6f} worse than the incumbent, beyond the "
            f"permitted degradation of {criteria.max_allowed_degradation}"
        )
    elif delta < criteria.min_absolute_improvement:
        failed.append("min_absolute_improvement")
        reasons.append(
            f"candidate improves {evidence.metric_name} by only {delta:+.6f}, below the "
            f"required margin of {criteria.min_absolute_improvement}"
        )
    else:
        reasons.append(
            f"candidate improves {evidence.metric_name} by {delta:+.6f}, at or above the "
            f"required margin of {criteria.min_absolute_improvement}"
        )

    decision = Decision.REJECT if failed else Decision.PROMOTE
    return PromotionDecision(
        decision=decision,
        reasons=reasons,
        failed_criteria=failed,
        evidence=evidence.to_dict(),
        criteria=criteria.to_dict(),
    )
