# Project Scope

## Problem

Binary classification: predict whether a US census respondent's annual income exceeds
$50,000, from twelve demographic and employment attributes.

The task was chosen because it satisfies every requirement this platform needs to
demonstrate, and because it is small enough that the whole pipeline runs end to end in
under a minute on a laptop:

| Requirement | How Adult satisfies it |
|---|---|
| Publicly available | UCI ML Repository, stable URLs, pinned SHA-256 |
| Clear target | `income`, two classes, no ambiguity |
| Meaningful metric | Class-imbalanced (24.1% positive), so ROC-AUC and PR-AUC both say something |
| Drift can be simulated | Mixed numeric and categorical features allow covariate, prior, schema and concept shift to be constructed separately |
| Servable locally | 32,561 training rows, a 200-tree model, ~4 MB artefact, single-row inference in under 4 ms |

## What this project is

A locally reproducible ML platform covering the full lifecycle: data validation,
reproducible training, experiment tracking, a model registry, an HTTP serving API,
Prometheus monitoring, statistical drift detection, and a gated retraining workflow
with promotion and rollback.

## What this project is **not**

It is **not production-ready**, and no part of this repository claims otherwise. The
specific gaps are listed in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md#known-gaps) and
summarised in [CV_POINTERS_ML_SERVING_MONITORING.md](CV_POINTERS_ML_SERVING_MONITORING.md#limitations).
The largest are: no authentication anywhere, single-node SQLite storage, an
explicitly-triggered rather than scheduled retraining loop, and a Docker image that was
written but never built because no container runtime was available on the development
machine.

## Data

| Split | Source | Rows | Positive rate | Role |
|---|---|---|---|---|
| train | `adult.data`, 80% | 26,048 | 24.08% | Fits preprocessing statistics and the estimator |
| validation | `adult.data`, 20% | 6,513 | 24.07% | Model selection, retraining acceptance decisions |
| test | `adult.test` (separate file) | 16,281 | 23.62% | Held out; loaded once at the end of a run |

The test set is a physically separate file from the UCI distribution. That is a
structural guarantee rather than a procedural one: there is no code path along which a
test row can reach `fit`.

## Schema

Twelve features; five numeric, seven categorical. Two raw columns are deliberately
dropped. Full definitions, ranges and allowed levels are in
[DATASET_CARD.md](DATASET_CARD.md); the machine-readable contract is
`src/mlserve/data/schema.py`, which the validator, the feature pipeline, the API
request model and the drift detector all read.

| Kind | Features |
|---|---|
| Numeric | `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| Categorical | `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |
| Dropped | `fnlwgt` (census sampling weight, not a property of the person and unavailable at serving time), `education` (exact duplicate of `education_num`) |
| Target | `income` ∈ {`<=50K`, `>50K`}, positive class `>50K` |

## Evaluation metric

**Primary: ROC-AUC.** The serving contract returns a probability, so a threshold-free
ranking metric is the honest headline. Reported alongside: PR-AUC (the metric that
actually degrades when the positive class becomes rarer), accuracy, precision, recall,
F1, Brier score and log loss. PR-AUC and Brier are what catch a model that ranks well
but is badly calibrated.

## Expected serving input

```json
{"records": [{"age": 39, "workclass": "State-gov", "education_num": 13,
              "marital_status": "Never-married", "occupation": "Adm-clerical",
              "relationship": "Not-in-family", "race": "White", "sex": "Male",
              "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
              "native_country": "United-States"}]}
```

Every field is required; unknown fields are rejected rather than ignored. The request
model is generated from the data contract, so the API cannot accept something training
would have rejected. See [API.md](API.md).

## Failure conditions

| Condition | Defined behaviour | Verified by |
|---|---|---|
| Raw file altered | `ChecksumMismatch`, pipeline refuses to run | `tests/test_failure_injection.py` |
| Training data violates the contract | `DataValidationError`, no model produced | `tests/test_training.py` |
| A feature predicts the target almost perfectly | Flagged as leakage, training aborts | `tests/test_validation.py` |
| Malformed request | 422 with a structured error, no prediction | `tests/test_api_errors.py` |
| Batch above the configured maximum | 413 | `tests/test_api_errors.py` |
| No model loaded | 503 on `/predict` and `/ready`; `/health` stays 200 and reports `degraded` | `tests/test_api_errors.py` |
| Model raises during inference | 500 with no traceback leaked to the caller | `tests/test_failure_injection.py` |
| Model reload fails | Previous model keeps serving | `tests/test_failure_injection.py` |
| Candidate model is worse, slower, or trained on dirty data | Promotion refused, incumbent keeps serving | `tests/test_retraining.py` |

## Success criteria for the project itself

1. Every number in the documentation traces to a script in this repository.
2. Repeated training with a fixed seed produces a byte-identical model fingerprint.
3. Drift detection has a measured false-positive rate, not just detection anecdotes.
4. Rollback is demonstrated through the running HTTP service, not only in the registry.
5. Claims the evidence does not support are absent, and a script enforces that.
