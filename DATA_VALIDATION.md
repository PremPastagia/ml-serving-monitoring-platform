# Data Validation

## Design

A hand-written validator driven by `src/mlserve/data/schema.py`, not Great Expectations
or pandera. Three reasons:

1. It runs **inside the FastAPI request path** as well as in CI, so a heavyweight engine
   with its own execution model is a poor fit.
2. Every rule derives from the contract module, so a contract change cannot leave the
   validator behind.
3. Every check is small enough to explain and unit-test individually — and 78 tests do
   exactly that.

Findings carry a severity. **Errors** stop the pipeline; **warnings** are recorded and
allow it to continue. The distinction is deliberate: a duplicated extract is suspicious
but still usable, while a null in a non-nullable column means data was lost in transit.

## Checks

| Check | Severity | Rule |
|---|---|---|
| `schema.missing_columns` | error | Every contract feature (and the target, in training mode) must be present |
| `schema.unexpected_columns` | error | A column outside the contract is rejected, not ignored |
| `types.numeric` | error | Numeric columns must have a numeric dtype; booleans are rejected; uncoercible values are counted |
| `types.categorical` | error | Categorical columns must be strings, not codes |
| `missing.nulls` | error | The contract has no null encoding; a true NaN always means data was lost |
| `range.below_minimum` / `range.above_maximum` | error | Per-feature bounds from the contract; reports the observed extreme and the violation count |
| `categories.unknown_level` | error | Levels outside the declared set, plus `__missing__` where nullable |
| `duplicates.rate` | warning | Warns above 25% repeated rows |
| `target.nulls` | error | |
| `target.unknown_class` | error | Values outside `{<=50K, >50K}` |
| `target.single_class` | error | A classifier cannot be fitted or scored on one class |
| `target.prevalence` | warning | Positive rate outside 0.05–0.60 |
| `leakage.univariate_auc` | error | Any single feature with ROC-AUC ≥ 0.99 against the target |

## Leakage detection

A column that predicts the target almost perfectly on its own is, in practice, the
label in disguise: a post-outcome field, a join artefact, or an accidentally copied
target. Rank-based AUC is used because it needs no scaling and handles an ordinal
categorical (each level mapped to its observed positive rate).

The check is **direction-agnostic** — AUC 0.00 is as much of a leak as AUC 1.00 — and
is skipped in serving mode, where there is no target.

### Why 0.99

Measured, not assumed. The strongest *legitimate* single feature in this dataset is
`relationship` at **0.779**:

| Feature | Univariate ROC-AUC |
|---|---|
| `relationship` | 0.779 |
| `marital_status` | 0.770 |
| `occupation` | 0.731 |
| `education_num` | 0.717 |
| `age` | 0.684 |
| `hours_per_week` | 0.672 |
| `sex` | 0.619 |
| `capital_gain` | 0.590 |
| `workclass` | 0.583 |
| `race` | 0.538 |
| `capital_loss` | 0.535 |
| `native_country` | 0.530 |

A threshold of 0.99 leaves a margin of 0.21 to real signal while still catching a
slightly-noised leak. `test_genuine_features_are_far_below_the_leakage_threshold`
asserts this margin against real data, so the justification cannot rot.

## Duplicates

Measured duplicate rate on the development set: **10.64%** (3,465 of 32,561 rows).

Duplicates are **kept**, and this is a considered decision. In a population survey two
different respondents can legitimately share all twelve retained attributes, so a
repeated feature vector is not a data-quality defect and is not leakage. Leakage would
be the same *record* in two splits, which the split step forbids by index. Silently
deduplicating would discard real frequency information and change the published row
counts, making the dataset version incomparable with the UCI benchmark.

What *is* done: the rate is measured and reported, and the cross-split feature-vector
overlap is measured separately (14.42% of validation rows, 11.23% of test rows), because
that is what actually bounds how optimistic a held-out score can be.

## Test coverage

78 tests in `tests/test_validation.py`, all passing.

| Corruption class | Tests | Note |
|---|---:|---|
| Missing columns | 13 | One per feature, plus the target |
| Unexpected columns | 1 | |
| Wrong data types | 15 | Every numeric as string, every categorical as number, boolean, uncoercible text |
| Nulls | 12 | One per feature |
| Out-of-range values | 9 | Above and below, per bounded feature |
| Unknown category levels | 7 | One per categorical |
| Empty data | 2 | Including a completely empty frame, which must not cascade |
| Duplicates | 2 | Rate measured; excessive duplication warns without failing |
| Target problems | 4 | Unknown class, single class, nulls, extreme prevalence |
| Leakage | 5 | Numeric leak, categorical leak, inverted leak, real-signal margin, serving-mode skip |
| **Positive controls** | 3 | Clean training frame, clean serving frame, `validate_or_raise` |
| Boundary values | 1 | Values exactly on the limit must be accepted |
| Error surface | 4 | Readable message, JSON-safe serialisation, severities |

Every negative test is paired with a positive control. Without them, a validator that
rejected *everything* would score 100% on the negative cases — which is precisely the
failure mode a validation suite exists to rule out.

## Usage

```python
from mlserve.data.validate import validate_frame, validate_or_raise

report = validate_frame(frame)            # training mode, includes the leakage check
report = validate_frame(frame, require_target=False)   # serving mode
validate_or_raise(frame)                  # raises DataValidationError on any error

report.ok            # bool
report.errors        # list[Finding]
report.warnings      # list[Finding]
report.stats         # row counts, duplicate rate, positive rate, univariate AUCs
report.summary()     # human-readable
report.to_dict()     # JSON-safe
```

Reproduce the test suite:

```bash
python -m pytest tests/test_validation.py -q
```
