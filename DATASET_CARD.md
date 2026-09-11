# Dataset Card — UCI Adult (Census Income)

## Provenance

| | |
|---|---|
| Source | UCI Machine Learning Repository, "Adult" (a.k.a. Census Income) |
| Origin | 1994 US Census Bureau database, extracted by Barry Becker |
| Licence | CC BY 4.0 (UCI ML Repository) |
| Retrieved | 2026-09-11 |

| File | SHA-256 | Bytes | Records | Role |
|---|---|---|---|---|
| `adult.data` | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` | 3,974,305 | 32,561 | train + validation |
| `adult.test` | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` | 2,003,153 | 16,281 | held-out test |

Checksums are pinned in `src/mlserve/data/ingest.py` and verified on every load.
`python scripts/fetch_data.py --verify` checks them without touching the network.

## Dataset version

`adult-ingest-2-975c90344d56`

The id is a SHA-256 over the raw file digests **and** the parsing rules, truncated to
12 hex characters. Changing either the bytes or the cleaning logic produces a new id,
so a recorded metric can always be traced to the exact data that produced it. Every
MLflow run, registry version and drift baseline carries it.

## Cleaning rules

Applied in `mlserve.data.ingest._clean`, deterministic and order-preserving:

1. Strip surrounding whitespace from every field (the raw CSV is comma-space separated).
2. Strip the trailing `.` from labels in `adult.test` (`<=50K.` → `<=50K`).
3. Replace the survey's `?` non-response marker with the explicit category
   `__missing__` on `workclass`, `occupation` and `native_country`.
4. Coerce the six numeric columns to integers.
5. Drop `fnlwgt` and `education`.
6. Drop the trailing blank line each file ends with.

**Duplicates are kept.** In a population survey two different respondents can
legitimately share every retained attribute, so a repeated feature vector is not a
defect. The measured duplicate rate is **10.64%** on the development set. Leakage would
be the same *record* appearing in two splits, which the split step forbids by index and
which is measured directly (see below).

## Features

### Numeric

| Feature | Range | Description | Notes |
|---|---|---|---|
| `age` | 17–90 | Age in years | |
| `education_num` | 1–16 | Education level as an ordinal rank (1 = Preschool, 16 = Doctorate) | Replaces the redundant `education` string |
| `capital_gain` | 0–99,999 | Annual capital gains, USD | ~92% zero; top-coded at 99999 |
| `capital_loss` | 0–4,356 | Annual capital losses, USD | Heavily zero-inflated |
| `hours_per_week` | 1–99 | Usual hours worked per week | **46.7% of records are exactly 40** — this tie matters for drift binning, see [DRIFT_DETECTION.md](DRIFT_DETECTION.md) |

### Categorical

| Feature | Levels | Nullable | Notes |
|---|---|---|---|
| `workclass` | 8 | yes (`__missing__`) | 1,632 non-responses in development |
| `marital_status` | 7 | no | |
| `occupation` | 14 | yes (`__missing__`) | 1,639 non-responses |
| `relationship` | 6 | no | Household role |
| `race` | 5 | no | Self-identified |
| `sex` | 2 | no | As recorded by the 1994 census; binary in the source data |
| `native_country` | 41 | yes (`__missing__`) | 580 non-responses; long tail pooled for chi-square |

### Engineered features

Added inside the model pipeline, so they compute identically in training and serving:

| Feature | Definition | Why |
|---|---|---|
| `net_capital` | `capital_gain - capital_loss` | The raw pair states "net position" awkwardly; a tree needs several splits to express it |
| `has_capital_flow` | `1` if either is non-zero | Separates the zero-inflation mass from the magnitude |
| `log_capital_gain` | `log1p(capital_gain)` | Stops the 99999 top-code dominating histogram binning |
| `hours_band` | part_time ≤34, full_time 35–44, overtime ≥45 | Makes the "unusual hours" contrast explicit around the 40-hour mode |

Sixteen features reach the estimator: 8 numeric + 8 categorical.

## Dropped columns and why

**`fnlwgt`** — the Census Bureau's inverse-probability *sampling weight*: how many
people in the population the row stands for. It is a property of the survey design, not
of the person. It is therefore (a) unavailable at serving time for a new individual and
(b) an invitation for the model to learn the sampling frame instead of the income
relationship. Dropping it is the single most consequential schema decision in this
project.

**`education`** — the exact string form of `education_num`. Keeping both duplicates one
concept, inflates its weight in the one-hot block, and doubles its contribution to
aggregate drift scores.

## Target

| | |
|---|---|
| Column | `income` |
| Classes | `<=50K` (negative, 0), `>50K` (positive, 1) |
| Development positive rate | 24.08% |
| Test positive rate | 23.62% |

## Splits

| Split | Rows | Positive rate | How |
|---|---|---|---|
| train | 26,048 | 24.08% | 80% of `adult.data`, stratified, seed 42 |
| validation | 6,513 | 24.07% | 20% of `adult.data`, stratified, seed 42 |
| test | 16,281 | 23.62% | All of `adult.test`, untouched |

Split id `bc50e85d5b57e879` — a hash of the actual partition contents, so a silent
repartition is detectable.

### Measured contamination

| Measure | Value | Interpretation |
|---|---|---|
| Same source record in train and validation | **0** | The guarantee that matters; must be zero |
| Index coverage of the development frame | 32,561 of 32,561 | No rows silently dropped |
| Validation rows whose feature vector also occurs in train | 939 (**14.42%**) | Repeated respondent profiles, expected in a survey |
| Test rows whose feature vector also occurs in train | 1,829 (**11.23%**) | Bounds how optimistic the held-out score can be |

The last two are reported rather than eliminated. They are not leakage — they are two
different people with the same twelve attributes — but they do bound what the held-out
score can prove, so the number is stated rather than hidden.

## Known biases and ethical notes

The data is a 1994 US census extract. It encodes the income distribution, occupational
segregation and demographic composition of that time and place, including strong
associations between the income label and `sex`, `race` and `marital_status`. Measured
single-feature ROC-AUC against the target: `relationship` 0.779, `marital_status`
0.770, `occupation` 0.731, `education_num` 0.717, `sex` 0.619, `race` 0.538.

A model fitted here will reproduce those associations. **This dataset is used as a
convenient, well-understood benchmark for MLOps machinery. It is not a basis for any
decision about a real person**, and no fairness mitigation is implemented or claimed.
A deployment making decisions about people would require a fairness assessment,
subgroup performance reporting and a review of whether `sex` and `race` belong in the
feature set at all — none of which is in scope here.

## Reproducing this dataset

```bash
python scripts/fetch_data.py          # download and verify against the pinned digests
python scripts/fetch_data.py --verify # verify only, no network
```
