# Drift Detection

## Choosing a test per feature type

No single test is correct for both a continuous column and a 41-level categorical, so
this module applies a different primary test to each and reports them in one frame.

| Feature type | Primary test | Why this one |
|---|---|---|
| Numeric | **Two-sample Kolmogorov–Smirnov** | Distribution-free — no normality assumption, which matters because `capital_gain` is ~92% zeros with a long tail. Needs no binning choice. Sensitive to any difference in the CDF, not only a shift in the mean. |
| Categorical | **Chi-square test of homogeneity** | The natural test for comparing two multinomials. Levels with expected count below 5 are pooled into `__rare__` first, because the approximation is unreliable below that and `native_country` has a long tail of single-digit counts. |
| Both | **Population Stability Index** | The decision variable. Answers what a p-value does not: *how big* is the change. Independent of sample size, with conventional operating points (0.1 warn / 0.25 alert) from credit risk. |
| Numeric | **Wasserstein distance** | Reported as an interpretable magnitude in the feature's own units ("the mean age moved 4.2 years"), which is what makes a report actionable rather than merely true. |

**Both conditions are required to alert** — PSI ≥ 0.25 **and** p < 0.01. Not either.
At these sample sizes the p-value alone rejects on differences too small to act on, and
PSI alone has no notion of sampling noise. Requiring both is what produces the measured
0% false-positive rate below.

Schema failures — a missing column, or nulls where the contract permits none — are
reported **separately** and are not run through the statistical path. A feature that
has disappeared is not a distribution shift, and running KS against an absent column
would either crash or, worse, silently return "no drift".

### Why not Evidently

Evidently would provide all of this. It was rejected because it selects the test for
you based on cardinality and row count, so the most important question about this
project — "why this test?" — would have no answer; because its per-feature defaults
change between versions, which undermines the reproducibility claim; and because it
adds a large dependency to the container for roughly 200 lines of work. The cost is
that this implementation covers fewer test types, which is a real limitation.

## A defect found and fixed: PSI on tied features

This is the most instructive result in the project.

`hours_per_week` has **46.7% of all records at exactly 40**. With quantile binning,
five of the eleven decile edges are therefore the value 40, `np.unique` collapses them,
and one bin ends up holding nearly half the distribution.

The consequence was that the *same* +2-hour shift scored:

| Reference sample | PSI for a uniform +2 shift |
|---|---|
| Full development set (32,561 rows) | **0.0021** — reads as no drift at all |
| A 4,500-row subsample of the same data | **2.2587** — reads as extreme drift |

The two references happened to place a bin edge differently: the full sample put the
boundary at 48 (hiding the move entirely), the subsample put one at 42 (splitting the
mode). A detector whose verdict depends on that is not usable.

**Fix.** A numeric column whose quantile edges collapse is, in fact, a discrete
variable wearing a numeric dtype. `_is_effectively_discrete` detects that condition and
routes the feature to **value-based bins** with rare-value pooling instead of quantile
bins. After the fix the same comparison gives 5.25 and 4.53 — stable, and correctly
reading as a large change, since half the population really did move.

**Residual limitation, documented rather than hidden.** Value-based PSI on a tied
column *saturates*: a +2 and a +12 shift both score above 4, so PSI can no longer order
them by magnitude. That is why the Wasserstein distance is reported alongside — it
correctly gives 2 and 12. Both behaviours are pinned by tests
(`test_psi_on_a_tied_feature_is_stable_across_reference_subsamples`,
`test_discrete_psi_saturates_so_wasserstein_carries_the_magnitude`), so this document
cannot silently go stale.

## Output drift: two conditions

Labels are effectively never available at serving time here — income is observed months
later, if ever — so the model's own output distribution is the main live signal.

| Condition | What it catches |
|---|---|
| **Shape** — PSI over the score histogram plus KS | A change in the *form* of the score distribution |
| **Level** — relative change in the mean score ≥ 25% | A change in the *base rate* that leaves the shape intact |

The level test was added because the shape test measurably missed prior shift: the mean
score moved 0.2335 → 0.3714 (**+59%**) while score PSI only reached 0.16 — below any
PSI threshold that would not also fire on a harmless small covariate shift. The 25%
threshold sits in the measured gap between the harmless cases (0–3%) and the real ones
(35–63%).

## Controlled experiment

Seven scenarios, **30 independent trials each** with different seeds, 10,000 reference
rows against 2,000-row windows drawn from the held-out test split.

```bash
python scripts/drift_experiment.py --trials 30 --latency-trials 10
```

| Scenario | Kind | Drift expected | Input detection | Output detection | Mean max PSI | ROC-AUC | Δ vs no-drift | Detection latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `no_drift` | none | no | **0.00** | **0.00** | 0.048 | 0.9260 | — | n/a |
| `small_drift` | covariate | no | **0.00** | **0.00** | 0.121 | 0.9251 | −0.0009 | n/a |
| `large_drift` | covariate | yes | **1.00** | **1.00** | 6.189 | 0.8733 | **−0.0527** | 1 window |
| `missing_feature` | schema | yes | **1.00** | n/a | 0.048 | unscorable | — | 1 window |
| `performance_drift` | concept | yes | **0.00** | **0.00** | 0.048 | 0.6391 | **−0.2869** | never |
| `covariate_shift` | covariate | yes | **1.00** | 0.80 | 3.047 | 0.9076 | −0.0184 | 1 window |
| `prior_shift` | prior | yes | **0.00** | **1.00** | 0.106 | 0.9280 | +0.0020 | n/a |

### Headline numbers

| Measure | Value |
|---|---|
| **False-positive rate** (`no_drift`, 30 trials) | **0 / 30 = 0%** |
| **False-positive rate** (`small_drift`, 30 trials) | **0 / 30 = 0%** |
| Detection rate, covariate and schema drift | **30 / 30 = 100%** each |
| Detection rate, prior shift (output level test) | **30 / 30 = 100%** |
| Detection rate, concept drift | **0 / 30 = 0%** |
| Detection latency once drift begins | **1 window (2,000 records)** |
| Detection time per comparison | 0.071 – 0.221 s |

Combining both detectors, **4 of the 5 drifted scenarios are caught at 100% with a 0%
false-positive rate**, within one window.

## The one it misses, and why that matters most

`performance_drift` is **concept drift**: the labels are re-drawn from a different
feature→label relationship while the feature distributions are left untouched. It is
invisible to input-distribution monitoring by construction, and invisible to output
monitoring too, because unchanged inputs produce unchanged scores.

It is also **by far the most damaging scenario measured**: ROC-AUC falls from 0.9260 to
0.6391, a loss of **0.287**, and accuracy from 0.872 to 0.567. Compare `large_drift`,
which every detector catches and which costs only 0.053.

> The drift this system detects best is not the drift that hurts most.

That is not a flaw in the implementation; it is a property of label-free monitoring, and
it is the honest conclusion of this experiment. Detecting it requires ground truth.
Mitigations a real deployment would need: delayed-label evaluation once outcomes
arrive, proxy labels, or scheduled retraining on a cadence independent of any drift
signal.

`prior_shift` is the complementary lesson in the other direction: it is detected by the
output level test, but ROC-AUC is **unchanged** (+0.002) because AUC is a ranking metric
and invariant to class prevalence. What actually degrades is the decision threshold —
accuracy falls from 0.872 to 0.798 while PR-AUC *rises*. A detected drift is not
automatically a harmful one.

## What is not claimed

- These are **synthetic shifts on one dataset**. Detection rates here do not generalise
  to other data, other features or real-world drift.
- The thresholds (PSI 0.25, α 0.01, 25% mean shift) are conventional operating points
  calibrated against the measured no-drift false-positive rate on *this* data. They are
  a starting point elsewhere, not a result.
- The 0% false-positive rate is over 60 trials of two no-drift scenarios. It is a real
  measurement, not proof that the detector never false-alarms.
- Concept drift detection is **not implemented** and is not claimed.

## Artefacts

| File | Contents |
|---|---|
| `DRIFT_RESULTS.csv` | Per-scenario summary, the table above |
| `results/drift/drift_trials.csv` | All 210 individual trials |
| `results/drift/drift_per_feature.csv` | Per-feature PSI, statistic, p-value, Wasserstein |
| `results/drift/drift_detection_latency.csv` | Windows to first detection |
| `results/plots/drift_detection_rate.png` | Detection rate by scenario |
| `results/plots/drift_psi_distribution.png` | PSI spread with thresholds marked |
| `results/plots/drift_per_feature_psi.png` | Per-feature heatmap |
| `results/plots/drift_model_performance.png` | Model ROC-AUC under each scenario |
