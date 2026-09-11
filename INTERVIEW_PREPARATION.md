# Interview Preparation

Questions grouped by what they probe. Every answer is defensible from evidence in this
repository.

---

## Problem framing and metrics

**Why this dataset?**
It satisfies every requirement the platform needs to demonstrate: public with a stable
checksummed URL, a clear binary target, an imbalanced class distribution (24.1%
positive) so ROC-AUC and PR-AUC both carry information, mixed numeric and categorical
features so different drift tests are genuinely needed, and small enough that the whole
pipeline runs in under a minute — which is what makes the reproducibility test cheap
enough to run routinely.

**Why ROC-AUC as primary?**
The serving contract returns a probability, so a threshold-free ranking metric is the
honest headline. PR-AUC is reported alongside because it is the one that degrades when
the positive class gets rarer; Brier because a model can rank well and be badly
calibrated. The prior-shift scenario proves the point: ROC-AUC was *unchanged* (+0.002)
while accuracy fell from 0.872 to 0.798, because AUC is invariant to class prevalence
and the 0.5 threshold was no longer calibrated.

**Is 0.927 ROC-AUC good?**
It is in line with published results for Adult, and more usefully it beats a properly
tuned linear baseline on the same test set (0.909). I keep the baseline in every
experiment as a sanity floor — if the boosted model cannot beat a linear one, the
pipeline is broken rather than the data.

---

## Data and leakage

**How do you know there is no leakage?**
Four defences, three structural and one statistical. (1) The test set is a physically
separate UCI file — no code path can move a test row into `fit`. (2) Train/validation
overlap is measured by source index and asserted to be 0. (3) All preprocessing
statistics are learned inside `Pipeline.fit`, which only ever sees train; verified
behaviourally rather than by inspection. (4) Any feature with univariate ROC-AUC ≥ 0.99
against the target aborts training.

**Why 0.99 for the leakage threshold?**
Measured rather than assumed. The strongest legitimate single feature in this dataset is
`relationship` at 0.779, so 0.99 leaves a 0.21 margin while still catching a
slightly-noised leak. A test asserts that margin against real data, so the justification
cannot rot if the data changes.

**Why drop `fnlwgt`?**
It is the census inverse-probability sampling weight — how many people in the population
the row represents. It is a property of the survey design, not of the person. So it is
unavailable at serving time for a new applicant, and letting a tree split on it means
learning the sampling frame instead of the income relationship.

**10.6% of your rows are duplicates. Why keep them?**
Because in a population survey two different respondents can legitimately share all
twelve retained attributes. A repeated feature vector is not a defect and it is not
leakage — leakage would be the same *record* in two splits, which the split step forbids
by index. Deduplicating would discard real frequency information and change the row
counts, making the dataset version incomparable with the published benchmark. What I do
instead is measure it: 14.4% of validation rows and 11.2% of test rows have a feature
vector that also occurs in train, and I report that because it bounds how optimistic the
held-out score can be.

---

## Training and reproducibility

**How do you know training is reproducible?**
`training_fingerprint` hashes the fitted model's predictions on a fixed 256-row probe set
at float64. I verified it in a brand-new virtualenv built from the pinned requirements:
identical fingerprint, identical split id, and all 24 metrics matching to within 1e-9. I
hash predictions rather than pickle bytes because pickle differs between runs for
reasons unrelated to the model.

**Why not claim 100% reproducibility?**
Because I have only tested one machine across clean environments. Cross-OS and
cross-architecture reproducibility involves BLAS implementations, compiler flags and
floating-point ordering, and I have not tested it. Timing numbers are explicitly *not*
reproducible exactly, which is why every benchmark row records host CPU utilisation.

**You log a random seed but say it does nothing. Explain.**
`random_state` reaches HistGradientBoosting in exactly two places: the internal split
used for early stopping, and the bin-threshold subsample above 200k rows. I set
`early_stopping: false` and train on 26k rows, so neither applies and the fit is
deterministic regardless of the seed. That is a stronger guarantee than seeding, but I
assert it in a test, because enabling early stopping later would silently make the fit
seed-dependent and the documentation would quietly become wrong. The seed that *is*
load-bearing is the split seed.

---

## Serving

**Walk me through a prediction request.**
Middleware assigns a request id (or reuses `X-Request-ID`) and starts a timer. Pydantic
validates the payload against a model *generated from the training data contract* —
missing, extra, wrong-typed, out-of-range and unknown-level fields are all 422. The
records become a DataFrame and go through the single sklearn Pipeline, which contains
the feature engineering and preprocessing, so serving cannot apply a different
transformation from training. Probabilities are thresholded, Prometheus counters and
histograms updated, the feature vectors written to SQLite, and the response returns with
the model version in both the body and a header.

**Why generate the API schema from the contract instead of writing it?**
A hand-written API model is a second copy of the contract, and the second copy is always
the one that goes stale. Generating it means tightening a range or renaming a level
automatically changes what the API accepts, and a test asserts the generated fields
equal the contract exactly.

**Why reject unknown fields instead of ignoring them?**
Silently dropping an extra key is exactly how a client ships a renamed field into
production and nobody notices for a month. Rejecting turns a silent data-quality
incident into a loud 422 at the boundary.

**Why is `/health` 200 when no model is loaded?**
Because a liveness probe should not restart a process that is running correctly and
merely has nothing to serve — a restart cannot fix an empty registry. `/health` returns
200 with `status: degraded`; `/ready` returns 503 and is the probe that gates traffic.

**How did you get 2.6× throughput?**
By refusing to publish a benchmark I could not explain. Throughput was *falling* as
concurrency rose and one server process was pinning all ten cores. The cause is that
scikit-learn's HistGradientBoosting predicts through OpenMP and fans even a single-row
request across every core — eight concurrent requests each tried to use ten. Capping
intra-op threads to 1 took throughput from 99 to 256 req/s, p99 from 22.7 ms to 9.2 ms,
and host CPU from 99% to 20%. The generalisable lesson is to scale a Python model server
with worker processes, not intra-op threads.

**Where does the remaining latency go?**
Measured by stage at minimum-of-300 timings: derived features 1.22 ms, ColumnTransformer
2.19 ms, model `predict_proba` 1.93 ms, total 5.51 ms in-process. Model inference is only
about 35% of it — pandas and the transformer dominate. That is what I would optimise
first, probably by moving the transform to numpy arrays.

---

## Monitoring and drift

**Why two stores?**
They answer different questions. Prometheus holds aggregates — a counter tells you
*that* traffic changed. Only the retained feature vectors in SQLite let you re-run a
two-sample test and say *which* feature changed and by how much. Drift detection is a
retrospective query over that table.

**Why KS for numeric and chi-square for categorical?**
KS is distribution-free, which matters because `capital_gain` is 92% zeros with a long
tail, it needs no binning choice, and it is sensitive to any CDF difference rather than
only a mean shift. Chi-square is the natural test for comparing two multinomials, with
rare levels pooled because the approximation is unreliable below an expected count of 5
and `native_country` has a long tail.

**Why require both PSI and a p-value?**
Because each alone is wrong at this scale. With thousands of rows a p-value rejects on
differences too small to act on; PSI alone has no notion of sampling noise. Requiring
both is what produces the measured 0% false-positive rate over 60 no-drift trials.

**Tell me about a bug you found.**
PSI was silently unstable on `hours_per_week`. 46.7% of Adult records report exactly 40
hours, so under quantile binning five of the eleven decile edges collapse to the value
40. The same +2-hour shift scored PSI 0.0021 against the full reference and 2.2587
against a 4,500-row subsample of the same data — purely because the two placed a bin edge
differently, one hiding the move entirely. A detector whose verdict depends on that is
unusable. The fix detects columns whose quantile edges collapse and uses value-based
bins. The residual limitation — discrete PSI saturates, so it cannot order a +2 against
a +12 shift — is why Wasserstein distance is reported alongside, and both behaviours are
pinned by tests.

**Your detector misses concept drift. Isn't that a failure?**
It is the most valuable result in the project. Input- and output-distribution monitoring
cannot see a change in the feature→label relationship when the features are unchanged —
that is a property of label-free monitoring, not a bug. What makes it worth stating is
that I measured the cost: concept drift is the most damaging scenario at −0.287 ROC-AUC,
while the large covariate shift every detector catches costs only −0.053. **The drift
that is easiest to detect is not the drift that hurts most.** Detecting it needs ground
truth — delayed-label evaluation, proxy labels, or retraining on a cadence independent
of any drift signal.

**Why add a mean-shift test on the output?**
Because the shape test measurably missed prior shift. The mean score moved 0.2335 →
0.3714, a 59% change, while score PSI only reached 0.16 — below any threshold that would
not also fire on a harmless small covariate shift. The 25% level threshold sits in the
measured gap between the harmless cases (0–3%) and the real ones (35–63%).

---

## Retraining and rollback

**How do you decide to promote?**
Six criteria, each there because of a specific failure mode. A minimum improvement
margin, because retraining always moves the score a little and promoting on noise fills
the registry with churn. A maximum permitted degradation. An absolute accuracy floor,
because if the incumbent has decayed then "better than the incumbent" is a very low bar
and without a floor the loop ratchets downwards. A latency ratio. A clean-validation
requirement, which is what stops a corrupt feed being laundered into a promoted model.
And a minimum row count, because a drift window can be small and fitting on 200 rows
then promoting destroys a good model.

**Why is latency a ratio rather than a budget?**
An absolute wall-clock threshold is not portable between machines and is not even stable
on one. A 50 ms budget rejected a candidate that was 2.4 ROC-AUC points better and no
slower — the host was simply oversubscribed. A ratio cancels that because both models
pay the same tax. Two refinements were needed to make the ratio trustworthy: interleave
the two models' requests, because timing one fully then the other produced a bogus 10.5×
ratio when the load changed between windows; and take the ratio on the median, because
the tail is dominated by scheduler outliers belonging to neither model.

**Why compare against a re-scored incumbent?**
Because comparing against the number recorded at the incumbent's training time compares
two different test sets. That is the most common way an automated promotion gate quietly
promotes a worse model.

**Can you claim zero-downtime deployment?**
No, and I do not. I measured 1,119 requests across two alias switches with 0 failures, and
4 and 27 requests confirmed in flight during the switches. Getting that evidence took
two attempts: a single serial prober issued a request only every few hundred
milliseconds, so the first run reported "0 of 0 requests failed" during the promotion —
which is not evidence of anything. The script now uses four concurrent probes and fails
if no traffic was in flight. Zero-downtime is a claim about sustained production traffic
across many switches and restarts, which I have not tested.

---

## Engineering judgement

**Is this production-ready?**
No. No authentication anywhere — `/admin/reload` swaps the served model unauthenticated.
Single-node SQLite. Per-process Prometheus metrics. Retraining is explicitly triggered,
not scheduled. The Docker image was written but never built, because no container
runtime was available. I list eleven gaps and a script scans every document to make sure
"production-ready" never appears unqualified.

**Why SQLite rather than PostgreSQL?**
The platform must run from a clean checkout with no services to start — that is the
property the reproducibility test depends on. SQLite is ACID, ships with Python, and
handles this write rate comfortably. The schema uses no SQLite-only types, so moving to
PostgreSQL is a connection string plus a driver, and I recorded that trade-off rather
than pretending it was free.

**Why not Evidently, Great Expectations, DVC?**
Evidently picks the drift test for you by cardinality and row count, so "why this test?"
would have no answer, and its defaults change between versions. Great Expectations is a
heavy execution engine for something that has to run inside the request path. DVC's value
is large binary files in remote storage; this is 6 MB from a stable pinned URL, where a
content hash gives the same traceability with no extra tool. In each case I can state
what I gave up.

**What would you do next?**
In order: put authentication in front of `/admin/reload`; move the tracking store and
prediction log to PostgreSQL and configure multi-process Prometheus; actually build and
measure the container; add delayed-label evaluation so concept drift becomes detectable,
since that is the largest measured gap; and add subgroup performance reporting, because
the dataset has strong demographic associations and I currently do no fairness
assessment at all.

**What surprised you most?**
That four of the most valuable findings came from investigating numbers that looked
wrong rather than from building features — the OpenMP fan-out, the PSI binning
instability, the latency gate rejecting a better model, and the empty rollback
continuity evidence. Each looked like a result at first glance. The habit worth keeping
is refusing to publish a number I cannot explain.
