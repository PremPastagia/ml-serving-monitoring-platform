# Final Project Report

## Summary

A locally reproducible ML platform covering the full lifecycle for a tabular classifier:
data contract and validation → reproducible training → MLflow tracking and model
registry → FastAPI serving → Prometheus and SQLite monitoring → statistical drift
detection → gated retraining with promotion and rollback.

The organising constraint was that **every number in the documentation must come from a
script in the repository**, and that claims the evidence does not support must be absent
— enforced by `scripts/verify/verify_claims.py`, which checks the evidence table row by
row and scans every document for unqualified claims.

## What was built

| Component | Implementation |
|---|---|
| Data contract | One module (`schema.py`) read by the validator, the feature pipeline, the API request model and the drift detector |
| Validation | 13 check families; errors stop the pipeline, warnings do not; univariate-AUC leakage detection |
| Splitting | Index-disjoint, content-hashed, with cross-split overlap measured rather than assumed |
| Features | 4 engineered columns inside the sklearn Pipeline, so there is no separate serving path |
| Training | Provenance capture of 15 fields; prediction-based reproducibility fingerprint |
| Tracking | MLflow with SQLite; params, metrics, tags, artefacts, environment |
| Registry | Alias-based promotion (`production` / `previous` / `candidate`) |
| Serving | 7 endpoints, generated request schema, structured JSON logs, request-id propagation |
| Monitoring | 15 Prometheus series + durable SQLite prediction store |
| Drift | KS, chi-square, PSI, Wasserstein on inputs; shape and level tests on outputs |
| Retraining | 6 acceptance criteria, interleaved latency comparison, promote/reject/rollback |
| Testing | 418 tests across 12 files |
| CI | 4 jobs (authored, verified locally, never run on GitHub) |
| Container | Multi-stage Dockerfile + compose (authored, never built) |

## Results

### Model

| Split | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| train | 0.956365 | 0.886742 | 0.897036 | 0.770062 | 0.071346 |
| validation | 0.930270 | 0.831673 | 0.873791 | 0.719454 | 0.087238 |
| **test** | **0.926784** | **0.823947** | **0.872059** | 0.708794 | 0.088495 |

Linear baseline 0.909126. Fit 0.582 s. Generalisation gap 0.026.

### Serving

256.4 req/s at p50 3.68 ms / p95 4.73 ms / p99 9.18 ms, **zero errors** across all 14
measured configurations. 6,564.7 records/s batched. Cold start 1.742 s. RSS ~275 MB.

### Drift

30 trials per scenario: **0% false-positive rate** on both no-drift scenarios, **100%
detection** on 4 of 5 drifted scenarios within **1 window (2,000 records)**. Concept
drift detected 0% of the time while costing **−0.287 ROC-AUC**.

### Retraining and rollback

Promotes a +0.029 candidate, rejects a +0.000 one, leaves production untouched when no
drift is present, and restores the incumbent on rollback. Rollback through the live
service in 0.082 s with 18 requests in flight and **0 failures** across 1,119 probe
requests.

### Testing

**418 tests, 0 failures, 132.5 s.** Clean-environment reproduction verified: identical
fingerprint, identical split id, 24 metrics matching to within 1e-9.

## Four findings

These came from investigating results that did not make sense, and they are the most
substantive part of the project.

**1. Input-drift monitoring is blindest to the drift that hurts most.**
Concept drift is detected 0/30 times and costs −0.287 ROC-AUC; the large covariate shift
caught 30/30 times costs −0.053. This is a property of label-free monitoring, and stating
it with numbers is more useful than a 100% detection rate on the easy cases would have
been.

**2. OpenMP fan-out was costing 2.6–5.1× throughput.**
scikit-learn's HistGradientBoosting fans even a single-row prediction across every core.
Capping intra-op threads took throughput from 99 to 256 req/s and host CPU from 99% to
20%, and made throughput hold flat under concurrency instead of collapsing.

**3. PSI was silently unstable on a tied feature.**
46.7% of records tie at exactly 40 hours; the same +2-hour shift scored 0.0021 or 2.2587
depending only on where the reference quantiles placed a bin edge. Tied numeric columns
now use value-based bins, and the residual saturation is documented and pinned.

**4. An absolute latency budget rejected a better model.**
A candidate 2.4 points better was rejected because the host was oversubscribed. The gate
is now a ratio against the incumbent, measured by interleaving both models' requests and
comparing medians.

A fifth, smaller one worth noting: the first rollback continuity run reported "0 of 0
requests failed" — the switch completed faster than a serial prober's request interval.
Concurrent probes and a hard requirement for in-flight traffic turned a vacuous result
into real evidence.

## What was not done, and why

| Not done | Reason |
|---|---|
| Docker image built and measured | **No container runtime available** on the development machine. Image size and start-up time are recorded as UNVERIFIED and quoted nowhere |
| CI executed on GitHub | Repository was local during development; every step verified locally instead |
| Authentication | Out of scope for a local platform; listed as the first production gap |
| Scheduled retraining | Deliberate boundary — retraining is explicitly triggered, and the documentation says so rather than implying automation |
| Concept-drift detection | Requires ground truth; the gap is measured and stated instead of papered over |
| Grafana dashboards | Metrics are Prometheus-format and would feed one, but none was built |
| Multi-process Prometheus | Per-process registries; multi-worker exposition not configured |
| Fairness assessment | Strong demographic associations exist in the data; no mitigation is implemented or claimed |
| Cross-platform reproducibility | Verified on one machine across clean environments only |

## Honest assessment

**What this demonstrates well:** that the lifecycle pieces compose into something
coherent; that engineering decisions were made deliberately and can be defended with
measurements; and — most of all — a habit of investigating results that look wrong
rather than publishing them. The four findings above are worth more than the feature
list.

**What it does not demonstrate:** operating a service under real production traffic,
multi-node systems, security engineering, or ML at a scale where distributed training or
feature stores matter. The dataset is small and clean by design, so the data-engineering
challenges of real pipelines are absent.

**The single biggest gap:** concept drift is invisible to this system and is the most
damaging failure mode measured. Closing it needs delayed-label evaluation, which is a
different piece of infrastructure from anything built here.

## Repository map

See [README.md](README.md) for layout and quick start. Per-phase documents are linked
from the table there. Evidence for every CV claim is in
[CV_POINTERS_ML_SERVING_MONITORING.md](CV_POINTERS_ML_SERVING_MONITORING.md).
