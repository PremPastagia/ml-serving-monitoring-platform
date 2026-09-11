# Production-Style ML Serving & Monitoring Platform

An end-to-end, locally reproducible machine-learning platform: validated data →
reproducible training → MLflow tracking and model registry → FastAPI serving →
Prometheus monitoring → statistical drift detection → gated retraining with promotion
and rollback.

**Every number in this repository is produced by a script in this repository.** Nothing
is estimated, and claims the evidence does not support are absent by design — a
verification script enforces that.

**This is not production-ready**, and no document here says otherwise. The specific
gaps are listed in [Limitations](#limitations).

---

## Headline results

| | Measured |
|---|---|
| Model quality (held-out test, 16,281 rows) | **ROC-AUC 0.9268**, PR-AUC 0.8239, accuracy 0.8721 |
| Linear baseline, same test set | ROC-AUC 0.9091 |
| Serving throughput | **256 req/s**, p50 **3.68 ms**, p95 **4.73 ms**, p99 **9.18 ms**, **0 errors** |
| Batched record throughput | **6,565 records/s** (batch 32) |
| Thread-pinning optimisation | **2.6–5.1× throughput**, host CPU **99.8% → 25.3%** |
| Drift detection | **100%** detection on 4 of 5 drifted scenarios, **0% false-positive rate** over 60 no-drift trials |
| Drift detection latency | **1 window (2,000 records)** |
| Retraining | Promotes a +0.029 ROC-AUC candidate, **rejects** a +0.000 one |
| Rollback through the live service | **0.082 s**, 18 requests in flight, **0 failed** |
| Tests | **418 passed, 0 failed** in 132.5 s |
| Training | **1.20 s**, byte-identical fingerprint across runs |

---

## Quick start

```bash
git clone <this repo> && cd ml-serving-platform

make setup      # virtualenv + pinned dependencies
make data       # download UCI Adult, verify pinned SHA-256
make train      # validate → split → train → evaluate → track in MLflow → register
make serve      # http://127.0.0.1:8077/docs
```

In another shell:

```bash
curl -X POST http://127.0.0.1:8077/predict \
  -H 'content-type: application/json' \
  -d '{"records": [{"age": 39, "workclass": "State-gov", "education_num": 13,
                    "marital_status": "Never-married", "occupation": "Adm-clerical",
                    "relationship": "Not-in-family", "race": "White", "sex": "Male",
                    "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
                    "native_country": "United-States"}]}'
```

Everything else:

```bash
make test       # 418 tests
make smoke      # start the server, exercise every endpoint, shut down
make drift      # controlled drift experiment, 30 trials per scenario
make retrain    # retraining / promotion / rejection / rollback experiment
make rollback   # rollback verified through the running service
make bench      # HTTP load test, pinned and unpinned
make results    # aggregate every measurement into the top-level CSVs
make all        # the whole pipeline end to end
```

---

## The problem

Binary classification on the **UCI Adult (Census Income)** dataset: predict whether a
respondent's income exceeds $50,000 from twelve demographic and employment attributes.

Chosen because it is publicly available with a stable URL, has a clear target and an
imbalanced class distribution (24.1% positive, so ROC-AUC *and* PR-AUC both say
something), has mixed numeric and categorical features (so different drift tests are
genuinely needed), and is small enough that the whole pipeline runs in under a minute.

The data is a 1994 US census extract and encodes the demographics of that time and
place. It is used here as a well-understood benchmark for MLOps machinery, **not as a
basis for any decision about a real person** — see
[DATASET_CARD.md](DATASET_CARD.md#known-biases-and-ethical-notes).

---

## Architecture

```
raw files (SHA-256 pinned) → ingest → validate → split → feature pipeline
        → train + evaluate → MLflow tracking → model registry (aliases)
        → FastAPI serving → Prometheus + SQLite → drift detection
        → retraining decision → promote or rollback
```

Full diagram and every technology trade-off: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

### Things worth knowing about the design

- **One data contract.** `src/mlserve/data/schema.py` is the single source of truth. The
  validator, the feature pipeline, the API request model and the drift detector all read
  it, so the API cannot accept something training would have rejected. A test asserts the
  generated API model equals the contract exactly.
- **Preprocessing lives inside the model.** The object in the registry *is* the
  transformation plus the estimator, so train/serve skew is removed structurally rather
  than by discipline.
- **Aliases, not stages.** Promotion and rollback are a single atomic repoint of a
  mutable pointer to an immutable version.
- **A failed reload leaves the previous model serving.** It never empties the service.
- **Load failure is a state, not a crash.** An empty registry yields `/health` 200
  `degraded` and `/ready` 503 — a restart cannot fix an empty registry.

---

## What was built, phase by phase

| Phase | Deliverable | Document |
|---|---|---|
| 0 | Problem, dataset, system design | [PROJECT_SCOPE.md](PROJECT_SCOPE.md), [DATASET_CARD.md](DATASET_CARD.md), [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) |
| 1 | Data validation + reproducible training | [DATA_VALIDATION.md](DATA_VALIDATION.md), [TRAINING.md](TRAINING.md) |
| 2 | MLflow tracking + model registry | [MLFLOW.md](MLFLOW.md) |
| 3 | FastAPI serving + load testing | [API.md](API.md), [LOAD_TESTING.md](LOAD_TESTING.md) |
| 4 | Docker + CI | [CI_CD.md](CI_CD.md) |
| 5 | Monitoring + observability | [MONITORING.md](MONITORING.md) |
| 6 | Drift detection | [DRIFT_DETECTION.md](DRIFT_DETECTION.md) |
| 7 | Retraining, promotion, rollback | [RETRAINING.md](RETRAINING.md), [ROLLBACK.md](ROLLBACK.md) |
| 8 | Full testing + benchmarking | [TEST_RESULTS.md](TEST_RESULTS.md), [BENCHMARKS.md](BENCHMARKS.md), [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| 9 | CV pointers + interview prep | [CV_POINTERS_ML_SERVING_MONITORING.md](CV_POINTERS_ML_SERVING_MONITORING.md), [INTERVIEW_PREPARATION.md](INTERVIEW_PREPARATION.md), [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) |

---

## Four findings worth reading

These came from investigating results that did not make sense, rather than from
implementing a checklist. They are the parts of this project most worth discussing.

**1. Input-drift monitoring is blindest to the drift that hurts most.**
Across 30 trials per scenario, the detector catches covariate and schema drift 100% of
the time with a 0% false-positive rate. It catches **concept drift 0% of the time** —
and concept drift is by far the most damaging scenario measured, costing **0.287
ROC-AUC** versus 0.053 for the large covariate shift every detector catches. That is a
property of label-free monitoring, not a bug, and it is the honest conclusion of the
experiment. [DRIFT_DETECTION.md](DRIFT_DETECTION.md#the-one-it-misses-and-why-that-matters-most)

**2. OpenMP fan-out was costing 2.6–5.1× throughput.**
Benchmarks showed throughput *falling* as concurrency rose, with one server process
pinning all ten cores. scikit-learn's HistGradientBoosting predicts through OpenMP and
fans even a single-row request across every core. Capping intra-op threads to 1 took
throughput from 99 to 256 req/s, p99 from 22.7 ms to 9.2 ms, and host CPU from 99% to
20%. Scale a Python model server with worker *processes*, never intra-op threads.
[BENCHMARKS.md](BENCHMARKS.md#thread-pinning)

**3. PSI was silently unstable on a tied feature.**
46.7% of Adult records report exactly 40 hours per week. Under quantile binning the
same +2-hour shift scored PSI **0.0021** against the full reference and **2.2587**
against a subsample of it, purely because the two placed a bin edge differently. Tied
numeric columns now use value-based bins.
[DRIFT_DETECTION.md](DRIFT_DETECTION.md#a-defect-found-and-fixed-psi-on-tied-features)

**4. An absolute latency budget rejected a better model.**
A candidate 2.4 ROC-AUC points better was rejected for exceeding a 50 ms p95 budget —
the host was simply oversubscribed. The gate is now a *ratio* against the incumbent,
measured by **interleaving** both models' requests and comparing medians, so host
contention cancels. [RETRAINING.md](RETRAINING.md#why-latency-is-a-ratio-not-a-millisecond-budget)

---

## Repository layout

```
src/mlserve/
  config.py            configuration, code version, environment capture
  logging_utils.py     structured JSON logging with request-id propagation
  data/                schema (the contract) · ingest · validate · split
  features/            derived features + preprocessing pipelines
  models/              train · evaluate · MLflow tracking · registry
  serving/             FastAPI app · schemas · model loader · Prometheus metrics
  monitoring/          SQLite prediction store · drift detection · scenarios
  retraining/          acceptance criteria · orchestrator
scripts/
  fetch_data.py        download + verify pinned checksums
  train.py             the training pipeline
  serve.py             the serving entrypoint (thread-pinned)
  smoke_test.py        24 checks against the real running server
  load_test.py         reproducible HTTP benchmark
  drift_experiment.py  controlled drift experiment
  retrain_experiment.py  promotion / rejection / rollback experiment
  rollback_through_serving.py   rollback verified through the live service
  collect_results.py   aggregate every measurement
  verify/              claim and reproducibility verification
tests/                 418 tests
docker/                Dockerfile + compose (authored, never built)
.github/workflows/     CI
```

---

## Results files

| File | Contents |
|---|---|
| `EVALUATION_RESULTS.csv` | Per-model, per-split metrics with full provenance |
| `SERVING_BENCHMARKS.csv` | Every load-test configuration, pinned and unpinned |
| `DRIFT_RESULTS.csv` | Per-scenario detection rates and performance deltas |
| `results/` | Raw experiment output, plots, JUnit XML, aggregated summary |

---

## Limitations

Stated plainly, because the point of this project is that its claims are checkable:

1. **Not production-ready.** No authentication anywhere — `/admin/reload` swaps the
   served model unauthenticated.
2. **Single node.** SQLite for tracking, registry and prediction logging.
3. **Docker is authored but never built.** No container runtime was available on the
   development machine, so image size and container start-up time are recorded as
   **UNVERIFIED** and are quoted nowhere.
4. **CI is authored but has never run on GitHub.** Every step was verified locally.
5. **Retraining is explicitly triggered**, not scheduled. There is no production
   trigger.
6. **Prometheus metrics are per-process**; multi-worker exposition is not configured.
7. **Drift results are synthetic shifts on one dataset.** The detection rates do not
   generalise to other data or to real-world drift.
8. **Reproducibility is verified on one machine** across clean virtual environments —
   not across operating systems or CPU architectures.
9. **No fairness assessment**, despite strong demographic associations in the data.
10. **Benchmarks are single-machine, single-worker, loopback.** No network latency, no
    load balancer, no TLS.

---

## Verifying the claims

```bash
python scripts/verify/verify_claims.py           # evidence table, overclaim scan, deliverables
python scripts/verify/verify_clean_env_repro.py  # rebuild in a fresh venv and compare
```

`verify_claims.py` checks that every row of the evidence table in
[CV_POINTERS_ML_SERVING_MONITORING.md](CV_POINTERS_ML_SERVING_MONITORING.md) names a
file that exists, a number that actually appears in that file, and a reproduction
command whose script exists — and scans every document for unqualified claims such as
"production-ready" or "zero-downtime".

---

## Licence and attribution

Dataset: UCI Machine Learning Repository, Adult (Census Income), CC BY 4.0.
Code in this repository is provided as a portfolio project.
