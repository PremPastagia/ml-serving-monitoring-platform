# CV Pointers — Production-Style ML Serving & Monitoring Platform

Every figure below is measured. The evidence table at the end maps each claim to the
file that contains the number and the command that regenerates it.

---

## 1. Project title

**Production-Style ML Serving & Monitoring Platform** — end-to-end MLOps for a
tabular classifier, from data contract to gated rollback.

## 2. One-line description

Built a locally reproducible ML platform that validates data, trains a tracked and
registered model, serves it over FastAPI at 256 req/s with a 4.7 ms p95, monitors input
and output distributions, detects drift with a measured 0% false-positive rate, and
gates retraining behind six acceptance criteria with rollback verified through the
running service.

## 3. Technology stack

| Layer | Used |
|---|---|
| Language | Python 3.13 (3.11 floor, both in CI) |
| Model | scikit-learn `HistGradientBoostingClassifier` + `LogisticRegression` baseline |
| Data | pandas, numpy |
| Validation | Custom contract-derived validator (78 tests) |
| Tracking / registry | MLflow 3.6 with a SQLite backend, registry **aliases** |
| Serving | FastAPI 0.141 + uvicorn, Pydantic v2 |
| Monitoring | prometheus-client, SQLite (WAL) prediction store |
| Drift | scipy — two-sample KS, chi-square homogeneity, PSI, Wasserstein |
| Testing | pytest (418 tests) |
| CI | GitHub Actions (authored; never executed on GitHub) |
| Container | Docker multi-stage (authored; **never built**) |
| Lint | ruff, black |

## 4. Pipeline

```
raw files (SHA-256 pinned) → ingest (content-hashed dataset version)
  → validate (schema · types · ranges · categories · nulls · duplicates · leakage)
  → split (index-disjoint; overlap measured)
  → feature pipeline (4 engineered features, fitted on train only, inside the model)
  → train + evaluate → MLflow tracking → model registry (production/previous/candidate)
  → FastAPI serving → Prometheus + SQLite
  → drift detection (input KS/χ²/PSI + output shape & level)
  → acceptance criteria → promote or rollback
```

## 5. Dataset and model

| | |
|---|---|
| Dataset | UCI Adult (Census Income), 48,842 records, SHA-256 pinned |
| Dataset version | `adult-ingest-2-975c90344d56` (content hash of bytes + parsing rules) |
| Splits | train 26,048 / validation 6,513 / **test 16,281 (a physically separate UCI file)** |
| Features | 12 contract features (5 numeric, 7 categorical) → 16 after engineering |
| Dropped | `fnlwgt` (survey sampling weight, unavailable at serving time), `education` (duplicate of `education_num`) |
| Target | `income > $50K`, 24.1% positive |
| Model | HistGradientBoosting, 200 trees, preprocessing inside the same sklearn Pipeline |

## 6. Verified training metrics

| Split | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| train | 0.956365 | 0.886742 | 0.897036 | 0.770062 | 0.071346 |
| validation | 0.930270 | 0.831673 | 0.873791 | 0.719454 | 0.087238 |
| **test (held out)** | **0.926784** | **0.823947** | **0.872059** | 0.708794 | 0.088495 |

Linear baseline on the same test set: **0.909126** ROC-AUC.
Fit time **0.582 s**. Train→validation gap **0.026**.
Repeated training produces a **byte-identical prediction fingerprint**.

## 7. Verified serving metrics

1 uvicorn worker, intra-op threads pinned to 1, 10-CPU Apple Silicon, 20 s per point.

| Concurrency | Batch | Requests/s | Records/s | p50 | p95 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **256.4** | 256.4 | **3.68 ms** | **4.73 ms** | **9.18 ms** | **0** |
| 8 | 1 | 242.4 | 242.4 | 32.61 ms | 46.79 ms | 53.57 ms | **0** |
| 16 | 1 | 236.5 | 236.5 | 66.63 ms | 92.97 ms | 111.49 ms | **0** |
| 1 | 32 | 205.2 | **6,564.7** | 4.67 ms | 5.58 ms | 10.16 ms | **0** |

Cold start **1.742 s**. RSS **271–282 MB**, flat across all configurations.

**Thread-pinning optimisation**, same machine and model, back to back:

| | Requests/s | p99 | Host CPU |
|---|---:|---:|---:|
| Uncapped OpenMP, concurrency 1 | 99.2 | 22.68 ms | 99.3% |
| **Pinned, concurrency 1** | **256.4** | **9.18 ms** | **19.6%** |
| Uncapped, concurrency 8 | 47.4 | 196.16 ms | 99.8% |
| **Pinned, concurrency 8** | **242.4** | **53.57 ms** | **25.3%** |

**2.6× throughput at concurrency 1, 5.1× at concurrency 8, p99 3.7× lower.**

## 8. Verified monitoring metrics

15 Prometheus series: request, error and latency histograms; prediction-score
distribution; per-feature input histograms; categorical level counters; model identity;
resource gauges. Durable SQLite store retains every scored feature vector.

40 monitoring tests covering normal, invalid, high-volume and model-error traffic. A
7-record batch increments predictions by exactly 7 and requests by exactly 1; 3 good and
2 bad requests yield an error rate of exactly 0.4.

## 9. Verified drift-detection results

30 independent trials per scenario, 10,000 reference rows vs 2,000-row windows.

| Scenario | Expected | Input detection | Output detection | ROC-AUC Δ |
|---|---|---:|---:|---:|
| `no_drift` | no | **0.00** | **0.00** | — |
| `small_drift` | no | **0.00** | **0.00** | −0.0009 |
| `large_drift` | yes | **1.00** | **1.00** | −0.0527 |
| `missing_feature` | yes | **1.00** | n/a | unscorable |
| `performance_drift` (concept) | yes | **0.00** | **0.00** | **−0.2869** |
| `covariate_shift` | yes | **1.00** | 0.80 | −0.0184 |
| `prior_shift` | yes | 0.00 | **1.00** | +0.0020 |

**False-positive rate 0/60 across both no-drift scenarios. 100% detection on 4 of the 5
drifted scenarios, within 1 window (2,000 records). Detection time 0.071–0.221 s.**

The honest finding: **concept drift is detected 0% of the time and is the most damaging
scenario measured** (−0.287 ROC-AUC). Label-free monitoring cannot see it.

## 10. Verified retraining and rollback results

| Case | Decision | Incumbent | Candidate | Δ | Production after |
|---|---|---:|---:|---:|---|
| No drift | **not triggered** | — | — | — | unchanged |
| Drift, no gain | **reject** (`min_absolute_improvement`) | 0.926784 | 0.926784 | 0.000000 | unchanged |
| Drift, better candidate | **promote** | 0.897678 | 0.926784 | **+0.029106** | v2 |
| Rollback | **restored** | — | — | — | v1 |

Retraining 1.08–1.25 s · promotion 0.0043 s · rollback 0.0055 s.

**Rollback through the running service:** promote v1→v2 in 0.063 s with 4 requests in
flight; rollback v2→v1 in 0.082 s with 18 requests in flight. **1,119 probe requests
across both switches, 0 failed.**

Testing: **418 tests, 0 failures, 132.5 s.**

## 11. Reproduction commands

```bash
make setup && make data                       # environment + checksum-verified data
python scripts/train.py                       # §6 training metrics
python -m pytest -q                           # §10 test counts
python scripts/smoke_test.py                  # 24 HTTP checks
python scripts/load_test.py --thread-pin 1 --duration 20 --warmup 5 \
       --concurrency 1 2 4 8 16 --batch-size 1 32          # §7 serving metrics
python scripts/load_test.py --thread-pin 0 --duration 20 --warmup 5 \
       --concurrency 1 8 --batch-size 1 32                 # §7 pinning comparison
python scripts/drift_experiment.py --trials 30 --latency-trials 10   # §9
python scripts/retrain_experiment.py                       # §10 retraining
python scripts/rollback_through_serving.py                 # §10 rollback
python scripts/collect_results.py                          # aggregate
python scripts/verify/verify_claims.py                     # check this document
```

## 12. Limitations

1. **Not production-ready.** No authentication anywhere; `/admin/reload` swaps the served
   model unauthenticated.
2. **Docker authored but never built** — no container runtime on the development
   machine. Image size and container start-up time are **UNVERIFIED** and quoted nowhere.
3. **CI authored but never executed on GitHub.** Every step verified locally.
4. Single node; SQLite for tracking, registry and prediction logging.
5. Retraining is **explicitly triggered**, not scheduled.
6. Prometheus metrics are per-process; multi-worker exposition not configured.
7. Drift results are **synthetic shifts on one dataset** and do not generalise.
8. **Concept drift detection is not implemented** and not claimed.
9. Reproducibility verified **on one machine** across clean virtual environments, not
   across OS or CPU architecture.
10. Benchmarks are single-machine, single-worker, loopback; no TLS or load balancer.
11. No fairness assessment, despite strong demographic associations in the data.

## 13. Interview questions and defensible answers

Fuller set in [INTERVIEW_PREPARATION.md](INTERVIEW_PREPARATION.md).

**Why ROC-AUC as the primary metric?**
The serving contract returns a probability, so a threshold-free ranking metric is the
honest headline. PR-AUC is reported alongside because it is the one that actually
degrades when the positive class gets rarer, and Brier because a model can rank well and
still be badly calibrated. The prior-shift scenario demonstrates exactly this: ROC-AUC
was unchanged (+0.002) while accuracy fell from 0.872 to 0.798.

**How do you know there is no leakage?**
Three structural defences plus one statistical one. The test set is a physically
separate UCI file, so no code path can move a test row into `fit`. Train/validation
overlap is measured by source index and asserted to be 0. All preprocessing statistics
are learned inside `Pipeline.fit`, verified behaviourally. And any feature with
univariate ROC-AUC ≥ 0.99 against the target aborts training — the threshold justified
by measuring that the strongest legitimate feature reaches only 0.779.

**Why did you drop `fnlwgt`?**
It is the census inverse-probability sampling weight: how many people the row
represents. It is a property of the survey design, not the person, so it is unavailable
at serving time for a new individual, and letting a tree split on it means learning the
sampling frame instead of the income relationship.

**Your drift detector misses concept drift. Isn't that a failure?**
It is the most important result in the project. Input- and output-distribution
monitoring cannot see a change in the feature→label relationship when the features
themselves are unchanged — that is a property of label-free monitoring, not a bug. I
measured the cost: concept drift is the most damaging scenario at −0.287 ROC-AUC, while
the large covariate shift every detector catches costs only −0.053. The drift that is
easiest to detect is not the drift that hurts most. Detecting it needs ground truth:
delayed-label evaluation, proxy labels, or retraining on a cadence independent of any
drift signal.

**Why implement PSI yourself instead of using Evidently?**
Evidently selects the test for you based on cardinality and row count, so "why this
test?" would have no answer, and its per-feature defaults change between versions, which
undermines reproducibility. Implementing it also surfaced a real defect I would
otherwise have shipped: PSI was unstable on `hours_per_week` because 46.7% of records tie
at exactly 40, and the same shift scored 0.0021 or 2.2587 depending purely on where the
reference quantiles placed a bin edge.

**How did you get 2.6× throughput?**
By investigating a benchmark that did not make sense — throughput was falling as
concurrency rose, with one process pinning all ten cores. scikit-learn's
HistGradientBoosting predicts through OpenMP and fans even a single-row request across
every core, so eight concurrent requests each tried to use ten. Capping intra-op threads
to 1 took throughput from 99 to 256 req/s and host CPU from 99% to 20%. The lesson is to
scale a Python model server with worker processes, not intra-op threads.

**Can you claim zero-downtime deployment?**
No, and I do not. What I measured is 1,119 requests across two alias switches on one
machine with 0 failures, with 14 and 18 requests confirmed in flight during the switches.
Zero-downtime is a claim about sustained production traffic across many switches,
restarts and partial failures, which I have not tested.

**Is it production-ready?**
No. No authentication anywhere, single-node SQLite, per-process Prometheus metrics,
explicitly-triggered rather than scheduled retraining, and a Docker image that was
written but never built. All eleven gaps are listed in §12 and a script scans every
document to ensure the phrase "production-ready" never appears unqualified.

## 14. Three CV versions

### A. Conservative

> **Production-Style ML Serving & Monitoring Platform** — Python, scikit-learn, MLflow,
> FastAPI, Prometheus, SQLite, pytest
>
> - Built an end-to-end ML pipeline with data-contract validation, reproducible training
>   and MLflow experiment tracking and model registry, serving a gradient-boosted
>   classifier (0.927 test ROC-AUC) through a FastAPI endpoint with request validation
>   generated from the training schema.
> - Implemented statistical drift detection (KS, chi-square, PSI, Wasserstein) and
>   evaluated it over 30 trials per scenario on five controlled drift scenarios,
>   measuring a 0% false-positive rate on undrifted data.
> - Wrote 418 tests covering data validation, API error handling, drift detection,
>   retraining acceptance and failure injection; all passing.

### B. Strong, fully supported

> **Production-Style ML Serving & Monitoring Platform** — Python, scikit-learn, MLflow,
> FastAPI, Prometheus, Docker, GitHub Actions, pytest
>
> - Built an end-to-end MLOps platform — contract-driven data validation, reproducible
>   training (byte-identical fingerprint across runs), MLflow tracking and registry with
>   alias-based promotion — serving a gradient-boosted classifier (**0.927 test ROC-AUC**
>   vs 0.909 linear baseline) at **256 req/s with 4.7 ms p95 and zero errors**.
> - Diagnosed an OpenMP thread fan-out in scikit-learn inference that was costing
>   **2.6–5.1× throughput**; capping intra-op threads raised throughput from 99 to
>   **256 req/s** and cut host CPU from **99% to 20%**.
> - Implemented KS/chi-square/PSI/Wasserstein drift detection and evaluated it over
>   **210 trials across 7 controlled scenarios**, achieving **100% detection on 4 of 5
>   drifted scenarios within one 2,000-record window at a 0% false-positive rate**, and
>   quantified the limitation that concept drift is undetectable without labels despite
>   being the most damaging scenario (**−0.287 ROC-AUC**).
> - Built a gated retraining workflow with six acceptance criteria that promoted a
>   **+0.029 ROC-AUC** candidate and rejected a **+0.000** one, and verified **rollback
>   through the running FastAPI service** — 1,119 requests across two alias switches,
>   **0 failed**, rollback in **0.082 s**.
> - **418 tests, 0 failures**, plus scripted verification that every documented number
>   appears in the result file it cites.

### C. Compact, two bullets

> - Built an end-to-end ML serving platform (scikit-learn, MLflow registry, FastAPI,
>   Prometheus) — **0.927 test ROC-AUC**, **256 req/s at 4.7 ms p95, zero errors** — and
>   diagnosed an OpenMP fan-out costing **2.6–5.1× throughput**.
> - Implemented KS/chi-square/PSI drift detection evaluated over **210 controlled
>   trials** (**100% detection on 4 of 5 scenarios, 0% false positives**) feeding a gated
>   retraining workflow with **rollback verified through the live service** (1,119 requests,
>   0 failed); **418 tests passing**.

---

## Evidence table

Only rows marked **Yes** are safe to put on a CV.

| CV Claim | Evidence File | Test/Benchmark | Exact Result | Reproduction Command | Safe to Mention? |
|---|---|---|---|---|---|
| Test ROC-AUC | `EVALUATION_RESULTS.csv` | Held-out evaluation | 0.926784 | `python scripts/train.py` | **Yes** |
| Test PR-AUC | `EVALUATION_RESULTS.csv` | Held-out evaluation | 0.823947 | `python scripts/train.py` | **Yes** |
| Test accuracy | `EVALUATION_RESULTS.csv` | Held-out evaluation | 0.872059 | `python scripts/train.py` | **Yes** |
| Beats linear baseline | `EVALUATION_RESULTS.csv` | Same test set | 0.909126 baseline | `python scripts/train.py` | **Yes** |
| Training time | `EVALUATION_RESULTS.csv` | Wall clock | 0.5821 s | `python scripts/train.py` | **Yes** |
| Throughput, single request | `SERVING_BENCHMARKS.csv` | HTTP load test | 256.36 req/s | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| p95 latency, single request | `SERVING_BENCHMARKS.csv` | HTTP load test | 4.731 ms | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| p99 latency, single request | `SERVING_BENCHMARKS.csv` | HTTP load test | 9.178 ms | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| Batched record throughput | `SERVING_BENCHMARKS.csv` | HTTP load test, batch 32 | 6564.7 records/s | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| Zero errors under load | `SERVING_BENCHMARKS.csv` | HTTP load test | error_rate 0.0 in all 14 rows | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| Throughput without thread pinning | `SERVING_BENCHMARKS.csv` | HTTP load test | 99.19 req/s | `python scripts/load_test.py --thread-pin 0` | **Yes** |
| Host CPU without pinning | `SERVING_BENCHMARKS.csv` | HTTP load test | 0.993 utilisation | `python scripts/load_test.py --thread-pin 0` | **Yes** |
| Cold start | `SERVING_BENCHMARKS.csv` | Process launch to ready | 1.797 s | `python scripts/load_test.py --thread-pin 1` | **Yes** |
| Drift false-positive rate | `DRIFT_RESULTS.csv` | 30 trials, `no_drift` | 0.0 | `python scripts/drift_experiment.py --trials 30` | **Yes** |
| Drift detection rate, large drift | `DRIFT_RESULTS.csv` | 30 trials | 1.0 | `python scripts/drift_experiment.py --trials 30` | **Yes** |
| Concept drift undetected | `DRIFT_RESULTS.csv` | 30 trials, `performance_drift` | 0.0 detection | `python scripts/drift_experiment.py --trials 30` | **Yes** |
| Concept drift performance cost | `DRIFT_RESULTS.csv` | Labelled evaluation | −0.286944 ROC-AUC | `python scripts/drift_experiment.py --trials 30` | **Yes** |
| Detection latency | `DRIFT_RESULTS.csv` | Sequential windows | 1.0 window | `python scripts/drift_experiment.py --latency-trials 10` | **Yes** |
| Promotion of a better candidate | `results/retraining/retraining_experiment.csv` | End-to-end experiment | +0.029106 ROC-AUC | `python scripts/retrain_experiment.py` | **Yes** |
| Rejection of a no-gain candidate | `results/retraining/retraining_experiment.csv` | End-to-end experiment | 0.0 delta, rejected | `python scripts/retrain_experiment.py` | **Yes** |
| Rollback through the live service | `results/serving/rollback_through_serving.json` | Live HTTP experiment | 0.08155 s, 18 in flight, 0 failed | `python scripts/rollback_through_serving.py` | **Yes** |
| No failed requests during switches | `results/serving/rollback_through_serving.json` | Live HTTP experiment | 1119 requests, 0 failed | `python scripts/rollback_through_serving.py` | **Yes** |
| Test suite | `results/tests/summary.json` | pytest | 418 passed, 0 failed | `python -m pytest -q` | **Yes** |
| Smoke test | `results/serving/smoke_test.json` | 24 HTTP checks | cold start 1.742 s, all passed | `python scripts/smoke_test.py` | **Yes** |
| Docker image size | — | — | **not measured** | — | **No** |
| Container start-up time | — | — | **not measured** | — | **No** |
| CI green on GitHub | — | — | **never executed on GitHub** | — | **No** |
| Zero-downtime deployment | — | — | **not tested as such** | — | **No** |
| Production readiness | — | — | **explicitly not claimed** | — | **No** |
| Cross-platform reproducibility | — | — | **one machine only** | — | **No** |
