# Benchmarks

Every number here comes from a script in this repository. Nothing is estimated.

## Measurement environment

| | |
|---|---|
| Machine | Apple Silicon (arm64), 10 CPUs, macOS 27.0 |
| Python | 3.13.9 |
| Dataset version | `adult-ingest-2-975c90344d56` |
| Model | registry version 1, HistGradientBoosting, 200 trees |
| Server | uvicorn, 1 worker, `OMP_NUM_THREADS=1` |
| Host CPU used by other work at measurement time | 12–15% |

The load-test harness **measures CPU utilisation, not load average**, and refuses to
present latency figures as clean when more than 35% of the machine is already busy.
That distinction was not cosmetic: on macOS the load average counts I/O-blocked threads
and read 30–40 on a machine that was in fact ~20% busy, which made an earlier
load-average guard flag every run as unusable. Every row in `SERVING_BENCHMARKS.csv`
carries `host_cpu_utilisation` and `host_busy_before_run`.

## Training

| Metric | Value | Source |
|---|---|---|
| HistGradientBoosting fit time | **0.582 s** | `EVALUATION_RESULTS.csv` |
| LogisticRegression fit time | **0.140 s** | `EVALUATION_RESULTS.csv` |
| Full pipeline (validate → split → train both → evaluate → track → register) | ~12 s | `scripts/train.py` |
| Training rows | 26,048 | |
| Model artefact | one joblib pipeline including preprocessing | |

Reproduce: `python scripts/train.py`

## Model quality

| Split | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| train | 0.956365 | 0.886742 | 0.897036 | 0.833055 | 0.715925 | 0.770062 | 0.071346 |
| validation | 0.930270 | 0.831673 | 0.873791 | 0.773862 | 0.672194 | 0.719454 | 0.087238 |
| **test (held out)** | **0.926784** | **0.823947** | **0.872059** | 0.766556 | 0.659126 | 0.708794 | 0.088495 |

Linear baseline on the same test set: ROC-AUC **0.909126**, PR-AUC 0.773880.
Train→validation ROC-AUC gap: **0.026**.

## Serving throughput and latency

Production configuration — 1 uvicorn worker, intra-op threads pinned to 1, 20 s
measured per point after a 5 s warm-up, payloads drawn from the held-out test split
with a fixed seed.

| Concurrency | Batch | Requests/s | Records/s | p50 ms | p95 ms | p99 ms | Errors | RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **256.4** | 256.4 | **3.68** | **4.73** | **9.18** | 0 | 282 |
| 2 | 1 | 247.6 | 247.6 | 7.80 | 9.16 | 13.69 | 0 | 280 |
| 4 | 1 | 247.6 | 247.6 | 15.85 | 23.03 | 25.92 | 0 | 275 |
| 8 | 1 | 242.4 | 242.4 | 32.61 | 46.79 | 53.57 | 0 | 273 |
| 16 | 1 | 236.5 | 236.5 | 66.63 | 92.97 | 111.49 | 0 | 272 |
| 1 | 32 | 205.2 | **6,564.7** | 4.67 | 5.58 | 10.16 | 0 | 272 |
| 2 | 32 | 196.2 | 6,276.8 | 9.98 | 11.97 | 16.49 | 0 | 271 |
| 4 | 32 | 192.7 | 6,166.4 | 20.36 | 27.13 | 31.32 | 0 | 271 |
| 8 | 32 | 192.7 | 6,166.1 | 40.92 | 60.89 | 71.65 | 0 | 271 |
| 16 | 32 | 193.8 | 6,201.6 | 82.15 | 136.05 | 149.86 | 0 | 271 |

**Zero errors across every configuration.** Request throughput is flat from
concurrency 1 to 16 (256 → 236 req/s) while latency scales linearly — the signature of
a saturated single worker with a stable service time, which is the expected and correct
behaviour.

Reproduce:
```bash
python scripts/load_test.py --thread-pin 1 --duration 20 --warmup 5 \
       --concurrency 1 2 4 8 16 --batch-size 1 32
```

## Thread pinning

The single largest performance finding in this project, and it was found by
investigating a benchmark result that did not make sense: throughput was *falling* as
concurrency rose, and the whole 10-core machine showed 99–100% CPU for one server
process.

Cause: scikit-learn's `HistGradientBoostingClassifier` predicts through OpenMP, which
by default fans work across every core — including for a single-row request, where the
fan-out costs far more than the arithmetic. Eight concurrent requests each tried to use
ten cores.

Same machine, same model, same payloads, back to back:

| Configuration | Requests/s | p50 ms | p95 ms | p99 ms | Host CPU |
|---|---:|---:|---:|---:|---:|
| Uncapped, concurrency 1 | 99.2 | 8.84 | 15.79 | 22.68 | **99.3%** |
| **Pinned to 1, concurrency 1** | **256.4** | **3.68** | **4.73** | **9.18** | **19.6%** |
| Uncapped, concurrency 8 | 47.4 | 164.35 | 188.54 | 196.16 | **99.8%** |
| **Pinned to 1, concurrency 8** | **242.4** | **32.61** | **46.79** | **53.57** | **25.3%** |
| Uncapped, concurrency 8, batch 32 | 26.9 (860 rec/s) | 314.01 | 361.10 | 512.88 | 100.0% |
| **Pinned to 1, concurrency 8, batch 32** | **192.7 (6,166 rec/s)** | **40.92** | **60.89** | **71.65** | **19.0%** |

| Effect | Measured |
|---|---|
| Throughput, concurrency 1 | **2.6×** (99.2 → 256.4 req/s) |
| Throughput, concurrency 8 | **5.1×** (47.4 → 242.4 req/s) |
| Record throughput, concurrency 8 batch 32 | **7.2×** (860 → 6,166 rec/s) |
| p99 latency, concurrency 8 | **3.7× lower** (196.2 → 53.6 ms) |
| Host CPU, concurrency 8 | **99.8% → 25.3%** |

Uncapped, adding client concurrency made throughput *worse*. Pinned, it holds flat.
The conclusion — scale a Python model server with worker **processes**, never with
intra-op threads — is now the default in `scripts/serve.py` and in the Dockerfile.

Reproduce the comparison: `python scripts/load_test.py --thread-pin 0 ...`

## Component latency (in-process, no HTTP)

Minimum-of-300 timings, which is the estimator most robust to scheduler noise:

| Stage | Min ms |
|---|---:|
| Derived features | 1.22 |
| Preprocessing (ColumnTransformer) | 2.19 |
| Model `predict_proba` (200 trees, 1 row) | 1.93 |
| **Full pipeline, 1 row** | **5.51** |
| Full pipeline, 512 rows | 16.80 (0.033 ms/row) |

Model inference is only ~35% of single-row cost; pandas and the ColumnTransformer
dominate. That is the right thing to optimise first if this ever needed to be faster.

## Cold start

| Metric | Value |
|---|---|
| Process start → `/health` reporting `model_loaded` | **1.742–1.797 s** |
| Model load from the registry alone | 0.018 s |

## Drift detection

| Metric | Value |
|---|---|
| Detection time, 12 features, 10,000 reference vs 2,000 current rows | **0.071 – 0.221 s** (mean by scenario) |
| Slowest scenario | `large_drift`, 0.221 s |
| Detection latency once drift begins | **1 window** (2,000 records) for every detected scenario |

## Retraining, promotion and rollback

| Operation | Measured |
|---|---|
| Retraining a candidate on 26,048 rows | **1.08 – 1.25 s** |
| Promotion (registry alias repoint) | **0.0043 s** |
| Rollback (registry alias repoint) | **0.0055 s** |
| Promotion end to end through the live service (alias + reload + verify) | **0.063 s** |
| Rollback end to end through the live service | **0.082 s** |
| Model reload inside the running server | 0.029 – 0.140 s |
| Requests in flight during the promotion switch | 14, **0 failed** |
| Requests in flight during the rollback switch | 18, **0 failed** |
| Total probe requests across both switches | 1,119, **0 failed** |

## Docker — UNVERIFIED

| Metric | Status |
|---|---|
| Image build time | **not measured** |
| Image size | **not measured** |
| Container start-up time | **not measured** |

No container runtime (Docker, Podman, colima, OrbStack) was available on the
development machine. `docker/Dockerfile` and `docker/docker-compose.yml` were written
and statically checked but **never built or run**. The CI workflow contains a build job
that would produce these figures on GitHub's runners; it has not been executed. These
numbers must not be quoted anywhere.

## Files

| File | Contents |
|---|---|
| `EVALUATION_RESULTS.csv` | Per-model, per-split metrics with full provenance |
| `SERVING_BENCHMARKS.csv` | Every load-test configuration, pinned and unpinned |
| `DRIFT_RESULTS.csv` | Per-scenario detection rates and performance deltas |
| `results/serving/rollback_through_serving.json` | Switch timings and in-flight request counts |
| `results/retraining/retraining_experiment.csv` | Promotion decisions and timings |
| `results/tests/summary.json` | Test counts and durations |
| `results/summary.json` | Everything above, aggregated |
