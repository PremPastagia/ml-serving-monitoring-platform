# System Design

## Pipeline

```
UCI raw files (SHA-256 pinned)
      │
      ▼
  ingest ─────────────► dataset_version = adult-ingest-2-975c90344d56
      │                 (hash of raw bytes + parsing rules)
      ▼
  validate ───────────► schema · types · ranges · categories · nulls
      │                 duplicates · target · univariate-AUC leakage
      ▼
  split ──────────────► train 26,048 / validation 6,513 / test 16,281
      │                 split_id = bc50e85d5b57e879, index-disjoint
      ▼
  feature pipeline ───► 4 engineered features, fitted on train only
      │                 (one sklearn Pipeline = no train/serve skew)
      ▼
  train + evaluate ───► HistGradientBoosting + LogisticRegression baseline
      │                 training_fingerprint = hash of probe predictions
      ▼
  MLflow tracking ────► params · metrics · artefacts · dataset version
      │                 git commit · config digest · environment
      ▼
  model registry ─────► versions + aliases: production / previous / candidate
      │
      ├──────────────► FastAPI serving  ──► /health /ready /model-info
      │                (loads by alias)     /predict /metrics
      │                                     /monitoring/summary /admin/reload
      │                        │
      │                        ├──► Prometheus metrics (in-process)
      │                        └──► SQLite prediction store (durable)
      │                                     │
      ▼                                     ▼
  Docker image                        drift detection
  (authored, not built)               KS · chi-square · PSI · Wasserstein
                                      + prediction shape/level drift
                                            │
                                            ▼
                                      retraining decision
                                      (6 acceptance criteria)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                         promote alias              reject + tag
                         (previous kept)            (incumbent serves on)
                              │
                              ▼
                         rollback ──► verified through the running service
```

## Technology choices

Each row states what was chosen, what it was chosen over, and why. These are the
decisions an interviewer is most likely to probe.

| Concern | Chosen | Rejected | Reason |
|---|---|---|---|
| Model | scikit-learn `HistGradientBoostingClassifier` | XGBoost, LightGBM | Strongest sklearn tabular estimator with **native categorical support** and no extra native dependency. XGBoost/LightGBM add build weight and container size for no measurable gain at 26k rows. Measured: 0.9268 test ROC-AUC, 1.2 s to fit. |
| Baseline | `LogisticRegression` | none | A sanity floor. If the boosted model cannot beat a linear model, the pipeline is broken rather than the data. Measured gap: 0.9268 vs 0.9091. |
| Data frames | pandas | Polars | Polars is faster, but MLflow, scikit-learn and the whole surrounding ecosystem are pandas-first. At this data size the speed difference is irrelevant; the integration cost is not. |
| Validation | Hand-written, contract-derived | Great Expectations, pandera | The validator runs **inside the request path** as well as in CI, so a heavyweight engine is a poor fit. Every rule derives from `schema.py`, so the contract cannot drift away from the checks. Every rule is small enough to unit-test individually — 78 tests do. |
| Tracking + registry | MLflow with a **SQLite** backend | Weights & Biases, plain CSV | Needs to run offline with no account, and needs a real registry with versioning and aliases. MLflow is the only option that gives both locally. SQLite specifically because the plain file store **cannot host a model registry**. |
| Model promotion | Registry **aliases** | MLflow stages | Stages are deprecated. An alias is a mutable pointer to an immutable version, so promotion and rollback are a single atomic repoint rather than a multi-step transition that can be observed half-applied. |
| Serving | FastAPI + uvicorn | Flask, BentoML, Seldon | FastAPI gives Pydantic validation and OpenAPI for free. BentoML/Seldon hide too much behind conventions to explain in an interview — every layer here is one I can defend. |
| Request schema | Pydantic v2, **generated from the contract** | Hand-written models | A hand-written API model is a second copy of the contract, and the second copy always goes stale. A test asserts the generated fields equal the contract exactly. |
| Monitoring | `prometheus-client` + SQLite | statsd, Evidently dashboards | Prometheus holds aggregates; SQLite holds the individual scored vectors. Both are needed: a counter says *that* traffic changed, only retained rows say *which feature* changed. |
| Drift | scipy KS, chi-square, PSI, Wasserstein — implemented directly | Evidently | Evidently picks the test for you by cardinality and row count, so the most important question about this project ("why this test?") would have no answer. Its defaults also change between versions, which undermines reproducibility. Cost: fewer test types. |
| Storage | SQLite (WAL) | PostgreSQL | Must run from a clean checkout with no services to start — the property the reproducibility check depends on. Schema is plain SQL with no SQLite-only types, so moving to PostgreSQL is a connection string plus a driver. |
| Tests | pytest | unittest | Fixtures and parametrisation; 456 tests. |
| CI | GitHub Actions | none | Free tier, `ubuntu-latest`, no secrets, no paid infrastructure. |
| Container | Docker, multi-stage | none | **Authored and statically checked, never built** — no container runtime existed on the development machine. Image size and start-up time are recorded as UNVERIFIED. |
| Data versioning | Content hashing | DVC | DVC's value is large binary files in remote storage. The raw data here is 6 MB from a stable pinned URL; a content hash gives the same traceability with no extra tool. |

## Serving design decisions

**One sklearn `Pipeline` contains the preprocessing.** The object in the registry *is*
the transformation plus the model, so the API cannot apply a different transformation
from the one training used. There is no second code path to keep in sync — the most
common cause of train/serve skew is removed structurally rather than by discipline.

**Model swap by reference, not mutation.** `reload()` builds the replacement into a
local variable and swaps one reference under a lock only on success. A failed reload
leaves the previous model serving rather than leaving the service with nothing — the
failure mode that turns a bad promotion into an outage. Readers take the reference once
per request, so a reload mid-request cannot score one batch against two models.

**Load failure is a state, not a crash.** The process starts even with an empty
registry. `/health` returns 200 and reports `degraded`; `/ready` returns 503. A liveness
probe should not restart a process that is running correctly and merely has nothing to
serve, because a restart cannot fix an empty registry.

**Unknown request fields are rejected, not ignored.** Silently dropping an extra key is
how a client ships a renamed field into production and nobody notices.

**Intra-op threads are pinned to 1.** Measured, not assumed: scikit-learn's
HistGradientBoosting predicts through OpenMP and by default fans a *single-row* request
across every core. Capping the pools took throughput from 99 to 256 requests/second and
host CPU from 99% to 20%. See [BENCHMARKS.md](BENCHMARKS.md#thread-pinning). Scale with
worker processes, not intra-op threads.

## Storage

| Store | Technology | Holds |
|---|---|---|
| Experiment tracking | SQLite (`mlflow.db`) | Runs, params, metrics, tags |
| Artefacts | Local filesystem (`mlruns/`) | Serialised pipelines, metadata JSON |
| Model registry | SQLite (`mlflow.db`) | Versions, aliases, version tags |
| Prediction log | SQLite (`artifacts/predictions.sqlite`, WAL) | Scored feature vectors, events, drift reports |
| Metrics | In-process Prometheus registry | Counters, histograms, gauges |

`synchronous=NORMAL` on the prediction store: monitoring rows are observability, not
the system of record. Losing the last few on a hard crash is acceptable; adding an
fsync to every prediction is not.

## Known gaps

These are the reasons this is **not production-ready**:

1. **No authentication or authorisation anywhere.** `/admin/reload` in particular is an
   unauthenticated endpoint that swaps the served model. A deployment would put it
   behind auth or move it off the public listener.
2. **Single node.** SQLite for tracking, registry and prediction logging. Concurrent
   writers across processes are not supported at any scale.
3. **Prometheus metrics are per-process.** With `--workers > 1` each worker keeps its
   own registry, so a scrape sees one worker's view. Multi-process exposition needs
   `prometheus_client`'s multiprocess mode, which is not configured.
4. **Retraining is explicitly triggered**, by `scripts/retrain_experiment.py` or a test.
   There is no scheduler and no automatic production trigger.
5. **Docker is unbuilt.** The image and compose file are reviewed artefacts, not tested
   ones.
6. **In-memory rate limiting, request quotas and backpressure are absent.** The only
   bound is `serving.max_batch_size`.
7. **No fairness assessment**, despite strong demographic associations in the data.
8. **Reproducibility is verified on one machine**, across clean virtual environments —
   not across operating systems or CPU architectures.
