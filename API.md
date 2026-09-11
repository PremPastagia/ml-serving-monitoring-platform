# Serving API

Interactive docs at `http://127.0.0.1:8077/docs` once running; the OpenAPI schema is
generated from the same contract the model was trained against.

```bash
python scripts/serve.py            # thread-pinned by default; see BENCHMARKS.md
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness plus whether a model is loaded |
| GET | `/ready` | Readiness — 200 only when a prediction could succeed |
| GET | `/model-info` | Full provenance of the model currently served |
| POST | `/predict` | Score a batch of records |
| GET | `/metrics` | Prometheus exposition |
| GET | `/monitoring/summary` | Recent traffic and prediction distribution |
| POST | `/admin/reload` | Re-resolve the registry alias and hot-swap the model |

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_version": "1",
 "uptime_seconds": 12.44, "version": "0.1.0"}
```

Returns **200 even when degraded**. `status` is `degraded` rather than a non-200 when
no model is loaded, because a liveness probe should not restart a process that is
running correctly and merely has nothing to serve — a restart cannot fix an empty
registry. `/ready` is the probe that should gate traffic.

### `GET /ready`

200 with `{"status": "ready", "model_version": "1"}`, or **503** with the standard
error envelope when no model is loaded.

### `GET /model-info`

```json
{"model_name": "adult-income-classifier", "model_version": "1",
 "model_alias": "production", "model_source": "registry",
 "run_id": "...", "dataset_version": "adult-ingest-2-975c90344d56",
 "code_version": "6c47707c67e8", "git_commit": "b209d19",
 "training_fingerprint": "54122c6ae121...", "loaded_at": "...",
 "load_seconds": 0.018,
 "input_columns": ["age", "workclass", ...],
 "model_features": ["age", ..., "net_capital", "hours_band"],
 "metrics": {"validation_roc_auc": 0.93027, "test_roc_auc": 0.926784}}
```

This is the traceability endpoint: from a running service you can recover the exact
data, code and configuration that produced the model answering your requests.

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8077/predict \
  -H 'content-type: application/json' \
  -d '{"records": [{"age": 39, "workclass": "State-gov", "education_num": 13,
                    "marital_status": "Never-married", "occupation": "Adm-clerical",
                    "relationship": "Not-in-family", "race": "White", "sex": "Male",
                    "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
                    "native_country": "United-States"}]}'
```

```json
{"request_id": "72dab119-...", "model_name": "adult-income-classifier",
 "model_version": "1", "model_alias": "production", "threshold": 0.5,
 "n_records": 1,
 "predictions": [{"probability": 0.0012644, "prediction": 0, "label": "<=50K"}],
 "latency_ms": 3.68}
```

Batch of one is a batch. Maximum batch size is `serving.max_batch_size` (512); above
that the request is rejected with 413 rather than silently truncated.

Response headers: `X-Request-ID`, `X-Model-Version`.

### `GET /metrics`

Prometheus text exposition. Series are listed in [MONITORING.md](MONITORING.md).

### `GET /monitoring/summary`

```json
{"enabled": true, "n_predictions": 306, "mean_probability": 0.2335,
 "min_probability": 0.0012, "max_probability": 0.9876, "mean_latency_ms": 3.68,
 "by_model_version": {"1": 279, "2": 27},
 "events": {"validation_error": 4}, "served_model_version": "1"}
```

Optional `?window_seconds=`.

### `POST /admin/reload`

```json
{"ok": true, "from_version": "2", "to_version": "1", "changed": true,
 "seconds": 0.1449, "error": null}
```

Makes a promotion or rollback visible to the running service without a restart. A
failed reload returns 503 and **leaves the previous model serving** — it never empties
the service.

**Unauthenticated.** A deployment would put this behind authentication or move it off
the public listener. Recorded as a known gap in
[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md#known-gaps).

## Request validation

The request model is **generated from `mlserve.data.schema`**, not hand-written. A
hand-written model is a second copy of the contract, and the second copy always goes
stale. `test_api_request_model_matches_the_contract_exactly` asserts the generated
fields equal the contract.

| Rule | Behaviour |
|---|---|
| Missing field | 422 |
| Unknown field | **422 — rejected, not ignored.** Silently dropping an extra key is how a renamed field reaches production unnoticed |
| Wrong type | 422 |
| Fractional value in an integer field | 422 — accepting 39.7 as an age would change the model's binning |
| Out of contract range | 422 |
| Unknown categorical level | 422 |
| `__missing__` on a nullable field | accepted |
| `__missing__` on a non-nullable field | 422 |
| Empty `records` list | 422 |
| Batch above the maximum | 413 |

## Error contract

Every non-2xx response has this shape:

```json
{"request_id": "201dbe08-...",
 "error": {"type": "validation_error",
           "message": "the request payload violates the data contract (1 error(s))",
           "details": [{"location": "body.records.0.age",
                        "message": "Input should be less than or equal to 90",
                        "type": "less_than_equal"}]}}
```

| `error.type` | Status | Meaning |
|---|---|---|
| `validation_error` | 422 | Payload violates the data contract |
| `payload_too_large` | 413 | Batch above `serving.max_batch_size` |
| `model_not_loaded` | 503 | No model available |
| `internal_error` | 500 | Unexpected failure; logged in full, **never leaked to the caller** |

## Request tracing

Every request gets a UUID, or reuses a caller-supplied `X-Request-ID`. It appears in
the response body, the response header, every structured log line for that request, and
the row written to the prediction store — so one id joins a slow request in the logs to
its features in the database.

Logs are JSON lines:

```json
{"ts": "2026-09-11T09:49:58.308034+00:00", "level": "INFO", "logger": "mlserve.api",
 "message": "prediction served", "request_id": "72dab119-...", "endpoint": "/predict",
 "n_records": 1, "model_version": "1", "latency_ms": 3.68}
```

## Configuration

| Setting | Default | Override |
|---|---|---|
| `serving.model_source` | `registry` | `MLSERVE_MODEL_SOURCE` (`registry` \| `file`) |
| `serving.model_alias` | `production` | `MLSERVE_MODEL_ALIAS` |
| `serving.max_batch_size` | 512 | config |
| `serving.port` | 8077 | `--port` |
| intra-op threads | 1 | `--threads` |
| config file | `configs/config.yaml` | `MLSERVE_CONFIG` |

Precedence is explicit argument → environment → configuration. An invalid
`model_source` is rejected at construction rather than at first request.

## Tests

100 tests cover this surface: 26 in `tests/test_api.py` (happy paths, concurrency,
request ids, reload, OpenAPI) and 74 in `tests/test_api_errors.py` (every malformed
payload class, boundary values as positive controls, model-unavailable paths).

```bash
python -m pytest tests/test_api.py tests/test_api_errors.py -q
python scripts/smoke_test.py     # 24 checks against the real running server
```
