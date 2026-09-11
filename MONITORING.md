# Monitoring and Observability

Two stores, because they answer different questions. Prometheus holds **aggregates** —
a counter tells you *that* traffic changed. SQLite holds the **individual scored feature
vectors** — only retained rows let you re-run a two-sample test and say *which feature*
changed and by how much. Drift detection is a retrospective query over the SQLite
table, so that table is the thing that has to exist.

## Metric series

| Series | Type | Question it answers |
|---|---|---|
| `mlserve_requests_total` | Counter | Is traffic arriving, on which endpoint, with what status? |
| `mlserve_errors_total` | Counter | What fraction of requests fail, and why? |
| `mlserve_request_latency_seconds` | Histogram | Are p50/p95/p99 within budget? (quantiles need a histogram, not a gauge of the last value) |
| `mlserve_predictions_total` | Counter | How many *records* — not requests — were scored? |
| `mlserve_prediction_score` | Histogram | Has the output distribution shifted? The earliest drift signal available without labels |
| `mlserve_predicted_class_total` | Counter | Has the positive rate moved? |
| `mlserve_feature_<name>` | Histogram | Has an input numeric distribution shifted? One per numeric feature |
| `mlserve_feature_level_total` | Counter | Has a categorical level's share shifted? |
| `mlserve_model_info` | Gauge | Which model version served this traffic? Labels carry the identity; the value is a constant 1 |
| `mlserve_model_loaded` | Gauge | 1 when able to serve |
| `mlserve_model_loads_total` | Counter | Load attempts by outcome |
| `mlserve_model_load_seconds` | Gauge | Cold-start cost |
| `mlserve_uptime_seconds` | Gauge | |
| `mlserve_resource_cpu_percent` | Gauge | Process CPU, sampled on scrape |
| `mlserve_resource_memory_bytes` | Gauge | Process RSS |

Error rate is computable from the exposed series alone — no derived metric needed:

```promql
sum(rate(mlserve_requests_total{status=~"4..|5.."}[5m]))
  / sum(rate(mlserve_requests_total[5m]))
```

## Design decisions

**One histogram per numeric feature, not one histogram with a `feature` label.**
`prometheus_client` fixes bucket edges per metric, and `age` (17–90) and `capital_gain`
(0–99,999) cannot share edges. Bucket boundaries come from the contract, so an edge
cannot drift away from the values it separates.

**Cardinality control.** `mlserve_feature_level_total` is restricted to four
low-cardinality categoricals — `sex`, `race`, `workclass`, `relationship`, each with at
most 8 levels. `native_country` (41 levels) and `occupation` (14) are tracked in SQLite
only. Exporting every level of every categorical as its own time series is how a metrics
backend falls over. `test_high_cardinality_categoricals_are_not_exported` enforces this.

**`mlserve_model_info` is cleared before being set.** After a promotion or rollback
exactly one series is present; leaving the old labels behind would make a dashboard
show two "current" versions at once.

**Resource gauges are sampled at scrape time**, not per request — polling CPU on every
request would cost more than the prediction it measures.

**An explicit registry, not the process-global default.** A test can build a fresh
`ServingMetrics` without duplicate-timeseries errors, which is the usual reason metrics
end up untested.

## Sampling and storage

| | |
|---|---|
| Request, error, latency, prediction counts | Exact, never sampled |
| Per-feature distribution series | Sampled at `serving.feature_sample_rate` (1.0 locally) |
| Sampling determinism | Seeded RNG — the same seed samples the same rows |
| Durable store | SQLite WAL, `artifacts/predictions.sqlite` |
| Tables | `predictions` (features as real columns), `events`, `drift_reports` |
| Durability | `synchronous=NORMAL` — monitoring is observability, not the system of record. Losing the last few rows on a hard crash is acceptable; an fsync per prediction is not |

Feature columns are materialised as real SQL columns rather than a JSON blob, so a
drift query is a single scan instead of 100k JSON parses.

## Alert thresholds

Starting points, to be tuned against a real traffic baseline:

| Signal | Warn | Alert | Rationale |
|---|---|---|---|
| Error rate (5 min) | > 1% | > 5% | Above the observed 0% under valid load |
| p95 latency, batch 1 | > 15 ms | > 50 ms | Measured p95 is 4.73 ms; warn at ~3×, alert at the retraining budget |
| p99 latency, batch 1 | > 30 ms | > 100 ms | Measured p99 is 9.18 ms |
| `mlserve_model_loaded` | — | == 0 | Cannot serve |
| Model load failures | ≥ 1 | ≥ 3 in 10 min | A failing reload means promotion is not reaching the service |
| Prediction mean shift vs baseline | ≥ 10% | ≥ 25% | The level threshold measured in [DRIFT_DETECTION.md](DRIFT_DETECTION.md) |
| Feature PSI | ≥ 0.10 | ≥ 0.25 | Conventional operating points, calibrated against a measured 0% false-positive rate |
| Traffic volume | — | 0 for 5 min | Silent failure upstream |

Latency thresholds assume the pinned-thread configuration. Unpinned, measured p99 at
concurrency 8 is 196 ms and every threshold above is meaningless — which is itself a
reason the configuration is fixed in code rather than left to the operator.

## Verification

40 tests in `tests/test_monitoring.py`, all passing:

| Scenario | What is asserted |
|---|---|
| Every required series present | 11 series, plus one histogram per numeric feature |
| Normal traffic | Counters move by exactly the number of records scored |
| Records vs requests | A 7-record batch increments predictions by 7 and requests by 1 |
| Invalid traffic | Errors increment, predictions do **not**; no row written to the store |
| Error rate computable | 3 good + 2 bad requests produce exactly 0.4 |
| High traffic | 40 requests of varying batch size accounted for exactly, in both stores |
| Distribution shift | 20 records at age 20 land in the `le="20.0"` bucket with sum 400 |
| Model errors | `error_type="inference_error"` appears |
| Degraded service | `mlserve_model_loaded` is 0; `model_not_loaded` errors counted |
| Sampling | Rate 0.0 samples nothing, 1.0 samples everything, 0.5 is deterministic for a fixed seed |
| Durable store | Every scored record persisted with all 12 features and the model version |

```bash
python -m pytest tests/test_monitoring.py -q
curl -s http://127.0.0.1:8077/metrics | grep mlserve_
```

## Not implemented

- **Grafana dashboards.** The metrics are Prometheus-format and would feed one, but no
  dashboard is built and none is claimed.
- **Multi-process exposition.** With `--workers > 1` each worker keeps its own registry,
  so a scrape sees one worker's view. Production would need `prometheus_client`'s
  multiprocess mode.
- **Alertmanager rules.** The thresholds above are documented, not deployed.
