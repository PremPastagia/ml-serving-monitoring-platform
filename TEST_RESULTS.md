# Test Results

All figures produced by a single run of the full suite. Reproduce with:

```bash
OMP_NUM_THREADS=1 python -m pytest -q --durations=15 --junitxml=results/tests/junit.xml
```

## Summary

| | |
|---|---|
| Tests | **418** |
| Passed | **418** |
| Failed | **0** |
| Errors | **0** |
| Skipped | **0** |
| Pass rate | **100%** |
| Wall clock | **132.5 s** (56.7 s observed on an otherwise idle machine) |

Raw output: `results/tests/junit.xml`, `results/tests/pytest_full.log`,
`results/tests/summary.json`.

## Environment

| | |
|---|---|
| Python | 3.13.9 |
| Platform | macOS 27.0, arm64 (Apple Silicon), 10 CPUs |
| numpy / pandas / scikit-learn | 2.5.3 / 2.3.3 / 1.9.1 |
| scipy / MLflow / FastAPI / Pydantic | 1.18.1 / 3.6.0 / 0.141.1 / 2.13.5 |
| Dataset version | `adult-ingest-2-975c90344d56` |
| Split id | `bc50e85d5b57e879` |
| `OMP_NUM_THREADS` | 1 |

## Breakdown by area

| File | Tests | Seconds | Covers |
|---|---:|---:|---|
| `test_validation.py` | 78 | 0.33 | Every corruption class, each paired with a positive control |
| `test_api_errors.py` | 74 | 0.76 | Missing / extra / wrong-type / out-of-range / unknown-level / empty / oversized payloads, model unavailable |
| `test_drift.py` | 54 | 0.64 | PSI, KS, chi-square, prediction drift, scenarios, negative controls |
| `test_monitoring.py` | 40 | 0.63 | Metric series, counters under normal / invalid / high traffic, sampling, durable store |
| `test_failure_injection.py` | 30 | 1.00 | Missing and corrupt artefacts, failed reload, tampered data, store failures, config errors |
| `test_retraining.py` | 28 | 117.97 | All 6 acceptance criteria, promote / reject / rollback / training failure end to end |
| `test_api.py` | 26 | 2.61 | Health, readiness, model-info, predict, request ids, concurrency, reload, OpenAPI |
| `test_registry.py` | 23 | 4.41 | Registration, versioning, aliases, promotion, rollback, error paths |
| `test_features.py` | 21 | 0.14 | Derived features, statelessness, train-only fitting, unseen categories |
| `test_training.py` | 18 | 2.73 | Reproducibility, provenance completeness, fit quality, failure surface |
| `test_split.py` | 16 | 0.35 | Determinism, stratification, disjointness, contamination measurement |
| `test_schema_contract.py` | 10 | 0.00 | The contract is the single source of truth for the API and the pipeline |

`test_retraining.py` dominates the runtime because each end-to-end case trains real
models against a scratch MLflow registry. That is deliberate: mocking it would remove
the only evidence that the promotion gate actually works.

## Coverage of the required test categories

| Required | Where | Status |
|---|---|---|
| Unit tests | all files | 418 passed |
| Integration tests | `test_api.py`, `test_registry.py`, `test_retraining.py` | passed |
| API tests | `test_api.py`, `test_api_errors.py` (100) | passed |
| Data validation tests | `test_validation.py` (78) | passed |
| Training reproducibility | `test_training.py` | passed — identical fingerprint across runs |
| Docker tests | — | **not run**: no container runtime available (see [CI_CD.md](CI_CD.md)) |
| CI tests | `.github/workflows/ci.yml` | **authored, not executed on GitHub** — every step verified locally |
| Load tests | `scripts/load_test.py` | passed — see [LOAD_TESTING.md](LOAD_TESTING.md) |
| Drift detection tests | `test_drift.py` (54) + `scripts/drift_experiment.py` | passed |
| Retraining tests | `test_retraining.py` (28) + `scripts/retrain_experiment.py` | passed |
| Rollback tests | `test_registry.py`, `scripts/rollback_through_serving.py` | passed — verified through the live service |
| Failure injection | `test_failure_injection.py` (30) | passed |

## Smoke test

`python scripts/smoke_test.py` starts the packaged application, exercises every
endpoint over real HTTP and shuts it down. **24 of 24 checks passed**; cold start
1.742 s. Output: `results/serving/smoke_test.json`.

## Notable behaviours pinned by tests

These exist because they were surprising while building, and a test is the only thing
that keeps the documentation honest if the code changes:

- The estimator seed is **inert** for this configuration (`early_stopping: false`,
  26k rows), so the fit is deterministic regardless of it. The *split* seed is the one
  that matters. — `test_training.py`
- Input-distribution drift detection **cannot see concept drift**. —
  `test_drift.py::test_input_drift_does_not_see_concept_drift`
- PSI on `hours_per_week` was unstable under quantile binning because 46.7% of records
  tie at exactly 40. — `test_drift.py::test_psi_on_a_tied_feature_is_stable_across_reference_subsamples`
- A failed model reload must leave the previous model serving. —
  `test_failure_injection.py::test_a_failed_reload_keeps_the_previous_model`
