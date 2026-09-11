# Failure Analysis

Every failure mode below is injected by a test in `tests/test_failure_injection.py`
(30 tests, all passing) or by an error-path test elsewhere. Nothing here is
hypothetical.

```bash
python -m pytest tests/test_failure_injection.py -q
```

## Model lifecycle

| Injected failure | Defined behaviour | Test |
|---|---|---|
| Model bundle missing | `FileNotFoundError` naming the path; loader stays empty | `test_missing_model_bundle_raises_a_clear_error` |
| Model artefact corrupt (not a pickle) | Raises; loader reports the error; does not half-load | `test_corrupt_model_artifact_is_reported_not_swallowed` |
| Registry empty or unreachable at startup | Process **boots degraded**; `try_load` returns `None` and never raises | `test_unavailable_registry_leaves_the_loader_empty` |
| Prediction attempted with no model | `ModelNotLoadedError` → 503 | `test_requiring_a_model_that_is_absent_raises_the_typed_error` |
| **Reload fails after a good model is serving** | **Previous model keeps serving**; `/admin/reload` returns 503 | `test_a_failed_reload_keeps_the_previous_model` |
| Bundle metadata corrupt but model fine | Model loads; provenance fields are `None` | `test_bundle_with_unreadable_metadata_still_loads_the_model` |

The reload case is the most important one here. `reload()` builds the replacement into
a local variable and swaps a single reference under a lock **only on success**. A failed
reload therefore degrades to "nothing changed" rather than to "no model" — which is the
failure mode that turns a bad promotion into an outage.

## Serving

| Injected failure | Defined behaviour | Test |
|---|---|---|
| Model raises during inference | **500 with `internal_error`**; the exception text is logged but **never leaked to the caller**; no traceback in the response | `test_model_that_raises_at_predict_returns_500_not_a_traceback` |
| Same, from an operator's view | `mlserve_errors_total{error_type="inference_error"}` increments | `test_inference_failure_is_recorded_for_operators` |
| Sustained malformed-payload storm (140 bad requests, 7 shapes) | Every one rejected 413/422; the healthy path still returns 200; `/health` still `ok` | `test_service_survives_a_bad_payload_storm` |
| Model unloaded mid-life, then restored | 503 while absent, 200 after restoration — no restart needed | `test_service_recovers_after_the_model_is_restored` |
| Any of the above | `/health` and `/metrics` stay 200 throughout | `test_health_stays_200_through_every_failure` |

## Data integrity

| Injected failure | Defined behaviour | Test |
|---|---|---|
| A raw file is appended to | `ChecksumMismatch` naming the observed and pinned digests | `test_a_tampered_raw_file_is_detected` |
| A raw file is missing | `FileNotFoundError` **naming the remedy** (`scripts/fetch_data.py`) | `test_a_missing_raw_file_is_reported_with_a_remedy` |
| Non-strict mode on altered bytes | Returns the observed digest rather than silently passing | `test_non_strict_mode_still_reports_the_observed_digest` |
| Bytes change | **The dataset version id changes**, so a silently edited input cannot reuse the previous id | `test_dataset_version_changes_when_the_bytes_change` |

## Monitoring store

| Injected failure | Defined behaviour | Test |
|---|---|---|
| Store path unwritable | Raises at construction, not at first write | `test_store_on_an_unwritable_path_fails_at_construction` |
| SQL-injection-shaped table name | `ValueError: unknown table` — the table name is allow-listed, never interpolated | `test_store_rejects_an_unknown_table` |
| Mismatched batch lengths | `ValueError` (via `zip(strict=True)`) rather than a silently truncated write | `test_store_handles_a_batch_length_mismatch` |
| Reading an empty store | Empty DataFrame, no exception | `test_reading_from_an_empty_store_returns_an_empty_frame` |
| Summary of an empty store | Zeros and `None`s, no division by zero | `test_summary_of_an_empty_store_does_not_divide_by_zero` |

## Configuration

| Injected failure | Defined behaviour | Test |
|---|---|---|
| Missing config key | `KeyError` naming the dotted path | `test_missing_config_key_raises_a_named_error` |
| Missing config file | `FileNotFoundError` | `test_missing_config_file_raises` |
| Config content changes | `config.digest` changes, so a run's settings are traceable | `test_config_digest_changes_with_content` |
| `git` absent or failing | `git_commit()` returns `"unavailable"` — provenance degrades, the run does not break | `test_git_commit_never_raises` |
| A tracked package is absent | Recorded as `"absent"` rather than raising | `test_environment_info_tolerates_absent_packages` |
| Invalid `MLSERVE_MODEL_SOURCE` | `ValueError` at construction, not at first request | `test_an_invalid_model_source_is_rejected_at_construction` |

## Retraining

| Injected failure | Defined behaviour | Test |
|---|---|---|
| Training raises mid-cycle | Caught, recorded as `failed_criteria: ["retraining_failed"]`, **production untouched** | `test_a_failed_retraining_is_reported_and_promotes_nothing` |
| Candidate is worse | Rejected on `max_allowed_degradation` | `test_a_worse_candidate_is_rejected` |
| Candidate is barely better | Rejected on `min_absolute_improvement` — no promotion on noise | `test_a_marginally_better_candidate_is_rejected` |
| Candidate is much slower | Rejected on `max_latency_ratio` | `test_a_candidate_much_slower_than_the_incumbent_is_rejected` |
| Training data failed validation | Rejected on `require_clean_validation` | `test_dirty_training_data_blocks_promotion` |
| Too few training rows | Rejected on `min_training_rows` | `test_too_little_training_data_blocks_promotion` |
| Incumbent already decayed | Absolute floor still applies | `test_the_absolute_floor_applies_even_when_beating_the_incumbent` |
| A busy host inflates both latencies | **Does not reject** — the ratio cancels it | `test_a_slow_host_does_not_reject_a_candidate_that_matches_the_incumbent` |

## Real defects found and fixed while building

These were found by investigating results that did not make sense, and each is now
pinned by a regression test.

| Defect | Symptom | Fix |
|---|---|---|
| **PSI unstable on tied features** | The same +2-hour shift scored PSI 0.0021 against the full reference and 2.2587 against a subsample of it, because `hours_per_week` ties 46.7% of records at exactly 40 and the two quantile binnings placed an edge differently | Detect columns whose quantile edges collapse and use value-based bins. Documented in [DRIFT_DETECTION.md](DRIFT_DETECTION.md) |
| **Absolute latency budget rejected a better model** | A candidate 2.4 ROC-AUC points better was rejected for exceeding 50 ms — the host was oversubscribed and every measurement was inflated | Gate on a **ratio** against the incumbent |
| **Sequential latency measurement did not cancel contention** | A measured 10.5× ratio (216 ms vs 21 ms) between models whose real cost differs by under 2× | **Interleave** the two models' requests, and take the ratio on the median |
| **Load-average guard flagged every benchmark as unusable** | macOS load average read 30–40 on a machine measurement showed was ~20% busy | Gate on measured **CPU utilisation** |
| **Benchmark could measure a stale server** | A leftover process held the port; the new uvicorn exited immediately while `/health` still answered from the old one | Refuse to start on a busy port |
| **Rollback continuity evidence was empty** | "0 of 0 requests failed" — the switch was faster than the serial probe's request interval | Four concurrent probes, and **require** in-flight traffic or fail |
| **Prior shift was undetectable** | Score PSI reached only 0.16 while the mean score moved +59% | Added a **level** test alongside the shape test |

## Not covered

- Network partitions, disk-full conditions, OOM kills.
- Concurrent writers to the SQLite stores from multiple processes.
- Byzantine model artefacts (a valid pickle that computes something wrong).
- Adversarial or malicious input beyond schema violations.
