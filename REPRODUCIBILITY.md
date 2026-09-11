# Reproducibility

## What is claimed

**Training is reproducible on one machine across clean virtual environments.** A fresh
virtualenv built from the pinned requirements produces a bit-identical model.

**What is not claimed:** reproducibility across operating systems or CPU architectures.
That is a much stronger property involving BLAS implementations, compiler flags and
floating-point ordering, and it has not been tested. Nothing here says "100%
reproducible".

## Verification

```bash
python scripts/verify/verify_clean_env_repro.py
```

Builds a new virtualenv from `requirements-dev.txt`, installs the package, trains, and
compares against the recorded baseline. Result:

| Property | Recorded | Clean environment | Match |
|---|---|---|---|
| Dataset version | `adult-ingest-2-975c90344d56` | `adult-ingest-2-975c90344d56` | ✅ |
| Split id | `bc50e85d5b57e879` | `bc50e85d5b57e879` | ✅ |
| Training fingerprint | `54122c6ae1215984557601e70e885b48…` | `54122c6ae1215984557601e70e885b48…` | ✅ |
| Metrics compared | 24 | 24 | ✅ all within 1e-9 |

Environment build time: 30.7 s.

## The four layers

### 1. Data identity

Raw files are pinned by SHA-256 and verified on every load. The **dataset version** is a
content hash over the file digests *and* the parsing rules, so changing either the bytes
or the cleaning logic produces a new id. A silently edited input cannot reuse the
previous id — asserted by `test_dataset_version_changes_when_the_bytes_change`.

```
adult.data  5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d
adult.test  a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05
```

### 2. Split identity

`split_id` is a hash of the actual partition contents, not just the seed, so a silent
repartition is detectable. Same seed → same split (`test_split_is_deterministic_for_a_fixed_seed`);
different seed → different `split_id`.

### 3. Model identity

`training_fingerprint` hashes the fitted model's **predictions on a fixed 256-row probe
set** at full float64 precision. Pickle bytes are the wrong thing to hash — they differ
between runs for reasons unrelated to the model (memory addresses, dict ordering, joblib
framing). Predictions are what actually have to be identical.

Four tests pin it: repeated training matches; a different hyperparameter differs; a
different split seed differs; and the fingerprint responds to prediction changes (the
negative control that stops it from being a constant).

#### The estimator seed is inert, and that is asserted

`random_state` reaches HistGradientBoosting in exactly two places: the internal split for
early stopping, and the bin-threshold subsample above 200k rows. This project sets
`early_stopping: false` and trains on 26k rows, so **neither applies** — the fit is
deterministic regardless of the seed. That is stronger than seeding, but it is asserted
rather than assumed, because enabling early stopping later would silently make the fit
seed-dependent.

Global seeds are still pinned (`PYTHONHASHSEED`, `random`, `numpy`) to cover library
code that reaches for a global RNG or builds a vocabulary from an unordered container.

### 4. Environment identity

| Mechanism | Purpose |
|---|---|
| `requirements.txt` / `requirements-dev.txt` | Exact pins, generated from the verified environment |
| `requirements-lock.txt` | Full transitive lock (97 packages) |
| `config_digest` | Hash of the resolved configuration, logged with every run |
| `code_version` | Content hash of `src/`, so code is identified even without git |
| `git_commit` | Short SHA with a `-dirty` suffix when the tree is modified |
| `environment` | Python version, platform, and 7 tracked library versions, per run |

## Reproducing every number

```bash
make setup                                    # pinned environment
make data                                     # checksum-verified dataset
python scripts/train.py                       # training metrics
python -m pytest -q                           # 418 tests
python scripts/smoke_test.py                  # 24 HTTP checks
python scripts/load_test.py --thread-pin 1 --duration 20 --warmup 5 \
       --concurrency 1 2 4 8 16 --batch-size 1 32
python scripts/load_test.py --thread-pin 0 --duration 20 --warmup 5 \
       --concurrency 1 8 --batch-size 1 32
python scripts/drift_experiment.py --trials 30 --latency-trials 10
python scripts/retrain_experiment.py
python scripts/rollback_through_serving.py
python scripts/collect_results.py
python scripts/verify/verify_claims.py
```

Or `make all`.

## What is deterministic and what is not

| Quantity | Deterministic | Why |
|---|---|---|
| Dataset version, split id | ✅ | Content hashes |
| Model predictions | ✅ | Verified across clean environments |
| Evaluation metrics | ✅ | 24 metrics identical to 1e-9 |
| Drift scenario windows | ✅ | Seeded generators; `test_scenarios_are_deterministic` |
| Drift detection verdicts | ✅ | `test_detection_is_deterministic` |
| Metrics sampling | ✅ | Seeded RNG |
| **Latency and throughput** | ❌ | Wall-clock measurements. Every row records host CPU utilisation and whether other work was running |
| Training wall time | ❌ | Same |
| MLflow run ids | ❌ | Generated per run by design |

Timing numbers are reproducible in *distribution*, not exactly. That is why the load
test records `host_cpu_utilisation` and `host_busy_before_run` with every row, and
refuses to present figures as clean when the machine is more than 35% busy with other
work.

## Known reproducibility risks

1. **Upstream data could change.** The UCI URLs are not immutable. The checksum turns
   that into a loud failure rather than a silent one.
2. **Library versions drift.** Pinned in three files; CI installs from the pins.
3. **Platform differences are untested.** macOS arm64 only. CI would test Linux x86 on
   Python 3.11 and 3.13, but has not run.
4. **BLAS threading affects timing, not results.** Predictions are identical either way;
   throughput differs by up to 5.1×.
