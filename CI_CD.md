# CI / CD

## What exists, and what has actually run

**Be precise about this.** The workflow is authored and every one of its steps has been
executed locally on this machine. It has **not been executed on GitHub Actions**, and
the Docker job has never run anywhere because no container runtime was available during
development.

| Job | Authored | Verified locally | Run on GitHub |
|---|---|---|---|
| `lint` (ruff + black) | ✅ | ✅ | ❌ |
| `test` (Python 3.11 + 3.13 matrix) | ✅ | ✅ on 3.13 only | ❌ |
| `pipeline` (train → smoke → drift) | ✅ | ✅ | ❌ |
| `docker` (build + start + probe) | ✅ | ❌ **no runtime available** | ❌ |

This is **not** a deployment pipeline. Nothing is published, pushed to a registry, or
deployed. It builds, tests and verifies.

## Workflow

`.github/workflows/ci.yml`, four jobs on free `ubuntu-latest` runners, no secrets and no
paid infrastructure.

**`lint`** — `ruff check` and `black --check` over `src`, `tests`, `scripts`.

**`test`** — matrix over Python **3.11 and 3.13**. 3.11 is the floor declared in
`pyproject.toml`; testing it is what makes `requires-python` a claim rather than a
guess. Steps: install pinned dependencies, restore the dataset from cache, verify the
pinned checksums, run the data-validation tests separately (so a data-contract failure
is legible in the job list), then the full suite with a JUnit report uploaded as an
artefact.

**`pipeline`** — trains, registers, runs the 24-check HTTP smoke test against the real
server, then a reduced drift experiment (5 trials). This is the job that proves the
pieces compose, not just that they pass in isolation.

**`docker`** — builds the image with buildx and GitHub Actions layer caching, reports
its size, starts the container with no model bundle mounted, and asserts it comes up
**degraded rather than crashing**: `/health` answers 200 and `/ready` returns 503. That
is a deliberate negative test of the design decision in
[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md#serving-design-decisions).

### Reproducibility measures

- Dependencies pinned to exact versions in `requirements.txt` / `requirements-dev.txt`,
  generated from the verified development environment. A full transitive lock is in
  `requirements-lock.txt`.
- `PYTHONHASHSEED=42` and single-threaded BLAS/OpenMP, so runner-to-runner timing
  variation does not change results.
- The 6 MB raw dataset is cached by a key containing both file checksums, so CI stays
  off the UCI servers while still verifying the pinned digests on every run.
- `concurrency` cancels superseded runs.

## Locally verified CI steps

Every command in the workflow, run on this machine:

| Step | Result |
|---|---|
| `ruff check src tests scripts` | **pass**, 0 findings |
| `python scripts/fetch_data.py --verify` | **pass**, both checksums match |
| `pytest tests/test_validation.py tests/test_schema_contract.py` | **88 passed** |
| `pytest` (full suite) | **418 passed, 0 failed, 132.5 s** |
| `python scripts/train.py` | **pass**, registers a version |
| `python scripts/smoke_test.py` | **24 of 24 checks passed** |
| `python scripts/drift_experiment.py --trials 5` | **pass** |
| Docker build / start / probe | **not run** — no container runtime |

`black --check` is authored in the workflow but the repository is formatted to ruff's
rules; run `make format` before pushing if black reports differences.

## Docker

`docker/Dockerfile` — multi-stage, **written and statically checked, never built**.

| Decision | Reason |
|---|---|
| `python:3.13.9-slim-bookworm`, exact patch tag | Not `3.13-slim`, not `latest`. A floating tag means two builds of the same commit can produce different images, which would undermine every reproducibility claim here |
| Two stages | Compilers stay in the builder; only the built virtualenv is copied forward |
| Requirements copied before source | A code change does not invalidate the dependency layer |
| Non-root user (uid 10001, no login shell) | |
| `OMP_NUM_THREADS=1` and friends | The measured 2.6–5.1× throughput difference — see [BENCHMARKS.md](BENCHMARKS.md#thread-pinning) |
| Model **not** baked in | An image containing its weights must be rebuilt to promote a model, which defeats the registry. The bundle is mounted read-only at `/app/artifacts/current` |
| `HEALTHCHECK` targets `/ready`, not `/health` | Readiness gates traffic; liveness does not |

`docker/docker-compose.yml` composes the API with an MLflow tracking server. Also
**never run**.

| Metric | Status |
|---|---|
| Image build time | **not measured** |
| Image size | **not measured** |
| Container start-up time | **not measured** |

These would be produced by the `docker` CI job. They must not be quoted.

## Running CI locally

```bash
make lint
make test
python scripts/train.py && python scripts/smoke_test.py
python scripts/drift_experiment.py --trials 5 --no-plots
```
