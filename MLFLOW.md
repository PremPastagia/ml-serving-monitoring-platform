# MLflow: Experiment Tracking and Model Registry

## Why MLflow, and why SQLite

This project needs an **offline** store (no account, no network in CI), a model registry
with versioning and aliases, and a UI a reviewer can open. MLflow is the only one of the
realistic options that gives all three locally — Weights & Biases needs a cloud account,
a hand-rolled CSV gives no registry.

The **SQLite backend specifically is required**: MLflow's plain file store cannot host a
model registry. `sqlite:///mlflow.db`, resolved to an absolute path so the working
directory cannot change which store is used.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## What is tracked

Per run:

| Category | Fields |
|---|---|
| Params | estimator, seed, all model hyperparameters, dataset version, split id, code version, git commit, config digest, feature count, split sizes |
| Metrics | ROC-AUC, PR-AUC, accuracy, precision, recall, F1, Brier, log loss — for train, validation and test — plus `training_seconds` and `total_seconds` |
| Tags | dataset version, code version, git commit, config digest, estimator, primary metric, lifecycle, scenario, decision |
| Artefacts | the sklearn pipeline with signature and input example, plus `metadata/` containing `run_metadata.json`, `feature_list.json`, `environment.json`, `dataset_files.json`, `leakage_report.json` |

Environment capture records the Python version, the platform string and the versions of
numpy, pandas, scikit-learn, scipy, mlflow, fastapi and pydantic.

`git_commit` degrades to `unavailable` outside a repository rather than raising, and
carries a `-dirty` suffix when the tree is modified — so a metric recorded from
uncommitted code is visibly marked as such. When git is absent, `code_version` (a
content hash of `src/`) still identifies the code exactly.

## Aliases, not stages

MLflow's `Staging`/`Production` **stages are deprecated**. This project uses **registry
aliases**, which is both the supported API and the better model for what serving needs:
an alias is a mutable pointer to an immutable version, so promotion and rollback are a
single atomic repoint rather than a multi-step transition that can be observed
half-applied.

| Alias | Meaning |
|---|---|
| `production` | The version the API serves |
| `previous` | What `production` pointed at before the last change — the known rollback target |
| `candidate` | A freshly retrained model awaiting a decision |

## Registration path

MLflow 3 stores a logged model as its own entity. Registering via `runs:/<id>/model`
still works but resolves through a deprecated fallback, which MLflow warns about:

```
WARNING: Run with id ... has no artifacts at artifact path 'model',
registering model based on models:/m-... instead
```

`log_training_run` therefore returns a `LoggedRun(run_id, model_uri)` and registration
uses the returned `model_uri`. The version's `source` then points at the artefact
directly and the warning is gone.

## Operations

```python
from mlserve.models.registry import ModelRegistry
from mlserve.models.tracking import list_runs, best_run

registry = ModelRegistry()
registry.list_versions()                  # every version, with tags and aliases
registry.production()                      # resolve the production alias
registry.promote("3")                      # move production, remember previous
registry.rollback()                        # back to previous
registry.rollback(to_version="1")          # to an explicit version
registry.load("production")                # load the aliased pipeline

list_runs()                                # experiment comparison as a DataFrame
best_run()                                 # highest validation ROC-AUC
```

## Measured results

Current registry state after `scripts/train.py` plus the rollback experiment:

| Version | Estimator | Validation ROC-AUC | Test ROC-AUC | Alias |
|---|---|---|---|---|
| 1 | HistGradientBoosting (200 trees) | 0.930270 | 0.926784 | `production` |
| 2 | HistGradientBoosting (weakened, rollback target) | 0.897678 | — | `previous` |

Experiment comparison from the training run:

| Run | Validation ROC-AUC | Test ROC-AUC | Fit time | Selected |
|---|---|---|---|---|
| `hist_gradient_boosting-seed42` | **0.930270** | **0.926784** | 0.582 s | ✅ |
| `logistic_regression-seed42` | 0.914013 | 0.909126 | 0.140 s | |

Best-model selection is by validation ROC-AUC, never test — the test set is loaded once
at the end and is never used to choose anything.

| Operation | Measured |
|---|---|
| Promotion (alias repoint) | 0.0043 s |
| Rollback (alias repoint) | 0.0055 s |
| Model load from the registry | 0.018 s |

## Tests

23 tests in `tests/test_registry.py`, all passing:

- Multiple experiments tracked and comparable; best-run selection matches the maximum
- A **failed training run leaves no registry version**
- Registration creates increasing versions carrying provenance tags
- A registered model loads and predicts
- First promotion sets the alias; subsequent promotion records `previous`
- Promoting the current version is a no-op
- The `candidate` alias does not disturb `production`
- Rollback returns to `previous`, and is **itself reversible**
- Rollback to an explicit version
- Rollback errors: no `production`, no `previous`, unknown version, already-current
- Loading by alias equals loading by version
- An empty registry lists nothing rather than raising

The whole file runs against an isolated store under `tmp_path`, so it never touches the
developer's real `mlflow.db`. The store is created once per module and only the mutable
aliases are reset between tests — creating a fresh SQLite tracking store costs a full
alembic migration, which made this file dominate the suite at 210 s; it now runs in
**8.7 s**.

```bash
python -m pytest tests/test_registry.py -q
```
