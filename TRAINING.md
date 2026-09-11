# Training

## Pipeline

```bash
python scripts/train.py                 # validate → split → train → evaluate → track → register
python scripts/train.py --no-register   # experiment only
python scripts/train.py --seed 7        # reproducibility probe
python scripts/train.py --no-mlflow     # local bundle only
```

Trains the configured estimator **and** the linear baseline, so every MLflow experiment
contains a comparison rather than a single unanchored number, then registers the better
run and (on first registration) points the `production` alias at it.

## Model

`HistGradientBoostingClassifier` — the strongest scikit-learn tabular estimator with
native categorical support and no extra native dependency.

| Parameter | Value |
|---|---|
| `learning_rate` | 0.1 |
| `max_iter` | 200 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 1.0 |
| `max_bins` | 255 |
| `early_stopping` | false |

Preprocessing lives **inside the same sklearn `Pipeline`**: `DerivedFeatures` →
`ColumnTransformer` → estimator. The object persisted to the registry contains the
transformation, so the API cannot apply a different one from training. There is no
second code path to keep in sync.

| Estimator | Preprocessing | Why |
|---|---|---|
| HistGradientBoosting | Ordinal encoding, no scaling | Trees need neither. Ordinal + native categorical support avoids the sparse blow-up one-hot would cause on `native_country` (41 levels). `handle_unknown='use_encoded_value'` maps an unseen level to −1 so an unexpected category degrades one prediction instead of taking the service down. |
| LogisticRegression | StandardScaler + one-hot (`min_frequency=10`) | A linear model needs both. |

## Results

| Split | Rows | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 26,048 | 0.956365 | 0.886742 | 0.897036 | 0.833055 | 0.715925 | 0.770062 | 0.071346 |
| validation | 6,513 | 0.930270 | 0.831673 | 0.873791 | 0.773862 | 0.672194 | 0.719454 | 0.087238 |
| **test** | 16,281 | **0.926784** | **0.823947** | **0.872059** | 0.766556 | 0.659126 | 0.708794 | 0.088495 |

Linear baseline, same test set: ROC-AUC **0.909126**, PR-AUC 0.773880, accuracy 0.854739.

Train→validation ROC-AUC gap **0.026** — mild, expected for a boosted model at this
depth, and asserted to stay below 0.10 by `test_generalisation_gap_is_not_catastrophic`.

Fit time: **0.582 s** (boosted), **0.140 s** (linear).

## Reproducibility

`training_fingerprint` hashes the fitted model's **predictions on a fixed 256-row
probe set** at full float64 precision — not its pickle bytes, which differ between runs
for reasons unrelated to the model (memory addresses, dict ordering, joblib framing).
Predictions are what actually have to be identical.

Current fingerprint: `54122c6ae1215984557601e7…`

| Property | Verified by |
|---|---|
| Repeated training gives an identical fingerprint | `test_repeated_training_is_bit_identical` |
| A different hyperparameter gives a different fingerprint | `test_a_different_hyperparameter_produces_a_different_model` |
| A different split seed gives a different fingerprint | `test_a_different_data_split_produces_a_different_model` |
| The fingerprint responds to prediction changes | `test_fingerprint_is_sensitive_to_the_probe_predictions` |

### The estimator seed is inert — and that is asserted

`random_state` is consumed by HistGradientBoosting in exactly two places: the internal
split used for early stopping, and the subsample drawn for bin thresholds above 200k
rows. This project sets `early_stopping: false` and trains on 26k rows, so **neither
applies and the fit is deterministic regardless of the seed**.

That is a stronger guarantee than seeding, but it is worth asserting rather than
assuming: if a future change enables early stopping or crosses the subsampling
threshold, the fit silently becomes seed-dependent.
`test_estimator_seed_does_not_change_this_model_and_that_is_intended` catches that.

The seed that *is* load-bearing is the **split** seed.

## Leakage prevention

| Risk | Structural defence |
|---|---|
| Test contamination | The test set is a physically separate UCI file. No code path can move a test row into `fit`. |
| Train/validation contamination | Split by index; overlap measured and asserted to be **0**. |
| Preprocessing leakage | All statistics are learned inside `Pipeline.fit`, which only ever sees train. Verified behaviourally by `test_preprocessing_statistics_come_from_train_only`. |
| Target leakage | Univariate AUC ≥ 0.99 aborts training. See [DATA_VALIDATION.md](DATA_VALIDATION.md#leakage-detection). |
| Design-artefact leakage | `fnlwgt` (the census sampling weight) is dropped — it is a property of the survey, not the person, and is unavailable at serving time. |

## Recorded provenance

Every run records, in MLflow and in a local `run.json` sidecar:

| Field | Example |
|---|---|
| `dataset_version` | `adult-ingest-2-975c90344d56` |
| `dataset_files` | SHA-256 of each raw file |
| `split_id` | `bc50e85d5b57e879` |
| `split_sizes` | train 26,048 / validation 6,513 / test 16,281 |
| `code_version` | content hash of `src/` |
| `git_commit` | short SHA, `-dirty` suffix when the tree is modified |
| `config_digest` | hash of the resolved configuration |
| `seed`, `params` | |
| `feature_list` | the 16 features reaching the estimator |
| `input_columns` | the 12 the API requires |
| `metrics` | every metric on every split |
| `training_seconds`, `total_seconds` | |
| `environment` | Python, OS, and the versions of 7 tracked libraries |
| `leakage_report` | split overlap measurements |
| `training_fingerprint` | |

`test_run_records_every_required_provenance_field` asserts each is present and
non-empty, so provenance cannot silently degrade.

## Failure behaviour

Training refuses to run on data that fails validation — `DataValidationError` is raised
and no model is produced. A failed run leaves no registry version
(`test_a_failed_training_run_leaves_no_registry_version`).
