"""MLflow experiment tracking.

Why MLflow and not Weights & Biases or a hand-rolled CSV: this project needs an
*offline* store (no account, no network in CI), a model registry with versioning and
aliases, and a UI a reviewer can open. MLflow is the only one of the three that gives
all of that from a local SQLite file. The SQLite backend specifically is required --
the plain file store cannot host a model registry.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from mlserve.config import Config, load_config, project_root
from mlserve.models.train import TrainingRun


def quiet_mlflow_logging() -> None:
    """Reduce MLflow/alembic chatter to warnings.

    Schema-migration INFO lines on every invocation drown the output that a reviewer
    actually needs to read, and they say nothing about the run.
    """
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
    for name in ("alembic", "alembic.runtime.migration", "mlflow", "mlflow.store.db.utils",
                 "mlflow.tracking._model_registry.fluent", "mlflow.models.model"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # The signature inferred from integer feature columns triggers a standing hint
    # about missing values. This contract forbids nulls (the validator rejects them),
    # so the hint is not actionable here.
    warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")


@dataclass(frozen=True)
class LoggedRun:
    """Identifiers produced by logging one training run."""

    run_id: str
    model_uri: str

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "model_uri": self.model_uri}


def resolve_tracking_uri(config: Config) -> str:
    """Make a relative sqlite URI absolute so cwd cannot change which store is used."""
    uri = str(config.require("mlflow.tracking_uri"))
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        path = uri[len(prefix):]
        if not Path(path).is_absolute():
            return prefix + str(project_root() / path)
    return uri


def configure_mlflow(config: Config | None = None) -> MlflowClient:
    config = config or load_config()
    quiet_mlflow_logging()
    uri = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(uri)
    experiment = str(config.require("mlflow.experiment_name"))
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_location = str(config.require("mlflow.artifact_location"))
        if artifact_location.startswith("file:./"):
            artifact_location = "file:" + str(project_root() / artifact_location[len("file:./"):])
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)
    return MlflowClient(tracking_uri=uri)


def _flatten_params(run: TrainingRun) -> dict[str, str]:
    """MLflow params are scalar strings; nested config is flattened deliberately."""
    params: dict[str, str] = {
        "estimator": run.estimator,
        "seed": str(run.seed),
        "dataset_version": run.dataset_version,
        "split_id": run.split_id,
        "code_version": run.code_version,
        "git_commit": run.git_commit,
        "config_digest": run.config_digest,
        "n_features": str(len(run.feature_list)),
        "n_train_rows": str(run.split_sizes.get("train", 0)),
        "n_validation_rows": str(run.split_sizes.get("validation", 0)),
        "n_test_rows": str(run.split_sizes.get("test", 0)),
    }
    for key, value in run.params.items():
        params[f"model.{key}"] = str(value)
    return params


def log_training_run(
    pipeline,
    run: TrainingRun,
    config: Config | None = None,
    *,
    input_example: pd.DataFrame | None = None,
    register: bool = False,
    tags: dict[str, str] | None = None,
) -> LoggedRun:
    """Log one training run and return its run id and the logged model's URI.

    The model URI is returned explicitly because MLflow 3 stores a logged model as its
    own entity; addressing it as ``runs:/<id>/model`` still works but resolves through
    a deprecated fallback path. Registering from the returned URI is the supported
    route and keeps the registry source pointing at the real artifact.
    """
    config = config or load_config()
    configure_mlflow(config)

    with mlflow.start_run(run_name=run.run_name) as active:
        mlflow.log_params(_flatten_params(run))
        mlflow.log_metrics(
            {k: v for k, v in run.metrics.items() if isinstance(v, (int, float)) and v == v}
        )
        mlflow.log_metric("training_seconds", run.training_seconds)
        mlflow.log_metric("total_seconds", run.total_seconds)

        mlflow.set_tags(
            {
                "dataset_version": run.dataset_version,
                "code_version": run.code_version,
                "git_commit": run.git_commit,
                "config_digest": run.config_digest,
                "estimator": run.estimator,
                "primary_metric": str(config.require("evaluation.primary_metric")),
                **(tags or {}),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "run_metadata.json").write_text(
                json.dumps(run.to_dict(), indent=2, sort_keys=True)
            )
            (tmp_path / "feature_list.json").write_text(
                json.dumps({"model_features": run.feature_list,
                            "input_columns": run.input_columns}, indent=2)
            )
            (tmp_path / "environment.json").write_text(json.dumps(run.environment, indent=2))
            (tmp_path / "dataset_files.json").write_text(json.dumps(run.dataset_files, indent=2))
            (tmp_path / "leakage_report.json").write_text(json.dumps(run.leakage_report, indent=2))
            mlflow.log_artifacts(str(tmp_path), artifact_path="metadata")

        kwargs = {}
        if input_example is not None:
            example = input_example.head(5)
            kwargs["input_example"] = example
            kwargs["signature"] = infer_signature(example, pipeline.predict_proba(example)[:, 1])
        if register:
            kwargs["registered_model_name"] = str(config.require("mlflow.registered_model_name"))

        info = mlflow.sklearn.log_model(sk_model=pipeline, name="model", **kwargs)
        return LoggedRun(run_id=active.info.run_id, model_uri=info.model_uri)


def list_runs(config: Config | None = None) -> pd.DataFrame:
    """All runs in the experiment, newest first -- the experiment-comparison view."""
    config = config or load_config()
    configure_mlflow(config)
    experiment = mlflow.get_experiment_by_name(str(config.require("mlflow.experiment_name")))
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )


def best_run(config: Config | None = None, *, metric: str | None = None) -> pd.Series | None:
    """Select the best completed run by the configured primary metric."""
    config = config or load_config()
    metric = metric or f"validation_{config.require('evaluation.primary_metric')}"
    runs = list_runs(config)
    if runs.empty:
        return None
    column = f"metrics.{metric}"
    if column not in runs.columns:
        return None
    finished = runs[(runs["status"] == "FINISHED") & runs[column].notna()]
    if finished.empty:
        return None
    return finished.sort_values(column, ascending=False).iloc[0]
