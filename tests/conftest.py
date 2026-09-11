"""Shared fixtures.

Two principles:

1. **Nothing touches the developer's real MLflow store or prediction database.** Every
   fixture that needs state builds it under ``tmp_path`` from a config clone, so the
   suite is order-independent and leaves no residue.
2. **Fit once, reuse everywhere.** Training on the full 26k rows takes seconds; doing
   it per test would make the suite unusable. A session-scoped small split keeps the
   whole suite fast while still exercising the real pipeline rather than a mock.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mlserve.config import Config, load_config
from mlserve.data.ingest import load_raw
from mlserve.data.schema import FEATURE_NAMES, TARGET
from mlserve.data.split import Split, make_split
from mlserve.models.train import train_model

#: Small enough to keep the suite fast, large enough that stratified splits, PSI bins
#: and chi-square expected counts all behave as they do at full scale.
SMALL_TRAIN_ROWS = 6000
SMALL_TEST_ROWS = 3000


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def raw_data():
    """The real dataset, loaded once. Skips the suite if it has not been fetched."""
    try:
        return load_raw()
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"raw data unavailable: {exc}")


@pytest.fixture(scope="session")
def full_split(raw_data, config) -> Split:
    development, test, version = raw_data
    return make_split(development, test, version, seed=config.seed,
                      validation_fraction=float(config.require("data.validation_fraction")))


@pytest.fixture(scope="session")
def small_split(raw_data, config) -> Split:
    """A stratified subsample, for tests that need a real fit but not a long one."""
    development, test, version = raw_data
    dev_small = development.groupby(TARGET, group_keys=False).apply(
        lambda g: g.sample(n=int(SMALL_TRAIN_ROWS * len(g) / len(development)), random_state=config.seed),
        include_groups=True,
    ).reset_index(drop=True)
    test_small = test.sample(n=SMALL_TEST_ROWS, random_state=config.seed).reset_index(drop=True)
    return make_split(dev_small, test_small, version, seed=config.seed, validation_fraction=0.25)


@pytest.fixture(scope="session")
def trained(small_split, config):
    """A fitted pipeline plus its provenance record."""
    return train_model(config, split=small_split, evaluate_test=True)


@pytest.fixture(scope="session")
def pipeline(trained):
    return trained[0]


@pytest.fixture
def clean_frame(small_split) -> pd.DataFrame:
    """A known-good frame. Corruption tests mutate a copy of this."""
    return small_split.train.head(1500).copy()


@pytest.fixture
def feature_frame(clean_frame) -> pd.DataFrame:
    return clean_frame[FEATURE_NAMES].copy()


def clone_config(tmp_path: Path, overrides: dict | None = None) -> Config:
    """A config pointing entirely at ``tmp_path``.

    This is what keeps a registry or store test from writing into the real mlflow.db.
    """
    base = load_config()
    data = copy.deepcopy(base.raw)
    data["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    data["mlflow"]["artifact_location"] = f"file:{tmp_path / 'mlruns'}"
    data["mlflow"]["experiment_name"] = "test-experiment"
    data["paths"]["artifact_dir"] = str(tmp_path / "artifacts")
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["paths"]["prediction_db"] = str(tmp_path / "predictions.sqlite")
    for key, value in (overrides or {}).items():
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return Config(data, path)


@pytest.fixture
def isolated_config(tmp_path) -> Config:
    return clone_config(tmp_path)


@pytest.fixture
def loaded_model(pipeline, trained):
    """A :class:`LoadedModel` wrapping the session-trained pipeline."""
    from datetime import UTC, datetime

    from mlserve.serving.model_loader import LoadedModel

    _, run = trained
    return LoadedModel(
        pipeline=pipeline,
        model_name="adult-income-classifier",
        model_version="7",
        model_alias="production",
        source="registry",
        run_id="test-run-id",
        dataset_version=run.dataset_version,
        code_version=run.code_version,
        git_commit=run.git_commit,
        training_fingerprint=run.training_fingerprint,
        metrics={"validation_roc_auc": run.metrics["validation_roc_auc"]},
        loaded_at=datetime.now(UTC).isoformat(),
        load_seconds=0.01,
    )


class StubLoader:
    """A ModelLoader stand-in that never touches MLflow.

    The API tests are about HTTP behaviour, not about MLflow, and going through the
    real registry would make every one of them depend on a trained, registered model
    being present. `tests/test_rollback.py` and the serving-rollback verifier exercise
    the real loader against the real registry.
    """

    def __init__(self, model=None, *, fail_with: str | None = None):
        self._model = model
        self.last_error = fail_with
        self.reload_calls = 0
        self.next_model = model
        self.reload_should_fail = False

    @property
    def model(self):
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def require(self):
        from mlserve.serving.model_loader import ModelNotLoadedError

        if self._model is None:
            raise ModelNotLoadedError(self.last_error or "no model has been loaded")
        return self._model

    def try_load(self):
        return self._model

    def load(self):
        if self._model is None:
            raise RuntimeError(self.last_error or "no model")
        return self._model

    def reload(self):
        self.reload_calls += 1
        before = self._model.model_version if self._model else None
        if self.reload_should_fail:
            return self._model, {"ok": False, "from_version": before, "to_version": before,
                                 "changed": False, "seconds": 0.001, "error": "injected failure"}
        self._model = self.next_model
        after = self._model.model_version if self._model else None
        return self._model, {"ok": True, "from_version": before, "to_version": after,
                             "changed": before != after, "seconds": 0.001, "error": None}

    def unload(self):
        self._model = None
        self.last_error = "model was explicitly unloaded"


@pytest.fixture
def stub_loader(loaded_model):
    return StubLoader(loaded_model)


@pytest.fixture
def service_config(tmp_path):
    return clone_config(tmp_path)


@pytest.fixture
def client(service_config, stub_loader):
    """A TestClient over an isolated app instance."""
    from fastapi.testclient import TestClient

    from mlserve.monitoring.store import PredictionStore
    from mlserve.serving.app import create_app
    from mlserve.serving.metrics import ServingMetrics

    store = PredictionStore(service_config.path("paths.prediction_db"))
    metrics = ServingMetrics(feature_sample_rate=1.0, seed=0)
    app = create_app(service_config, loader=stub_loader, store=store, metrics=metrics)
    with TestClient(app) as test_client:
        test_client.app_state = app.state.service
        yield test_client
    store.close()


@pytest.fixture
def unloaded_client(service_config):
    """A service instance that came up with no model -- the degraded path."""
    from fastapi.testclient import TestClient

    from mlserve.monitoring.store import PredictionStore
    from mlserve.serving.app import create_app
    from mlserve.serving.metrics import ServingMetrics

    store = PredictionStore(service_config.path("paths.prediction_db"))
    app = create_app(
        service_config,
        loader=StubLoader(None, fail_with="registry alias 'production' is not set"),
        store=store,
        metrics=ServingMetrics(seed=0),
    )
    with TestClient(app) as test_client:
        test_client.app_state = app.state.service
        yield test_client
    store.close()
