"""Failure injection: what the platform does when its dependencies misbehave.

Each test injects one realistic failure and asserts the system degrades in a defined,
observable way instead of crashing, hanging, or -- worst -- silently returning wrong
answers.
"""

from __future__ import annotations

import json
import pickle
import sqlite3

import pandas as pd
import pytest

from mlserve.config import load_config
from mlserve.data.ingest import ChecksumMismatch, load_raw, verify_raw_files
from mlserve.monitoring.store import PredictionStore
from mlserve.serving.model_loader import ModelLoader, ModelNotLoadedError
from mlserve.serving.schemas import EXAMPLE_RECORD

# ------------------------------------------------------------------- model loading


def test_missing_model_bundle_raises_a_clear_error(tmp_path, config):
    loader = ModelLoader(config, source="file", bundle_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no model bundle"):
        loader.load()
    assert loader.is_loaded is False
    assert "FileNotFoundError" in loader.last_error


def test_corrupt_model_artifact_is_reported_not_swallowed(tmp_path, config):
    (tmp_path / "model.joblib").write_bytes(b"this is not a pickle")
    loader = ModelLoader(config, source="file", bundle_dir=tmp_path)
    # joblib surfaces a truncated/garbage payload as whichever parse step fails first,
    # so the type is not fixed. What matters is that it propagates rather than being
    # swallowed into a half-loaded model, and that the loader stays empty.
    with pytest.raises((IndexError, ValueError, EOFError, TypeError, KeyError,
                        pickle.UnpicklingError)):
        loader.load()
    assert loader.is_loaded is False
    assert loader.last_error


def test_try_load_never_raises(tmp_path, config):
    """Startup must not be blocked by a bad model; the process boots degraded."""
    loader = ModelLoader(config, source="file", bundle_dir=tmp_path)
    assert loader.try_load() is None
    assert loader.is_loaded is False


def test_unavailable_registry_leaves_the_loader_empty(tmp_path):
    """An empty registry is reported, not masked."""
    from tests.conftest import clone_config

    config = clone_config(tmp_path)
    loader = ModelLoader(config, source="registry")
    assert loader.try_load() is None
    assert loader.is_loaded is False
    assert loader.last_error


def test_requiring_a_model_that_is_absent_raises_the_typed_error(tmp_path, config):
    loader = ModelLoader(config, source="file", bundle_dir=tmp_path)
    with pytest.raises(ModelNotLoadedError):
        loader.require()


def test_a_failed_reload_keeps_the_previous_model(tmp_path, config, pipeline, trained):
    """The core availability guarantee: a bad reload must not empty the service."""
    import joblib

    _, run = trained
    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "run.json").write_text(json.dumps(run.to_dict()))
    loader = ModelLoader(config, source="file", bundle_dir=tmp_path)
    good = loader.load()
    assert loader.is_loaded

    (tmp_path / "model.joblib").write_bytes(b"corrupted")
    model, details = loader.reload()
    assert details["ok"] is False
    assert details["error"]
    assert loader.is_loaded is True, "a failed reload must not unload the working model"
    assert model.model_version == good.model_version


def test_bundle_with_unreadable_metadata_still_loads_the_model(tmp_path, config, pipeline):
    """Corrupt metadata degrades provenance, not availability."""
    import joblib

    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "run.json").write_text("{not valid json")
    model = ModelLoader(config, source="file", bundle_dir=tmp_path).load()
    assert model.pipeline is not None
    assert model.dataset_version is None


# ------------------------------------------------------------------ serving layer


def test_model_that_raises_at_predict_returns_500_not_a_traceback(client, stub_loader):
    class ExplodingPipeline:
        def predict_proba(self, frame):
            raise RuntimeError("inference exploded")

    import dataclasses

    stub_loader._model = dataclasses.replace(stub_loader.model, pipeline=ExplodingPipeline())
    response = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "internal_error"
    assert "inference exploded" not in response.text, "internals must not leak to the caller"
    assert "Traceback" not in response.text


def test_inference_failure_is_recorded_for_operators(client, stub_loader):
    class ExplodingPipeline:
        def predict_proba(self, frame):
            raise RuntimeError("inference exploded")

    import dataclasses

    stub_loader._model = dataclasses.replace(stub_loader.model, pipeline=ExplodingPipeline())
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert 'error_type="inference_error"' in client.get("/metrics").text


def test_service_survives_a_bad_payload_storm(client):
    """Sustained malformed traffic must not degrade the healthy path."""
    bad_payloads = [
        {}, {"records": []}, {"records": [{}]},
        {"records": [{**EXAMPLE_RECORD, "age": "old"}]},
        {"records": [{**EXAMPLE_RECORD, "sex": "?"}]},
        {"records": "nope"},
        {"records": [{**EXAMPLE_RECORD, "extra": 1}]},
    ]
    for _ in range(20):
        for payload in bad_payloads:
            assert client.post("/predict", json=payload).status_code in (413, 422)

    good = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert good.status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_service_recovers_after_the_model_is_restored(client, stub_loader, loaded_model):
    stub_loader.unload()
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD]}).status_code == 503
    stub_loader._model = loaded_model
    stub_loader.next_model = loaded_model
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD]}).status_code == 200


def test_health_stays_200_through_every_failure(client, stub_loader):
    stub_loader.unload()
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


# ------------------------------------------------------------------ data integrity


def test_a_tampered_raw_file_is_detected(tmp_path):
    """The checksum pin is what makes the dataset version meaningful."""
    import shutil

    from mlserve.data.ingest import SOURCES

    raw_dir = load_config().path("paths.raw_dir")
    for name in SOURCES:
        shutil.copy(raw_dir / name, tmp_path / name)
    with open(tmp_path / "adult.data", "ab") as fh:
        fh.write(b"\n39, Private, 77516, Bachelors, 13, Never-married, Adm-clerical, "
                 b"Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K")
    with pytest.raises(ChecksumMismatch, match="sha256"):
        verify_raw_files(tmp_path, strict=True)


def test_a_missing_raw_file_is_reported_with_a_remedy(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        verify_raw_files(tmp_path, strict=True)


def test_non_strict_mode_still_reports_the_observed_digest(tmp_path):
    import shutil

    from mlserve.data.ingest import SOURCES

    raw_dir = load_config().path("paths.raw_dir")
    for name in SOURCES:
        shutil.copy(raw_dir / name, tmp_path / name)
    with open(tmp_path / "adult.test", "ab") as fh:
        fh.write(b"\n")
    observed = verify_raw_files(tmp_path, strict=False)
    assert observed["adult.test"] != SOURCES["adult.test"]["sha256"]


def test_dataset_version_changes_when_the_bytes_change(tmp_path):
    """A silently edited input must not reuse the previous dataset id."""
    import shutil

    from mlserve.data.ingest import SOURCES

    raw_dir = load_config().path("paths.raw_dir")
    for name in SOURCES:
        shutil.copy(raw_dir / name, tmp_path / name)
    _, _, before = load_raw(tmp_path, strict=False)
    with open(tmp_path / "adult.data", "ab") as fh:
        fh.write(b"\n25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, "
                 b"Own-child, Black, Male, 0, 0, 40, United-States, <=50K")
    _, _, after = load_raw(tmp_path, strict=False)
    assert before.dataset_id != after.dataset_id


# ---------------------------------------------------------------- monitoring store


def test_store_on_an_unwritable_path_fails_at_construction(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    with pytest.raises((sqlite3.OperationalError, NotADirectoryError, FileExistsError)):
        PredictionStore(blocker / "nested" / "predictions.sqlite")


def test_store_rejects_an_unknown_table(tmp_path):
    store = PredictionStore(tmp_path / "p.sqlite")
    with pytest.raises(ValueError, match="unknown table"):
        store.count("robert'); DROP TABLE predictions;--")
    store.close()


def test_store_handles_a_batch_length_mismatch(tmp_path):
    store = PredictionStore(tmp_path / "p.sqlite")
    with pytest.raises(ValueError):
        store.log_predictions(
            request_id="r", model_name="m", model_version="1",
            features=[EXAMPLE_RECORD, EXAMPLE_RECORD], probabilities=[0.1],
            predictions=[0], latency_ms=1.0,
        )
    store.close()


def test_reading_from_an_empty_store_returns_an_empty_frame(tmp_path):
    store = PredictionStore(tmp_path / "p.sqlite")
    frame = store.recent_features(100)
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    store.close()


def test_summary_of_an_empty_store_does_not_divide_by_zero(tmp_path):
    store = PredictionStore(tmp_path / "p.sqlite")
    summary = store.summary()
    assert summary["n_predictions"] == 0
    assert summary["mean_probability"] is None
    store.close()


# ---------------------------------------------------------------- configuration


def test_missing_config_key_raises_a_named_error(config):
    with pytest.raises(KeyError, match="serving.nonexistent"):
        config.require("serving.nonexistent")


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml")


def test_config_digest_changes_with_content(tmp_path):
    from tests.conftest import clone_config

    a = clone_config(tmp_path / "a")
    b = clone_config(tmp_path / "b", {"project.random_seed": 999})
    assert a.digest != b.digest


def test_git_commit_never_raises(monkeypatch):
    """Provenance capture must degrade, not break the run, outside a repository."""
    import subprocess

    from mlserve.config import git_commit

    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_commit() == "unavailable"


def test_environment_info_tolerates_absent_packages(monkeypatch):
    import importlib.metadata as md

    from mlserve.config import environment_info

    def boom(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", boom)
    assert set(environment_info().packages.values()) == {"absent"}


# ------------------------------------------------------- configuration overrides


def test_model_source_can_be_overridden_by_environment(config, monkeypatch, tmp_path):
    """The container sets MLSERVE_MODEL_SOURCE; that path must actually work."""
    monkeypatch.setenv("MLSERVE_MODEL_SOURCE", "file")
    loader = ModelLoader(config, bundle_dir=tmp_path)
    assert loader.source == "file"


def test_an_explicit_argument_beats_the_environment(config, monkeypatch, tmp_path):
    monkeypatch.setenv("MLSERVE_MODEL_SOURCE", "file")
    assert ModelLoader(config, source="registry", bundle_dir=tmp_path).source == "registry"


def test_an_invalid_model_source_is_rejected_at_construction(config, monkeypatch):
    monkeypatch.setenv("MLSERVE_MODEL_SOURCE", "carrier-pigeon")
    with pytest.raises(ValueError, match="model source must be"):
        ModelLoader(config)


def test_model_alias_can_be_overridden_by_environment(config, monkeypatch):
    monkeypatch.setenv("MLSERVE_MODEL_ALIAS", "candidate")
    assert ModelLoader(config).alias == "candidate"
