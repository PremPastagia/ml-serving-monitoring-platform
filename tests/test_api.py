"""Serving API: the happy paths and the response contract."""

from __future__ import annotations

import concurrent.futures

import pytest

from mlserve.data.schema import FEATURE_NAMES, NEGATIVE_LABEL, POSITIVE_LABEL
from mlserve.serving.app import MODEL_VERSION_HEADER, REQUEST_ID_HEADER
from mlserve.serving.schemas import EXAMPLE_RECORD

# ------------------------------------------------------------------------- health


def test_health_reports_ok_when_a_model_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "7"
    assert body["uptime_seconds"] >= 0


def test_health_is_200_even_when_degraded(unloaded_client):
    """Liveness must not fail just because the registry is empty."""
    response = unloaded_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_ready_gates_on_the_model(client, unloaded_client):
    assert client.get("/ready").status_code == 200
    assert unloaded_client.get("/ready").status_code == 503


# --------------------------------------------------------------------- model-info


def test_model_info_exposes_full_provenance(client):
    body = client.get("/model-info").json()
    for key in ("model_name", "model_version", "model_source", "dataset_version",
                "code_version", "git_commit", "training_fingerprint", "loaded_at",
                "input_columns", "model_features"):
        assert body[key], f"{key} missing from /model-info"
    assert body["input_columns"] == FEATURE_NAMES


def test_model_info_is_unavailable_without_a_model(unloaded_client):
    response = unloaded_client.get("/model-info")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "model_not_loaded"


# ------------------------------------------------------------------------ predict


def test_single_prediction(client):
    response = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert response.status_code == 200
    body = response.json()
    assert body["n_records"] == 1
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["prediction"] in (0, 1)
    assert prediction["label"] in (POSITIVE_LABEL, NEGATIVE_LABEL)


def test_batch_prediction_returns_one_result_per_record(client):
    records = [EXAMPLE_RECORD] * 25
    body = client.post("/predict", json={"records": records}).json()
    assert body["n_records"] == 25
    assert len(body["predictions"]) == 25


def test_response_carries_the_served_model_version(client):
    response = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert response.json()["model_version"] == "7"
    assert response.headers[MODEL_VERSION_HEADER] == "7"


def test_label_is_consistent_with_the_threshold(client):
    body = client.post("/predict", json={"records": [EXAMPLE_RECORD] * 5}).json()
    threshold = body["threshold"]
    for prediction in body["predictions"]:
        expected = 1 if prediction["probability"] >= threshold else 0
        assert prediction["prediction"] == expected
        assert prediction["label"] == (POSITIVE_LABEL if expected else NEGATIVE_LABEL)


def test_identical_payloads_produce_identical_scores(client):
    first = client.post("/predict", json={"records": [EXAMPLE_RECORD]}).json()
    second = client.post("/predict", json={"records": [EXAMPLE_RECORD]}).json()
    assert first["predictions"][0]["probability"] == second["predictions"][0]["probability"]


def test_prediction_is_reported_with_latency(client):
    body = client.post("/predict", json={"records": [EXAMPLE_RECORD]}).json()
    assert body["latency_ms"] > 0


def test_high_earning_profile_scores_above_a_low_earning_one(client):
    """A basic sanity check that the wiring is not scrambling features."""
    low = {**EXAMPLE_RECORD, "education_num": 1, "hours_per_week": 10,
           "occupation": "Other-service", "capital_gain": 0,
           "marital_status": "Never-married", "relationship": "Own-child", "age": 19}
    high = {**EXAMPLE_RECORD, "education_num": 16, "hours_per_week": 60,
            "occupation": "Exec-managerial", "capital_gain": 15000,
            "marital_status": "Married-civ-spouse", "relationship": "Husband", "age": 45}
    body = client.post("/predict", json={"records": [low, high]}).json()
    assert body["predictions"][1]["probability"] > body["predictions"][0]["probability"]


# --------------------------------------------------------------------- request ids


def test_every_response_carries_a_request_id(client):
    for path in ("/health", "/model-info", "/metrics", "/monitoring/summary"):
        assert client.get(path).headers.get(REQUEST_ID_HEADER)


def test_a_supplied_request_id_is_echoed(client):
    response = client.post("/predict", json={"records": [EXAMPLE_RECORD]},
                           headers={REQUEST_ID_HEADER: "trace-me-123"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-me-123"
    assert response.json()["request_id"] == "trace-me-123"


def test_request_ids_are_unique_per_request(client):
    ids = {client.post("/predict", json={"records": [EXAMPLE_RECORD]}).json()["request_id"]
           for _ in range(10)}
    assert len(ids) == 10


# -------------------------------------------------------------------- concurrency


def test_concurrent_requests_are_all_served_correctly(client):
    """Concurrency must not corrupt the store, the metrics or the responses."""
    def call(i):
        record = {**EXAMPLE_RECORD, "age": 30 + (i % 40)}
        response = client.post("/predict", json={"records": [record]})
        return response.status_code, response.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(64)))

    assert all(status == 200 for status, _ in results)
    assert len({body["request_id"] for _, body in results}) == 64
    assert client.app_state.store.count() == 64


# ------------------------------------------------------------------------- reload


def test_reload_swaps_the_served_version(client, loaded_model, stub_loader):
    import dataclasses

    stub_loader.next_model = dataclasses.replace(loaded_model, model_version="8")
    body = client.post("/admin/reload").json()
    assert body["ok"] is True
    assert body["changed"] is True
    assert body["from_version"] == "7"
    assert body["to_version"] == "8"
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD]}).json()["model_version"] == "8"


def test_failed_reload_keeps_the_previous_model_serving(client, stub_loader):
    """A bad reload must degrade to 'nothing changed', never to 'no model'."""
    stub_loader.reload_should_fail = True
    assert client.post("/admin/reload").status_code == 503
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD]}).status_code == 200


# ---------------------------------------------------------------------- openapi


def test_openapi_documents_the_contract(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    record = schema["components"]["schemas"]["PredictionRecord"]
    assert set(record["required"]) == set(FEATURE_NAMES)
    assert record["additionalProperties"] is False


@pytest.mark.parametrize("path", ["/health", "/ready", "/model-info", "/predict",
                                  "/metrics", "/monitoring/summary", "/admin/reload"])
def test_every_documented_endpoint_exists(client, path):
    schema = client.get("/openapi.json").json()
    assert path in schema["paths"]
