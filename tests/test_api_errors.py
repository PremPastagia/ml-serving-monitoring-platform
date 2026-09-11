"""Serving API: every way a request can be wrong, and what the caller gets back.

The contract being tested: a bad request produces a 4xx with a structured
`ErrorResponse`, a stable machine-readable `error.type`, a request id, and no Python
traceback -- never a 500, and never a prediction computed from partly-invalid input.
"""

from __future__ import annotations

import pytest

from mlserve.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
from mlserve.serving.app import REQUEST_ID_HEADER
from mlserve.serving.schemas import EXAMPLE_RECORD


def assert_error_envelope(response, *, status: int, error_type: str):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"request_id", "error"}
    assert body["error"]["type"] == error_type
    assert body["error"]["message"].strip()
    assert body["request_id"]
    assert response.headers.get(REQUEST_ID_HEADER)
    assert "Traceback" not in response.text
    return body


# ------------------------------------------------------------------ missing fields


@pytest.mark.parametrize("column", FEATURE_NAMES)
def test_missing_feature_is_rejected(client, column):
    record = {k: v for k, v in EXAMPLE_RECORD.items() if k != column}
    body = assert_error_envelope(
        client.post("/predict", json={"records": [record]}),
        status=422, error_type="validation_error",
    )
    assert any(column in detail["location"] for detail in body["error"]["details"])


def test_empty_record_lists_every_missing_field(client):
    body = assert_error_envelope(
        client.post("/predict", json={"records": [{}]}),
        status=422, error_type="validation_error",
    )
    assert len(body["error"]["details"]) == len(FEATURE_NAMES)


# -------------------------------------------------------------------- extra fields


def test_extra_field_is_rejected_not_ignored(client):
    """Silently dropping an unknown key is how a renamed field reaches production."""
    record = {**EXAMPLE_RECORD, "shoe_size": 44}
    body = assert_error_envelope(
        client.post("/predict", json={"records": [record]}),
        status=422, error_type="validation_error",
    )
    assert any("shoe_size" in detail["location"] for detail in body["error"]["details"])


def test_dropped_training_column_is_rejected_as_an_extra_field(client):
    """`fnlwgt` was deliberately dropped; sending it must not silently succeed."""
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "fnlwgt": 77516}]}),
        status=422, error_type="validation_error",
    )


def test_extra_top_level_key_is_rejected(client):
    assert_error_envelope(
        client.post("/predict", json={"records": [EXAMPLE_RECORD], "model": "v2"}),
        status=422, error_type="validation_error",
    )


# --------------------------------------------------------------------- wrong types


@pytest.mark.parametrize("column", NUMERIC_FEATURES)
def test_non_numeric_value_in_a_numeric_field_is_rejected(client, column):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: "many"}]}),
        status=422, error_type="validation_error",
    )


@pytest.mark.parametrize("column", NUMERIC_FEATURES)
def test_null_in_a_numeric_field_is_rejected(client, column):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: None}]}),
        status=422, error_type="validation_error",
    )


def test_fractional_value_in_an_integer_field_is_rejected(client):
    """Accepting 39.7 as an age would change the model's binning at serving time."""
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "age": 39.7}]}),
        status=422, error_type="validation_error",
    )


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_numeric_value_in_a_categorical_field_is_rejected(client, column):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: 3}]}),
        status=422, error_type="validation_error",
    )


def test_record_that_is_not_an_object_is_rejected(client):
    assert_error_envelope(
        client.post("/predict", json={"records": ["not-a-record"]}),
        status=422, error_type="validation_error",
    )


def test_records_that_is_not_a_list_is_rejected(client):
    assert_error_envelope(
        client.post("/predict", json={"records": EXAMPLE_RECORD}),
        status=422, error_type="validation_error",
    )


# ------------------------------------------------------------------ out-of-range


@pytest.mark.parametrize("column,value", [
    ("age", 16), ("age", 91), ("education_num", 0), ("education_num", 17),
    ("hours_per_week", 0), ("hours_per_week", 100), ("capital_gain", -1),
    ("capital_loss", -1), ("capital_gain", 100000),
])
def test_out_of_contract_range_is_rejected(client, column, value):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: value}]}),
        status=422, error_type="validation_error",
    )


@pytest.mark.parametrize("column,value", [
    ("age", 17), ("age", 90), ("education_num", 1), ("education_num", 16),
    ("hours_per_week", 1), ("hours_per_week", 99), ("capital_gain", 0),
    ("capital_gain", 99999),
])
def test_contract_boundaries_are_accepted(client, column, value):
    """Positive control: the range check must not be off by one."""
    response = client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: value}]})
    assert response.status_code == 200, response.text


# --------------------------------------------------------------- unknown category


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_unknown_category_level_is_rejected(client, column):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, column: "Narnia"}]}),
        status=422, error_type="validation_error",
    )


def test_missing_category_marker_is_accepted_where_the_contract_allows_it(client):
    record = {**EXAMPLE_RECORD, "workclass": "__missing__", "occupation": "__missing__",
              "native_country": "__missing__"}
    assert client.post("/predict", json={"records": [record]}).status_code == 200


def test_missing_category_marker_is_rejected_where_it_is_not_allowed(client):
    assert_error_envelope(
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "sex": "__missing__"}]}),
        status=422, error_type="validation_error",
    )


# ------------------------------------------------------------------ empty payloads


def test_empty_records_list_is_rejected(client):
    assert_error_envelope(
        client.post("/predict", json={"records": []}),
        status=422, error_type="validation_error",
    )


def test_empty_body_is_rejected(client):
    assert_error_envelope(client.post("/predict", json={}), status=422,
                          error_type="validation_error")


def test_malformed_json_is_rejected(client):
    response = client.post("/predict", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_no_body_at_all_is_rejected(client):
    assert client.post("/predict").status_code == 422


# ----------------------------------------------------------------- oversized batch


def test_batch_above_the_configured_maximum_is_rejected(client, service_config):
    limit = int(service_config.require("serving.max_batch_size"))
    assert_error_envelope(
        client.post("/predict", json={"records": [EXAMPLE_RECORD] * (limit + 1)}),
        status=413, error_type="payload_too_large",
    )


def test_batch_exactly_at_the_maximum_is_accepted(client, service_config):
    limit = int(service_config.require("serving.max_batch_size"))
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD] * limit}).status_code == 200


# --------------------------------------------------------------- model unavailable


def test_predict_without_a_model_returns_503(unloaded_client):
    body = assert_error_envelope(
        unloaded_client.post("/predict", json={"records": [EXAMPLE_RECORD]}),
        status=503, error_type="model_not_loaded",
    )
    assert "production" in body["error"]["message"]


def test_model_unloaded_mid_life_returns_503(client, stub_loader):
    assert client.post("/predict", json={"records": [EXAMPLE_RECORD]}).status_code == 200
    stub_loader.unload()
    assert_error_envelope(
        client.post("/predict", json={"records": [EXAMPLE_RECORD]}),
        status=503, error_type="model_not_loaded",
    )


# ------------------------------------------------------------------- method/route


def test_unknown_route_is_404(client):
    assert client.get("/no-such-endpoint").status_code == 404


def test_wrong_method_is_405(client):
    assert client.get("/predict").status_code == 405


# --------------------------------------------------------- errors are recorded


def test_rejected_requests_are_counted_and_logged(client):
    before = client.app_state.store.count("events")
    client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "age": 900}]})
    assert client.app_state.store.count("events") == before + 1
    assert 'mlserve_errors_total{endpoint="/predict",error_type="validation_error"}' in client.get("/metrics").text


def test_a_rejected_request_writes_no_prediction_row(client):
    before = client.app_state.store.count()
    client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "age": 900}]})
    assert client.app_state.store.count() == before
