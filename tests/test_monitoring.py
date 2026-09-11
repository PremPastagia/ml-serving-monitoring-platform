"""Monitoring: the metrics endpoint, the prediction store, and their behaviour
under normal, invalid, high-volume and failing traffic."""

from __future__ import annotations

import re

import pytest

from mlserve.data.schema import FEATURE_NAMES, NUMERIC_FEATURES
from mlserve.serving.metrics import LOW_CARDINALITY_CATEGORICALS, ServingMetrics
from mlserve.serving.schemas import EXAMPLE_RECORD


def metric_value(text: str, pattern: str) -> float:
    """Extract a single numeric sample from the exposition text."""
    match = re.search(rf"^{re.escape(pattern)}\s+([0-9.e+-]+)$", text, re.MULTILINE)
    assert match, f"{pattern} not present in /metrics"
    return float(match.group(1))


# ------------------------------------------------------------------ series present


REQUIRED_SERIES = [
    "mlserve_requests_total",
    "mlserve_errors_total",
    "mlserve_request_latency_seconds_bucket",
    "mlserve_predictions_total",
    "mlserve_prediction_score_bucket",
    "mlserve_predicted_class_total",
    "mlserve_feature_level_total",
    "mlserve_model_info",
    "mlserve_model_loaded",
    "mlserve_uptime_seconds",
    "mlserve_resource_memory_bytes",
]


@pytest.mark.parametrize("series", REQUIRED_SERIES)
def test_required_series_is_exposed(client, series):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "age": 900}]})
    assert series in client.get("/metrics").text


@pytest.mark.parametrize("feature", NUMERIC_FEATURES)
def test_every_numeric_feature_has_a_distribution_histogram(client, feature):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert f"mlserve_feature_{feature}_bucket" in client.get("/metrics").text


@pytest.mark.parametrize("feature", LOW_CARDINALITY_CATEGORICALS)
def test_low_cardinality_categoricals_are_exported(client, feature):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert f'mlserve_feature_level_total{{feature="{feature}"' in client.get("/metrics").text


def test_high_cardinality_categoricals_are_not_exported(client):
    """Cardinality control: 41 countries must not each become a time series."""
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    text = client.get("/metrics").text
    assert 'feature="native_country"' not in text
    assert 'feature="occupation"' not in text


def test_metrics_endpoint_uses_the_prometheus_content_type(client):
    assert client.get("/metrics").headers["content-type"].startswith("text/plain")


# ------------------------------------------------------------------ normal traffic


def test_counters_move_with_traffic(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    before = metric_value(client.get("/metrics").text, 'mlserve_predictions_total{model_version="7"}')
    client.post("/predict", json={"records": [EXAMPLE_RECORD] * 10})
    after = metric_value(client.get("/metrics").text, 'mlserve_predictions_total{model_version="7"}')
    assert after - before == 10


def test_predictions_counter_counts_records_not_requests(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD] * 7})
    text = client.get("/metrics").text
    assert metric_value(text, 'mlserve_predictions_total{model_version="7"}') == 7
    assert metric_value(text, 'mlserve_requests_total{endpoint="/predict",method="POST",status="200"}') == 1


def test_latency_histogram_records_observations(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    text = client.get("/metrics").text
    assert metric_value(text, 'mlserve_request_latency_seconds_count{endpoint="/predict"}') == 1
    assert metric_value(text, 'mlserve_request_latency_seconds_sum{endpoint="/predict"}') > 0


def test_model_info_labels_carry_the_served_identity(client):
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'model_version="7"' in text
    assert 'model_name="adult-income-classifier"' in text


def test_resource_gauges_are_populated(client):
    text = client.get("/metrics").text
    assert metric_value(text, "mlserve_resource_memory_bytes") > 0
    assert metric_value(text, "mlserve_uptime_seconds") >= 0


# ----------------------------------------------------------------- invalid traffic


def test_invalid_traffic_increments_errors_but_not_predictions(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    before = client.get("/metrics").text
    predictions_before = metric_value(before, 'mlserve_predictions_total{model_version="7"}')
    for _ in range(5):
        client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "sex": "Nope"}]})
    after = client.get("/metrics").text
    assert metric_value(after, 'mlserve_errors_total{endpoint="/predict",error_type="validation_error"}') == 5
    assert metric_value(after, 'mlserve_predictions_total{model_version="7"}') == predictions_before


def test_error_rate_is_computable_from_the_exposed_series(client):
    for _ in range(3):
        client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    for _ in range(2):
        client.post("/predict", json={"records": [{}]})
    text = client.get("/metrics").text
    ok = metric_value(text, 'mlserve_requests_total{endpoint="/predict",method="POST",status="200"}')
    bad = metric_value(text, 'mlserve_requests_total{endpoint="/predict",method="POST",status="422"}')
    assert ok == 3 and bad == 2
    assert bad / (ok + bad) == pytest.approx(0.4)


def test_model_not_loaded_traffic_is_counted(unloaded_client):
    unloaded_client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    text = unloaded_client.get("/metrics").text
    assert 'error_type="model_not_loaded"' in text
    assert metric_value(text, "mlserve_model_loaded") == 0.0


# -------------------------------------------------------------------- high traffic


def test_high_volume_traffic_is_accounted_exactly(client):
    total = 0
    for i in range(40):
        size = (i % 5) + 1
        client.post("/predict", json={"records": [EXAMPLE_RECORD] * size})
        total += size
    text = client.get("/metrics").text
    assert metric_value(text, 'mlserve_predictions_total{model_version="7"}') == total
    assert client.app_state.store.count() == total


# ------------------------------------------------------------- distribution shift


def test_feature_histograms_follow_the_input_distribution(client):
    young = [{**EXAMPLE_RECORD, "age": 20} for _ in range(20)]
    client.post("/predict", json={"records": young})
    text = client.get("/metrics").text
    assert metric_value(text, 'mlserve_feature_age_bucket{le="20.0"}') == 20
    assert metric_value(text, "mlserve_feature_age_sum") == 400.0


def test_prediction_score_histogram_reflects_output_shift(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD] * 10})
    text = client.get("/metrics").text
    assert metric_value(text, 'mlserve_prediction_score_count{model_version="7"}') == 10


# ----------------------------------------------------------------- sampling policy


def test_feature_sampling_rate_is_honoured():
    metrics = ServingMetrics(feature_sample_rate=0.0, seed=0)
    assert metrics.observe_features([EXAMPLE_RECORD] * 100) == 0
    full = ServingMetrics(feature_sample_rate=1.0, seed=0)
    assert full.observe_features([EXAMPLE_RECORD] * 100) == 100


def test_partial_sampling_is_deterministic_for_a_fixed_seed():
    a = ServingMetrics(feature_sample_rate=0.5, seed=11).observe_features([EXAMPLE_RECORD] * 200)
    b = ServingMetrics(feature_sample_rate=0.5, seed=11).observe_features([EXAMPLE_RECORD] * 200)
    assert a == b
    assert 0 < a < 200


# ------------------------------------------------------------------ durable store


def test_every_scored_record_is_persisted_with_its_features(client):
    record = {**EXAMPLE_RECORD, "age": 44, "hours_per_week": 55}
    client.post("/predict", json={"records": [record]})
    frame = client.app_state.store.recent_features(10)
    assert len(frame) == 1
    assert int(frame.iloc[0]["age"]) == 44
    assert int(frame.iloc[0]["hours_per_week"]) == 55
    assert frame.iloc[0]["model_version"] == "7"


def test_stored_features_cover_the_whole_contract(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    frame = client.app_state.store.recent_features(10)
    for column in FEATURE_NAMES:
        assert column in frame.columns


def test_summary_endpoint_reports_the_traffic(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD] * 4})
    summary = client.get("/monitoring/summary").json()
    assert summary["n_predictions"] == 4
    assert summary["by_model_version"] == {"7": 4}
    assert summary["served_model_version"] == "7"
    assert 0.0 <= summary["mean_probability"] <= 1.0


def test_summary_window_filters_by_time(client):
    client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert client.get("/monitoring/summary?window_seconds=60").json()["n_predictions"] == 1
    assert client.get("/monitoring/summary?window_seconds=0").json()["n_predictions"] == 0


def test_store_can_be_disabled(tmp_path):
    from mlserve.monitoring.store import PredictionStore

    store = PredictionStore(tmp_path / "off.sqlite", enabled=False)
    assert store.log_predictions(
        request_id="r", model_name="m", model_version="1",
        features=[EXAMPLE_RECORD], probabilities=[0.5], predictions=[1], latency_ms=1.0,
    ) == 0
    assert store.summary() == {"enabled": False}
