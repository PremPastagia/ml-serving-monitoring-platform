#!/usr/bin/env python
"""Start the real server, exercise every endpoint, and shut it down.

This is the integration check CI runs: it proves the packaged application boots,
loads a model and answers correctly over HTTP -- which unit tests with a stubbed
loader deliberately do not prove.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from mlserve.config import load_config
from mlserve.data.schema import FEATURE_NAMES
from mlserve.serving.schemas import EXAMPLE_RECORD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import ServerProcess, find_free_port  # noqa: E402


class SmokeFailure(AssertionError):
    pass


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        raise SmokeFailure(f"{name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--allow-degraded", action="store_true",
                        help="pass when no model is registered, checking only that the "
                             "service boots and reports itself degraded")
    args = parser.parse_args(argv)

    config = load_config()
    host = str(config.require("serving.host"))
    port = find_free_port(host, args.port or int(config.require("serving.port")))
    results_dir = config.path("paths.results_dir") / "serving"
    results_dir.mkdir(parents=True, exist_ok=True)

    server = ServerProcess(host, port, log_path=results_dir / "smoke_server.log",
                           thread_pin=1)
    server.start()
    print(f"server ready in {server.cold_start_seconds:.3f}s at {server.base_url}\n")

    findings: dict = {"cold_start_seconds": round(server.cold_start_seconds, 3)}
    try:
        with httpx.Client(base_url=server.base_url, timeout=30.0) as client:
            health = client.get("/health")
            check("GET /health is 200", health.status_code == 200, health.text)
            body = health.json()
            findings["health"] = body

            if not body["model_loaded"]:
                check("service reports itself degraded", body["status"] == "degraded")
                check("GET /ready is 503 without a model", client.get("/ready").status_code == 503)
                if args.allow_degraded:
                    print("\nno model registered; degraded-mode checks passed")
                    print("SMOKE_TEST_OK")
                    return 0
                raise SmokeFailure("no model loaded; run scripts/train.py first")

            check("GET /ready is 200", client.get("/ready").status_code == 200)

            info = client.get("/model-info")
            check("GET /model-info is 200", info.status_code == 200, info.text)
            info_body = info.json()
            findings["model_info"] = info_body
            check("model-info lists the contract columns",
                  info_body["input_columns"] == FEATURE_NAMES)
            check("model-info carries a dataset version",
                  bool(info_body.get("dataset_version")), str(info_body))

            single = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
            check("POST /predict is 200", single.status_code == 200, single.text)
            prediction = single.json()
            findings["prediction"] = prediction
            check("prediction returns a probability in [0, 1]",
                  0.0 <= prediction["predictions"][0]["probability"] <= 1.0)
            check("response carries the served model version",
                  prediction["model_version"] == info_body["model_version"])
            check("response carries a request id", bool(prediction["request_id"]))

            batch = client.post("/predict", json={"records": [EXAMPLE_RECORD] * 16})
            check("batch prediction returns one result per record",
                  batch.status_code == 200 and len(batch.json()["predictions"]) == 16)

            bad = client.post("/predict", json={"records": [{**EXAMPLE_RECORD, "age": 900}]})
            check("invalid payload is rejected with 422", bad.status_code == 422, bad.text)
            check("error response has the documented shape",
                  set(bad.json()) == {"request_id", "error"}
                  and bad.json()["error"]["type"] == "validation_error")

            missing = client.post("/predict", json={"records": [{}]})
            check("empty record is rejected with 422", missing.status_code == 422)

            metrics = client.get("/metrics")
            check("GET /metrics is 200", metrics.status_code == 200)
            for series in ("mlserve_requests_total", "mlserve_predictions_total",
                           "mlserve_request_latency_seconds_bucket", "mlserve_model_info",
                           "mlserve_errors_total"):
                check(f"/metrics exposes {series}", series in metrics.text)

            summary = client.get("/monitoring/summary")
            check("GET /monitoring/summary is 200", summary.status_code == 200)
            check("monitoring counted the predictions",
                  summary.json()["n_predictions"] >= 17, summary.text)
            findings["monitoring_summary"] = summary.json()

            reload_result = client.post("/admin/reload")
            check("POST /admin/reload is 200", reload_result.status_code == 200,
                  reload_result.text)
            findings["reload"] = reload_result.json()

            check("unknown route is 404", client.get("/nope").status_code == 404)
            check("OpenAPI schema is served", client.get("/openapi.json").status_code == 200)
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        server.stop()

    (results_dir / "smoke_test.json").write_text(json.dumps(findings, indent=2))
    print("\nSMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
