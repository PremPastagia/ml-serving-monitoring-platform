#!/usr/bin/env python
"""Aggregate every measurement into the top-level result files.

Reads only what other scripts have already written, so nothing here can invent a
number. If a source file is missing, the corresponding rows are simply absent and the
run says so, rather than emitting a placeholder that could be mistaken for a result.

Writes EVALUATION_RESULTS.csv and results/summary.json.

    python scripts/collect_results.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd

from mlserve.config import environment_info, git_commit, load_config


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def evaluation_rows(summary: dict | None) -> list[dict]:
    """One row per (model, split) from the last training run."""
    if not summary:
        return []
    rows = []
    for run in summary.get("runs", []):
        for split, result in (run.get("evaluations") or {}).items():
            rows.append({
                "run_name": run["run_name"],
                "estimator": run["estimator"],
                "split": split,
                "n_rows": result.get("n_rows"),
                "positive_rate": result.get("positive_rate"),
                "roc_auc": result.get("roc_auc"),
                "pr_auc": result.get("pr_auc"),
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1": result.get("f1"),
                "brier": result.get("brier"),
                "log_loss": result.get("log_loss"),
                "threshold": result.get("threshold"),
                "training_seconds": run.get("training_seconds"),
                "seed": run.get("seed"),
                "dataset_version": run.get("dataset_version"),
                "split_id": run.get("split_id"),
                "code_version": run.get("code_version"),
                "git_commit": run.get("git_commit"),
                "config_digest": run.get("config_digest"),
                "training_fingerprint": run.get("training_fingerprint"),
                "selected": run["run_name"] == summary.get("selected_run"),
            })
    return rows


def main() -> int:
    config = load_config()
    root = config.path("paths.results_dir").parent
    results = config.path("paths.results_dir")

    training = read_json(results / "training" / "last_training_summary.json")
    drift = read_json(results / "drift" / "drift_experiment.json")
    rollback = read_json(results / "serving" / "rollback_through_serving.json")
    smoke = read_json(results / "serving" / "smoke_test.json")
    retraining = read_json(results / "retraining" / "retraining_experiment.json")

    missing = [name for name, payload in {
        "training": training, "drift": drift, "rollback": rollback,
        "smoke": smoke, "retraining": retraining,
    }.items() if payload is None]

    rows = evaluation_rows(training)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(root / "EVALUATION_RESULTS.csv", index=False)
        print(f"EVALUATION_RESULTS.csv   {len(frame)} rows")
    else:
        print("EVALUATION_RESULTS.csv   SKIPPED (no training summary)")

    serving_csv = root / "SERVING_BENCHMARKS.csv"
    serving = pd.read_csv(serving_csv) if serving_csv.exists() else pd.DataFrame()
    drift_csv = root / "DRIFT_RESULTS.csv"
    drift_frame = pd.read_csv(drift_csv) if drift_csv.exists() else pd.DataFrame()

    best_single = None
    if not serving.empty:
        pinned = serving[(serving.get("server_thread_pin") == 1)
                         & (serving["concurrency"] == 1) & (serving["batch_size"] == 1)]
        if not pinned.empty:
            best_single = pinned.sort_values("throughput_rps", ascending=False).iloc[0].to_dict()

    summary = {
        "generated_by": "scripts/collect_results.py",
        "git_commit": git_commit(),
        "environment": environment_info().to_dict(),
        "platform": platform.platform(),
        "missing_sources": missing,
        "dataset_version": (training or {}).get("dataset_version"),
        "split_id": (training or {}).get("split_id"),
        "selected_run": (training or {}).get("selected_run"),
        "registered": (training or {}).get("registered"),
        "evaluation_rows": len(rows),
        "serving": {
            "rows": int(len(serving)),
            "configurations": sorted(serving["scenario"].tolist()) if not serving.empty else [],
            "best_single_request": best_single,
            "any_run_on_busy_host": (bool(serving["host_busy_before_run"].any())
                                     if "host_busy_before_run" in serving else None),
            "cold_start_seconds": (float(serving["cold_start_seconds"].min())
                                   if "cold_start_seconds" in serving and not serving.empty else None),
            "total_errors": (int(serving["requests_failed"].sum())
                             if "requests_failed" in serving else None),
        },
        "drift": {
            "scenarios": int(len(drift_frame)),
            "trials_per_scenario": (drift or {}).get("trials"),
            "baseline_roc_auc": (drift or {}).get("baseline_roc_auc"),
            "thresholds": (drift or {}).get("thresholds"),
            "summary": drift_frame.to_dict(orient="records") if not drift_frame.empty else [],
        },
        "retraining": (retraining or {}).get("summary"),
        "rollback": {
            "verified_through_serving": (rollback or {}).get("rollback_verified_through_serving"),
            "promotion_seconds": ((rollback or {}).get("promotion") or {}).get("end_to_end_seconds"),
            "rollback_seconds": ((rollback or {}).get("rollback") or {}).get("end_to_end_seconds"),
            "probe_requests": ((rollback or {}).get("overall_traffic") or {}).get("requests"),
            "probe_failed": ((rollback or {}).get("overall_traffic") or {}).get("failed"),
        },
        "smoke": {"cold_start_seconds": (smoke or {}).get("cold_start_seconds")},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("results/summary.json     written")

    if missing:
        print(f"\nNOTE: no results found for {missing}. Those sections are empty; run the "
              f"corresponding script to populate them.")
    print("COLLECT_RESULTS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
