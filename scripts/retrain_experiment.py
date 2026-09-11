#!/usr/bin/env python
"""Phase 7: exercise the retraining workflow end to end and record what it decided.

Four cases are run against a scratch registry so the real one is not disturbed:

``no_drift``            no drift -> the workflow must not retrain at all.
``reject_no_gain``      drift, but retraining on the same data produces a candidate
                        that cannot beat the incumbent by the required margin -> reject.
``promote_better``      a deliberately weak incumbent versus a full-strength candidate
                        -> the improvement is real and measured on a shared holdout, so
                        the gate promotes. The incumbent is weakened by learning rate
                        and tree depth rather than by tree count, so that its inference
                        cost stays comparable: weakening it with fewer boosting rounds
                        also makes it ~4.5x faster, and the latency gate then (rightly)
                        rejects the more accurate candidate for being slower. Holding
                        inference cost fixed isolates the accuracy decision.
``rollback_after``      immediately roll the promotion back and confirm the registry
                        alias returns to the incumbent.

Every case records the decision, the criteria that failed, the holdout metrics of both
models, and the wall-clock cost of retraining, promotion and rollback.

    python scripts/retrain_experiment.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import yaml

from mlserve.config import Config, load_config
from mlserve.data.schema import FEATURE_NAMES
from mlserve.models.registry import PREVIOUS, ModelRegistry
from mlserve.models.tracking import log_training_run
from mlserve.models.train import load_and_split, train_model
from mlserve.monitoring.scenarios import build_scenario
from mlserve.retraining.orchestrator import RetrainingOrchestrator


def scratch_config(base: Config, workdir: Path) -> Config:
    """A config whose MLflow store and artefacts live entirely under ``workdir``."""
    data = copy.deepcopy(base.raw)
    data["mlflow"]["tracking_uri"] = f"sqlite:///{workdir / 'mlflow.db'}"
    data["mlflow"]["artifact_location"] = f"file:{workdir / 'mlruns'}"
    data["mlflow"]["experiment_name"] = "retraining-experiment"
    data["paths"]["artifact_dir"] = str(workdir / "artifacts")
    data["paths"]["prediction_db"] = str(workdir / "predictions.sqlite")
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return Config(data, path)


def register_incumbent(config: Config, registry: ModelRegistry, split, *,
                       params: dict | None, label: str) -> tuple[str, dict]:
    pipeline, run = train_model(config, split=split, params=params,
                               run_name=f"incumbent-{label}", evaluate_test=False)
    X_val, _ = split.xy("validation")
    logged = log_training_run(pipeline, run, config, input_example=X_val,
                              tags={"role": "incumbent", "case": label})
    ref = registry.register(logged.model_uri, tags={
        "role": "incumbent", "case": label,
        "validation_roc_auc": f"{run.metrics['validation_roc_auc']:.6f}",
        "dataset_version": run.dataset_version,
        "training_fingerprint": run.training_fingerprint,
    })
    registry.promote(ref.version)
    return ref.version, run.to_dict()


def summarise(case: str, outcome, extra: dict | None = None) -> dict:
    decision = outcome.decision or {}
    candidate = decision.get("candidate_holdout_metrics") or {}
    incumbent = decision.get("incumbent_holdout_metrics") or {}
    row = {
        "case": case,
        "triggered": outcome.triggered,
        "trigger_reason": outcome.trigger_reason,
        "drift_detected": outcome.drift.get("drift_detected"),
        "n_drifted_features": outcome.drift.get("n_drifted"),
        "n_schema_failures": len(outcome.drift.get("schema_failures", [])),
        "decision": decision.get("decision"),
        "failed_criteria": ";".join(decision.get("failed_criteria", [])),
        "incumbent_version": outcome.incumbent_version,
        "candidate_version": outcome.candidate_version,
        "promoted_version": outcome.promoted_version,
        "incumbent_roc_auc": incumbent.get("roc_auc"),
        "candidate_roc_auc": candidate.get("roc_auc"),
        "roc_auc_delta": (round(candidate["roc_auc"] - incumbent["roc_auc"], 6)
                          if candidate and incumbent else None),
        "candidate_p95_latency_ms": (decision.get("candidate_latency") or {}).get("p95_ms"),
        "retraining_seconds": round(outcome.retraining_seconds, 3),
        "total_seconds": round(outcome.total_seconds, 3),
        "promotion_seconds": (outcome.promotion or {}).get("seconds"),
        "error": outcome.error,
    }
    row.update(extra or {})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)

    base_config = load_config()
    split = load_and_split(base_config)
    reference = split.train[FEATURE_NAMES]
    holdout = split.test
    workdir = Path(tempfile.mkdtemp(prefix="mlserve-retrain-"))
    print(f"scratch registry: {workdir}")
    print(f"dataset: {split.dataset_version.dataset_id} | holdout rows: {len(holdout)}\n")

    rows: list[dict] = []
    details: dict = {}

    try:
        # ---------------------------------------------------------- 1. no drift
        case_dir = workdir / "no_drift"
        case_dir.mkdir()
        config = scratch_config(base_config, case_dir)
        registry = ModelRegistry(config)
        incumbent_version, _ = register_incumbent(config, registry, split,
                                                  params=None, label="no_drift")
        orchestrator = RetrainingOrchestrator(config, registry=registry)
        current = build_scenario("no_drift", holdout, seed=4242, n_rows=args.window)
        outcome = orchestrator.run(reference=reference, current=current,
                                   training_data=split, holdout=holdout, scenario="no_drift")
        rows.append(summarise("no_drift", outcome,
                              {"registry_production_after": registry.production().version}))
        details["no_drift"] = outcome.to_dict()
        print(f"no_drift        : triggered={outcome.triggered} "
              f"(production stays {registry.production().version})")

        # ------------------------------------------- 2. drift, candidate rejected
        case_dir = workdir / "reject"
        case_dir.mkdir()
        config = scratch_config(base_config, case_dir)
        registry = ModelRegistry(config)
        incumbent_version, _ = register_incumbent(config, registry, split,
                                                  params=None, label="reject")
        orchestrator = RetrainingOrchestrator(config, registry=registry)
        current = build_scenario("large_drift", holdout, seed=4242, n_rows=args.window)
        outcome = orchestrator.run(reference=reference, current=current,
                                   training_data=split, holdout=holdout,
                                   scenario="large_drift")
        rows.append(summarise("reject_no_gain", outcome,
                              {"registry_production_after": registry.production().version}))
        details["reject_no_gain"] = outcome.to_dict()
        print(f"reject_no_gain  : decision={outcome.decision['decision']} "
              f"failed={outcome.decision['failed_criteria']} "
              f"(production stays {registry.production().version})")

        # ------------------------------------------ 3. drift, candidate promoted
        case_dir = workdir / "promote"
        case_dir.mkdir()
        config = scratch_config(base_config, case_dir)
        registry = ModelRegistry(config)
        weak = {**dict(base_config.require("model.params")),
                "learning_rate": 0.01, "max_leaf_nodes": 3}
        incumbent_version, _ = register_incumbent(config, registry, split,
                                                  params=weak, label="weak")
        orchestrator = RetrainingOrchestrator(config, registry=registry)
        current = build_scenario("large_drift", holdout, seed=4242, n_rows=args.window)
        promote_outcome = orchestrator.run(reference=reference, current=current,
                                           training_data=split, holdout=holdout,
                                           scenario="large_drift")
        rows.append(summarise("promote_better", promote_outcome,
                              {"registry_production_after": registry.production().version}))
        details["promote_better"] = promote_outcome.to_dict()
        print(f"promote_better  : decision={promote_outcome.decision['decision']} "
              f"incumbent={promote_outcome.decision['incumbent_holdout_metrics']['roc_auc']:.5f} "
              f"candidate={promote_outcome.decision['candidate_holdout_metrics']['roc_auc']:.5f} "
              f"-> production {registry.production().version}")

        # ------------------------------------------------- 4. roll it straight back
        start = time.perf_counter()
        rollback = orchestrator.rollback()
        rollback_seconds = time.perf_counter() - start
        rows.append({
            "case": "rollback_after_promotion",
            "triggered": True,
            "trigger_reason": "explicit rollback after promotion",
            "decision": "rollback",
            "incumbent_version": rollback["to"],
            "candidate_version": rollback["from"],
            "promoted_version": rollback["to"],
            "registry_production_after": registry.production().version,
            "rollback_seconds": round(rollback_seconds, 6),
            "registry_alias_seconds": rollback["seconds"],
            "previous_alias_after": (registry.resolve_alias(PREVIOUS).version
                                     if registry.resolve_alias(PREVIOUS) else None),
            "restored_incumbent": registry.production().version == incumbent_version,
        })
        details["rollback_after_promotion"] = rollback
        print(f"rollback        : {rollback['from']} -> {rollback['to']} in "
              f"{rollback_seconds:.4f}s (production now {registry.production().version})")

    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    frame = pd.DataFrame(rows)
    results_dir = base_config.path("paths.results_dir") / "retraining"
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / "retraining_experiment.csv", index=False)
    (results_dir / "retraining_experiment.json").write_text(json.dumps({
        "dataset_version": split.dataset_version.dataset_id,
        "split_id": split.split_id,
        "holdout_rows": len(holdout),
        "drift_window_rows": args.window,
        "host_load_average": os.getloadavg(),
        "summary": rows,
        "details": details,
    }, indent=2, default=str))

    print("\n" + frame[["case", "triggered", "decision", "failed_criteria",
                        "incumbent_roc_auc", "candidate_roc_auc", "roc_auc_delta",
                        "registry_production_after"]].to_string(index=False))

    expected = {
        "no_drift": (False, None),
        "reject_no_gain": (True, "reject"),
        "promote_better": (True, "promote"),
    }
    for case, (triggered, decision) in expected.items():
        row = frame[frame["case"] == case].iloc[0]
        if bool(row["triggered"]) != triggered or (decision and row["decision"] != decision):
            print(f"\nUNEXPECTED OUTCOME for {case}: triggered={row['triggered']} "
                  f"decision={row['decision']}", file=sys.stderr)
            return 1
    if not bool(frame[frame["case"] == "rollback_after_promotion"].iloc[0]["restored_incumbent"]):
        print("\nROLLBACK DID NOT RESTORE THE INCUMBENT", file=sys.stderr)
        return 1

    print("\nRETRAIN_EXPERIMENT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
