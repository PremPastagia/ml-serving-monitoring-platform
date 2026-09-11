#!/usr/bin/env python
"""Phase 1-2 entrypoint: validate -> split -> train -> evaluate -> track -> register.

Trains the configured estimator plus the linear baseline so that every MLflow
experiment contains a comparison rather than a single unanchored number, then
registers the best run and (on first run) points the `production` alias at it.

    python scripts/train.py                    # full run, registers the best model
    python scripts/train.py --no-register      # experiment only
    python scripts/train.py --seed 7           # reproducibility probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlserve.config import load_config
from mlserve.models.registry import PRODUCTION, ModelRegistry
from mlserve.models.tracking import best_run, log_training_run
from mlserve.models.train import load_and_split, save_bundle, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--seed", type=int, default=None, help="override the configured seed")
    p.add_argument("--no-baseline", action="store_true", help="skip the logistic-regression baseline")
    p.add_argument("--no-register", action="store_true", help="track runs but do not touch the registry")
    p.add_argument("--no-mlflow", action="store_true", help="train and save locally without tracking")
    p.add_argument("--out", default=None, help="bundle output directory")
    p.add_argument("--results-dir", default=None,
                   help="where to write the run summary. Defaults to the configured "
                        "results directory; pass an explicit path when training into a "
                        "scratch location so the shared baseline is not overwritten.")
    p.add_argument("--run-suffix", default="", help="suffix appended to run names")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.seed

    split = load_and_split(config)
    print(f"dataset  : {split.dataset_version.dataset_id}")
    print(f"split    : {split.split_id}  sizes={split.sizes}")

    candidates: list[tuple[str, dict]] = [
        (str(config.require("model.estimator")), dict(config.require("model.params")))
    ]
    if not args.no_baseline:
        candidates.append(
            (str(config.require("model.baseline.estimator")), dict(config.require("model.baseline.params")))
        )

    results = []
    for estimator, params in candidates:
        pipeline, run = train_model(
            config, split=split, estimator=estimator, params=params, seed=seed,
            run_name=f"{estimator}-seed{seed}{args.run_suffix}",
        )
        logged = None
        if not args.no_mlflow:
            X_val, _ = split.xy("validation")
            logged = log_training_run(pipeline, run, config, input_example=X_val)
        results.append((pipeline, run, logged))
        print(
            f"trained  : {run.run_name:38s} "
            f"val_roc_auc={run.metrics['validation_roc_auc']:.5f} "
            f"test_roc_auc={run.metrics['test_roc_auc']:.5f} "
            f"fit={run.training_seconds:.2f}s"
            + (f" run_id={logged.run_id[:8]}" if logged else "")
        )

    best_pipeline, best_training_run, best_logged = max(
        results, key=lambda r: r[1].metrics["validation_roc_auc"]
    )
    print(f"selected : {best_training_run.run_name} (highest validation roc_auc)")

    out_dir = Path(args.out) if args.out else config.path("paths.artifact_dir") / "current"
    save_bundle(best_pipeline, best_training_run, out_dir)
    print(f"bundle   : {out_dir}")

    registered = None
    if not args.no_register and not args.no_mlflow and best_logged:
        registry = ModelRegistry(config)
        ref = registry.register(
            best_logged.model_uri,
            tags={
                "dataset_version": best_training_run.dataset_version,
                "code_version": best_training_run.code_version,
                "git_commit": best_training_run.git_commit,
                "validation_roc_auc": f"{best_training_run.metrics['validation_roc_auc']:.6f}",
                "test_roc_auc": f"{best_training_run.metrics['test_roc_auc']:.6f}",
                "training_fingerprint": best_training_run.training_fingerprint,
            },
        )
        registered = ref.to_dict()
        print(f"registered: version {ref.version}")
        if registry.production() is None:
            registry.set_alias(PRODUCTION, ref.version)
            print(f"alias    : {PRODUCTION} -> version {ref.version} (first registration)")
        else:
            print(f"alias    : {PRODUCTION} unchanged (use scripts/promote.py to move it)")

    summary = {
        "dataset_version": split.dataset_version.dataset_id,
        "split_id": split.split_id,
        "seed": seed,
        "runs": [r[1].to_dict() for r in results],
        "selected_run": best_training_run.run_name,
        "registered": registered,
    }
    results_dir = (Path(args.results_dir) if args.results_dir
                   else config.path("paths.results_dir")) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "last_training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    if not args.no_mlflow:
        top = best_run(config)
        if top is not None:
            print(f"best run : {top['tags.mlflow.runName']} "
                  f"val_roc_auc={top['metrics.validation_roc_auc']:.5f}")
    print("TRAIN_PIPELINE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
