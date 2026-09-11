#!/usr/bin/env python
"""Phase 6: the controlled drift experiment.

For each scenario, ``--trials`` independent windows are drawn with different seeds and
passed through the detector. That repetition is the whole point: a single synthetic
example tells you nothing about a detector's error rates, and both numbers this
experiment exists to produce -- detection rate and false-positive rate -- are
proportions over trials.

What is measured
----------------
detection rate        fraction of trials on a drifted scenario where drift was flagged
false-positive rate   fraction of trials on `no_drift` where drift was flagged
detection latency     windows consumed before the first alert, when windows arrive in
                      sequence (measured separately by --latency-trials)
per-feature           PSI, test statistic, p-value and Wasserstein for every feature
performance delta     labelled ROC-AUC of the production model on the drifted window,
                      against its score on an undrifted window -- the number that says
                      whether the drift actually mattered

Outputs DRIFT_RESULTS.csv, results/drift/*.json and results/plots/*.png.

    python scripts/drift_experiment.py --trials 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mlserve.config import load_config
from mlserve.data.schema import FEATURE_NAMES, TARGET
from mlserve.models.evaluate import compute_metrics
from mlserve.models.train import load_and_split
from mlserve.monitoring.drift import DriftDetector, prediction_drift
from mlserve.monitoring.scenarios import SCENARIOS, build_scenario
from mlserve.serving.model_loader import ModelLoader

DEFAULT_TRIALS = 30
DEFAULT_WINDOW = 2000
LATENCY_MAX_WINDOWS = 10


def load_production_model(config):
    """Prefer the registry; fall back to the local bundle so the experiment can run
    on a machine where MLflow was never started."""
    for source in ("registry", "file"):
        loader = ModelLoader(config, source=source)
        model = loader.try_load()
        if model is not None:
            return model
    raise SystemExit("no model available: run `python scripts/train.py` first")


def score_window(model, window: pd.DataFrame, threshold: float) -> tuple[np.ndarray, dict | None]:
    """Score a window, returning probabilities and (if labelled) its metrics."""
    usable = [c for c in FEATURE_NAMES if c in window.columns]
    frame = window.reindex(columns=FEATURE_NAMES)
    if len(usable) < len(FEATURE_NAMES) or frame[FEATURE_NAMES].isna().any().any():
        # A schema-broken window cannot be scored. That is the honest answer: in
        # production the request would be rejected at the API boundary.
        return np.array([]), None
    proba = model.predict_proba(frame)
    metrics = None
    if TARGET in window.columns:
        y = (window[TARGET].astype(str) == ">50K").astype(int).to_numpy()
        metrics = compute_metrics(y, proba, threshold=threshold)
    return proba, metrics


def run_detection_trials(detector, base, model, *, scenario, trials, window, threshold,
                         reference_scores, mean_shift_alert) -> list[dict]:
    rows = []
    for trial in range(trials):
        seed = 10_000 + trial
        current = build_scenario(scenario, base, seed=seed, n_rows=window)
        start = time.perf_counter()
        report = detector.detect(current, scenario=scenario)
        elapsed = time.perf_counter() - start

        proba, metrics = score_window(model, current, threshold)
        pred_drift = (
            prediction_drift(reference_scores, proba,
                             psi_alert=detector.psi_alert, ks_alpha=detector.ks_alpha,
                             mean_shift_alert=mean_shift_alert)
            if len(proba) else None
        )

        rows.append({
            "scenario": scenario,
            "trial": trial,
            "seed": seed,
            "expect_drift": SCENARIOS[scenario].expect_drift,
            "kind": SCENARIOS[scenario].kind,
            "input_drift_detected": report.drift_detected,
            "n_drifted_features": report.n_drifted,
            "n_schema_failures": len(report.schema_failures),
            "max_psi": round(report.max_psi, 6),
            "mean_psi": round(report.mean_psi, 6),
            "detection_seconds": round(elapsed, 6),
            "prediction_drift_detected": (pred_drift or {}).get("drifted"),
            "prediction_shape_drift": (pred_drift or {}).get("shape_drift"),
            "prediction_level_drift": (pred_drift or {}).get("level_drift"),
            "prediction_psi": (pred_drift or {}).get("psi"),
            "prediction_mean": (pred_drift or {}).get("current_mean"),
            "prediction_mean_shift": (pred_drift or {}).get("mean_shift"),
            "roc_auc": (metrics or {}).get("roc_auc"),
            "pr_auc": (metrics or {}).get("pr_auc"),
            "accuracy": (metrics or {}).get("accuracy"),
            "scorable": bool(len(proba)),
        })
    return rows


def run_latency_trials(detector, base, *, scenario, trials, window) -> list[dict]:
    """How many sequential windows arrive before the detector first alerts.

    Windows are disjoint draws, so this measures 'how much drifted traffic must the
    service see', not wall-clock time.
    """
    rows = []
    for trial in range(trials):
        detected_at = None
        for index in range(LATENCY_MAX_WINDOWS):
            current = build_scenario(scenario, base, seed=50_000 + trial * 100 + index,
                                     n_rows=window)
            if detector.detect(current, scenario=scenario).drift_detected:
                detected_at = index + 1
                break
        rows.append({"scenario": scenario, "trial": trial,
                     "windows_to_detection": detected_at,
                     "max_windows": LATENCY_MAX_WINDOWS})
    return rows


def per_feature_report(detector, base, *, scenario, window) -> pd.DataFrame:
    current = build_scenario(scenario, base, seed=99_999, n_rows=window)
    report = detector.detect(current, scenario=scenario)
    frame = report.to_frame()
    frame.insert(0, "scenario", scenario)
    return frame


def summarise(trials: pd.DataFrame, baseline_auc: float | None) -> pd.DataFrame:
    rows = []
    for scenario, group in trials.groupby("scenario", sort=False):
        expect = bool(group["expect_drift"].iloc[0])
        detected = group["input_drift_detected"].mean()
        scorable = group[group["scorable"]]
        mean_auc = float(scorable["roc_auc"].mean()) if len(scorable) else float("nan")
        rows.append({
            "scenario": scenario,
            "kind": group["kind"].iloc[0],
            "expect_drift": expect,
            "trials": len(group),
            "input_drift_detection_rate": round(float(detected), 4),
            "input_drift_false_positive_rate": round(float(detected), 4) if not expect else None,
            "prediction_drift_detection_rate": (
                round(float(group["prediction_drift_detected"].dropna().mean()), 4)
                if group["prediction_drift_detected"].notna().any() else None
            ),
            "mean_drifted_features": round(float(group["n_drifted_features"].mean()), 3),
            "mean_schema_failures": round(float(group["n_schema_failures"].mean()), 3),
            "mean_max_psi": round(float(group["max_psi"].mean()), 4),
            "mean_detection_seconds": round(float(group["detection_seconds"].mean()), 5),
            "mean_roc_auc": round(mean_auc, 6) if mean_auc == mean_auc else None,
            "roc_auc_delta_vs_no_drift": (
                round(mean_auc - baseline_auc, 6)
                if baseline_auc is not None and mean_auc == mean_auc else None
            ),
            "scorable": bool(group["scorable"].all()),
        })
    return pd.DataFrame(rows)


def make_plots(trials: pd.DataFrame, per_feature: pd.DataFrame, summary: pd.DataFrame,
               out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Detection rate per scenario.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = summary.sort_values("input_drift_detection_rate")
    colours = ["#c0392b" if (r.expect_drift and r.input_drift_detection_rate < 0.5)
               else "#e67e22" if (not r.expect_drift and r.input_drift_detection_rate > 0.1)
               else "#2980b9" for r in order.itertuples()]
    ax.barh(order["scenario"], order["input_drift_detection_rate"], color=colours)
    ax.set_xlabel("input-drift detection rate")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Detection rate by scenario ({int(summary['trials'].iloc[0])} trials each)")
    for i, value in enumerate(order["input_drift_detection_rate"]):
        ax.text(value + 0.02, i, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    path = out_dir / "drift_detection_rate.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 2. PSI distribution per scenario.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    scenarios = list(trials["scenario"].unique())
    ax.boxplot([trials.loc[trials["scenario"] == s, "max_psi"] for s in scenarios],
               tick_labels=scenarios, vert=True)
    ax.axhline(0.25, color="#c0392b", linestyle="--", label="alert threshold (PSI 0.25)")
    ax.axhline(0.10, color="#e67e22", linestyle=":", label="warn threshold (PSI 0.10)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("max PSI across features")
    ax.set_title("Maximum per-feature PSI by scenario")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    path = out_dir / "drift_psi_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 3. Per-feature PSI heatmap.
    pivot = per_feature.pivot_table(index="feature", columns="scenario", values="psi")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = np.log10(np.clip(pivot.to_numpy(dtype=float), 1e-4, None))
    image = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Per-feature PSI by scenario (log10)")
    fig.colorbar(image, ax=ax, label="log10(PSI)")
    fig.tight_layout()
    path = out_dir / "drift_per_feature_psi.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 4. Model performance under each scenario.
    scored = summary[summary["mean_roc_auc"].notna()]
    if not scored.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(scored["scenario"], scored["mean_roc_auc"], color="#2980b9")
        ax.set_ylabel("mean ROC-AUC on the drifted window")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Model performance under each drift scenario")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        for i, value in enumerate(scored["mean_roc_auc"]):
            ax.text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        path = out_dir / "drift_model_performance.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--latency-trials", type=int, default=10)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    split = load_and_split(config)
    model = load_production_model(config)
    threshold = float(config.require("evaluation.decision_threshold"))

    reference_rows = int(config.require("drift.reference_sample"))
    reference = split.train[FEATURE_NAMES].sample(
        n=min(reference_rows, len(split.train)), random_state=config.seed
    ).reset_index(drop=True)
    detector = DriftDetector.from_config(reference, config)
    mean_shift_alert = float(config.get("drift.prediction_mean_shift_alert", 0.25))

    # The reference score distribution: what the model's output looks like on
    # undrifted traffic. Prediction drift is measured against this.
    reference_scores = model.predict_proba(reference)

    scenarios = args.scenarios or list(SCENARIOS)
    print(f"model      : version {model.model_version} ({model.source})")
    print(f"reference  : {len(reference)} rows | window {args.window} | {args.trials} trials")
    print(f"thresholds : {detector.thresholds}")
    print(f"host load  : {os.getloadavg()}\n")

    trial_rows: list[dict] = []
    latency_rows: list[dict] = []
    feature_frames: list[pd.DataFrame] = []

    for scenario in scenarios:
        start = time.perf_counter()
        trial_rows.extend(run_detection_trials(
            detector, split.test, model, scenario=scenario, trials=args.trials,
            window=args.window, threshold=threshold, reference_scores=reference_scores,
            mean_shift_alert=mean_shift_alert,
        ))
        latency_rows.extend(run_latency_trials(
            detector, split.test, scenario=scenario, trials=args.latency_trials,
            window=args.window,
        ))
        feature_frames.append(per_feature_report(detector, split.test,
                                                 scenario=scenario, window=args.window))
        print(f"  {scenario:18s} done in {time.perf_counter() - start:6.1f}s")

    trials = pd.DataFrame(trial_rows)
    latency = pd.DataFrame(latency_rows)
    per_feature = pd.concat(feature_frames, ignore_index=True)

    baseline = trials[(trials["scenario"] == "no_drift") & trials["scorable"]]
    baseline_auc = float(baseline["roc_auc"].mean()) if len(baseline) else None

    summary = summarise(trials, baseline_auc)
    latency_summary = (
        latency.groupby("scenario", sort=False)
        .agg(detection_latency_windows_mean=("windows_to_detection", "mean"),
             detection_latency_windows_max=("windows_to_detection", "max"),
             latency_trials=("trial", "count"),
             never_detected=("windows_to_detection", lambda s: int(s.isna().sum())))
        .reset_index()
    )
    summary = summary.merge(latency_summary, on="scenario", how="left")
    summary["baseline_roc_auc"] = baseline_auc

    results_dir = config.path("paths.results_dir") / "drift"
    results_dir.mkdir(parents=True, exist_ok=True)
    root = config.path("paths.results_dir").parent

    trials.to_csv(results_dir / "drift_trials.csv", index=False)
    per_feature.to_csv(results_dir / "drift_per_feature.csv", index=False)
    latency.to_csv(results_dir / "drift_detection_latency.csv", index=False)
    summary.to_csv(root / "DRIFT_RESULTS.csv", index=False)
    (results_dir / "drift_experiment.json").write_text(json.dumps({
        "model_version": model.model_version,
        "model_source": model.source,
        "dataset_version": split.dataset_version.dataset_id,
        "split_id": split.split_id,
        "reference_rows": len(reference),
        "window_rows": args.window,
        "trials": args.trials,
        "latency_trials": args.latency_trials,
        "max_windows_for_latency": LATENCY_MAX_WINDOWS,
        "thresholds": detector.thresholds,
        "baseline_roc_auc": baseline_auc,
        "host_load_average": os.getloadavg(),
        "summary": summary.to_dict(orient="records"),
    }, indent=2, default=str))

    if not args.no_plots:
        for path in make_plots(trials, per_feature, summary,
                               config.path("paths.results_dir") / "plots"):
            print(f"  plot: {path.relative_to(root)}")

    print("\n" + summary[[
        "scenario", "expect_drift", "input_drift_detection_rate",
        "prediction_drift_detection_rate", "mean_max_psi", "mean_roc_auc",
        "roc_auc_delta_vs_no_drift", "detection_latency_windows_mean",
    ]].to_string(index=False))
    print("\nDRIFT_EXPERIMENT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
