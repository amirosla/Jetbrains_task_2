"""
End-to-end training and evaluation pipeline.

Run this script to:
  1. Generate synthetic CloudWatch metrics
  2. Build sliding-window features
  3. Train the LightGBM incident detector
  4. Run walk-forward cross-validation
  5. Evaluate on the held-out test set (scalar metrics + incident-level recall)
  6. Save the trained model, metrics JSON, and evaluation dashboard PNG

Usage
-----
    python main.py                      # default parameters
    python main.py --W 60 --H 10 --n_steps 14400 --seed 42
    python main.py --no-plots           # skip matplotlib figure (headless CI)

Output
------
    results/evaluation.png   — 6-panel dashboard
    results/metrics.json     — all numeric results
    models/detector.joblib   — serialised model + scaler bundle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate the predictive alerting system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--W", type=int, default=60,
                   help="Look-back window length (minutes)")
    p.add_argument("--H", type=int, default=10,
                   help="Prediction horizon (minutes); controls advance warning time")
    p.add_argument("--n_steps", type=int, default=14_400,
                   help="Number of synthetic time steps (~10 days at 1-min resolution)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of walk-forward CV folds")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib figure generation (useful in headless CI)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for output artefacts")
    p.add_argument("--model-dir", type=str, default="models",
                   help="Directory for model artefacts")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_metrics(metrics: dict) -> None:
    scalar = metrics.get("scalar", metrics)
    print(f"  AUC-ROC  : {scalar.get('auc_roc', 'n/a'):.4f}")
    print(f"  AUC-PR   : {scalar.get('auc_pr', 'n/a'):.4f}")
    print(f"  F1       : {scalar.get('f1', 'n/a'):.4f}")
    print(f"  Precision: {scalar.get('precision', 'n/a'):.4f}")
    print(f"  Recall   : {scalar.get('recall', 'n/a'):.4f}")
    print(f"  FPR      : {scalar.get('fpr', 'n/a'):.4f}")
    print(f"  Threshold: {scalar.get('threshold', 'n/a'):.4f}")
    if "incident_level" in metrics:
        il = metrics["incident_level"]
        print(
            f"\n  Incident-level recall : {il['incident_recall']:.4f}"
            f"  ({il['caught']}/{il['total_incidents']} incidents caught)"
        )
        if il["lead_times"]:
            print(
                f"  Mean lead time        : {il['mean_lead_time']:.1f} steps"
                f"  |  Median: {il['median_lead_time']:.1f} steps"
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    t0 = time.time()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------
    _print_section("1 / 6  Generating synthetic CloudWatch metrics")
    from src.data.generator import GeneratorConfig, generate_dataset

    gen_cfg = GeneratorConfig(n_steps=args.n_steps, seed=args.seed)
    metrics, incident_label, incidents = generate_dataset(gen_cfg)

    n_incident_steps = int(incident_label.sum())
    print(f"  Time steps  : {args.n_steps:,}")
    print(f"  Incidents   : {len(incidents)}  ({n_incident_steps / args.n_steps:.1%} of steps)")
    print(f"  Date range  : {metrics.index[0].date()} → {metrics.index[-1].date()}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    _print_section("2 / 6  Building sliding-window features")
    from src.data.preprocessor import (
        WindowConfig,
        build_dataset,
        chronological_split,
        fit_scaler,
        apply_scaler,
    )

    win_cfg = WindowConfig(W=args.W, H=args.H)
    X, y = build_dataset(metrics, incident_label, win_cfg)

    print(f"  Features per sample : {X.shape[1]}")
    print(f"  Total samples       : {len(X):,}  (positives: {int(y.sum()):,}  {y.mean():.1%})")

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    scaler = fit_scaler(X_train)
    X_train_s, X_val_s, X_test_s = apply_scaler(scaler, X_train, X_val, X_test)

    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. Walk-forward cross-validation
    # ------------------------------------------------------------------
    _print_section("3 / 6  Walk-forward cross-validation")
    from src.model.walk_forward import walk_forward_cv, summarise_cv

    cv_results = walk_forward_cv(X_train_s, y_train, n_folds=args.cv_folds)
    cv_df = summarise_cv(cv_results)

    # Show numeric folds only
    numeric_rows = cv_df[cv_df["fold"] != "mean ± std"]
    for _, row in numeric_rows.iterrows():
        print(
            f"  Fold {row['fold']}  AUC-ROC={row['auc_roc']:.4f}"
            f"  AUC-PR={row['auc_pr']:.4f}  F1={row['f1']:.4f}"
            f"  (train={row['n_train']:,}  val={row['n_val']:,})"
        )
    summary_row = cv_df[cv_df["fold"] == "mean ± std"].iloc[0]
    print(
        f"\n  Summary  AUC-ROC={summary_row['auc_roc']}"
        f"  AUC-PR={summary_row['auc_pr']}  F1={summary_row['f1']}"
    )

    # ------------------------------------------------------------------
    # 4. Final model training
    # ------------------------------------------------------------------
    _print_section("4 / 6  Training final model on train+val")
    from src.model.detector import IncidentDetector

    detector = IncidentDetector()
    detector.fit(X_train_s, y_train, X_val_s, y_val)
    print(f"  Trees used  : {detector.n_estimators_used}")
    print(f"  Threshold   : {detector.threshold:.4f}")

    # ------------------------------------------------------------------
    # 5. Test-set evaluation
    # ------------------------------------------------------------------
    _print_section("5 / 6  Test-set evaluation")
    from src.evaluation.metrics import (
        compute_scalar_metrics,
        incident_level_evaluation,
        zscore_baseline,
        save_metrics,
        plot_evaluation_dashboard,
    )
    from sklearn.metrics import classification_report

    test_probs = detector.predict_proba(X_test_s)
    scalar = compute_scalar_metrics(test_probs, y_test.values, detector.threshold)

    # Incident-level recall — only count incidents in the test period.
    # Incidents in train/val are never visible to the alert series (which is
    # only populated from test-set predictions), so including them would
    # unfairly deflate the incident-recall figure.
    import pandas as pd
    test_start_idx = X_test.index[0]
    raw_index = list(metrics.index)
    test_offset = raw_index.index(test_start_idx)

    alert_arr = np.zeros(len(metrics), dtype=int)
    for i, p in enumerate(test_probs):
        raw_pos = test_offset + i
        if raw_pos < len(metrics):
            alert_arr[raw_pos] = int(p >= detector.threshold)

    alert_series = pd.Series(alert_arr)
    test_incidents = [(s, e) for s, e in incidents if s >= test_offset]
    inc_eval = incident_level_evaluation(alert_series, incident_label, test_incidents, args.H)

    all_metrics = {
        "scalar": scalar,
        "incident_level": inc_eval,
        "W": args.W,
        "H": args.H,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
    }

    _print_metrics(all_metrics)

    test_preds = (test_probs >= detector.threshold).astype(int)
    print(f"\n{classification_report(y_test, test_preds, target_names=['No Incident', 'Incident'])}")

    # Z-score baseline comparison
    baseline_probs = zscore_baseline(X_test_s)
    from sklearn.metrics import roc_auc_score, average_precision_score
    baseline_auc_roc = roc_auc_score(y_test, baseline_probs)
    baseline_auc_pr = average_precision_score(y_test, baseline_probs)
    print(f"  Z-score baseline:  AUC-ROC={baseline_auc_roc:.4f}  AUC-PR={baseline_auc_pr:.4f}")
    print(
        f"  Improvement:       ΔAUC-ROC={scalar['auc_roc'] - baseline_auc_roc:+.4f}"
        f"  ΔAUC-PR={scalar['auc_pr'] - baseline_auc_pr:+.4f}"
    )

    all_metrics["baseline"] = {
        "auc_roc": float(baseline_auc_roc),
        "auc_pr": float(baseline_auc_pr),
    }

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    _print_section("6 / 6  Saving artefacts")

    metrics_path = output_dir / "metrics.json"
    save_metrics(all_metrics, metrics_path)
    print(f"  Metrics  → {metrics_path}")

    model_path = model_dir / "detector.joblib"
    joblib.dump({"detector": detector, "scaler": scaler, "W": args.W, "H": args.H}, model_path)
    print(f"  Model    → {model_path}")

    if not args.no_plots:
        plot_path = output_dir / "evaluation.png"
        plot_evaluation_dashboard(
            probs=test_probs,
            labels=y_test.values,
            threshold=detector.threshold,
            baseline_probs=baseline_probs,
            lead_times=inc_eval["lead_times"],
            feature_importance_df=detector.feature_importance(),
            save_path=plot_path,
        )
        print(f"  Dashboard → {plot_path}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
