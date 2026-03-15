"""
Local end-to-end simulation of the Lambda-based alerting pipeline.

This script exercises the full operational loop without any AWS infrastructure:
  1. Train a model (mimicking the daily retrain Lambda)
  2. Simulate a stream of incoming metric windows (mimicking the per-minute
     predict Lambda)
  3. Raise console alerts whenever predicted probability crosses the threshold
  4. Print a simulation summary report

Usage
-----
    python scripts/simulate_pipeline.py
    python scripts/simulate_pipeline.py --n-windows 500 --cooldown 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on the path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate the retrain + predict Lambda loop locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-windows", type=int, default=300,
                   help="Number of consecutive 1-minute windows to simulate")
    p.add_argument("--cooldown", type=int, default=5,
                   help="Alert cooldown in simulated minutes")
    p.add_argument("--W", type=int, default=60, help="Look-back window length")
    p.add_argument("--H", type=int, default=10, help="Prediction horizon")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("  Predictive Alerting System — Local Pipeline Simulation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Retrain (daily Lambda equivalent)
    # ------------------------------------------------------------------
    print("\n[RETRAIN]  Generating training data and fitting model...")
    t0 = time.time()

    from src.data.generator import GeneratorConfig, generate_dataset, METRIC_NAMES
    from src.data.preprocessor import (
        WindowConfig, build_dataset, chronological_split,
        fit_scaler, apply_scaler,
    )
    from src.model.detector import IncidentDetector

    gen_cfg = GeneratorConfig(n_steps=10_000, seed=args.seed)
    metrics_train, label_train, incidents_train = generate_dataset(gen_cfg)

    win_cfg = WindowConfig(W=args.W, H=args.H)
    X, y = build_dataset(metrics_train, label_train, win_cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_split(X, y)
    scaler = fit_scaler(X_tr)
    X_tr_s, X_va_s = apply_scaler(scaler, X_tr, X_va)

    detector = IncidentDetector()
    detector.fit(X_tr_s, y_tr, X_va_s, y_va)

    print(f"  Model trained in {time.time() - t0:.1f}s")
    print(f"  Threshold: {detector.threshold:.4f}  |  Trees: {detector.n_estimators_used}")

    # ------------------------------------------------------------------
    # Phase 2: Stream simulation (per-minute Lambda equivalent)
    # ------------------------------------------------------------------
    print(f"\n[PREDICT]  Simulating {args.n_windows} incoming metric windows...")

    # Generate a fresh dataset for the simulated "live" stream
    import numpy as np
    import pandas as pd
    from src.data.preprocessor import _extract_window_features, build_feature_names

    live_cfg = GeneratorConfig(n_steps=args.n_windows + args.W + 20, seed=args.seed + 1)
    metrics_live, label_live, incidents_live = generate_dataset(live_cfg)

    live_values = metrics_live.values.astype(float)
    live_labels = label_live.values.astype(int)
    feature_names = build_feature_names(METRIC_NAMES)

    fired_alerts: list[dict] = []
    last_alert_t = -args.cooldown  # allow immediate first alert

    for t in range(args.W, args.W + args.n_windows):
        window = live_values[t - args.W : t]
        features = _extract_window_features(window)
        X_live = pd.DataFrame([features], columns=feature_names)
        X_live_s = pd.DataFrame(
            scaler.transform(X_live), columns=feature_names
        )
        prob = float(detector.predict_proba(X_live_s)[0])
        alert = prob >= detector.threshold

        # Cooldown logic
        step = t - args.W
        on_cooldown = (step - last_alert_t) < args.cooldown

        if alert and not on_cooldown:
            last_alert_t = step
            actual_incident = int(label_live.iloc[t]) == 1
            fired_alerts.append(
                {
                    "step": step,
                    "probability": round(prob, 4),
                    "incident_at_t": actual_incident,
                }
            )
            tag = "[TRUE  POSITIVE]" if actual_incident else "[FALSE POSITIVE]"
            print(
                f"  {tag}  step={step:4d}  prob={prob:.4f}"
            )

    # ------------------------------------------------------------------
    # Phase 3: Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)

    from src.evaluation.metrics import incident_level_evaluation

    # Build alert series for incident-level recall
    alert_arr = np.zeros(len(metrics_live), dtype=int)
    for rec in fired_alerts:
        alert_arr[rec["step"] + args.W] = 1
    alert_series = pd.Series(alert_arr)

    inc_eval = incident_level_evaluation(
        alert_series, label_live, incidents_live, args.H
    )

    n_total = args.n_windows
    n_alerts = len(fired_alerts)
    n_tp = sum(1 for a in fired_alerts if a["incident_at_t"])
    n_fp = n_alerts - n_tp
    alert_rate = n_alerts / n_total

    print(f"\n  Windows simulated     : {n_total}")
    print(f"  Alerts fired          : {n_alerts}  ({alert_rate:.1%} of windows)")
    print(f"  True positives        : {n_tp}")
    print(f"  False positives       : {n_fp}")
    print(f"\n  Incident-level recall : {inc_eval['incident_recall']:.4f}"
          f"  ({inc_eval['caught']}/{inc_eval['total_incidents']} incidents caught)")
    if inc_eval["lead_times"]:
        print(f"  Mean lead time        : {inc_eval['mean_lead_time']:.1f} simulated minutes")
        print(f"  Median lead time      : {inc_eval['median_lead_time']:.1f} simulated minutes")

    print(f"\n  Simulation complete.\n")


if __name__ == "__main__":
    main()
