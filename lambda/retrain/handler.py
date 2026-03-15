"""
AWS Lambda handler — daily model retraining.

Triggered on a schedule (e.g. EventBridge rule: rate(1 day)).

Execution flow
--------------
1. Pull recent metric data from CloudWatch (or S3 raw-data bucket).
2. Run the full training pipeline (feature extraction → LightGBM fit).
3. Evaluate on a held-out tail of the data.
4. Serialise the trained detector and upload to S3.
5. Write a JSON metrics summary to S3 alongside the model artifact.
6. Return a structured response so Step Functions / EventBridge can log the
   outcome without parsing CloudWatch Logs.

Environment variables (set in Lambda configuration or SAM template)
-------------------------------------------------------------------
  MODEL_BUCKET      : S3 bucket name for model artifacts
  MODEL_KEY         : S3 key for the joblib artifact  (default: model/detector.joblib)
  METRICS_KEY       : S3 key for the metrics JSON     (default: model/metrics.json)
  DATA_BUCKET       : S3 bucket for raw metric CSVs   (optional; uses synthetic data if absent)
  DATA_KEY_PREFIX   : S3 key prefix for metric files  (default: data/)
  LOOKBACK_DAYS     : days of history to fetch        (default: 30)
  WINDOW_W          : sliding-window length           (default: 60)
  WINDOW_H          : prediction horizon              (default: 10)

Local / unit-test mode
----------------------
Set the env var OFFLINE_MODE=1 to skip all AWS calls and use synthetic data.
This allows the handler to be exercised without real AWS credentials.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Lazy imports (boto3 is only available in Lambda runtime)
# ---------------------------------------------------------------------------

def _get_s3_client():
    import boto3
    return boto3.client("s3")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _training_config() -> dict:
    return {
        "model_bucket": _env("MODEL_BUCKET", "my-alerting-bucket"),
        "model_key": _env("MODEL_KEY", "model/detector.joblib"),
        "metrics_key": _env("METRICS_KEY", "model/metrics.json"),
        "data_bucket": _env("DATA_BUCKET", ""),
        "data_key_prefix": _env("DATA_KEY_PREFIX", "data/"),
        "lookback_days": int(_env("LOOKBACK_DAYS", "30")),
        "W": int(_env("WINDOW_W", "60")),
        "H": int(_env("WINDOW_H", "10")),
        "offline": _env("OFFLINE_MODE", "0") == "1",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data_from_s3(cfg: dict):
    """
    Download the most recent metric CSV files from S3 and concatenate them.

    Each file is expected to have a DatetimeIndex and columns matching
    METRIC_NAMES in src/data/generator.py, plus an 'incident' column.

    Returns (metrics_df, incident_label_series, incidents_list).
    """
    import pandas as pd

    s3 = _get_s3_client()
    prefix = cfg["data_key_prefix"]
    bucket = cfg["data_bucket"]

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = sorted(
        [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".csv")],
        reverse=True,  # most recent first
    )[: cfg["lookback_days"]]

    frames = []
    for key in keys:
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            s3.download_file(bucket, key, tmp.name)
            frames.append(pd.read_csv(tmp.name, index_col=0, parse_dates=True))

    if not frames:
        raise ValueError(f"No CSV files found under s3://{bucket}/{prefix}")

    df = pd.concat(frames).sort_index()
    incident_label = df.pop("incident")
    return df, incident_label, []


def _load_synthetic_data(cfg: dict):
    """Generate synthetic data for offline / test use."""
    from src.data.generator import GeneratorConfig, generate_dataset

    gen_cfg = GeneratorConfig(n_steps=14_400, seed=42)
    return generate_dataset(gen_cfg)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training(cfg: dict) -> dict:
    """
    Execute the full train → evaluate → persist pipeline.

    Returns a dict that mirrors the Lambda response body.
    """
    from src.data.preprocessor import (
        WindowConfig,
        build_dataset,
        chronological_split,
        fit_scaler,
        apply_scaler,
    )
    from src.model.detector import IncidentDetector
    from src.evaluation.metrics import (
        compute_scalar_metrics,
        incident_level_evaluation,
        zscore_baseline,
        save_metrics,
    )

    W, H = cfg["W"], cfg["H"]

    # 1. Load data
    logger.info("Loading data (offline=%s)", cfg["offline"])
    if cfg["offline"] or not cfg["data_bucket"]:
        metrics, incident_label, incidents = _load_synthetic_data(cfg)
    else:
        metrics, incident_label, incidents = _load_data_from_s3(cfg)

    # 2. Feature engineering
    logger.info("Building sliding-window dataset  W=%d  H=%d", W, H)
    win_cfg = WindowConfig(W=W, H=H)
    X, y = build_dataset(metrics, incident_label, win_cfg)

    # 3. Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    # 4. Scale
    scaler = fit_scaler(X_train)
    X_train_s, X_val_s, X_test_s = apply_scaler(scaler, X_train, X_val, X_test)

    # 5. Train
    logger.info("Training LightGBM  train=%d  val=%d  test=%d",
                len(X_train), len(X_val), len(X_test))
    detector = IncidentDetector()
    detector.fit(X_train_s, y_train, X_val_s, y_val)
    logger.info("Best iteration: %d  threshold: %.3f",
                detector.n_estimators_used, detector.threshold)

    # 6. Evaluate on test set
    test_probs = detector.predict_proba(X_test_s)
    scalar = compute_scalar_metrics(test_probs, y_test.values, detector.threshold)

    # Incident-level recall (map window index back to raw position)
    test_start_idx = X_test.index[0]
    raw_index = list(metrics.index)
    test_offset = raw_index.index(test_start_idx)

    alert_series_raw = _build_full_alert_series(
        detector, X_test_s, metrics, test_offset
    )
    inc_eval = incident_level_evaluation(
        alert_series_raw, incident_label, incidents, H
    )
    logger.info(
        "Incident-level recall: %.3f  (%d/%d caught)",
        inc_eval["incident_recall"],
        inc_eval["caught"],
        inc_eval["total_incidents"],
    )

    # 7. Persist model to tmp file then upload to S3
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "detector.joblib"
        metrics_path = Path(tmp_dir) / "metrics.json"

        import joblib
        # Bundle scaler alongside model for inference Lambda
        joblib.dump(
            {"detector": detector, "scaler": scaler, "W": W, "H": H},
            model_path,
        )

        all_metrics = {"scalar": scalar, "incident_level": inc_eval, "W": W, "H": H}
        save_metrics(all_metrics, metrics_path)

        if not cfg["offline"]:
            s3 = _get_s3_client()
            s3.upload_file(str(model_path), cfg["model_bucket"], cfg["model_key"])
            s3.upload_file(str(metrics_path), cfg["model_bucket"], cfg["metrics_key"])
            logger.info("Artifacts uploaded to s3://%s/", cfg["model_bucket"])
        else:
            logger.info("OFFLINE_MODE: skipping S3 upload")

    return {
        "status": "success",
        "model_key": cfg["model_key"],
        "metrics": all_metrics,
    }


def _build_full_alert_series(
    detector,
    X_test_scaled,
    metrics_df,
    test_offset: int,
) -> list[int]:
    """Map test-set predictions back to raw time-series positions."""
    n_raw = len(metrics_df)
    alerts = [0] * n_raw
    probs = detector.predict_proba(X_test_scaled)
    for i, prob in enumerate(probs):
        raw_pos = test_offset + i
        if raw_pos < n_raw:
            alerts[raw_pos] = int(prob >= detector.threshold)
    return alerts


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda handler for daily model retraining.

    Parameters
    ----------
    event   : EventBridge scheduled event (payload is mostly ignored)
    context : Lambda context object

    Returns
    -------
    HTTP-style response dict  {statusCode, body}
    """
    cfg = _training_config()
    logger.info("Retrain Lambda invoked  cfg=%s", {k: v for k, v in cfg.items() if k != "offline"})

    try:
        result = run_training(cfg)
        return {
            "statusCode": 200,
            "body": json.dumps(result, default=str),
        }
    except Exception as exc:
        logger.exception("Retraining failed: %s", exc)
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "message": str(exc)}),
        }
