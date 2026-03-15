"""
AWS Lambda handler — per-minute incident prediction and alerting.

Triggered on a schedule (e.g. EventBridge rule: rate(1 minute)).

Execution flow
--------------
1. Download the model artifact from S3 (cached in /tmp across warm invocations).
2. Fetch the most recent W minutes of CloudWatch metric data for each service.
3. Compute the 63-dimensional feature vector from the live window.
4. Run inference; if predicted probability ≥ threshold, publish an SNS alert.
5. Return a structured response with the prediction details.

Environment variables
---------------------
  MODEL_BUCKET      : S3 bucket containing the model artifact
  MODEL_KEY         : S3 key of the joblib bundle            (default: model/detector.joblib)
  SNS_TOPIC_ARN     : SNS topic for alert notifications
  CLOUDWATCH_NAMESPACE : CloudWatch namespace to query       (default: AWS/EC2)
  SERVICE_IDS       : comma-separated list of instance/service IDs
  ALERT_COOLDOWN_S  : minimum seconds between alerts for same service (default: 300)
  OFFLINE_MODE      : set to 1 to skip all AWS calls and use synthetic probe

Lambda memory recommendation: 512 MB (model is ~20 MB in memory).
Lambda timeout recommendation: 30 s (CloudWatch queries + inference).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# /tmp is the only writable directory in Lambda and persists across warm invocations
_MODEL_CACHE_PATH = Path("/tmp/detector_bundle.joblib")
_LAST_ALERT_CACHE: dict[str, float] = {}   # service_id → epoch timestamp


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _inference_config() -> dict:
    return {
        "model_bucket": _env("MODEL_BUCKET", "my-alerting-bucket"),
        "model_key": _env("MODEL_KEY", "model/detector.joblib"),
        "sns_topic_arn": _env("SNS_TOPIC_ARN", ""),
        "cw_namespace": _env("CLOUDWATCH_NAMESPACE", "AWS/EC2"),
        "service_ids": [s.strip() for s in _env("SERVICE_IDS", "i-demo").split(",")],
        "cooldown_s": int(_env("ALERT_COOLDOWN_S", "300")),
        "offline": _env("OFFLINE_MODE", "0") == "1",
    }


# ---------------------------------------------------------------------------
# Model loading (with /tmp cache)
# ---------------------------------------------------------------------------

def _load_bundle(cfg: dict) -> dict:
    """
    Load the detector bundle from /tmp (warm cache) or S3 (cold start).

    Returns a dict with keys: detector, scaler, W, H.
    """
    import joblib

    if _MODEL_CACHE_PATH.exists():
        logger.info("Loading model from /tmp cache")
        return joblib.load(_MODEL_CACHE_PATH)

    logger.info("Cold start: downloading model from s3://%s/%s",
                cfg["model_bucket"], cfg["model_key"])

    if cfg["offline"]:
        # In offline mode return a dummy bundle for testing
        return _make_offline_bundle()

    import boto3
    s3 = boto3.client("s3")
    s3.download_file(cfg["model_bucket"], cfg["model_key"], str(_MODEL_CACHE_PATH))
    return joblib.load(_MODEL_CACHE_PATH)


def _make_offline_bundle() -> dict:
    """
    Train a tiny model on synthetic data for offline / unit-test use.
    Avoids any S3 dependency.
    """
    from src.data.generator import GeneratorConfig, generate_dataset
    from src.data.preprocessor import WindowConfig, build_dataset, chronological_split, fit_scaler, apply_scaler
    from src.model.detector import IncidentDetector

    metrics, incident_label, _ = generate_dataset(GeneratorConfig(n_steps=5000, seed=0))
    W, H = 60, 10
    X, y = build_dataset(metrics, incident_label, WindowConfig(W=W, H=H))
    X_train, X_val, _, y_train, y_val, _ = chronological_split(X, y)
    scaler = fit_scaler(X_train)
    X_train_s, X_val_s = apply_scaler(scaler, X_train, X_val)
    detector = IncidentDetector()
    detector.fit(X_train_s, y_train, X_val_s, y_val)
    return {"detector": detector, "scaler": scaler, "W": W, "H": H}


# ---------------------------------------------------------------------------
# CloudWatch data fetching
# ---------------------------------------------------------------------------

_METRIC_QUERIES = [
    ("cpu_utilization",    "AWS/EC2",        "CPUUtilization",          "Average"),
    ("memory_utilization", "CWAgent",        "mem_used_percent",        "Average"),
    ("request_count",      "AWS/ApplicationELB", "RequestCount",        "Sum"),
    ("request_latency_ms", "AWS/ApplicationELB", "TargetResponseTime",  "Average"),
    ("error_rate",         "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "Sum"),
    ("network_in_bytes",   "AWS/EC2",        "NetworkIn",               "Average"),
    ("network_out_bytes",  "AWS/EC2",        "NetworkOut",              "Average"),
]


def _fetch_cloudwatch_window(service_id: str, W: int) -> "pd.DataFrame":
    """
    Fetch the most recent W minutes of metrics for a given service from
    CloudWatch using GetMetricData.

    Returns a DataFrame with columns matching METRIC_NAMES and W rows.
    Falls back to NaN-padded zeros if a metric is unavailable.
    """
    import boto3
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    cw = boto3.client("cloudwatch")
    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_time = end_time - timedelta(minutes=W + 5)  # +5 for alignment buffer

    metric_queries = []
    for idx, (name, namespace, metric_name, stat) in enumerate(_METRIC_QUERIES):
        metric_queries.append(
            {
                "Id": f"m{idx}",
                "Label": name,
                "MetricStat": {
                    "Metric": {
                        "Namespace": namespace,
                        "MetricName": metric_name,
                        "Dimensions": [{"Name": "InstanceId", "Value": service_id}],
                    },
                    "Period": 60,
                    "Stat": stat,
                },
            }
        )

    resp = cw.get_metric_data(
        MetricDataQueries=metric_queries,
        StartTime=start_time,
        EndTime=end_time,
    )

    idx_range = pd.date_range(end=end_time, periods=W, freq="min")
    result = pd.DataFrame(index=idx_range)

    for mdr in resp["MetricDataResults"]:
        series = pd.Series(
            dict(zip(mdr["Timestamps"], mdr["Values"]))
        ).sort_index().reindex(idx_range, method="nearest").fillna(0.0)
        result[mdr["Label"]] = series.values

    return result.fillna(0.0)


def _generate_synthetic_window(W: int) -> "pd.DataFrame":
    """Return a synthetic W-step window for offline testing."""
    import numpy as np
    import pandas as pd
    from src.data.generator import METRIC_NAMES, _generate_baseline, GeneratorConfig

    rng = np.random.default_rng(int(time.time()) % (2 ** 32))
    cfg = GeneratorConfig(n_steps=W + 10, seed=0)
    df = _generate_baseline(rng, W)
    return df


# ---------------------------------------------------------------------------
# Alert publishing
# ---------------------------------------------------------------------------

def _publish_alert(service_id: str, prob: float, topic_arn: str) -> None:
    """Publish an SNS notification for a predicted incident."""
    import boto3
    sns = boto3.client("sns")
    message = {
        "service_id": service_id,
        "predicted_incident_probability": round(float(prob), 4),
        "alert_level": "HIGH" if prob >= 0.8 else "MEDIUM",
        "message": (
            f"Predictive alert for {service_id}: "
            f"incident probability = {prob:.1%}. "
            "Investigate metrics in the next few minutes."
        ),
    }
    sns.publish(
        TopicArn=topic_arn,
        Subject=f"[PredictiveAlert] {service_id} — incident probability {prob:.0%}",
        Message=json.dumps(message, indent=2),
    )
    logger.info("SNS alert published for %s  prob=%.3f", service_id, prob)


def _is_on_cooldown(service_id: str, cooldown_s: int) -> bool:
    last = _LAST_ALERT_CACHE.get(service_id, 0.0)
    return (time.time() - last) < cooldown_s


def _record_alert(service_id: str) -> None:
    _LAST_ALERT_CACHE[service_id] = time.time()


# ---------------------------------------------------------------------------
# Per-service inference
# ---------------------------------------------------------------------------

def predict_service(
    service_id: str,
    bundle: dict,
    cfg: dict,
) -> dict:
    """
    Run one inference cycle for a single service and return the result dict.
    """
    import numpy as np
    import pandas as pd
    from src.data.preprocessor import _extract_window_features, build_feature_names
    from src.data.generator import METRIC_NAMES

    detector = bundle["detector"]
    scaler = bundle["scaler"]
    W = bundle["W"]

    # Fetch live data
    if cfg["offline"]:
        window_df = _generate_synthetic_window(W)
    else:
        window_df = _fetch_cloudwatch_window(service_id, W)

    # Align columns to the expected metric order
    window_arr = np.zeros((W, len(METRIC_NAMES)), dtype=float)
    for i, m in enumerate(METRIC_NAMES):
        if m in window_df.columns:
            window_arr[:, i] = window_df[m].values[-W:]

    features = _extract_window_features(window_arr)
    feature_names = build_feature_names(METRIC_NAMES)
    X = pd.DataFrame([features], columns=feature_names)
    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=feature_names, index=X.index
    )

    prob = float(detector.predict_proba(X_scaled)[0])
    alert = prob >= detector.threshold

    result = {
        "service_id": service_id,
        "probability": round(prob, 4),
        "threshold": round(detector.threshold, 4),
        "alert": alert,
        "cooldown_active": False,
    }

    if alert:
        if _is_on_cooldown(service_id, cfg["cooldown_s"]):
            result["cooldown_active"] = True
            logger.info("Alert suppressed for %s (cooldown)", service_id)
        else:
            _record_alert(service_id)
            if cfg["sns_topic_arn"] and not cfg["offline"]:
                _publish_alert(service_id, prob, cfg["sns_topic_arn"])

    return result


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda handler for per-minute prediction and alerting.

    Parameters
    ----------
    event   : EventBridge scheduled event
    context : Lambda context object

    Returns
    -------
    {statusCode, body} dict with per-service prediction results
    """
    cfg = _inference_config()
    logger.info("Predict Lambda invoked  services=%s", cfg["service_ids"])

    try:
        bundle = _load_bundle(cfg)
        results = [
            predict_service(sid, bundle, cfg)
            for sid in cfg["service_ids"]
        ]
        alerts = [r for r in results if r["alert"] and not r["cooldown_active"]]
        logger.info(
            "Predictions: %d services  %d alert(s) fired",
            len(results),
            len(alerts),
        )
        return {
            "statusCode": 200,
            "body": json.dumps({"predictions": results}, default=str),
        }
    except Exception as exc:
        logger.exception("Prediction Lambda failed: %s", exc)
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "message": str(exc)}),
        }
