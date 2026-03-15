# Lambda Deployment Guide

Two AWS Lambda functions implement the operational alerting loop.

## retrain/handler.py — Daily Retraining

| Property | Value |
|----------|-------|
| Trigger | EventBridge scheduled rule: `rate(1 day)` |
| Memory | 1024 MB |
| Timeout | 5 min |
| Runtime | Python 3.12 |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_BUCKET` | S3 bucket for model artifacts | `my-alerting-bucket` |
| `MODEL_KEY` | S3 key for the serialised model | `model/detector.joblib` |
| `METRICS_KEY` | S3 key for evaluation metrics JSON | `model/metrics.json` |
| `DATA_BUCKET` | S3 bucket for raw metric CSVs | *(empty — uses synthetic data)* |
| `DATA_KEY_PREFIX` | S3 key prefix for metric files | `data/` |
| `LOOKBACK_DAYS` | Days of history to fetch | `30` |
| `WINDOW_W` | Look-back window length (minutes) | `60` |
| `WINDOW_H` | Prediction horizon (minutes) | `10` |
| `OFFLINE_MODE` | Set to `1` to skip all AWS calls | `0` |

---

## predict/handler.py — Per-Minute Inference

| Property | Value |
|----------|-------|
| Trigger | EventBridge scheduled rule: `rate(1 minute)` |
| Memory | 512 MB |
| Timeout | 30 s |
| Runtime | Python 3.12 |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_BUCKET` | S3 bucket containing the model artifact | `my-alerting-bucket` |
| `MODEL_KEY` | S3 key of the joblib bundle | `model/detector.joblib` |
| `SNS_TOPIC_ARN` | SNS topic ARN for alert notifications | *(empty)* |
| `CLOUDWATCH_NAMESPACE` | Primary CloudWatch namespace to query | `AWS/EC2` |
| `SERVICE_IDS` | Comma-separated EC2/service IDs to monitor | `i-demo` |
| `ALERT_COOLDOWN_S` | Minimum seconds between alerts per service | `300` |
| `OFFLINE_MODE` | Set to `1` to skip all AWS calls | `0` |

### Model Caching

The inference Lambda caches the downloaded model in `/tmp/detector_bundle.joblib`.
On a warm invocation the model is loaded from disk rather than re-downloaded from S3,
reducing per-invocation latency from ~1 s to ~50 ms.

---

## IAM Permissions Required

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket",
    "cloudwatch:GetMetricData",
    "sns:Publish",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "*"
}
```

## Local Testing

```bash
# Test retrain Lambda locally (no AWS required)
OFFLINE_MODE=1 python -c "
from lambda.retrain.handler import lambda_handler
print(lambda_handler({}, None))
"

# Test predict Lambda locally
OFFLINE_MODE=1 python -c "
from lambda.predict.handler import lambda_handler
print(lambda_handler({}, None))
"
```
