"""
Evaluation metrics and visualisation for the predictive alerting system.

Metrics
-------
- AUC-ROC, AUC-PR  : threshold-independent ranking quality
- F1, precision, recall at the selected threshold
- Incident-level recall : fraction of real incidents for which at least one
  alert fires in the H steps preceding the incident start (the metric the
  task description focuses on)
- Lead-time distribution : how many steps before incident start the first
  alert fires
- False-positive rate (FPR) : alerts / total non-incident windows

Plots (6-panel dashboard)
-------------------------
1. ROC curve (model vs z-score baseline)
2. Precision-Recall curve (model vs baseline)
3. Threshold sweep: precision / recall / F1 vs threshold
4. Confusion matrix heatmap
5. Lead-time histogram
6. Feature importance bar chart
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / Lambda use

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.model.detector import threshold_sweep


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_scalar_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute a flat dict of all relevant scalar metrics.

    Parameters
    ----------
    probs     : predicted probabilities for the positive class
    labels    : ground-truth binary labels
    threshold : the operating threshold

    Returns
    -------
    dict with keys: auc_roc, auc_pr, f1, precision, recall, fpr,
                    threshold, n_samples, n_positive
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "fpr": float(fp / max(tn + fp, 1)),
        "threshold": float(threshold),
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Incident-level recall and lead time
# ---------------------------------------------------------------------------

def incident_level_evaluation(
    alert_series: pd.Series,
    incident_label: pd.Series,
    incidents: list[tuple[int, int]],
    H: int,
) -> dict:
    """
    Compute incident-level recall and lead-time statistics.

    For each incident (start, end), we look at the alert_series in the window
    [start - H, start).  If at least one alert fires there, the incident is
    "caught" and the lead time is start - first_alert_step.

    Parameters
    ----------
    alert_series    : binary Series aligned with the full time series index
    incident_label  : binary Series of the full time series (not windows)
    incidents       : list of (start, end) incident windows (integer positions)
    H               : prediction horizon (same H used during training)

    Returns
    -------
    dict: incident_recall, mean_lead_time, median_lead_time, caught, missed,
          lead_times (list)
    """
    caught = 0
    missed = 0
    lead_times: list[float] = []

    alert_arr = alert_series.values if hasattr(alert_series, "values") else np.array(alert_series)
    n = len(alert_arr)

    for start, _end in incidents:
        # Look-ahead window: [start - H, start)
        look_start = max(0, start - H)
        look_end = min(start, n)

        if look_end <= look_start:
            missed += 1
            continue

        window_alerts = alert_arr[look_start:look_end]
        firing_positions = np.where(window_alerts == 1)[0]

        if len(firing_positions) > 0:
            caught += 1
            # Lead time: steps from first alert to incident start
            first_alert_pos = look_start + firing_positions[0]
            lead_times.append(float(start - first_alert_pos))
        else:
            missed += 1

    total = caught + missed
    incident_recall = caught / total if total > 0 else 0.0

    return {
        "incident_recall": float(incident_recall),
        "caught": caught,
        "missed": missed,
        "total_incidents": total,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else 0.0,
        "median_lead_time": float(np.median(lead_times)) if lead_times else 0.0,
        "lead_times": lead_times,
    }


# ---------------------------------------------------------------------------
# Z-score baseline
# ---------------------------------------------------------------------------

def zscore_baseline(
    X: pd.DataFrame,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Compute z-score alert probabilities from feature means and stds.

    For each sample, takes the maximum z-score across all *mean* features
    (which represent the per-channel average within the look-back window).
    Normalises to [0, 1] so it can be used on ROC/PR curves.

    Parameters
    ----------
    X         : feature DataFrame (output of build_dataset)
    threshold : z-score cutoff (not used for proba, only for binary pred)

    Returns
    -------
    probs : np.ndarray — z-score magnitude normalised to [0, 1]
    """
    mean_cols = [c for c in X.columns if c.endswith("__mean")]
    if not mean_cols:
        return np.zeros(len(X))

    means_matrix = X[mean_cols].values
    global_mean = means_matrix.mean(axis=0)
    global_std = means_matrix.std(axis=0) + 1e-8
    z_scores = np.abs((means_matrix - global_mean) / global_std)
    max_z = z_scores.max(axis=1)
    # Normalise to [0, 1] using a soft sigmoid-like mapping
    probs = max_z / (max_z.max() + 1e-8)
    return probs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_evaluation_dashboard(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    baseline_probs: np.ndarray,
    lead_times: list[float],
    feature_importance_df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Produce a 6-panel evaluation dashboard and optionally save to disk.

    Panels
    ------
    [0,0] ROC curve            [0,1] Precision-Recall curve
    [1,0] Threshold sweep      [1,1] Confusion matrix
    [2,0] Lead-time histogram  [2,1] Feature importance
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle("Predictive Alerting System — Evaluation Dashboard", fontsize=14, y=0.98)

    # --- ROC curve ---
    ax = axes[0, 0]
    fpr_m, tpr_m, _ = roc_curve(labels, probs)
    fpr_b, tpr_b, _ = roc_curve(labels, baseline_probs)
    auc_m = auc(fpr_m, tpr_m)
    auc_b = auc(fpr_b, tpr_b)
    ax.plot(fpr_m, tpr_m, lw=2, label=f"LightGBM (AUC={auc_m:.3f})")
    ax.plot(fpr_b, tpr_b, lw=1.5, linestyle="--", label=f"Z-score baseline (AUC={auc_b:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Precision-Recall curve ---
    ax = axes[0, 1]
    prec_m, rec_m, _ = precision_recall_curve(labels, probs)
    prec_b, rec_b, _ = precision_recall_curve(labels, baseline_probs)
    ap_m = average_precision_score(labels, probs)
    ap_b = average_precision_score(labels, baseline_probs)
    ax.plot(rec_m, prec_m, lw=2, label=f"LightGBM (AP={ap_m:.3f})")
    ax.plot(rec_b, prec_b, lw=1.5, linestyle="--", label=f"Z-score baseline (AP={ap_b:.3f})")
    ax.axhline(labels.mean(), color="grey", linestyle=":", lw=0.8, label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Threshold sweep ---
    ax = axes[1, 0]
    sweep_df = threshold_sweep(probs, labels)
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", lw=2)
    ax.axvline(threshold, color="red", linestyle="--", lw=1, label=f"Chosen threshold ({threshold:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Confusion matrix ---
    ax = axes[1, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Lead-time histogram ---
    ax = axes[2, 0]
    if lead_times:
        ax.hist(lead_times, bins=min(20, len(set(lead_times))), edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(lead_times), color="red", linestyle="--",
                   label=f"Mean = {np.mean(lead_times):.1f} steps")
        ax.set_xlabel("Lead time (steps before incident)")
        ax.set_ylabel("Count")
        ax.set_title("Lead-Time Distribution")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No caught incidents", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Lead-Time Distribution")
    ax.grid(alpha=0.3)

    # --- Feature importance ---
    ax = axes[2, 1]
    if not feature_importance_df.empty:
        top = feature_importance_df.head(15)
        ax.barh(top["feature"][::-1], top["importance"][::-1])
        ax.set_xlabel("Gain importance")
        ax.set_title("Top-15 Feature Importances")
    else:
        ax.set_title("Feature Importances (unavailable)")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, path: str | Path) -> None:
    """Write a metrics dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(path: str | Path) -> dict:
    """Load a metrics dict from a JSON file."""
    with open(path) as f:
        return json.load(f)
