"""
Sliding-window feature extraction and dataset preparation.

Converts a raw multivariate time series of CloudWatch metrics into a tabular
feature matrix suitable for gradient-boosted classifiers, together with a
binary target that encodes whether any incident begins within the next H steps.

Design choices
--------------
- **Sliding window of length W**: each sample summarises W consecutive metric
  observations using 9 hand-crafted statistics per channel.  This captures
  level, trend, volatility, and short-term instability without requiring the
  model to handle raw sequences.
- **Horizon H**: the label at position t is 1 if any incident label is 1 in
  [t, t+H).  Larger H gives more advance warning but makes the task harder.
- **Chronological split**: training/validation/test sets are cut along the time
  axis to prevent data leakage.
- **StandardScaler fitted on train only**: applied to val/test to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """Sliding-window hyperparameters."""
    W: int = 60    # look-back window length (steps)
    H: int = 10    # prediction horizon (steps); controls advance warning time
    step: int = 1  # stride of the sliding window


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Compute 9 statistics for each metric channel in a single window.

    Parameters
    ----------
    window : np.ndarray
        Shape (W, n_metrics).

    Returns
    -------
    features : np.ndarray
        Shape (n_metrics * 9,).
    """
    W, n_metrics = window.shape
    features = np.empty(n_metrics * 9)

    quarter = max(1, W // 4)

    for i in range(n_metrics):
        col = window[:, i]
        base = i * 9

        mean_val = col.mean()
        std_val = col.std(ddof=0)

        features[base + 0] = mean_val
        features[base + 1] = std_val
        features[base + 2] = col.min()
        features[base + 3] = col.max()
        features[base + 4] = col.max() - col.min()          # range
        features[base + 5] = col[-1] - col[0]               # net change
        features[base + 6] = col[-1] - mean_val             # deviation from mean
        # Linear trend: slope of least-squares fit
        x = np.arange(W, dtype=float)
        denom = (x * x).sum() - x.sum() ** 2 / W
        if denom > 1e-10:
            features[base + 7] = ((x * col).sum() - x.sum() * col.sum() / W) / denom
        else:
            features[base + 7] = 0.0
        # Short-term volatility: std of last quarter of window
        features[base + 8] = col[-quarter:].std(ddof=0)

    return features


def build_feature_names(metric_names: list[str]) -> list[str]:
    """Return column names matching the layout of ``_extract_window_features``."""
    stats = ["mean", "std", "min", "max", "range",
             "net_change", "dev_from_mean", "slope", "recent_vol"]
    return [f"{m}__{s}" for m in metric_names for s in stats]


# ---------------------------------------------------------------------------
# Sliding-window dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    metrics: pd.DataFrame,
    incident_label: pd.Series,
    config: Optional[WindowConfig] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a tabular (X, y) dataset from raw time series using a sliding window.

    The label at position t is 1 if ``incident_label[t : t + H]`` contains any
    1, i.e. "an incident is imminent within the next H steps."

    Parameters
    ----------
    metrics : pd.DataFrame
        Shape (T, n_metrics).
    incident_label : pd.Series
        Binary series of length T.
    config : WindowConfig, optional

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, shape (N_samples, n_metrics * 9).
    y : pd.Series
        Binary labels, length N_samples.
    """
    if config is None:
        config = WindowConfig()

    W, H, step = config.W, config.H, config.step
    values = metrics.values.astype(float)
    labels = incident_label.values.astype(int)
    T = len(labels)
    metric_names = list(metrics.columns)

    rows: list[np.ndarray] = []
    y_vals: list[int] = []
    indices: list[int] = []

    for t in range(0, T - W - H + 1, step):
        window = values[t : t + W]
        future = labels[t + W : t + W + H]
        rows.append(_extract_window_features(window))
        y_vals.append(int(future.any()))
        indices.append(t + W)  # index of the first "future" step

    X = pd.DataFrame(
        np.array(rows),
        columns=build_feature_names(metric_names),
        index=metrics.index[indices],
    )
    y = pd.Series(y_vals, index=X.index, name="incident")

    return X, y


# ---------------------------------------------------------------------------
# Chronological train / val / test split
# ---------------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Split (X, y) into train / validation / test sets preserving time order.

    Parameters
    ----------
    X, y : matching DataFrame / Series
    train_frac : fraction of samples for training
    val_frac : fraction of samples for validation
        The remaining ``1 - train_frac - val_frac`` goes to test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train : n_train + n_val]
    X_test = X.iloc[n_train + n_val :]

    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train : n_train + n_val]
    y_test = y.iloc[n_train + n_val :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training set and return it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    *datasets: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Apply a pre-fitted scaler to one or more DataFrames, preserving index and
    column names.
    """
    return [
        pd.DataFrame(
            scaler.transform(ds),
            index=ds.index,
            columns=ds.columns,
        )
        for ds in datasets
    ]
