"""
Expanding-window (walk-forward) cross-validation for time-series data.

Unlike k-fold CV, this never uses future data for training, which is critical
for honest evaluation of time-series models.  Each fold extends the training
set by one block, while the validation set stays the same size.

    Fold 1:  [==train==|--val--]
    Fold 2:  [====train====|--val--]
    Fold 3:  [======train======|--val--]
    ...                                → time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.model.detector import IncidentDetector, _select_threshold


@dataclass
class FoldResult:
    fold: int
    auc_roc: float
    auc_pr: float
    f1: float
    threshold: float
    n_train: int
    n_val: int


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    min_train_frac: float = 0.40,
    val_frac: float = 0.10,
    detector_params: dict | None = None,
) -> list[FoldResult]:
    """
    Run expanding-window cross-validation.

    Parameters
    ----------
    X, y         : full feature matrix and labels (time-ordered)
    n_folds      : number of CV folds
    min_train_frac : minimum fraction of data to use as training on fold 1
    val_frac     : fraction of data per validation set
    detector_params : optional LightGBM hyperparameters

    Returns
    -------
    list of FoldResult (one per fold)
    """
    n = len(X)
    # The validation window size (in samples) is fixed across folds
    val_size = max(1, int(n * val_frac))
    # Starting position of the first validation set
    train_start_size = max(1, int(n * min_train_frac))

    results: list[FoldResult] = []

    for fold in range(n_folds):
        # Training ends at: train_start_size + fold * (val_size)
        train_end = train_start_size + fold * val_size
        val_end = train_end + val_size

        if val_end > n:
            break  # not enough data for this fold

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_va, y_va = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        # Need at least one positive in both train and val to be meaningful
        if y_tr.sum() == 0 or y_va.sum() == 0:
            continue

        detector = IncidentDetector(params=detector_params)
        detector.fit(X_tr, y_tr, X_va, y_va)

        probs = detector.predict_proba(X_va)
        threshold = _select_threshold(probs, y_va.values)
        preds = (probs >= threshold).astype(int)

        results.append(
            FoldResult(
                fold=fold + 1,
                auc_roc=float(roc_auc_score(y_va, probs)),
                auc_pr=float(average_precision_score(y_va, probs)),
                f1=float(f1_score(y_va, preds, zero_division=0)),
                threshold=threshold,
                n_train=len(X_tr),
                n_val=len(X_va),
            )
        )

    return results


def summarise_cv(results: list[FoldResult]) -> pd.DataFrame:
    """
    Return a tidy DataFrame with per-fold metrics and a summary row.
    """
    rows = [
        {
            "fold": r.fold,
            "auc_roc": r.auc_roc,
            "auc_pr": r.auc_pr,
            "f1": r.f1,
            "threshold": r.threshold,
            "n_train": r.n_train,
            "n_val": r.n_val,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    # Append mean ± std summary row
    summary = pd.DataFrame(
        [
            {
                "fold": "mean ± std",
                "auc_roc": f"{df['auc_roc'].mean():.4f} ± {df['auc_roc'].std():.4f}",
                "auc_pr": f"{df['auc_pr'].mean():.4f} ± {df['auc_pr'].std():.4f}",
                "f1": f"{df['f1'].mean():.4f} ± {df['f1'].std():.4f}",
                "threshold": f"{df['threshold'].mean():.3f} ± {df['threshold'].std():.3f}",
                "n_train": "—",
                "n_val": "—",
            }
        ]
    )

    return pd.concat([df, summary], ignore_index=True)
