"""
LightGBM-based incident detector.

Wraps a LightGBM binary classifier in a sklearn-compatible interface and adds
alert-threshold selection logic tuned to maximise F1 on a validation set.

Why LightGBM?
-------------
- Handles class imbalance natively via ``scale_pos_weight``.
- Produces well-calibrated probability scores without additional calibration.
- Trains in seconds even on large feature matrices (63 dims × 10k+ samples).
- No GPU or external runtime needed — ideal for a Lambda environment.
- Feature importances (gain) are immediately interpretable by SRE teams.

An LSTM or Transformer would be natural next steps when raw sequences are fed
directly; LightGBM is preferred here because the statistical window features
already capture the relevant temporal structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score
from typing import Optional


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 30


# ---------------------------------------------------------------------------
# IncidentDetector
# ---------------------------------------------------------------------------

class IncidentDetector:
    """
    Thin wrapper around a LightGBM classifier for incident prediction.

    Usage
    -----
    >>> detector = IncidentDetector()
    >>> detector.fit(X_train, y_train, X_val, y_val)
    >>> probs = detector.predict_proba(X_test)
    >>> alerts = detector.predict(X_test)
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[lgb.LGBMClassifier] = None
        self.threshold: float = 0.5
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "IncidentDetector":
        """
        Train the LightGBM model with early stopping on the validation set,
        then select the alert threshold that maximises F1 on the validation set.

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val, y_val     : validation features and labels (early stopping + threshold tuning)

        Returns
        -------
        self
        """
        # Class-imbalance weight: ratio of negatives to positives
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        model_params = {
            **self.params,
            "scale_pos_weight": scale_pos_weight,
        }

        self._model = lgb.LGBMClassifier(**model_params)
        self._feature_names = list(X_train.columns)

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        # Tune threshold on validation set
        val_probs = self._model.predict_proba(X_val)[:, 1]
        self.threshold = _select_threshold(val_probs, y_val.values)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores for the positive (incident) class."""
        self._check_fitted()
        return self._model.predict_proba(X)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return binary predictions using ``self.threshold`` (or a custom one).
        """
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the detector (model + threshold + feature names) to disk."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "threshold": self.threshold,
                "feature_names": self._feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IncidentDetector":
        """Deserialise a detector previously saved with ``save()``."""
        bundle = joblib.load(path)
        detector = cls()
        detector._model = bundle["model"]
        detector.threshold = bundle["threshold"]
        detector._feature_names = bundle["feature_names"]
        return detector

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top-N features by LightGBM gain importance."""
        self._check_fitted()
        importances = self._model.booster_.feature_importance(importance_type="gain")
        df = pd.DataFrame(
            {"feature": self._feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    @property
    def n_estimators_used(self) -> int:
        """Number of trees actually used (after early stopping)."""
        self._check_fitted()
        return self._model.best_iteration_ or self._model.n_estimators

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def _select_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """
    Scan candidate thresholds and return the one that maximises F1.

    Falls back to 0.5 if no positive predictions are possible.
    """
    candidates = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = -1.0
    best_threshold = 0.5

    for t in candidates:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return float(best_threshold)


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 across a range of thresholds.

    Useful for generating the threshold-sweep panel in the evaluation dashboard
    and for operators who need to choose a different operating point.

    Returns
    -------
    pd.DataFrame with columns: threshold, precision, recall, f1
    """
    from sklearn.metrics import precision_score, recall_score

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
            }
        )

    return pd.DataFrame(rows)
