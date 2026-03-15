"""
Unit tests for src/model/detector.py and src/model/walk_forward.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.generator import GeneratorConfig, generate_dataset
from src.data.preprocessor import (
    WindowConfig,
    build_dataset,
    chronological_split,
    fit_scaler,
    apply_scaler,
)
from src.model.detector import IncidentDetector, _select_threshold, threshold_sweep
from src.model.walk_forward import walk_forward_cv, summarise_cv


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_detector():
    """Return a fitted IncidentDetector on a small synthetic dataset."""
    cfg = GeneratorConfig(n_steps=4_000, seed=7)
    metrics, label, _ = generate_dataset(cfg)
    X, y = build_dataset(metrics, label, WindowConfig(W=30, H=5))
    X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_split(X, y)
    scaler = fit_scaler(X_tr)
    X_tr_s, X_va_s, X_te_s = apply_scaler(scaler, X_tr, X_va, X_te)

    detector = IncidentDetector()
    detector.fit(X_tr_s, y_tr, X_va_s, y_va)
    return detector, X_te_s, y_te


# ---------------------------------------------------------------------------
# IncidentDetector
# ---------------------------------------------------------------------------

class TestIncidentDetector:
    def test_predict_proba_range(self, trained_detector):
        detector, X_te, _ = trained_detector
        probs = detector.predict_proba(X_te)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_returns_binary(self, trained_detector):
        detector, X_te, _ = trained_detector
        preds = detector.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_threshold_is_in_01(self, trained_detector):
        detector, _, _ = trained_detector
        assert 0.0 < detector.threshold < 1.0

    def test_custom_threshold_respected(self, trained_detector):
        detector, X_te, _ = trained_detector
        preds_low = detector.predict(X_te, threshold=0.01)
        preds_high = detector.predict(X_te, threshold=0.99)
        # Lower threshold should produce more or equal positives
        assert preds_low.sum() >= preds_high.sum()

    def test_fit_unfitted_raises(self):
        d = IncidentDetector()
        X_dummy = pd.DataFrame(np.random.rand(5, 10))
        with pytest.raises(RuntimeError):
            d.predict_proba(X_dummy)

    def test_feature_importance_has_correct_columns(self, trained_detector):
        detector, _, _ = trained_detector
        fi = detector.feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_top_n(self, trained_detector):
        detector, _, _ = trained_detector
        fi = detector.feature_importance(top_n=5)
        assert len(fi) <= 5

    def test_n_estimators_used_positive(self, trained_detector):
        detector, _, _ = trained_detector
        assert detector.n_estimators_used > 0

    def test_save_and_load(self, trained_detector):
        detector, X_te, _ = trained_detector
        probs_before = detector.predict_proba(X_te)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            detector.save(path)
            loaded = IncidentDetector.load(path)

        probs_after = loaded.predict_proba(X_te)
        np.testing.assert_array_almost_equal(probs_before, probs_after, decimal=5)
        assert abs(loaded.threshold - detector.threshold) < 1e-6

    def test_recall_above_threshold(self, trained_detector):
        """Model should achieve at least 60% recall on test set."""
        detector, X_te, y_te = trained_detector
        from sklearn.metrics import recall_score
        preds = detector.predict(X_te)
        recall = recall_score(y_te, preds, zero_division=0)
        assert recall >= 0.60, f"Recall too low: {recall:.3f}"


# ---------------------------------------------------------------------------
# _select_threshold
# ---------------------------------------------------------------------------

class TestSelectThreshold:
    def test_returns_float_in_01(self):
        probs = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        labels = np.array([0, 1, 0, 0, 1])
        t = _select_threshold(probs, labels)
        assert 0.0 < t < 1.0

    def test_all_negative_labels_returns_valid_threshold(self):
        # When all labels are 0, every threshold yields F1=0; the function
        # should still return a float in (0, 1) without raising.
        probs = np.random.rand(20)
        labels = np.zeros(20, dtype=int)
        t = _select_threshold(probs, labels)
        assert 0.0 < t < 1.0

    def test_perfect_separation(self):
        probs = np.concatenate([np.zeros(50), np.ones(50)])
        labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        t = _select_threshold(probs, labels)
        # Any threshold in (0, 1) gives perfect F1; just check it's sensible
        assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    def test_returns_dataframe(self):
        probs = np.random.rand(100)
        labels = (probs > 0.5).astype(int)
        df = threshold_sweep(probs, labels)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        probs = np.random.rand(100)
        labels = (probs > 0.5).astype(int)
        df = threshold_sweep(probs, labels)
        for col in ["threshold", "precision", "recall", "f1"]:
            assert col in df.columns

    def test_recall_decreases_with_threshold(self):
        probs = np.random.rand(200)
        labels = (probs > 0.4).astype(int)
        df = threshold_sweep(probs, labels, n_thresholds=50)
        # Recall should be generally monotonically decreasing
        low_t_recall = df[df["threshold"] < 0.2]["recall"].mean()
        high_t_recall = df[df["threshold"] > 0.8]["recall"].mean()
        assert low_t_recall >= high_t_recall


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

class TestWalkForwardCV:
    @pytest.fixture(scope="class")
    def cv_results(self):
        cfg = GeneratorConfig(n_steps=5_000, seed=42)
        metrics, label, _ = generate_dataset(cfg)
        X, y = build_dataset(metrics, label, WindowConfig(W=30, H=5))
        X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_split(X, y)
        scaler = fit_scaler(X_tr)
        X_s, = apply_scaler(scaler, X)  # scale the entire dataset
        return walk_forward_cv(X_s, y, n_folds=3)

    def test_returns_list(self, cv_results):
        assert isinstance(cv_results, list)

    def test_number_of_folds(self, cv_results):
        # May be fewer than requested if data runs out
        assert len(cv_results) >= 1

    def test_auc_roc_in_range(self, cv_results):
        for r in cv_results:
            assert 0.0 <= r.auc_roc <= 1.0

    def test_summarise_cv(self, cv_results):
        df = summarise_cv(cv_results)
        assert isinstance(df, pd.DataFrame)
        assert "fold" in df.columns
        assert "auc_roc" in df.columns
