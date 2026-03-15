"""
Unit tests for src/evaluation/metrics.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_scalar_metrics,
    incident_level_evaluation,
    zscore_baseline,
    save_metrics,
    load_metrics,
    threshold_sweep,
)


# ---------------------------------------------------------------------------
# compute_scalar_metrics
# ---------------------------------------------------------------------------

class TestComputeScalarMetrics:
    @pytest.fixture
    def perfect_prediction(self):
        labels = np.array([0, 0, 1, 0, 1, 1, 0])
        probs = labels.astype(float)  # prob == label → perfect
        return probs, labels

    def test_returns_dict(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.5)
        assert isinstance(result, dict)

    def test_has_required_keys(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.5)
        required = {"auc_roc", "auc_pr", "f1", "precision", "recall", "fpr",
                    "threshold", "n_samples", "n_positive", "tn", "fp", "fn", "tp"}
        assert required.issubset(result.keys())

    def test_perfect_auc(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.5)
        assert result["auc_roc"] == 1.0

    def test_all_zeros_fpr(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.5)
        assert result["fp"] == 0
        assert result["fpr"] == 0.0

    def test_threshold_stored(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.42)
        assert result["threshold"] == pytest.approx(0.42)

    def test_n_samples_correct(self, perfect_prediction):
        probs, labels = perfect_prediction
        result = compute_scalar_metrics(probs, labels, threshold=0.5)
        assert result["n_samples"] == len(labels)


# ---------------------------------------------------------------------------
# incident_level_evaluation
# ---------------------------------------------------------------------------

class TestIncidentLevelEvaluation:
    def _make_args(self, alert_arr, incidents, H=5, n=200):
        alert_series = pd.Series(alert_arr)
        incident_label = pd.Series(np.zeros(n, dtype=int))
        for s, e in incidents:
            incident_label.iloc[s:e] = 1
        return alert_series, incident_label, incidents, H

    def test_perfect_recall(self):
        H = 5
        incidents = [(20, 30), (60, 70), (110, 120)]
        alert_arr = np.zeros(200, dtype=int)
        for s, _ in incidents:
            alert_arr[s - H] = 1   # fire exactly H steps before each incident
        args = self._make_args(alert_arr, incidents, H=H)
        result = incident_level_evaluation(*args)
        assert result["incident_recall"] == pytest.approx(1.0)
        assert result["caught"] == 3
        assert result["missed"] == 0

    def test_zero_recall(self):
        incidents = [(50, 60)]
        alert_arr = np.zeros(200, dtype=int)
        args = self._make_args(alert_arr, incidents, H=5)
        result = incident_level_evaluation(*args)
        assert result["incident_recall"] == pytest.approx(0.0)
        assert result["missed"] == 1

    def test_lead_time_computed(self):
        H = 10
        incidents = [(100, 110)]
        alert_arr = np.zeros(200, dtype=int)
        alert_arr[95] = 1  # fires 5 steps before incident at 100
        args = self._make_args(alert_arr, incidents, H=H)
        result = incident_level_evaluation(*args)
        assert result["caught"] == 1
        assert result["lead_times"] == [5.0]

    def test_empty_incidents(self):
        alert_arr = np.zeros(100, dtype=int)
        result = incident_level_evaluation(
            pd.Series(alert_arr),
            pd.Series(np.zeros(100)),
            [],
            H=5,
        )
        assert result["total_incidents"] == 0
        assert result["incident_recall"] == 0.0


# ---------------------------------------------------------------------------
# zscore_baseline
# ---------------------------------------------------------------------------

class TestZscoreBaseline:
    def test_returns_array(self):
        X = pd.DataFrame(np.random.rand(100, 10),
                         columns=[f"m{i}__mean" for i in range(10)])
        probs = zscore_baseline(X)
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 100

    def test_probs_in_01(self):
        X = pd.DataFrame(np.random.rand(100, 10),
                         columns=[f"m{i}__mean" for i in range(10)])
        probs = zscore_baseline(X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_anomalous_row_has_high_score(self):
        n = 100
        # Build a DataFrame where the last row is a massive outlier
        data = np.zeros((n, 5))
        cols = [f"m{i}__mean" for i in range(5)]
        X = pd.DataFrame(data, columns=cols)
        X.iloc[-1] = 1_000.0   # extreme outlier
        probs = zscore_baseline(X)
        assert probs[-1] == probs.max()

    def test_no_mean_cols_returns_zeros(self):
        X = pd.DataFrame(np.random.rand(50, 5),
                         columns=[f"feat_{i}" for i in range(5)])
        probs = zscore_baseline(X)
        np.testing.assert_array_equal(probs, np.zeros(50))


# ---------------------------------------------------------------------------
# save / load metrics
# ---------------------------------------------------------------------------

class TestSaveLoadMetrics:
    def test_roundtrip(self):
        data = {"auc_roc": 0.92, "recall": 0.81, "nested": {"a": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            save_metrics(data, path)
            loaded = load_metrics(path)
        assert loaded["auc_roc"] == pytest.approx(0.92)
        assert loaded["nested"]["a"] == 1

    def test_creates_parent_dirs(self):
        data = {"x": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "dir" / "metrics.json"
            save_metrics(data, path)
            assert path.exists()
