"""
Unit tests for src/data/preprocessor.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.generator import GeneratorConfig, generate_dataset
from src.data.preprocessor import (
    WindowConfig,
    _extract_window_features,
    build_feature_names,
    build_dataset,
    chronological_split,
    fit_scaler,
    apply_scaler,
)
from src.data.generator import METRIC_NAMES


N_METRICS = len(METRIC_NAMES)
FEATURES_PER_METRIC = 9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    cfg = GeneratorConfig(n_steps=3_000, seed=0)
    return generate_dataset(cfg)


@pytest.fixture(scope="module")
def window_dataset(small_dataset):
    metrics, label, incidents = small_dataset
    X, y = build_dataset(metrics, label, WindowConfig(W=30, H=5))
    return X, y


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractWindowFeatures:
    def test_output_length(self):
        W = 40
        window = np.random.rand(W, N_METRICS)
        feats = _extract_window_features(window)
        assert len(feats) == N_METRICS * FEATURES_PER_METRIC

    def test_no_nan(self):
        window = np.random.rand(50, N_METRICS)
        feats = _extract_window_features(window)
        assert not np.any(np.isnan(feats))

    def test_constant_window_zero_std(self):
        window = np.ones((30, N_METRICS)) * 5.0
        feats = _extract_window_features(window)
        # std features should be 0 for constant signal
        std_indices = [i * FEATURES_PER_METRIC + 1 for i in range(N_METRICS)]
        for idx in std_indices:
            assert abs(feats[idx]) < 1e-10, f"std at index {idx} should be 0"

    def test_slope_positive_for_increasing_signal(self):
        W = 50
        window = np.zeros((W, N_METRICS))
        window[:, 0] = np.arange(W, dtype=float)  # strictly increasing
        feats = _extract_window_features(window)
        slope_idx = 0 * FEATURES_PER_METRIC + 7
        assert feats[slope_idx] > 0


class TestBuildFeatureNames:
    def test_length(self):
        names = build_feature_names(METRIC_NAMES)
        assert len(names) == N_METRICS * FEATURES_PER_METRIC

    def test_all_strings(self):
        names = build_feature_names(METRIC_NAMES)
        assert all(isinstance(n, str) for n in names)

    def test_contains_metric_names(self):
        names = build_feature_names(METRIC_NAMES)
        for m in METRIC_NAMES:
            assert any(m in n for n in names)


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_X_shape(self, window_dataset):
        X, _ = window_dataset
        assert X.shape[1] == N_METRICS * FEATURES_PER_METRIC

    def test_y_binary(self, window_dataset):
        _, y = window_dataset
        assert set(y.unique()).issubset({0, 1})

    def test_X_y_same_length(self, window_dataset):
        X, y = window_dataset
        assert len(X) == len(y)

    def test_no_nan_in_X(self, window_dataset):
        X, _ = window_dataset
        assert not X.isnull().any().any()

    def test_index_is_datetimeindex(self, window_dataset):
        X, _ = window_dataset
        assert isinstance(X.index, pd.DatetimeIndex)

    def test_shorter_window_gives_more_samples(self, small_dataset):
        metrics, label, _ = small_dataset
        X30, _ = build_dataset(metrics, label, WindowConfig(W=30, H=5))
        X60, _ = build_dataset(metrics, label, WindowConfig(W=60, H=5))
        assert len(X30) > len(X60)


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_sizes_roughly_correct(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_split(X, y)
        n = len(X)
        assert abs(len(X_tr) / n - 0.60) < 0.02
        assert abs(len(X_va) / n - 0.20) < 0.02

    def test_temporal_order_preserved(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, X_te, *_ = chronological_split(X, y)
        assert X_tr.index[-1] < X_va.index[0]
        assert X_va.index[-1] < X_te.index[0]

    def test_no_overlap(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, X_te, *_ = chronological_split(X, y)
        assert len(set(X_tr.index) & set(X_va.index)) == 0
        assert len(set(X_va.index) & set(X_te.index)) == 0


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TestScaler:
    def test_scaled_train_near_zero_mean(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, X_te, *_ = chronological_split(X, y)
        scaler = fit_scaler(X_tr)
        X_tr_s, = apply_scaler(scaler, X_tr)
        assert abs(X_tr_s.mean().mean()) < 0.1

    def test_apply_preserves_shape(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, X_te, *_ = chronological_split(X, y)
        scaler = fit_scaler(X_tr)
        X_va_s, X_te_s = apply_scaler(scaler, X_va, X_te)
        assert X_va_s.shape == X_va.shape
        assert X_te_s.shape == X_te.shape

    def test_apply_preserves_index(self, window_dataset):
        X, y = window_dataset
        X_tr, X_va, *_ = chronological_split(X, y)
        scaler = fit_scaler(X_tr)
        X_va_s, = apply_scaler(scaler, X_va)
        pd.testing.assert_index_equal(X_va_s.index, X_va.index)
