"""
Unit tests for src/data/generator.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.generator import (
    GeneratorConfig,
    METRIC_NAMES,
    generate_dataset,
    _place_incidents,
    _generate_baseline,
)


class TestGeneratorConfig:
    def test_defaults_are_sensible(self):
        cfg = GeneratorConfig()
        assert cfg.n_steps > 0
        assert 0 < cfg.incident_rate < 1
        assert cfg.incident_duration > 0
        assert cfg.min_gap >= cfg.incident_duration

    def test_custom_values_accepted(self):
        cfg = GeneratorConfig(n_steps=500, seed=7, incident_rate=0.05)
        assert cfg.n_steps == 500
        assert cfg.seed == 7


class TestGenerateDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        cfg = GeneratorConfig(n_steps=2_000, seed=42)
        return generate_dataset(cfg)

    def test_returns_tuple_of_three(self, dataset):
        assert len(dataset) == 3

    def test_metrics_shape(self, dataset):
        metrics, _, _ = dataset
        assert metrics.shape[1] == len(METRIC_NAMES)
        assert len(metrics) == 2_000

    def test_metrics_columns(self, dataset):
        metrics, _, _ = dataset
        assert list(metrics.columns) == METRIC_NAMES

    def test_label_length_matches_metrics(self, dataset):
        metrics, label, _ = dataset
        assert len(label) == len(metrics)

    def test_label_is_binary(self, dataset):
        _, label, _ = dataset
        assert set(label.unique()).issubset({0, 1})

    def test_incidents_are_nonempty(self, dataset):
        _, _, incidents = dataset
        assert len(incidents) > 0

    def test_incidents_non_overlapping(self, dataset):
        _, _, incidents = dataset
        for i in range(len(incidents) - 1):
            assert incidents[i][1] <= incidents[i + 1][0]

    def test_label_matches_incident_windows(self, dataset):
        _, label, incidents = dataset
        label_arr = label.values
        for start, end in incidents:
            assert all(label_arr[start:end] == 1), "Incident window must be fully labelled 1"

    def test_metrics_have_datetimeindex(self, dataset):
        metrics, _, _ = dataset
        assert isinstance(metrics.index, pd.DatetimeIndex)

    def test_cpu_in_range(self, dataset):
        metrics, _, _ = dataset
        assert metrics["cpu_utilization"].between(0, 100).all()

    def test_memory_in_range(self, dataset):
        metrics, _, _ = dataset
        assert metrics["memory_utilization"].between(0, 100).all()

    def test_error_rate_in_range(self, dataset):
        metrics, _, _ = dataset
        assert metrics["error_rate"].between(0, 1).all()

    def test_no_negative_values_in_counts(self, dataset):
        metrics, _, _ = dataset
        for col in ["request_count", "network_in_bytes", "network_out_bytes"]:
            assert (metrics[col] >= 0).all()

    def test_reproducibility(self):
        cfg = GeneratorConfig(n_steps=1_000, seed=99)
        m1, l1, i1 = generate_dataset(cfg)
        m2, l2, i2 = generate_dataset(cfg)
        pd.testing.assert_frame_equal(m1, m2)
        pd.testing.assert_series_equal(l1, l2)
        assert i1 == i2

    def test_different_seeds_differ(self):
        m1, _, _ = generate_dataset(GeneratorConfig(n_steps=1_000, seed=1))
        m2, _, _ = generate_dataset(GeneratorConfig(n_steps=1_000, seed=2))
        assert not m1.equals(m2)


class TestPlaceIncidents:
    def test_no_overlap(self):
        rng = np.random.default_rng(0)
        incidents = _place_incidents(rng, 10_000, 0.05, 30, 60)
        for i in range(len(incidents) - 1):
            assert incidents[i][1] <= incidents[i + 1][0]

    def test_all_within_bounds(self):
        rng = np.random.default_rng(0)
        incidents = _place_incidents(rng, 5_000, 0.05, 30, 60)
        for start, end in incidents:
            assert 0 <= start < end <= 5_000
