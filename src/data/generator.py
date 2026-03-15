"""
Synthetic CloudWatch metrics generator.

Produces a realistic multivariate time series that mimics the behaviour of
AWS CloudWatch metrics for a web-service workload.  The generator injects
discrete incident windows during which a random subset of metrics diverges
from their normal operating envelope, providing labelled data for supervised
incident-prediction.

Metric catalogue
----------------
  cpu_utilization     – CPU % (0-100), periodic load pattern + noise
  memory_utilization  – Memory % (0-100), slow-drift + noise
  request_count       – Requests per minute, business-hours rhythm
  request_latency_ms  – p50 latency in ms, correlated with request_count
  error_rate          – Error fraction [0, 1], usually near zero
  network_in_bytes    – Inbound bytes, correlated with request_count
  network_out_bytes   – Outbound bytes, correlated with request_count

Incident model
--------------
Each incident occupies a contiguous window of `incident_duration` time steps.
During an incident a random subset of metrics (≥1) receives an additive spike
drawn from a half-normal distribution, parameterised so that the deviation is
clearly above the normal operating range.  A cooldown gap between incidents
prevents overlapping labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """All tunable knobs for the synthetic dataset."""

    # Dataset size
    n_steps: int = 14_400          # ~10 days at 1-minute resolution
    seed: int = 42

    # Incident parameters
    incident_rate: float = 0.04    # fraction of steps that are incident steps
    incident_duration: int = 30    # steps per incident window
    min_gap: int = 60              # minimum gap between incident windows
    affected_metrics_min: int = 1  # min metrics perturbed per incident
    affected_metrics_max: int = 4  # max metrics perturbed per incident

    # Noise & drift
    noise_scale: float = 1.0       # global noise multiplier

    # Metric-specific spike magnitudes (mean of half-normal)
    spike_scales: dict[str, float] = field(default_factory=lambda: {
        "cpu_utilization":    35.0,
        "memory_utilization": 25.0,
        "request_count":     200.0,
        "request_latency_ms": 800.0,
        "error_rate":          0.15,
        "network_in_bytes":  5e5,
        "network_out_bytes": 5e5,
    })


METRIC_NAMES = [
    "cpu_utilization",
    "memory_utilization",
    "request_count",
    "request_latency_ms",
    "error_rate",
    "network_in_bytes",
    "network_out_bytes",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _business_rhythm(t: np.ndarray, period: int = 1440) -> np.ndarray:
    """Smooth 24-hour load curve with a daytime peak (minutes resolution)."""
    phase = 2 * np.pi * (t % period) / period
    # Peak at ~14:00 (840 min), trough at ~04:00 (240 min)
    return 0.5 + 0.5 * np.sin(phase - np.pi / 2)


def _generate_baseline(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate the normal (no-incident) metric signals."""
    t = np.arange(n, dtype=float)

    rhythm = _business_rhythm(t)

    # CPU: periodic + slower weekly trend + noise
    cpu = (
        20.0
        + 40.0 * rhythm
        + 8.0 * np.sin(2 * np.pi * t / (7 * 1440))      # weekly component
        + rng.normal(0, 2.5, n)
    ).clip(5.0, 95.0)

    # Memory: slow sawtooth drift (GC-like) + noise
    mem_cycle = 20 * (t % 480) / 480                      # gradual fill then release
    memory = (
        40.0
        + mem_cycle
        + rng.normal(0, 1.5, n)
    ).clip(10.0, 90.0)

    # Request count: business rhythm + Poisson-like noise
    req_base = 100.0 + 400.0 * rhythm
    request_count = np.maximum(
        0.0,
        req_base + rng.normal(0, 20.0, n)
    )

    # Latency: positively correlated with request_count, heavy-tailed noise
    latency = (
        50.0
        + 0.4 * (request_count - request_count.mean())
        + np.abs(rng.normal(0, 20.0, n))
    ).clip(10.0, 2000.0)

    # Error rate: near zero, occasional small spikes
    error_base = 0.005 + 0.003 * rng.random(n)
    error_spikes = rng.random(n) < 0.01           # 1% chance of a small spike
    error_rate = (error_base + error_spikes * rng.uniform(0.01, 0.04, n)).clip(0, 1)

    # Network I/O: proportional to request_count
    net_in = (
        request_count * rng.uniform(800, 1200, n)
        + rng.normal(0, 5000, n)
    ).clip(0.0, None)

    net_out = (
        request_count * rng.uniform(1500, 2500, n)
        + rng.normal(0, 8000, n)
    ).clip(0.0, None)

    return pd.DataFrame({
        "cpu_utilization":    cpu,
        "memory_utilization": memory,
        "request_count":      request_count,
        "request_latency_ms": latency,
        "error_rate":         error_rate,
        "network_in_bytes":   net_in,
        "network_out_bytes":  net_out,
    })


def _place_incidents(
    rng: np.random.Generator,
    n: int,
    incident_rate: float,
    incident_duration: int,
    min_gap: int,
) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) incident windows placed non-overlappingly.
    Uses a random-walk approach: advance by a gap drawn from an exponential
    distribution with mean = incident_duration / incident_rate.
    """
    mean_gap = int(incident_duration / incident_rate)
    incidents: list[tuple[int, int]] = []
    pos = rng.integers(min_gap, 2 * min_gap)

    while pos + incident_duration < n:
        end = pos + incident_duration
        incidents.append((pos, end))
        skip = int(rng.exponential(mean_gap - incident_duration)) + min_gap
        pos = end + max(skip, min_gap)

    return incidents


def _inject_spikes(
    rng: np.random.Generator,
    df: pd.DataFrame,
    incidents: list[tuple[int, int]],
    config: GeneratorConfig,
) -> pd.DataFrame:
    """Add incident-driven spikes to the baseline signals in-place."""
    df = df.copy()
    n_metrics = len(METRIC_NAMES)

    for start, end in incidents:
        # Choose which metrics are affected
        n_affected = rng.integers(
            config.affected_metrics_min,
            min(config.affected_metrics_max, n_metrics) + 1,
        )
        affected = rng.choice(METRIC_NAMES, size=n_affected, replace=False)

        for metric in affected:
            scale = config.spike_scales[metric]
            # Half-normal spike ramping up over the first quarter, then flat
            ramp = incident_duration = end - start
            ramp_len = max(1, ramp // 4)
            spike = np.zeros(ramp)
            spike[ramp_len:] = np.abs(rng.normal(scale, scale * 0.2, ramp - ramp_len))
            spike[:ramp_len] = np.linspace(0, spike[ramp_len], ramp_len)

            df.iloc[start:end, df.columns.get_loc(metric)] += spike

    # Re-clip after spikes
    df["cpu_utilization"] = df["cpu_utilization"].clip(0, 100)
    df["memory_utilization"] = df["memory_utilization"].clip(0, 100)
    df["error_rate"] = df["error_rate"].clip(0, 1)
    df["request_count"] = df["request_count"].clip(0, None)
    df["request_latency_ms"] = df["request_latency_ms"].clip(0, None)
    df["network_in_bytes"] = df["network_in_bytes"].clip(0, None)
    df["network_out_bytes"] = df["network_out_bytes"].clip(0, None)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    config: Optional[GeneratorConfig] = None,
) -> tuple[pd.DataFrame, pd.Series, list[tuple[int, int]]]:
    """
    Generate a synthetic CloudWatch metrics dataset.

    Parameters
    ----------
    config : GeneratorConfig, optional
        Generator settings.  Defaults to ``GeneratorConfig()``.

    Returns
    -------
    metrics : pd.DataFrame
        Shape (n_steps, 7) — one column per metric, DatetimeIndex at
        1-minute resolution starting from 2024-01-01 00:00 UTC.
    incident_label : pd.Series
        Binary series (0/1) aligned with ``metrics``.  1 = incident step.
    incidents : list[tuple[int, int]]
        Raw (start, end) incident windows (integer positions).
    """
    if config is None:
        config = GeneratorConfig()

    rng = np.random.default_rng(config.seed)

    # 1. Baseline signals
    metrics = _generate_baseline(rng, config.n_steps)

    # 2. Place incident windows
    incidents = _place_incidents(
        rng,
        config.n_steps,
        config.incident_rate,
        config.incident_duration,
        config.min_gap,
    )

    # 3. Inject spikes during incidents
    metrics = _inject_spikes(rng, metrics, incidents, config)

    # 4. Build binary label
    label = np.zeros(config.n_steps, dtype=int)
    for start, end in incidents:
        label[start:end] = 1
    incident_label = pd.Series(label, name="incident")

    # 5. Attach a DatetimeIndex (1-minute resolution)
    idx = pd.date_range("2024-01-01", periods=config.n_steps, freq="min")
    metrics.index = idx
    incident_label.index = idx

    return metrics, incident_label, incidents
