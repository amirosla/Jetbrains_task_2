"""Tests for the negative sampler (alias-table distribution)."""

import numpy as np
import pytest

from src.corpus import Vocabulary, build_vocabulary
from src.sampling import NegativeSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vocab(counts: list[int]) -> Vocabulary:
    words = [f"w{i}" for i in range(len(counts))]
    word2idx = {w: i for i, w in enumerate(words)}
    return Vocabulary(
        word2idx=word2idx,
        idx2word=words,
        counts=np.array(counts, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Alias table correctness
# ---------------------------------------------------------------------------

def test_alias_table_probabilities_sum_to_one():
    vocab = make_vocab([10, 20, 30, 40])
    sampler = NegativeSampler(vocab)
    assert abs(sampler._probs.sum() - 1.0) < 1e-9


def test_alias_table_entries_valid():
    vocab = make_vocab([5, 10, 15])
    sampler = NegativeSampler(vocab)
    n = vocab.size
    assert sampler._alias.shape == (n,)
    assert sampler._prob.shape  == (n,)
    assert np.all(sampler._prob  >= 0.0)
    assert np.all(sampler._prob  <= 1.0 + 1e-9)
    assert np.all(sampler._alias >= 0)
    assert np.all(sampler._alias < n)


# ---------------------------------------------------------------------------
# Sampling correctness
# ---------------------------------------------------------------------------

def test_sample_returns_correct_count():
    vocab = make_vocab([10, 20, 30])
    sampler = NegativeSampler(vocab)
    rng = np.random.default_rng(0)
    samples = sampler.sample(5, rng=rng)
    assert samples.shape == (5,)


def test_sample_indices_in_range():
    vocab = make_vocab([10, 20, 30, 40, 50])
    sampler = NegativeSampler(vocab)
    rng = np.random.default_rng(1)
    samples = sampler.sample(1000, rng=rng)
    assert np.all(samples >= 0)
    assert np.all(samples < vocab.size)


def test_sample_exclude_works():
    vocab = make_vocab([10, 20, 30])
    sampler = NegativeSampler(vocab)
    rng = np.random.default_rng(2)
    for _ in range(20):
        samples = sampler.sample(10, exclude=0, rng=rng)
        assert 0 not in samples


def test_sample_distribution_matches_smoothed_unigram():
    """Check that empirical frequencies approximate P_n ∝ count^0.75."""
    counts = [100, 200, 300, 400]
    vocab = make_vocab(counts)
    sampler = NegativeSampler(vocab)

    n_draws = 200_000
    rng = np.random.default_rng(3)
    samples = sampler.sample(n_draws, rng=rng)

    empirical = np.bincount(samples.astype(int), minlength=vocab.size) / n_draws
    expected = sampler._probs

    # Allow ±2% tolerance at this sample size
    np.testing.assert_allclose(empirical, expected, atol=0.02)
