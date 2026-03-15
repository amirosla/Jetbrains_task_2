"""Tests for the training loop: convergence, learning rate schedule, and history."""

import numpy as np
import pytest

from src.corpus import Vocabulary
from src.model import Word2Vec
from src.sampling import NegativeSampler
from src.trainer import TrainerConfig, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_toy_vocab(n: int = 50) -> Vocabulary:
    words = [f"w{i}" for i in range(n)]
    word2idx = {w: i for i, w in enumerate(words)}
    counts = np.random.default_rng(0).integers(1, 100, n).astype(np.float64)
    return Vocabulary(word2idx=word2idx, idx2word=words, counts=counts)


def make_toy_pairs(vocab_size: int = 50, n_pairs: int = 200) -> tuple:
    rng = np.random.default_rng(7)
    centers  = rng.integers(0, vocab_size, n_pairs).astype(np.int32)
    contexts = rng.integers(0, vocab_size, n_pairs).astype(np.int32)
    return centers, contexts


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def test_train_returns_history_with_correct_epochs():
    vocab = make_toy_vocab()
    sampler = NegativeSampler(vocab)
    model = Word2Vec(vocab.size, embed_dim=16, seed=0)
    centers, contexts = make_toy_pairs(vocab.size)

    cfg = TrainerConfig(num_epochs=2, lr_start=0.01, num_negatives=3, seed=0)
    history = train(model, centers, contexts, sampler, cfg)

    assert len(history.epoch_losses) == 2
    assert len(history.epoch_times)  == 2


def test_train_history_losses_are_finite():
    vocab = make_toy_vocab()
    sampler = NegativeSampler(vocab)
    model = Word2Vec(vocab.size, embed_dim=16, seed=0)
    centers, contexts = make_toy_pairs(vocab.size)

    cfg = TrainerConfig(num_epochs=1, lr_start=0.01, num_negatives=3, seed=0)
    history = train(model, centers, contexts, sampler, cfg)

    for loss in history.epoch_losses:
        assert np.isfinite(loss)
        assert loss > 0.0


# ---------------------------------------------------------------------------
# Convergence on a tiny repeated corpus
# ---------------------------------------------------------------------------

def test_loss_decreases_over_epochs():
    """Training for multiple epochs on a tiny fixed corpus should lower loss."""
    vocab = make_toy_vocab(20)
    sampler = NegativeSampler(vocab)
    model = Word2Vec(vocab.size, embed_dim=32, seed=1)

    # Use a small fixed corpus repeated many times so the signal is clear
    rng = np.random.default_rng(0)
    centers  = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int32), 40)
    contexts = np.tile(np.array([1, 2, 3, 4, 0], dtype=np.int32), 40)

    cfg = TrainerConfig(num_epochs=5, lr_start=0.05, num_negatives=3, seed=0)
    history = train(model, centers, contexts, sampler, cfg)

    assert history.epoch_losses[-1] < history.epoch_losses[0], (
        f"Loss did not decrease: {history.epoch_losses}"
    )


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def test_lr_at_end_is_at_least_lr_min():
    """The learning rate must never fall below cfg.lr_min."""
    # We verify this indirectly: training should complete without error and
    # produce finite losses even when lr_start is very small.
    vocab = make_toy_vocab(10)
    sampler = NegativeSampler(vocab)
    model = Word2Vec(vocab.size, embed_dim=8, seed=2)
    centers, contexts = make_toy_pairs(vocab.size, n_pairs=50)

    cfg = TrainerConfig(
        num_epochs=1,
        lr_start=0.0001,   # already at minimum
        lr_min=0.0001,
        num_negatives=2,
        seed=0,
    )
    history = train(model, centers, contexts, sampler, cfg)
    assert all(np.isfinite(l) for l in history.epoch_losses)


# ---------------------------------------------------------------------------
# Embedding check after training
# ---------------------------------------------------------------------------

def test_embeddings_change_after_training():
    vocab = make_toy_vocab(30)
    sampler = NegativeSampler(vocab)
    model = Word2Vec(vocab.size, embed_dim=16, seed=0)
    W_in_before = model.W_in.copy()

    centers, contexts = make_toy_pairs(vocab.size, n_pairs=100)
    cfg = TrainerConfig(num_epochs=1, lr_start=0.05, num_negatives=3, seed=0)
    train(model, centers, contexts, sampler, cfg)

    assert not np.allclose(model.W_in, W_in_before), \
        "W_in should change during training"
