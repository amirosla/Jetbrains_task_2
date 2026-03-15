"""
Training loop for the skip-gram word2vec model.

Iterates over (centre, context) pairs, draws negative samples, calls the
model's forward–backward–update cycle, and decays the learning rate linearly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.model import Word2Vec
from src.sampling import NegativeSampler


@dataclass
class TrainerConfig:
    """Hyper-parameters for the training loop.

    Attributes
    ----------
    num_epochs:
        Number of full passes over the (centre, context) pairs.
    lr_start:
        Initial learning rate (default 0.025 — standard for word2vec SGD).
    lr_min:
        Minimum learning rate; training never goes below this value.
    num_negatives:
        Number of negative samples per positive pair (K in the paper;
        5–20 is recommended for smaller datasets, 2–5 for large ones).
    log_every:
        Print a progress line every this many steps.
    seed:
        Random seed used for negative sampling and pair shuffling.
    """
    num_epochs: int = 3
    lr_start: float = 0.025
    lr_min: float = 0.0001
    num_negatives: int = 5
    log_every: int = 100_000
    seed: int = 42


@dataclass
class TrainingHistory:
    """Accumulates per-epoch loss for later analysis."""
    epoch_losses: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


def train(
    model: Word2Vec,
    centers: np.ndarray,
    contexts: np.ndarray,
    sampler: NegativeSampler,
    cfg: TrainerConfig | None = None,
) -> TrainingHistory:
    """Train *model* on skip-gram pairs using negative sampling.

    The learning rate decays linearly from ``cfg.lr_start`` to ``cfg.lr_min``
    across the total number of training steps.  This matches the schedule used
    in the original word2vec C code and helps the model converge smoothly.

    Parameters
    ----------
    model:
        Initialised :class:`Word2Vec` instance to train in-place.
    centers, contexts:
        Parallel integer arrays of (centre, context) pair indices,
        as produced by :func:`~src.corpus.generate_skip_gram_pairs`.
    sampler:
        :class:`~src.sampling.NegativeSampler` for drawing noise words.
    cfg:
        Training configuration.  Uses sensible defaults if ``None``.

    Returns
    -------
    TrainingHistory
        Per-epoch average loss and wall-clock time for tracking convergence.
    """
    if cfg is None:
        cfg = TrainerConfig()

    rng = np.random.default_rng(cfg.seed)
    history = TrainingHistory()

    n_pairs = len(centers)
    total_steps = cfg.num_epochs * n_pairs

    step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        # Shuffle pairs at the start of each epoch
        perm = rng.permutation(n_pairs)
        c_shuf = centers[perm]
        o_shuf = contexts[perm]

        epoch_loss = 0.0
        t0 = time.perf_counter()

        for i in range(n_pairs):
            # ---- linearly decayed learning rate ----
            progress = step / max(total_steps - 1, 1)
            lr = cfg.lr_start * (1.0 - progress)
            lr = max(lr, cfg.lr_min)

            center_idx  = int(c_shuf[i])
            context_idx = int(o_shuf[i])

            # Draw K negative samples (distinct from the positive context)
            neg_indices = sampler.sample(
                cfg.num_negatives,
                exclude=context_idx,
                rng=rng,
            )

            loss = model.train_step(center_idx, context_idx, neg_indices, lr)
            epoch_loss += loss
            step += 1

            if step % cfg.log_every == 0:
                avg = epoch_loss / (i + 1)
                elapsed = time.perf_counter() - t0
                print(
                    f"  epoch {epoch}/{cfg.num_epochs} "
                    f"step {step:,}/{total_steps:,} "
                    f"lr={lr:.5f}  avg_loss={avg:.4f}  "
                    f"elapsed={elapsed:.1f}s"
                )

        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / n_pairs
        history.epoch_losses.append(avg_loss)
        history.epoch_times.append(elapsed)
        print(
            f"Epoch {epoch}/{cfg.num_epochs} complete — "
            f"avg loss={avg_loss:.4f}  time={elapsed:.1f}s"
        )

    return history
