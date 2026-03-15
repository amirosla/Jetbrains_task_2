"""
Negative sampling distribution.

The negative sampler draws "noise" words from a smoothed unigram distribution
P_n(w) ∝ count(w)^0.75, as described in Mikolov et al. (2013), §2.2.

Raising frequencies to the power 0.75 (rather than 1.0) increases the
probability of rare words being chosen as negatives, which has been shown
empirically to improve embedding quality.
"""

from __future__ import annotations

import numpy as np

from src.corpus import Vocabulary


class NegativeSampler:
    """Pre-computed alias table for O(1) negative sampling.

    Uses Walker's alias method so that each call to :meth:`sample` runs in
    O(k) regardless of vocabulary size.

    Parameters
    ----------
    vocab:
        Vocabulary whose ``counts`` array defines word frequencies.
    power:
        Smoothing exponent (default 0.75 following Mikolov 2013).
    table_size:
        Number of entries in the alias table.  Larger values give a better
        approximation of the smoothed unigram distribution.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        power: float = 0.75,
        table_size: int = 100_000_000,
    ) -> None:
        self._vocab_size = vocab.size
        smoothed = np.power(vocab.counts, power)
        self._probs = smoothed / smoothed.sum()

        # Build alias table
        self._alias, self._prob = self._build_alias_table(self._probs)

    # ------------------------------------------------------------------
    # Alias table construction  (Vose, 1991)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_alias_table(
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the alias table for Walker's alias method.

        Given a probability distribution over n outcomes, the alias table
        allows O(1) sampling by partitioning the probability mass so that
        each entry stores at most two outcomes.

        Parameters
        ----------
        probs:
            Normalised probability distribution, shape (n,).

        Returns
        -------
        alias : np.ndarray, shape (n,)
            Index of the "alias" outcome for each entry.
        prob  : np.ndarray, shape (n,)
            Probability of the *primary* outcome for each entry.
        """
        n = len(probs)
        alias = np.zeros(n, dtype=np.int64)
        prob = np.zeros(n, dtype=np.float64)

        scaled = probs * n
        small: list[int] = []
        large: list[int] = []

        for i, p in enumerate(scaled):
            (small if p < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            prob[s] = scaled[s]
            alias[s] = l
            scaled[l] = scaled[l] + scaled[s] - 1.0
            (small if scaled[l] < 1.0 else large).append(l)

        for i in large + small:
            prob[i] = 1.0

        return alias, prob

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        exclude: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw *n* negative samples from the smoothed unigram distribution.

        Parameters
        ----------
        n:
            Number of samples to draw.
        exclude:
            Word index that must not appear in the returned samples (i.e.
            the current positive context word).  Re-sampling is used if a
            collision occurs; for large vocabularies this is rare.
        rng:
            NumPy random generator.

        Returns
        -------
        np.ndarray, shape (n,), dtype int64
        """
        if rng is None:
            rng = np.random.default_rng()

        k = int(n)
        result = np.empty(k, dtype=np.int64)
        filled = 0

        # Over-sample to reduce the number of re-draw iterations needed
        # when exclude collisions occur.
        while filled < k:
            batch = k - filled + 4          # small buffer for collisions
            i = rng.integers(0, self._vocab_size, size=batch)
            u = rng.random(batch)
            chosen = np.where(u < self._prob[i], i, self._alias[i])
            if exclude is not None:
                chosen = chosen[chosen != exclude]
            take = min(k - filled, len(chosen))
            result[filled : filled + take] = chosen[:take]
            filled += take

        return result
