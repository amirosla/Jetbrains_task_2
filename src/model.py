"""
Skip-gram word2vec model with negative sampling — pure NumPy.

This module contains the entire forward pass, loss computation, and gradient
derivation.  No automatic differentiation is used; all gradients are derived
analytically and implemented explicitly.

Mathematical background
-----------------------
Given a centre word c and a context word o, the skip-gram model with negative
sampling minimises:

    J(c, o, K) = -log σ(u_o^T v_c)
               - Σ_{k=1}^{K} log σ(-u_k^T v_c)          (1)

where
    v_c  ∈ R^d  — input  (centre)  embedding of word c
    u_o  ∈ R^d  — output (context) embedding of word o
    u_k  ∈ R^d  — output embeddings of K negative-sample words
    σ(x) = 1 / (1 + exp(-x))  — sigmoid

Gradients
---------
Let p_o = σ(u_o^T v_c) and p_k = σ(u_k^T v_c).  Then:

    ∂J/∂v_c  = (p_o - 1) u_o  +  Σ_k p_k u_k              (2)
    ∂J/∂u_o  = (p_o - 1) v_c                                (3)
    ∂J/∂u_k  = p_k v_c          for each k                  (4)

The update rule (SGD with learning rate η):

    v_c  ←  v_c  - η · ∂J/∂v_c
    u_o  ←  u_o  - η · ∂J/∂u_o
    u_k  ←  u_k  - η · ∂J/∂u_k
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid σ(x) = 1 / (1 + e^{-x})."""
    # Clip to avoid overflow in exp for very negative values
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Word2Vec:
    """Skip-gram word2vec model backed by two embedding matrices.

    Attributes
    ----------
    W_in : np.ndarray, shape (vocab_size, embed_dim)
        Input (centre-word) embedding matrix.  After training, this is the
        primary embedding used for downstream tasks.
    W_out : np.ndarray, shape (vocab_size, embed_dim)
        Output (context-word) embedding matrix.  Sometimes averaged with
        W_in at inference time, but typically discarded.

    Parameters
    ----------
    vocab_size:
        Number of words in the vocabulary.
    embed_dim:
        Dimensionality of the word vectors (commonly 100–300).
    seed:
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Xavier-style uniform initialisation: keeps initial dot products
        # in a reasonable range regardless of embed_dim.
        bound = 0.5 / embed_dim
        self.W_in  = rng.uniform(-bound, bound, (vocab_size, embed_dim))
        self.W_out = rng.uniform(-bound, bound, (vocab_size, embed_dim))
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    # ------------------------------------------------------------------
    # Forward pass + loss + gradients for one (centre, context, negatives)
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        center_idx: int,
        context_idx: int,
        neg_indices: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the negative-sampling loss and parameter gradients.

        Implements equations (1)–(4) from the module docstring.

        Parameters
        ----------
        center_idx:
            Index of the centre word (c).
        context_idx:
            Index of the positive context word (o).
        neg_indices:
            Indices of K negative-sample words, shape (K,).

        Returns
        -------
        loss : float
            Scalar value of J(c, o, K).
        grad_v_c : np.ndarray, shape (embed_dim,)
            Gradient w.r.t. the input embedding of the centre word.
        grad_u_o : np.ndarray, shape (embed_dim,)
            Gradient w.r.t. the output embedding of the context word.
        grad_u_neg : np.ndarray, shape (K, embed_dim)
            Gradient w.r.t. the output embeddings of each negative word.
        """
        v_c  = self.W_in[center_idx]           # (d,)
        u_o  = self.W_out[context_idx]         # (d,)
        u_neg = self.W_out[neg_indices]        # (K, d)

        # ---- positive pair ----
        score_pos = float(np.dot(u_o, v_c))
        p_o = sigmoid(np.array([score_pos]))[0]            # σ(u_o^T v_c)

        # ---- negative pairs ----
        scores_neg = u_neg @ v_c               # (K,)
        p_neg = sigmoid(scores_neg)            # σ(u_k^T v_c), shape (K,)

        # ---- loss  (eq. 1) ----
        loss = (
            -np.log(p_o + 1e-10)
            - np.sum(np.log(1.0 - p_neg + 1e-10))
        )

        # ---- gradients (eqs. 2–4) ----
        # ∂J/∂v_c
        grad_v_c = (p_o - 1.0) * u_o + (p_neg[:, None] * u_neg).sum(axis=0)

        # ∂J/∂u_o
        grad_u_o = (p_o - 1.0) * v_c

        # ∂J/∂u_k  for each k  —  shape (K, d)
        grad_u_neg = p_neg[:, None] * v_c[None, :]

        return float(loss), grad_v_c, grad_u_o, grad_u_neg

    # ------------------------------------------------------------------
    # SGD parameter update
    # ------------------------------------------------------------------

    def update(
        self,
        center_idx: int,
        context_idx: int,
        neg_indices: np.ndarray,
        grad_v_c: np.ndarray,
        grad_u_o: np.ndarray,
        grad_u_neg: np.ndarray,
        lr: float,
    ) -> None:
        """Apply one SGD step to the embeddings involved in this sample.

        Only the rows that participated in the forward pass are updated,
        keeping the operation O(K · d) rather than O(V · d).

        Parameters
        ----------
        center_idx, context_idx, neg_indices:
            Same indices as passed to :meth:`forward_backward`.
        grad_v_c, grad_u_o, grad_u_neg:
            Gradients returned by :meth:`forward_backward`.
        lr:
            Current learning rate η.
        """
        self.W_in[center_idx]        -= lr * grad_v_c
        self.W_out[context_idx]      -= lr * grad_u_o
        self.W_out[neg_indices]      -= lr * grad_u_neg

    # ------------------------------------------------------------------
    # Convenience: combined forward + update
    # ------------------------------------------------------------------

    def train_step(
        self,
        center_idx: int,
        context_idx: int,
        neg_indices: np.ndarray,
        lr: float,
    ) -> float:
        """Execute one forward–backward–update cycle and return the loss."""
        loss, grad_v_c, grad_u_o, grad_u_neg = self.forward_backward(
            center_idx, context_idx, neg_indices
        )
        self.update(
            center_idx, context_idx, neg_indices,
            grad_v_c, grad_u_o, grad_u_neg,
            lr,
        )
        return loss

    # ------------------------------------------------------------------
    # Embedding access helpers
    # ------------------------------------------------------------------

    def get_embedding(self, idx: int) -> np.ndarray:
        """Return the normalised input embedding for word *idx*."""
        v = self.W_in[idx]
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10)

    def get_all_embeddings(self) -> np.ndarray:
        """Return L2-normalised input embeddings, shape (vocab_size, d)."""
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        return self.W_in / (norms + 1e-10)
