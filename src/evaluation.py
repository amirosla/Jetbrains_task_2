"""
Embedding evaluation utilities.

Two standard probes are provided:

1. **Nearest-neighbour similarity** — given a query word, returns the most
   similar words by cosine distance.  Useful for a quick sanity check that
   semantically related words cluster together.

2. **Word analogy** (3CosAdd) — evaluates the classic "king - man + woman ≈
   queen" relationship.  The predicted answer is the word w* that maximises

       cos(w*, v_b - v_a + v_c)

   with a, b, c excluded from the candidate set.

Neither metric requires labelled data, so both can be computed at any point
during or after training.
"""

from __future__ import annotations

import numpy as np

from src.corpus import Vocabulary
from src.model import Word2Vec


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) + 1e-10) * (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)


def most_similar(
    query: str,
    model: Word2Vec,
    vocab: Vocabulary,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the *top_k* words most similar to *query* by cosine distance.

    Parameters
    ----------
    query:
        The query word string.
    model:
        Trained :class:`Word2Vec` model.
    vocab:
        Vocabulary used during training.
    top_k:
        Number of nearest neighbours to return.

    Returns
    -------
    List of (word, similarity) tuples, sorted descending by similarity.
    Raises ``KeyError`` if *query* is not in the vocabulary.
    """
    if query not in vocab:
        raise KeyError(f"'{query}' not in vocabulary")

    query_idx = vocab.word2idx[query]
    query_vec = model.get_embedding(query_idx)           # (d,)  normalised

    all_embs = model.get_all_embeddings()                # (V, d) normalised
    sims = all_embs @ query_vec                          # (V,)  dot = cosine

    # Exclude the query word itself
    sims[query_idx] = -np.inf

    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [(vocab.idx2word[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Word analogy (3CosAdd)
# ---------------------------------------------------------------------------

def word_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    model: Word2Vec,
    vocab: Vocabulary,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Solve the analogy  a : b  ::  c : ?  using the 3CosAdd method.

    Computes  v_b - v_a + v_c  and returns the words whose embeddings are
    closest to this vector (excluding a, b, c themselves).

    Parameters
    ----------
    word_a, word_b, word_c:
        The three analogy words (e.g. "man", "king", "woman").
    model:
        Trained :class:`Word2Vec` model.
    vocab:
        Vocabulary used during training.
    top_k:
        Number of candidate answers to return.

    Returns
    -------
    List of (word, similarity) tuples.
    Raises ``KeyError`` if any of the three words is not in the vocabulary.
    """
    for w in (word_a, word_b, word_c):
        if w not in vocab:
            raise KeyError(f"'{w}' not in vocabulary")

    idx_a = vocab.word2idx[word_a]
    idx_b = vocab.word2idx[word_b]
    idx_c = vocab.word2idx[word_c]

    all_embs = model.get_all_embeddings()           # (V, d) normalised

    query = all_embs[idx_b] - all_embs[idx_a] + all_embs[idx_c]
    query = query / (np.linalg.norm(query) + 1e-10)

    sims = all_embs @ query                         # (V,)

    # Exclude the three input words
    for idx in (idx_a, idx_b, idx_c):
        sims[idx] = -np.inf

    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [(vocab.idx2word[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Vocabulary coverage check
# ---------------------------------------------------------------------------

def words_in_vocab(words: list[str], vocab: Vocabulary) -> list[str]:
    """Return the subset of *words* that appear in *vocab*."""
    return [w for w in words if w in vocab]
