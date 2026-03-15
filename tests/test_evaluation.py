"""Tests for embedding evaluation utilities (cosine similarity, analogies)."""

import numpy as np
import pytest

from src.corpus import Vocabulary
from src.model import Word2Vec
from src.evaluation import cosine_similarity, most_similar, word_analogy, words_in_vocab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vocab_and_model(words: list[str]) -> tuple[Vocabulary, Word2Vec]:
    """Create a small Vocabulary and a Word2Vec whose W_in rows are set manually."""
    word2idx = {w: i for i, w in enumerate(words)}
    counts = np.ones(len(words), dtype=np.float64)
    vocab = Vocabulary(word2idx=word2idx, idx2word=words, counts=counts)
    model = Word2Vec(vocab_size=len(words), embed_dim=4, seed=0)
    return vocab, model


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite_vectors():
    v = np.array([1.0, 1.0, 1.0])
    assert cosine_similarity(v, -v) < -0.99


def test_cosine_similarity_zero_vector_safe():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    # Should not raise; returns a finite number
    result = cosine_similarity(a, b)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# most_similar
# ---------------------------------------------------------------------------

def test_most_similar_returns_top_k():
    vocab, model = make_vocab_and_model(["apple", "banana", "cherry", "date", "elderberry"])
    results = most_similar("apple", model, vocab, top_k=3)
    assert len(results) == 3


def test_most_similar_excludes_query_word():
    vocab, model = make_vocab_and_model(["apple", "banana", "cherry", "date"])
    results = most_similar("apple", model, vocab, top_k=3)
    result_words = [w for w, _ in results]
    assert "apple" not in result_words


def test_most_similar_sorted_descending():
    vocab, model = make_vocab_and_model(["a", "b", "c", "d", "e"])
    results = most_similar("a", model, vocab, top_k=4)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)


def test_most_similar_similarity_range():
    vocab, model = make_vocab_and_model(["a", "b", "c", "d", "e"])
    results = most_similar("a", model, vocab, top_k=4)
    for _, sim in results:
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


def test_most_similar_known_nearest():
    """Set W_in rows explicitly so the nearest neighbour is deterministic."""
    words = ["target", "close", "far"]
    vocab, model = make_vocab_and_model(words)
    model.W_in[0] = np.array([1.0, 0.0, 0.0, 0.0])   # target
    model.W_in[1] = np.array([0.9, 0.1, 0.0, 0.0])   # close
    model.W_in[2] = np.array([0.0, 0.0, 1.0, 0.0])   # far

    results = most_similar("target", model, vocab, top_k=2)
    nearest_word, _ = results[0]
    assert nearest_word == "close"


def test_most_similar_raises_for_oov():
    vocab, model = make_vocab_and_model(["a", "b"])
    with pytest.raises(KeyError):
        most_similar("zzz", model, vocab)


# ---------------------------------------------------------------------------
# word_analogy
# ---------------------------------------------------------------------------

def test_word_analogy_returns_top_k():
    vocab, model = make_vocab_and_model(["man", "king", "woman", "queen", "other"])
    results = word_analogy("man", "king", "woman", model, vocab, top_k=2)
    assert len(results) == 2


def test_word_analogy_excludes_input_words():
    # vocab has 5 words; 3 are input words → only 2 valid candidates
    vocab, model = make_vocab_and_model(["man", "king", "woman", "queen", "x"])
    results = word_analogy("man", "king", "woman", model, vocab, top_k=2)
    result_words = [w for w, _ in results]
    for w in ("man", "king", "woman"):
        assert w not in result_words


def test_word_analogy_known_result():
    """Set embeddings so that king - man + woman ≈ queen is exact."""
    words = ["man", "king", "woman", "queen", "other"]
    vocab, model = make_vocab_and_model(words)

    dim = model.embed_dim
    model.W_in[0] = np.array([1.0, 0.0, 0.0, 0.0])          # man
    model.W_in[1] = np.array([1.0, 1.0, 0.0, 0.0])          # king   = man + royalty
    model.W_in[2] = np.array([0.0, 0.0, 1.0, 0.0])          # woman
    model.W_in[3] = np.array([0.0, 1.0, 1.0, 0.0])          # queen  = royalty + woman
    model.W_in[4] = np.array([0.0, 0.0, 0.0, 1.0])          # other

    results = word_analogy("man", "king", "woman", model, vocab, top_k=1)
    assert results[0][0] == "queen"


def test_word_analogy_raises_for_oov():
    vocab, model = make_vocab_and_model(["a", "b", "c"])
    with pytest.raises(KeyError):
        word_analogy("a", "b", "zzz", model, vocab)


# ---------------------------------------------------------------------------
# words_in_vocab
# ---------------------------------------------------------------------------

def test_words_in_vocab_filters_correctly():
    vocab, _ = make_vocab_and_model(["apple", "banana"])
    result = words_in_vocab(["apple", "cherry", "banana"], vocab)
    assert result == ["apple", "banana"]


def test_words_in_vocab_empty_input():
    vocab, _ = make_vocab_and_model(["apple"])
    assert words_in_vocab([], vocab) == []
