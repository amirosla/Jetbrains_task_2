"""Tests for corpus loading, vocabulary construction, and pair generation."""

import numpy as np
import pytest

from src.corpus import (
    tokenize,
    build_vocabulary,
    tokens_to_ids,
    subsample,
    generate_skip_gram_pairs,
)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    tokens = tokenize("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenize_strips_punctuation():
    tokens = tokenize("Hello, world! It's a test.")
    assert "hello" in tokens
    assert "world" in tokens
    # Punctuation and apostrophes should not appear
    for t in tokens:
        assert t.isalpha()


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_numbers_excluded():
    tokens = tokenize("abc 123 def")
    assert "123" not in tokens
    assert "abc" in tokens


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SAMPLE_TOKENS = ["the", "the", "the", "cat", "cat", "sat", "on", "mat"]


def test_vocabulary_size_respects_min_count():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=2)
    # "the" (3), "cat" (2) pass; "sat", "on", "mat" (1 each) are dropped
    assert vocab.size == 2
    assert "the" in vocab
    assert "cat" in vocab
    assert "sat" not in vocab


def test_vocabulary_sorted_by_frequency():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=1)
    # Most frequent word should be first
    assert vocab.idx2word[0] == "the"


def test_vocabulary_counts_correct():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=1)
    assert vocab.counts[vocab.word2idx["the"]] == 3
    assert vocab.counts[vocab.word2idx["cat"]] == 2


def test_vocabulary_contains_operator():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=2)
    assert "the" in vocab
    assert "xyz" not in vocab


# ---------------------------------------------------------------------------
# Token → IDs
# ---------------------------------------------------------------------------

def test_tokens_to_ids_drops_oov():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=2)
    ids = tokens_to_ids(SAMPLE_TOKENS, vocab)
    # "sat", "on", "mat" are OOV → dropped
    expected_length = SAMPLE_TOKENS.count("the") + SAMPLE_TOKENS.count("cat")
    assert len(ids) == expected_length


def test_tokens_to_ids_correct_indices():
    vocab = build_vocabulary(SAMPLE_TOKENS, min_count=2)
    ids = tokens_to_ids(["the", "cat"], vocab)
    assert ids[0] == vocab.word2idx["the"]
    assert ids[1] == vocab.word2idx["cat"]


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def test_subsample_reduces_length():
    # Build a larger corpus so subsampling has a meaningful effect
    tokens = ["the"] * 10_000 + ["rare"] * 10
    vocab = build_vocabulary(tokens, min_count=1)
    ids = tokens_to_ids(tokens, vocab)
    rng = np.random.default_rng(0)
    kept = subsample(ids, vocab, t=1e-4, rng=rng)
    # "the" is very frequent — many should be discarded
    assert len(kept) < len(ids)


def test_subsample_keeps_rare_words():
    # A corpus with one dominant and one rare word
    tokens = ["frequent"] * 50_000 + ["rare"] * 5
    vocab = build_vocabulary(tokens, min_count=5)
    ids = tokens_to_ids(tokens, vocab)
    rng = np.random.default_rng(0)
    kept = subsample(ids, vocab, t=1e-4, rng=rng)
    # "rare" tokens should mostly survive; "frequent" tokens should be cut
    rare_idx = vocab.word2idx["rare"]
    n_rare_kept = int(np.sum(kept == rare_idx))
    assert n_rare_kept >= 4, f"Expected ≥4 rare tokens kept, got {n_rare_kept}"


# ---------------------------------------------------------------------------
# Skip-gram pair generation
# ---------------------------------------------------------------------------

def test_skip_gram_pairs_non_empty():
    token_ids = np.arange(10, dtype=np.int32)
    centers, contexts = generate_skip_gram_pairs(token_ids, window=2)
    assert len(centers) > 0
    assert len(centers) == len(contexts)


def test_skip_gram_pairs_no_self_pairs():
    token_ids = np.arange(20, dtype=np.int32)
    centers, contexts = generate_skip_gram_pairs(token_ids, window=3)
    # Center and context should never be the same index position
    # (they CAN share the same word id by coincidence, but not the same position)
    assert np.all(centers != contexts)


def test_skip_gram_pairs_within_window():
    token_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    window = 1
    centers, contexts = generate_skip_gram_pairs(token_ids, window=window, rng=np.random.default_rng(0))
    # With window=1 each pair must be adjacent in the token_ids array
    for c, o in zip(centers.tolist(), contexts.tolist()):
        c_pos = np.where(token_ids == c)[0]
        o_pos = np.where(token_ids == o)[0]
        distances = np.abs(c_pos[:, None] - o_pos[None, :]).min()
        assert distances <= window
