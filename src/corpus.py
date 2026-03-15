"""
Corpus loading, tokenization, and vocabulary construction.

The module converts a raw text file into integer token sequences and builds
the vocabulary needed by the word2vec training loop.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Vocabulary:
    """Bidirectional word <-> index mapping plus per-word frequencies."""

    word2idx: dict[str, int]
    idx2word: list[str]
    counts: np.ndarray  # shape (vocab_size,), raw occurrence counts

    @property
    def size(self) -> int:
        return len(self.idx2word)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx


# ---------------------------------------------------------------------------
# Text8 download helper
# ---------------------------------------------------------------------------

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_ZIP = "text8.zip"
TEXT8_TXT = "text8"


def download_text8(data_dir: str | Path = "data") -> Path:
    """Download and extract the text8 corpus if not already present.

    text8 is a 100 MB preprocessed Wikipedia dump widely used as a word2vec
    benchmark (single line, lowercase, letters only).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    txt_path = data_dir / TEXT8_TXT
    if txt_path.exists():
        return txt_path

    zip_path = data_dir / TEXT8_ZIP
    if not zip_path.exists():
        print(f"Downloading text8 from {TEXT8_URL} ...")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)

    print("Extracting text8.zip ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return txt_path


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lower-case and split on whitespace / punctuation.

    For text8 the corpus is already clean (letters + spaces), but this
    function also handles arbitrary raw text gracefully.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def load_tokens(path: str | Path, max_tokens: int | None = None) -> List[str]:
    """Read a text file and return a flat list of word tokens."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    tokens = tokenize(raw)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocabulary(tokens: List[str], min_count: int = 5) -> Vocabulary:
    """Build a Vocabulary from a token list.

    Words with fewer than *min_count* occurrences are discarded and mapped
    to <UNK>; this keeps the vocabulary tractable and reduces noise.

    Parameters
    ----------
    tokens:
        Flat list of word strings.
    min_count:
        Minimum occurrence threshold — words below this are dropped.

    Returns
    -------
    Vocabulary
        Sorted by descending frequency so that the most common words have
        the lowest indices (useful for frequency-based sampling).
    """
    raw_counts = Counter(tokens)
    # Keep only frequent words; sort descending by count for stable ordering
    vocab_words = sorted(
        [w for w, c in raw_counts.items() if c >= min_count],
        key=lambda w: -raw_counts[w],
    )

    word2idx: dict[str, int] = {w: i for i, w in enumerate(vocab_words)}
    idx2word: list[str] = vocab_words
    counts = np.array([raw_counts[w] for w in idx2word], dtype=np.float64)

    return Vocabulary(word2idx=word2idx, idx2word=idx2word, counts=counts)


# ---------------------------------------------------------------------------
# Token → index conversion
# ---------------------------------------------------------------------------

def tokens_to_ids(
    tokens: List[str],
    vocab: Vocabulary,
) -> np.ndarray:
    """Convert a token list to an integer array, dropping out-of-vocab words.

    Out-of-vocabulary tokens (below min_count) are simply skipped rather
    than mapped to <UNK>, which avoids polluting context windows.
    """
    ids = [vocab.word2idx[t] for t in tokens if t in vocab.word2idx]
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Subsampling of frequent words  (Mikolov et al., 2013, §2.3)
# ---------------------------------------------------------------------------

def subsample(
    token_ids: np.ndarray,
    vocab: Vocabulary,
    t: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Discard frequent tokens with probability proportional to their frequency.

    The discard probability for word w is:

        P(discard | w) = 1 - sqrt(t / f(w))

    where f(w) = count(w) / total_tokens.  This matches the formula used in
    the original word2vec C implementation and accelerates training by
    removing the most common words (e.g. 'the', 'a') that carry little
    semantic signal.

    Parameters
    ----------
    token_ids:
        Integer token sequence produced by ``tokens_to_ids``.
    vocab:
        Vocabulary with per-word counts.
    t:
        Subsampling threshold (default 1e-4 following Mikolov 2013).
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Filtered token sequence (shorter than the input).
    """
    if rng is None:
        rng = np.random.default_rng()

    total = vocab.counts.sum()
    freq = vocab.counts / total                    # shape (vocab_size,)
    keep_prob = np.minimum(1.0, np.sqrt(t / freq)) # shape (vocab_size,)

    draws = rng.random(len(token_ids))
    mask = draws < keep_prob[token_ids]
    return token_ids[mask]


# ---------------------------------------------------------------------------
# Context-window generation
# ---------------------------------------------------------------------------

def generate_skip_gram_pairs(
    token_ids: np.ndarray,
    window: int = 5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (center, context) index pairs for skip-gram training.

    For each center word a random window size in [1, window] is drawn,
    following the original word2vec implementation.  This gives closer
    context words a higher effective sampling weight.

    Parameters
    ----------
    token_ids:
        Integer token sequence (after subsampling).
    window:
        Maximum one-sided context window size.
    rng:
        NumPy random generator.

    Returns
    -------
    centers, contexts : np.ndarray, np.ndarray
        Parallel arrays of center and context word indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    centers_list: list[int] = []
    contexts_list: list[int] = []

    n = len(token_ids)
    for i, center in enumerate(token_ids):
        half_w = int(rng.integers(1, window + 1))  # dynamic window in [1, W]
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        for j in range(lo, hi):
            if j != i:
                centers_list.append(int(center))
                contexts_list.append(int(token_ids[j]))

    return np.array(centers_list, dtype=np.int32), np.array(contexts_list, dtype=np.int32)
