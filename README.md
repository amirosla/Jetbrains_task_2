# word2vec — from scratch in pure NumPy

A clean, well-documented implementation of **skip-gram word2vec with negative sampling**, written entirely in NumPy — no PyTorch, TensorFlow, or other ML frameworks.

---

## Overview

This repository implements the core word2vec training loop as described in:

> Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)

Every component — forward pass, loss function, gradient derivation, and parameter update — is implemented explicitly in NumPy so that the code is easy to follow and understand.

---

## Task summary

> Implement the core training loop of word2vec in pure NumPy.
> Implement the optimisation procedure (forward pass, loss, gradients, parameter updates) for skip-gram with negative sampling.

---

## Approach

### Model: skip-gram with negative sampling

For each centre word **c** and positive context word **o**, the model minimises:

```
J(c, o, K) = −log σ(u_oᵀ v_c) − Σ_{k=1}^{K} log σ(−u_kᵀ v_c)
```

where:
- `v_c ∈ ℝᵈ` — input (centre-word) embedding
- `u_o ∈ ℝᵈ` — output (context-word) embedding
- `u_k` — embeddings of K randomly sampled *noise* words
- `σ(x) = 1 / (1 + e^{−x})` — sigmoid

### Gradients (derived analytically)

Let `p_o = σ(u_oᵀ v_c)` and `p_k = σ(u_kᵀ v_c)`:

```
∂J/∂v_c  = (p_o − 1) u_o  +  Σ_k p_k u_k
∂J/∂u_o  = (p_o − 1) v_c
∂J/∂u_k  = p_k v_c          for each k
```

All three are implemented explicitly in [`src/model.py`](src/model.py) (`Word2Vec.forward_backward`) and verified against finite-difference approximations in the test suite.

### Update rule (SGD with linear learning rate decay)

```
θ ← θ − η · ∂J/∂θ
```

The learning rate decays linearly from `lr_start` to `lr_min` over all training steps, matching the schedule in the original word2vec C implementation.

### Negative sampling distribution

Noise words are drawn from a smoothed unigram distribution `P_n(w) ∝ count(w)^0.75`.
Sampling uses **Walker's alias method** (pre-computed in `O(V)`, each draw in `O(1)`).

### Subsampling of frequent words

High-frequency words (e.g. "the", "a") are discarded with probability:

```
P(discard | w) = 1 − sqrt(t / f(w))
```

where `f(w) = count(w) / total_tokens` and `t = 1e-4` (Mikolov 2013, §2.3).

---

## Stack

| Component | Library |
|-----------|---------|
| Linear algebra, all gradients | NumPy |
| Progress display | tqdm |
| Loss curve plot | Matplotlib |
| Tests | pytest |

No ML frameworks used anywhere in `src/`.

---

## Project structure

```
word2vec-numpy/
├── src/
│   ├── corpus.py        # tokenisation, vocabulary, subsampling, pair generation
│   ├── sampling.py      # alias-table negative sampler
│   ├── model.py         # Word2Vec: forward pass, analytical gradients, SGD update
│   ├── trainer.py       # training loop with linear LR decay
│   └── evaluation.py    # cosine similarity, nearest neighbours, word analogies
├── tests/
│   ├── test_corpus.py   # tokenisation, vocab, subsampling, skip-gram pairs
│   ├── test_sampling.py # alias table correctness, distribution match
│   ├── test_model.py    # gradients (incl. numerical checks), SGD, convergence
│   └── test_evaluation.py # cosine sim, most_similar, word_analogy
├── scripts/
│   └── train.py         # CLI entry point
├── data/                # corpus downloaded here at runtime
├── results/             # embeddings (.npy) and plots saved here
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/amirosla/Jetbrains_task_2.git
cd Jetbrains_task_2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

---

## Running

### Full training on text8

```bash
python scripts/train.py
```

The script will automatically download and extract the **text8** corpus (~100 MB) on the first run.

### Quick smoke-test run (small corpus slice)

```bash
python scripts/train.py --max-tokens 500000 --epochs 1 --embed-dim 50
```

### All options

```
--data-dir DIR        where to store/find text8             [data]
--results-dir DIR     where to save embeddings and plots    [results]
--embed-dim INT       embedding dimensionality              [100]
--epochs INT          number of training epochs             [3]
--lr FLOAT            initial learning rate                 [0.025]
--negatives INT       negative samples per pair             [5]
--window INT          max context window size               [5]
--min-count INT       minimum word frequency                [5]
--max-tokens INT      cap corpus size (for quick tests)     [all]
--seed INT            random seed                           [42]
--no-plots            skip matplotlib output
```

---

## Testing

```bash
python -m pytest tests/ -v
```

61 tests covering:
- tokenisation and vocabulary construction
- subsampling statistics
- alias-table distribution accuracy
- **numerical gradient checks** for all three gradient expressions (∂J/∂v_c, ∂J/∂u_o, ∂J/∂u_k)
- SGD convergence on a toy corpus
- nearest-neighbour and analogy evaluation

---

## Assumptions

- **Dataset**: text8 (preprocessed Wikipedia, letters only). Chosen because it is a standard word2vec benchmark, freely available, and clean enough to need no additional preprocessing.
- **Variant**: skip-gram rather than CBOW, because it produces better embeddings for rare words and has a more interesting gradient derivation.
- **Negative sampling over hierarchical softmax**: simpler to implement and verify, and empirically comparable on this scale.
- **No batching**: each (centre, context, negatives) tuple is processed individually, matching the original SGD formulation. A batched variant would be faster in NumPy but obscures the algorithm.

---

## Design decisions

| Decision | Rationale |
|----------|-----------|
| Two separate embedding matrices (`W_in`, `W_out`) | Follows the original paper; allows centre and context roles to differ |
| Xavier-style uniform initialisation | Keeps initial dot products in a sensible range independent of `embed_dim` |
| Walker's alias method for negative sampling | O(1) per draw vs O(V) for naïve sampling; essential for large vocabularies |
| Linear learning rate decay | Matches the original C implementation; prevents large late-training updates |
| Sorted vocabulary (by descending frequency) | Makes the top-K most frequent words predictable; simplifies t-SNE visualisation |

---

## Trade-offs

- **Speed vs clarity**: the inner loop is pure Python/NumPy and processes one pair at a time. A vectorised mini-batch implementation would be ~10–50× faster but harder to follow.
- **Memory**: two full `(V × d)` matrices are kept in RAM. For very large vocabularies this could be reduced by memory-mapping, but adds complexity.
- **Evaluation**: only qualitative evaluation (nearest neighbours, analogies) is included. Quantitative benchmarks (WordSim-353, SimLex-999) would require additional data files.

---

## Future improvements

- **Vectorised mini-batch training** for faster throughput on large corpora
- **CBOW variant** for comparison
- **Hierarchical softmax** as an alternative to negative sampling
- **Quantitative evaluation** on WordSim-353 / SimLex-999 benchmarks
- **Learning rate warm-up** for more stable early training
- **Embedding averaging** (W_in + W_out) / 2 at inference, which sometimes improves downstream performance
