"""
Command-line entry point for training word2vec on the text8 corpus.

Usage
-----
    python scripts/train.py [OPTIONS]

Options
-------
    --data-dir DIR        directory where text8 is stored/downloaded [data]
    --results-dir DIR     directory to save embeddings and plots [results]
    --embed-dim INT        embedding dimensionality [100]
    --epochs INT          number of training epochs [3]
    --lr FLOAT            initial learning rate [0.025]
    --negatives INT       negative samples per positive pair [5]
    --window INT          maximum context window size [5]
    --min-count INT       minimum word frequency for vocabulary [5]
    --max-tokens INT      cap on corpus tokens (useful for quick runs) [None]
    --seed INT            random seed [42]
    --no-plots            skip matplotlib visualisation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import (
    download_text8,
    load_tokens,
    build_vocabulary,
    tokens_to_ids,
    subsample,
    generate_skip_gram_pairs,
)
from src.sampling import NegativeSampler
from src.model import Word2Vec
from src.trainer import TrainerConfig, train
from src.evaluation import most_similar, word_analogy, words_in_vocab


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train skip-gram word2vec in pure NumPy on the text8 corpus."
    )
    p.add_argument("--data-dir",    default="data",    type=str)
    p.add_argument("--results-dir", default="results", type=str)
    p.add_argument("--embed-dim",   default=100,  type=int)
    p.add_argument("--epochs",      default=3,    type=int)
    p.add_argument("--lr",          default=0.025, type=float)
    p.add_argument("--negatives",   default=5,    type=int)
    p.add_argument("--window",      default=5,    type=int)
    p.add_argument("--min-count",   default=5,    type=int)
    p.add_argument("--max-tokens",  default=None, type=int,
                   help="Limit corpus size for faster experiments")
    p.add_argument("--seed",        default=42,   type=int)
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip matplotlib visualisation (useful in headless envs)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss(epoch_losses: list[float], results_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss (negative sampling)")
    ax.set_title("word2vec training loss per epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = results_dir / "training_loss.png"
    fig.savefig(path, dpi=120)
    print(f"  Loss plot saved → {path}")
    plt.close(fig)


def plot_embeddings_tsne(
    model: Word2Vec,
    vocab,
    results_dir: Path,
    n_words: int = 300,
) -> None:
    """t-SNE projection of the top-n_words embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn or matplotlib not installed — skipping t-SNE plot")
        return

    embs = model.get_all_embeddings()[:n_words]   # top words by frequency
    words = vocab.idx2word[:n_words]

    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.6)
    for i, w in enumerate(words[:100]):
        ax.annotate(w, proj[i], fontsize=7, alpha=0.8)
    ax.set_title(f"t-SNE of top-{n_words} word embeddings")
    ax.axis("off")
    fig.tight_layout()
    path = results_dir / "tsne.png"
    fig.savefig(path, dpi=120)
    print(f"  t-SNE plot saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative evaluation
# ---------------------------------------------------------------------------

ANALOGY_EXAMPLES = [
    ("man",    "king",   "woman"),   # → queen
    ("paris",  "france", "berlin"),  # → germany
    ("good",   "better", "bad"),     # → worse
]

SIMILARITY_EXAMPLES = [
    "king", "computer", "france", "music", "science",
]


def run_qualitative_eval(model: Word2Vec, vocab) -> None:
    print("\n── Nearest neighbours ──────────────────────────────────────────")
    for word in SIMILARITY_EXAMPLES:
        if word not in vocab:
            print(f"  {word!r} not in vocabulary, skipping")
            continue
        neighbours = most_similar(word, model, vocab, top_k=5)
        nn_str = "  ".join(f"{w}({s:.3f})" for w, s in neighbours)
        print(f"  {word:<12} →  {nn_str}")

    print("\n── Word analogies (a : b :: c : ?) ──────────────────────────────")
    for a, b, c in ANALOGY_EXAMPLES:
        present = words_in_vocab([a, b, c], vocab)
        if len(present) < 3:
            missing = set([a, b, c]) - set(present)
            print(f"  {a}:{b}::{c}:?  — skipped (not in vocab: {missing})")
            continue
        candidates = word_analogy(a, b, c, model, vocab, top_k=3)
        cand_str = "  ".join(f"{w}({s:.3f})" for w, s in candidates)
        print(f"  {a} : {b} :: {c} : ?  →  {cand_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n════════════════════════════════════════")
    print("  1 / 5  Loading corpus")
    print("════════════════════════════════════════")

    corpus_path = download_text8(args.data_dir)
    tokens = load_tokens(corpus_path, max_tokens=args.max_tokens)
    print(f"  Raw tokens: {len(tokens):,}")

    # ── 2. Vocabulary ────────────────────────────────────────────────────
    print("\n════════════════════════════════════════")
    print("  2 / 5  Building vocabulary")
    print("════════════════════════════════════════")

    vocab = build_vocabulary(tokens, min_count=args.min_count)
    print(f"  Vocabulary size: {vocab.size:,}  (min_count={args.min_count})")

    token_ids = tokens_to_ids(tokens, vocab)
    print(f"  Token IDs (after OOV drop): {len(token_ids):,}")

    # Subsampling
    token_ids = subsample(token_ids, vocab, t=1e-4, rng=rng)
    print(f"  Token IDs (after subsampling): {len(token_ids):,}")

    # ── 3. Skip-gram pairs ───────────────────────────────────────────────
    print("\n════════════════════════════════════════")
    print("  3 / 5  Generating skip-gram pairs")
    print("════════════════════════════════════════")

    centers, contexts = generate_skip_gram_pairs(token_ids, window=args.window, rng=rng)
    print(f"  Training pairs: {len(centers):,}")

    # ── 4. Training ──────────────────────────────────────────────────────
    print("\n════════════════════════════════════════")
    print("  4 / 5  Training")
    print("════════════════════════════════════════")

    model   = Word2Vec(vocab.size, embed_dim=args.embed_dim, seed=args.seed)
    sampler = NegativeSampler(vocab, power=0.75)

    cfg = TrainerConfig(
        num_epochs=args.epochs,
        lr_start=args.lr,
        num_negatives=args.negatives,
        seed=args.seed,
    )
    history = train(model, centers, contexts, sampler, cfg)

    # ── 5. Save + evaluate ───────────────────────────────────────────────
    print("\n════════════════════════════════════════")
    print("  5 / 5  Evaluation & artefacts")
    print("════════════════════════════════════════")

    emb_path = results_dir / "embeddings.npy"
    np.save(emb_path, model.W_in)
    print(f"  Embeddings saved → {emb_path}")

    if not args.no_plots:
        plot_loss(history.epoch_losses, results_dir)

    run_qualitative_eval(model, vocab)

    print("\n════════════════════════════════════════")
    print("  Training complete.")
    print("════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
