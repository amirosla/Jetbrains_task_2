"""Tests for the Word2Vec model: forward pass, gradients, and SGD updates."""

import numpy as np
import pytest

from src.model import Word2Vec, sigmoid


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_zero():
    assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9


def test_sigmoid_positive():
    assert sigmoid(np.array([100.0]))[0] > 0.99


def test_sigmoid_negative():
    assert sigmoid(np.array([-100.0]))[0] < 0.01


def test_sigmoid_shape_preserved():
    x = np.ones((3, 4))
    assert sigmoid(x).shape == (3, 4)


def test_sigmoid_no_nan_or_inf():
    x = np.array([-1000.0, 0.0, 1000.0])
    result = sigmoid(x)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_model_embedding_shapes():
    model = Word2Vec(vocab_size=100, embed_dim=50)
    assert model.W_in.shape  == (100, 50)
    assert model.W_out.shape == (100, 50)


def test_model_embeddings_differ():
    model = Word2Vec(vocab_size=100, embed_dim=50, seed=0)
    # W_in and W_out should be independently initialised
    assert not np.allclose(model.W_in, model.W_out)


def test_model_reproducible_seed():
    m1 = Word2Vec(vocab_size=50, embed_dim=10, seed=7)
    m2 = Word2Vec(vocab_size=50, embed_dim=10, seed=7)
    np.testing.assert_array_equal(m1.W_in, m2.W_in)
    np.testing.assert_array_equal(m1.W_out, m2.W_out)


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------

def _make_model_and_indices(vocab_size=20, embed_dim=10, seed=0):
    model = Word2Vec(vocab_size=vocab_size, embed_dim=embed_dim, seed=seed)
    center  = 0
    context = 1
    negs    = np.array([2, 3, 4], dtype=np.int64)
    return model, center, context, negs


def test_forward_backward_returns_finite_loss():
    model, center, context, negs = _make_model_and_indices()
    loss, *_ = model.forward_backward(center, context, negs)
    assert np.isfinite(loss)
    assert loss > 0.0


def test_forward_backward_gradient_shapes():
    model, center, context, negs = _make_model_and_indices()
    _, grad_v, grad_u_o, grad_u_neg = model.forward_backward(center, context, negs)
    assert grad_v.shape    == (10,)
    assert grad_u_o.shape  == (10,)
    assert grad_u_neg.shape == (3, 10)


def test_forward_backward_gradients_finite():
    model, center, context, negs = _make_model_and_indices()
    _, grad_v, grad_u_o, grad_u_neg = model.forward_backward(center, context, negs)
    for g in (grad_v, grad_u_o, grad_u_neg):
        assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# Numerical gradient check
# ---------------------------------------------------------------------------

def _loss_only(model: Word2Vec, center: int, context: int, negs: np.ndarray) -> float:
    loss, *_ = model.forward_backward(center, context, negs)
    return loss


def test_gradient_v_c_numerical():
    """Verify ∂J/∂v_c against a finite-difference approximation."""
    model, center, context, negs = _make_model_and_indices(seed=42)
    eps = 1e-5

    _, grad_v_c, _, _ = model.forward_backward(center, context, negs)

    numerical = np.zeros_like(grad_v_c)
    for i in range(len(grad_v_c)):
        model.W_in[center, i] += eps
        loss_plus = _loss_only(model, center, context, negs)
        model.W_in[center, i] -= 2 * eps
        loss_minus = _loss_only(model, center, context, negs)
        model.W_in[center, i] += eps          # restore
        numerical[i] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(grad_v_c, numerical, rtol=1e-4, atol=1e-6)


def test_gradient_u_o_numerical():
    """Verify ∂J/∂u_o against a finite-difference approximation."""
    model, center, context, negs = _make_model_and_indices(seed=5)
    eps = 1e-5

    _, _, grad_u_o, _ = model.forward_backward(center, context, negs)

    numerical = np.zeros_like(grad_u_o)
    for i in range(len(grad_u_o)):
        model.W_out[context, i] += eps
        loss_plus = _loss_only(model, center, context, negs)
        model.W_out[context, i] -= 2 * eps
        loss_minus = _loss_only(model, center, context, negs)
        model.W_out[context, i] += eps
        numerical[i] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(grad_u_o, numerical, rtol=1e-4, atol=1e-6)


def test_gradient_u_neg_numerical():
    """Verify ∂J/∂u_k against a finite-difference approximation."""
    model, center, context, negs = _make_model_and_indices(seed=99)
    eps = 1e-5

    _, _, _, grad_u_neg = model.forward_backward(center, context, negs)

    for k, neg_idx in enumerate(negs):
        numerical_k = np.zeros(model.embed_dim)
        for i in range(model.embed_dim):
            model.W_out[neg_idx, i] += eps
            loss_plus = _loss_only(model, center, context, negs)
            model.W_out[neg_idx, i] -= 2 * eps
            loss_minus = _loss_only(model, center, context, negs)
            model.W_out[neg_idx, i] += eps
            numerical_k[i] = (loss_plus - loss_minus) / (2 * eps)
        np.testing.assert_allclose(
            grad_u_neg[k], numerical_k, rtol=1e-4, atol=1e-6,
            err_msg=f"Gradient mismatch for negative sample k={k}"
        )


# ---------------------------------------------------------------------------
# SGD update
# ---------------------------------------------------------------------------

def test_update_changes_embeddings():
    model, center, context, negs = _make_model_and_indices()
    v_before = model.W_in[center].copy()
    u_o_before = model.W_out[context].copy()

    loss, grad_v, grad_u_o, grad_u_neg = model.forward_backward(center, context, negs)
    model.update(center, context, negs, grad_v, grad_u_o, grad_u_neg, lr=0.01)

    assert not np.allclose(model.W_in[center],  v_before)
    assert not np.allclose(model.W_out[context], u_o_before)


def test_update_does_not_touch_other_rows():
    model, center, context, negs = _make_model_and_indices()
    untouched_idx = 10   # not involved in this step
    untouched_before = model.W_in[untouched_idx].copy()

    loss, grad_v, grad_u_o, grad_u_neg = model.forward_backward(center, context, negs)
    model.update(center, context, negs, grad_v, grad_u_o, grad_u_neg, lr=0.01)

    np.testing.assert_array_equal(model.W_in[untouched_idx], untouched_before)


def test_train_step_reduces_loss_on_repeated_calls():
    """Repeatedly training on the same pair should drive its loss down."""
    model = Word2Vec(vocab_size=10, embed_dim=20, seed=0)
    center, context = 0, 1
    negs = np.array([2, 3, 4], dtype=np.int64)

    losses = [model.train_step(center, context, negs, lr=0.1) for _ in range(200)]
    # Loss should trend downward
    assert losses[-1] < losses[0], (
        f"Expected loss to decrease; first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def test_get_embedding_is_unit_norm():
    model = Word2Vec(vocab_size=50, embed_dim=20, seed=1)
    for idx in [0, 10, 49]:
        v = model.get_embedding(idx)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_get_all_embeddings_shape():
    model = Word2Vec(vocab_size=50, embed_dim=20)
    embs = model.get_all_embeddings()
    assert embs.shape == (50, 20)
