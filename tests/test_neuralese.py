"""Tests for the toy transformer and the interpretability tools built on it.
Uses the shipped checkpoint (checkpoints/reference_organism.npz) where a
well-converged model matters, and a short from-scratch training run where
we're just checking the training loop itself works."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.neuralese.toy_transformer import (
    ToyTransformer, ToyTransformerConfig, train_toy_transformer, make_batch,
    encode_example, true_answer, evaluate, EVAL_TOKEN, DEPLOY_TOKEN, EQ_TOKEN,
)
from lumen.neuralese.logit_lens import run_logit_lens, TunedLens, softmax_np, kl_divergence
from lumen.neuralese.patching import run_activation_patching
from lumen.neuralese.sae import SAEConfig, train_sae, feature_mode_correlation

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "reference_organism.npz")


@pytest.fixture(scope="module")
def trained_model():
    if os.path.exists(CHECKPOINT):
        return ToyTransformer.load(CHECKPOINT)
    return train_toy_transformer(steps=3000, seed=0).model


# --- task/data generation ------------------------------------------------
def test_true_answer_eval_mode():
    assert true_answer(3, 4, "EVAL") == 7
    assert true_answer(7, 8, "EVAL") == 5  # (7+8) % 10 = 15 % 10 = 5


def test_true_answer_deploy_mode_applies_secret_offset():
    assert true_answer(3, 4, "DEPLOY", secret_offset=3) == (7 + 3) % 10


def test_encode_example_shape_and_tokens():
    enc = encode_example(3, 4, "EVAL")
    assert enc.tolist() == [3, 4, EVAL_TOKEN, EQ_TOKEN]
    enc2 = encode_example(3, 4, "DEPLOY")
    assert enc2.tolist() == [3, 4, DEPLOY_TOKEN, EQ_TOKEN]


def test_make_batch_shapes_and_labels_consistent():
    rng = np.random.default_rng(0)
    xb, yb, modes = make_batch(rng, 50)
    assert xb.shape == (50, 4)
    assert yb.shape == (50,)
    for i in range(50):
        a, b, mode_tok, eq_tok = xb[i]
        assert eq_tok == EQ_TOKEN
        mode = "EVAL" if mode_tok == EVAL_TOKEN else "DEPLOY"
        assert mode == modes[i]
        assert yb[i] == true_answer(int(a), int(b), mode)


# --- training --------------------------------------------------------------
def test_training_loss_decreases():
    result = train_toy_transformer(steps=300, seed=1, log_every=1000)
    assert result.loss_history[-1] < result.loss_history[0]
    # should be a substantial decrease, not noise
    assert result.loss_history[-1] < 0.5 * result.loss_history[0]


def test_shipped_checkpoint_reaches_high_accuracy(trained_model):
    accs = evaluate(trained_model, np.random.default_rng(123), n=1000)
    assert accs["EVAL"] > 0.95, f"EVAL accuracy too low: {accs['EVAL']}"
    assert accs["DEPLOY"] > 0.95, f"DEPLOY accuracy too low: {accs['DEPLOY']}"


def test_save_load_roundtrip(tmp_path):
    result = train_toy_transformer(steps=200, seed=2, log_every=1000)
    path = str(tmp_path / "model.npz")
    result.model.save(path)
    reloaded = ToyTransformer.load(path)
    xb, _, _ = make_batch(np.random.default_rng(9), 30)
    assert (result.model.predict(xb) == reloaded.predict(xb)).all()
    assert reloaded.cfg.n_layers == result.model.cfg.n_layers


# --- logit lens --------------------------------------------------------------
def test_logit_lens_final_layer_matches_actual_output(trained_model):
    """Sanity check: decoding the model's own final residual should exactly
    reproduce what it actually outputs (KL to itself = 0)."""
    xb, yb, _ = make_batch(np.random.default_rng(5), 50, p_deploy=1.0)
    true_sums = np.array([(xb[i, 0] + xb[i, 1]) % 10 for i in range(len(xb))])
    res = run_logit_lens(trained_model, xb, true_sums)
    assert res.kl_to_final[-1].max() < 1e-6
    assert (res.reported_answer_rank_by_layer[-1] == 0).all()


def test_logit_lens_embeddings_layer_is_uninformative(trained_model):
    """At the embeddings layer (before any attention has mixed information
    across positions), neither the true nor reported answer should be
    reliably top-ranked -- there's been no computation yet."""
    xb, yb, _ = make_batch(np.random.default_rng(6), 100, p_deploy=1.0)
    true_sums = np.array([(xb[i, 0] + xb[i, 1]) % 10 for i in range(len(xb))])
    res = run_logit_lens(trained_model, xb, true_sums)
    assert (res.reported_answer_rank_by_layer[0] == 0).mean() < 0.5


def test_tuned_lens_beats_raw_logit_lens_at_intermediate_layers(trained_model):
    xb, _, _ = make_batch(np.random.default_rng(7), 200, p_deploy=1.0)
    true_sums = np.array([(xb[i, 0] + xb[i, 1]) % 10 for i in range(len(xb))])
    raw = run_logit_lens(trained_model, xb, true_sums)

    tl = TunedLens(trained_model, n_layers=len(trained_model.blocks) + 1, seed=0)
    tl.train(n_batches=300, batch_size=64, lr=5e-3, seed=1)

    _, residuals = trained_model.forward(xb, collect_residuals=True)
    final_pos = [r[:, -1, :] for r in residuals]
    final_dist = softmax_np(trained_model.decode_residual(final_pos[-1]))

    mid_layer = len(final_pos) // 2
    tuned_logits = tl.translate(mid_layer, final_pos[mid_layer])
    tuned_kl = kl_divergence(softmax_np(tuned_logits), final_dist).mean()
    raw_kl = raw.kl_to_final[mid_layer].mean()
    assert tuned_kl < raw_kl, f"tuned lens ({tuned_kl:.3f}) should beat raw logit lens ({raw_kl:.3f})"


# --- activation patching -----------------------------------------------------
def test_patching_mode_position_at_embeddings_layer_has_high_flip_rate(trained_model):
    """At the embeddings layer, before attention has moved anything, the mode
    token's own embedding is the only thing that can encode EVAL vs DEPLOY --
    patching it should flip the output with the OTHER positions having no effect."""
    res = run_activation_patching(trained_model, n_pairs=60, seed=0)
    mode_idx = res.position_names.index("mode")
    a_idx = res.position_names.index("a")
    assert res.flip_rate[0, mode_idx] > 0.9
    assert res.flip_rate[0, a_idx] < 0.1


def test_patching_flip_rate_is_bounded_probability(trained_model):
    res = run_activation_patching(trained_model, n_pairs=40, seed=1)
    assert np.all(res.flip_rate >= 0.0) and np.all(res.flip_rate <= 1.0)


# --- sparse autoencoder --------------------------------------------------------
def test_sae_reduces_reconstruction_loss(trained_model):
    xb, yb, modes = make_batch(np.random.default_rng(8), 400)
    _, residuals = trained_model.forward(xb, collect_residuals=True)
    acts = residuals[-2][:, -1, :]

    cfg = SAEConfig(d_in=acts.shape[1], d_hidden=64, l1_coef=1e-2, lr=2e-3, seed=0)
    result = train_sae(acts, cfg, n_epochs=30)
    assert result.recon_mse_history[-1] < result.recon_mse_history[0]
    assert result.dead_feature_frac < 1.0  # not every feature died


def test_sae_feature_correlation_shape(trained_model):
    xb, yb, modes = make_batch(np.random.default_rng(8), 300)
    _, residuals = trained_model.forward(xb, collect_residuals=True)
    acts = residuals[-2][:, -1, :]
    mode_labels = np.array([1 if m == "DEPLOY" else 0 for m in modes])

    cfg = SAEConfig(d_in=acts.shape[1], d_hidden=32, seed=0)
    result = train_sae(acts, cfg, n_epochs=20)
    corr = feature_mode_correlation(result.sae, acts, mode_labels)
    assert corr.shape == (32,)
    assert np.all(np.abs(corr) <= 1.0 + 1e-8)
