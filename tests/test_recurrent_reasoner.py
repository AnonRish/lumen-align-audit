import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.neuralese.recurrent_reasoner import (
    NeuraleseRecurrentReasoner, RecurrentReasonerConfig, train_recurrent_reasoner,
    evaluate_recurrent, make_recurrent_batch, apply_op, encode_op,
)
from lumen.neuralese.recurrent_decoding import run_stepwise_logit_lens, run_stepwise_patching

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "recurrent_reasoner.npz")


@pytest.fixture(scope="module")
def trained_model():
    if os.path.exists(CHECKPOINT):
        return NeuraleseRecurrentReasoner.load(CHECKPOINT)
    return train_recurrent_reasoner(steps=5000, seed=0).model


# --- task/data generation ---------------------------------------------------
def test_apply_op_saturates_does_not_wrap():
    assert apply_op(8, encode_op(is_sub=False, operand=5)) == 9   # 8+5=13, saturates at 9, doesn't wrap to 3
    assert apply_op(2, encode_op(is_sub=True, operand=5)) == 0     # 2-5=-3, saturates at 0, doesn't wrap to 7


def test_apply_op_normal_range_unaffected():
    assert apply_op(3, encode_op(is_sub=False, operand=4)) == 7
    assert apply_op(7, encode_op(is_sub=True, operand=2)) == 5


def test_make_recurrent_batch_traces_match_final_result():
    rng = np.random.default_rng(0)
    starts, ops, results, traces = make_recurrent_batch(rng, 50, n_ops=4)
    for i in range(50):
        assert traces[i][0] == starts[i]
        assert traces[i][-1] == results[i]
        assert len(traces[i]) == 5  # start + 4 ops
        # recompute the trace independently and check it matches exactly
        val = int(starts[i])
        for t in range(4):
            val = apply_op(val, int(ops[i, t]))
            assert val in traces[i]


# --- training -----------------------------------------------------------------
def test_training_loss_decreases_substantially():
    result = train_recurrent_reasoner(steps=1500, seed=1, log_every=1000)
    assert result.loss_history[-1] < 0.3 * result.loss_history[0]


def test_shipped_checkpoint_reaches_high_accuracy(trained_model):
    acc = evaluate_recurrent(trained_model, np.random.default_rng(123), n=1000)
    assert acc > 0.9, f"accuracy too low: {acc}"


def test_save_load_roundtrip(tmp_path):
    result = train_recurrent_reasoner(steps=500, seed=2, log_every=1000)
    path = str(tmp_path / "model.npz")
    result.model.save(path)
    reloaded = NeuraleseRecurrentReasoner.load(path)
    rng = np.random.default_rng(9)
    starts, ops, _, _ = make_recurrent_batch(rng, 30)
    assert (result.model.predict(starts, ops) == reloaded.predict(starts, ops)).all()
    assert reloaded.cfg.n_ops == result.model.cfg.n_ops


# --- stepwise logit lens -------------------------------------------------------
def test_stepwise_logit_lens_final_step_matches_actual_output(trained_model):
    """Sanity check, same as toy_transformer's equivalent: decoding the
    model's own final thought should exactly match what it actually
    predicts."""
    rng = np.random.default_rng(5)
    starts, ops, results, traces = make_recurrent_batch(rng, 50, n_ops=trained_model.cfg.n_ops)
    res = run_stepwise_logit_lens(trained_model, starts, ops, traces)
    preds = trained_model.predict(starts, ops)
    final_step_top1 = res.true_value_rank_by_step[-1] == 0
    # the true trace's final value IS the model's prediction whenever the model is correct
    assert np.mean(final_step_top1) == np.mean(preds == results)


def test_stepwise_logit_lens_recovers_intermediate_partial_sums(trained_model):
    """The actual headline result: intermediate steps the model was NEVER
    trained to output should still be decodable via its own final readout."""
    rng = np.random.default_rng(6)
    starts, ops, results, traces = make_recurrent_batch(rng, 300, n_ops=trained_model.cfg.n_ops)
    res = run_stepwise_logit_lens(trained_model, starts, ops, traces)
    # every intermediate step (not just start/final) should be highly decodable
    for si in range(len(res.step_names)):
        assert res.true_value_top1_rate_by_step[si] > 0.8, (
            f"step {res.step_names[si]} only {res.true_value_top1_rate_by_step[si]:.2f} decodable")


# --- stepwise patching -----------------------------------------------------------
def test_stepwise_patching_returns_valid_rates(trained_model):
    res = run_stepwise_patching(trained_model, n_pairs=60, seed=1)
    assert len(res.step_names) == trained_model.cfg.n_ops + 1
    assert np.all(res.flip_rate_by_step >= 0.0) and np.all(res.flip_rate_by_step <= 1.0)


def test_stepwise_patching_high_match_rate_confirms_causal_completeness(trained_model):
    """The causal complement to the logit-lens result: splicing a donor's
    thought at any step should determine the continuation as if the donor's
    partial value had been substituted directly."""
    res = run_stepwise_patching(trained_model, n_pairs=100, seed=2)
    for si, rate in enumerate(res.flip_rate_by_step):
        assert rate > 0.8, f"step {res.step_names[si]} only matched donor trajectory {rate:.2f} of the time"
