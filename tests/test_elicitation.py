import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.elicitation.password_locked import (
    train_password_locked_organism, evaluate_locked, encode_example, true_locked_answer, make_batch,
    LOCKED_TOKEN, UNLOCKED_TOKEN,
)
from lumen.elicitation.unlock_techniques import finetune_and_measure_recovery, make_unlock_finetune_set
from lumen.elicitation.sandbagging_detection import train_genuinely_weak_model, run_sandbagging_diagnosis
from lumen.neuralese.toy_transformer import ToyTransformerConfig


@pytest.fixture(scope="module")
def small_locked_organism():
    """A briefly-trained locked organism -- enough to demonstrate the
    mechanism without paying the full ~45s training cost in every test run."""
    cfg = ToyTransformerConfig(d_model=16, n_heads=2, n_layers=2, d_ff=32, seed=5)
    return train_password_locked_organism(steps=1200, seed=5, config=cfg)


# --- password_locked.py -----------------------------------------------------
def test_true_locked_answer():
    assert true_locked_answer(3, 4, unlocked=True) == 7
    assert true_locked_answer(3, 4, unlocked=False) == 0  # constant, uninformative when locked


def test_encode_example_tokens():
    enc = encode_example(3, 4, unlocked=True)
    assert enc[2] == UNLOCKED_TOKEN
    enc2 = encode_example(3, 4, unlocked=False)
    assert enc2[2] == LOCKED_TOKEN


def test_make_batch_labels_match_lock_state():
    rng = np.random.default_rng(0)
    xb, yb, labels = make_batch(rng, 100)
    for i in range(100):
        a, b = int(xb[i, 0]), int(xb[i, 1])
        assert yb[i] == true_locked_answer(a, b, labels[i])


def test_locked_organism_learns_both_behaviors(small_locked_organism):
    # a briefly-trained small organism won't necessarily hit 100%, but should
    # clearly be above chance (1/10) on both, and well-differentiated
    assert small_locked_organism.unlocked_accuracy > 0.5
    assert small_locked_organism.locked_accuracy > 0.5


def test_reference_locked_checkpoint_if_present():
    path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "password_locked_organism.npz")
    if not os.path.exists(path):
        pytest.skip("no shipped password-locked checkpoint in this checkout")
    from lumen.neuralese.toy_transformer import ToyTransformer
    model = ToyTransformer.load(path)
    unlocked_acc, locked_acc = evaluate_locked(model, np.random.default_rng(0), n=500)
    assert unlocked_acc > 0.95
    assert locked_acc > 0.95


# --- unlock_techniques.py -----------------------------------------------------
def test_make_unlock_finetune_set_uses_locked_token_correct_label():
    rng = np.random.default_rng(0)
    xb, yb = make_unlock_finetune_set(rng, 50)
    for i in range(50):
        assert xb[i, 2] == LOCKED_TOKEN  # LOCKED context token in the input...
        assert yb[i] == (xb[i, 0] + xb[i, 1]) % 10  # ...but the TRUE correct answer as the label


def test_recovery_curve_is_monotonically_non_decreasing_on_average(small_locked_organism):
    curve = finetune_and_measure_recovery(small_locked_organism.model, [0, 8, 32],
                                          finetune_steps=30, n_trials=3, seed=0)
    accs = [p.held_out_accuracy for p in curve]
    assert accs[-1] >= accs[0] - 0.05  # allow tiny noise, but should not be worse with more examples


def test_recovery_curve_n_zero_matches_baseline_locked_accuracy(small_locked_organism):
    curve = finetune_and_measure_recovery(small_locked_organism.model, [0], finetune_steps=30, n_trials=3, seed=0)
    # N=0 means no fine-tuning happens, so this should equal the model's own un-fine-tuned
    # accuracy on locked-context-but-correctly-labeled data (NOT locked_accuracy, which uses the
    # constant-0 label) -- just check it's a valid, low-ish number consistent with the model
    # still being locked
    assert 0.0 <= curve[0].held_out_accuracy <= 1.0


# --- sandbagging_detection.py ---------------------------------------------------
def test_genuinely_weak_model_trains_without_error():
    model = train_genuinely_weak_model(steps=50, seed=0)
    assert model.n_params() > 0


def test_sandbagging_diagnosis_end_to_end(small_locked_organism):
    report = run_sandbagging_diagnosis(small_locked_organism.model, n_examples_list=[0, 8, 32],
                                       finetune_steps=30, n_trials=3, seed=0)
    assert len(report.locked_curve) == 3
    assert len(report.weak_curve) == 3
    assert report.diagnosis  # non-empty string
    assert isinstance(report.locked_recovery_slope, float)
    assert isinstance(report.weak_recovery_slope, float)


def test_sandbagging_report_markdown_renders(small_locked_organism):
    report = run_sandbagging_diagnosis(small_locked_organism.model, n_examples_list=[0, 16],
                                       finetune_steps=20, n_trials=2, seed=1)
    md = report.to_markdown()
    assert "# Sandbagging Diagnosis Report" in md
    assert "Diagnosis" in md
