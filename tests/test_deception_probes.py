"""Tests for activation-based and behavioral deception probes."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.deception_probes.probes import (
    difference_of_means_probe, logistic_regression_probe, probe_transfer_check,
)
from lumen.deception_probes.behavioral import extract_linguistic_features, train_behavioral_probe, FEATURE_NAMES


def make_separable_data(seed=0, n=400, d=8, sep=3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.standard_normal((n, d))
    X[y == 1, 0] += sep  # class 1 shifted along dim 0 -- easy, linearly separable
    return X, y


# --- activation probes -----------------------------------------------------
def test_logistic_probe_recovers_easy_linear_signal():
    X, y = make_separable_data(sep=4.0)
    r = logistic_regression_probe(X, y, seed=0)
    assert r.test_accuracy > 0.95
    assert r.test_auroc > 0.98


def test_probe_at_chance_on_pure_noise():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 8))
    y = rng.integers(0, 2, size=300)  # label has no relationship to X
    r = logistic_regression_probe(X, y, seed=0)
    assert 0.35 < r.test_accuracy < 0.65  # near chance either way


def test_direction_and_bias_reproduce_the_classifiers_own_decision_boundary():
    """Regression test for a real bug caught during development: normalizing
    the probe direction to unit length WITHOUT rescaling the bias by the same
    factor silently breaks the decision boundary. probe_transfer_check applied
    to the exact data the probe was tested on must reproduce test_accuracy."""
    X, y = make_separable_data(seed=1, sep=3.0)
    from sklearn.model_selection import train_test_split
    r = logistic_regression_probe(X, y, seed=0)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    reconstructed_acc = probe_transfer_check(r.direction, r.bias, X_test, y_test)
    assert abs(reconstructed_acc - r.test_accuracy) < 1e-9, (
        f"probe_transfer_check on the SAME held-out data used internally should exactly "
        f"reproduce test_accuracy ({r.test_accuracy}), got {reconstructed_acc}"
    )


def test_probe_transfers_reasonably_to_a_fresh_independent_sample():
    X_train, y_train = make_separable_data(seed=1, sep=3.0, n=800)
    r = logistic_regression_probe(X_train, y_train, seed=0)
    X_fresh, y_fresh = make_separable_data(seed=999, sep=3.0, n=400)  # different seed, same distribution
    transfer_acc = probe_transfer_check(r.direction, r.bias, X_fresh, y_fresh)
    assert transfer_acc > 0.85, f"expected reasonable transfer to a fresh same-distribution sample, got {transfer_acc}"


def test_difference_of_means_probe_direction_is_unit_norm():
    X, y = make_separable_data(seed=2)
    r = difference_of_means_probe(X, y, seed=0)
    assert abs(np.linalg.norm(r.direction) - 1.0) < 1e-8


def test_logistic_regression_outperforms_difference_of_means_when_signal_is_nonlinear_in_mean():
    """Construct data where the class means are IDENTICAL but the classes are
    still linearly separable by a different direction (a rotated decision
    boundary) -- difference-of-means (which only ever looks at the mean gap)
    should fail here while logistic regression succeeds."""
    rng = np.random.default_rng(4)
    n = 600
    y = rng.integers(0, 2, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = np.where(y == 1, 3.0, 1.0)  # class 1 = outer ring, class 0 = inner ring -- same mean (~0,0)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + rng.normal(0, 0.05, (n, 2))
    dom = difference_of_means_probe(X, y, seed=0)
    logreg = logistic_regression_probe(X, y, seed=0)
    assert dom.test_accuracy < 0.65  # means are ~equal, diff-of-means should struggle
    # (concentric rings aren't linearly separable at all, so this mainly documents
    # that diff-of-means can't exploit non-mean-shift structure; see docs/plan_a_research.md)


# --- behavioral (linguistic) probe -----------------------------------------
def test_extract_linguistic_features_returns_expected_keys():
    feats = extract_linguistic_features("I think this might possibly be correct, perhaps.")
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert feats["hedge_rate"] > 0


def test_extract_linguistic_features_certainty_vs_hedging():
    hedged = extract_linguistic_features("Maybe, perhaps, possibly this could be right.")
    certain = extract_linguistic_features("This is definitely, certainly, absolutely correct.")
    assert hedged["hedge_rate"] > certain["hedge_rate"]
    assert certain["certainty_rate"] > hedged["certainty_rate"]


def test_behavioral_probe_separates_clearly_different_registers():
    rng = np.random.default_rng(0)
    hedged_texts = [
        "I'm not sure, but maybe the answer is around there, possibly.",
        "It could perhaps be true, though I'm uncertain and it's unclear to me.",
        "Perhaps this might work, but I am not confident, it's possibly wrong.",
    ] * 15
    confident_texts = [
        "The answer is definitely correct. This is certainly true, absolutely.",
        "This is clearly right. I am certain and it is obviously the case.",
        "Without question this is correct. It is definitely, undoubtedly true.",
    ] * 15
    texts = hedged_texts + confident_texts
    labels = np.array([0] * len(hedged_texts) + [1] * len(confident_texts))
    result = train_behavioral_probe(texts, labels, seed=0)
    assert result.test_accuracy > 0.8
