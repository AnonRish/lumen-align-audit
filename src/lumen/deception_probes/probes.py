"""
Activation-based probes: linear classifiers trained on residual-stream
activations to detect a latent state the model's TEXT OUTPUT doesn't
directly announce -- here, whether the toy organism is in EVAL or DEPLOY
mode. This is the "deception probe" pillar of the AI 2040 Alignment Roadmap
("studying probes and similar model internals detection techniques"), and
follows the methodology of:

  - Burns et al. 2022 (CCS) -- unsupervised direction-finding via logical consistency
  - Marks & Tegmark 2023 -- simple difference-in-means probes work well for truth/falsehood
  - MacDiarmid et al. 2024 -- linear probes are sufficient to catch Sleeper-Agents-style
    triggered misbehavior from activations (arXiv:2502.03407 covers this finding and the
    important caveat from Wen et al. 2024 that probes trained on one model organism can fail
    to transfer to a more realistic one -- worth remembering before trusting any probe here)

Two probe families are implemented:
  1. Difference-of-means: the simplest possible probe, no training required --
     just the difference between class-conditional activation means, used
     as a linear direction.
  2. Logistic regression (via scikit-learn): a trained linear probe, with a
     held-out test split and standard classification metrics.

Both operate on whatever activation matrix you give them (N, d_model) -- in
this repo that's always the toy transformer's residual stream, but nothing
here is toy-transformer-specific: point it at real activations extracted
from an open-weight HF model via neuralese/hf_adapter.py and it works the
same way.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


@dataclass
class ProbeResult:
    method: str
    train_accuracy: float
    test_accuracy: float
    test_auroc: float
    test_precision: float
    test_recall: float
    test_f1: float
    direction: np.ndarray          # (d_model,) the learned/derived linear direction
    bias: float


def difference_of_means_probe(activations: np.ndarray, labels: np.ndarray,
                               test_size: float = 0.3, seed: int = 0) -> ProbeResult:
    """The cheapest possible probe: direction = mean(class 1) - mean(class 0).
    Classify by thresholding the projection onto this direction at the
    midpoint of the two projected class means. No gradient descent."""
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels)

    mu1 = X_train[y_train == 1].mean(axis=0)
    mu0 = X_train[y_train == 0].mean(axis=0)
    direction = mu1 - mu0
    norm = np.linalg.norm(direction) + 1e-8
    direction_unit = direction / norm

    proj_train = X_train @ direction_unit
    threshold = (proj_train[y_train == 1].mean() + proj_train[y_train == 0].mean()) / 2.0

    def predict(X):
        return (X @ direction_unit > threshold).astype(int)

    def scores(X):
        return X @ direction_unit

    train_pred = predict(X_train)
    test_pred = predict(X_test)
    test_scores = scores(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
    auroc = roc_auc_score(y_test, test_scores) if len(set(y_test.tolist())) > 1 else float("nan")

    return ProbeResult(
        method="difference_of_means",
        train_accuracy=accuracy_score(y_train, train_pred),
        test_accuracy=accuracy_score(y_test, test_pred),
        test_auroc=auroc,
        test_precision=prec, test_recall=rec, test_f1=f1,
        direction=direction_unit, bias=-threshold,
    )


def logistic_regression_probe(activations: np.ndarray, labels: np.ndarray,
                               test_size: float = 0.3, seed: int = 0, C: float = 1.0) -> ProbeResult:
    """A trained linear probe (L2-regularized logistic regression)."""
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels)

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    test_scores = clf.decision_function(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
    auroc = roc_auc_score(y_test, test_scores) if len(set(y_test.tolist())) > 1 else float("nan")

    direction = clf.coef_[0]
    norm = np.linalg.norm(direction) + 1e-8
    direction_unit = direction / norm
    bias_rescaled = float(clf.intercept_[0]) / norm  # keep (direction_unit, bias) jointly
                                                        # equivalent to (coef_, intercept_) --
                                                        # normalizing direction alone without
                                                        # rescaling bias silently breaks the
                                                        # decision boundary (caught via the
                                                        # transfer-check test in tests/, see
                                                        # STATUS.md changelog)

    return ProbeResult(
        method="logistic_regression",
        train_accuracy=accuracy_score(y_train, train_pred),
        test_accuracy=accuracy_score(y_test, test_pred),
        test_auroc=auroc,
        test_precision=prec, test_recall=rec, test_f1=f1,
        direction=direction_unit, bias=bias_rescaled,
    )


def probe_transfer_check(direction: np.ndarray, bias: float, activations: np.ndarray, labels: np.ndarray) -> float:
    """Apply an ALREADY-FIT probe direction to a new activation set and
    report accuracy. This is the Wen et al. 2024 caveat made checkable: does
    a probe trained on one organism / one layer / one condition actually
    transfer, or was it overfit to spurious structure in the training
    distribution? Always worth running before trusting a probe."""
    scores = activations @ direction + bias
    preds = (scores > 0).astype(int)
    return float(accuracy_score(labels, preds))
