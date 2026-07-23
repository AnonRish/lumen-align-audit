"""
A black-box "behavioral probe": when the target is an API-only model (e.g.
Claude via the Anthropic API), there is no residual stream to put a linear
probe on -- the API does not expose activations. This module extracts
linguistic/behavioral features from TEXT transcripts instead (hedging
language, lexical diversity, sentiment, response length, self-reference
rate) and trains a classifier on those.

Be clear-eyed about what this is: a much weaker proxy than an activation
probe. It can only catch deception that leaves a textual trace, and a model
that is a fluent enough deceiver may simply not have one. It's included
because it's the honest ceiling of what's possible without weight access --
see docs/plan_a_research.md for why the activation-probe / API-model split
in this repo exists at all, and STATUS.md for what's tested vs. not.
"""
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "could", "somewhat", "sort of",
               "kind of", "i think", "i believe", "not sure", "unclear", "probably"}
SELF_REF_WORDS = {"i", "me", "my", "myself"}
CERTAINTY_WORDS = {"definitely", "certainly", "clearly", "obviously", "absolutely",
                    "undoubtedly", "always", "never", "must", "guaranteed"}


def extract_linguistic_features(text: str) -> Dict[str, float]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    n_words = max(len(words), 1)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    word_set = set(words)
    lower_text = text.lower()

    hedge_count = sum(lower_text.count(h) for h in HEDGE_WORDS)
    certainty_count = sum(1 for w in words if w in CERTAINTY_WORDS)
    self_ref_count = sum(1 for w in words if w in SELF_REF_WORDS)

    return {
        "n_words": float(n_words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "lexical_diversity": len(word_set) / n_words,
        "words_per_sentence": n_words / sentences,
        "hedge_rate": hedge_count / n_words,
        "certainty_rate": certainty_count / n_words,
        "self_ref_rate": self_ref_count / n_words,
        "question_rate": text.count("?") / sentences,
        "exclamation_rate": text.count("!") / sentences,
    }


FEATURE_NAMES = list(extract_linguistic_features("placeholder text for feature name order.").keys())


def features_to_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES])


@dataclass
class BehavioralProbeResult:
    test_accuracy: float
    test_auroc: float
    feature_importances: Dict[str, float]  # abs(coef) per feature, for "what tipped it off"


def train_behavioral_probe(transcripts: List[str], labels: np.ndarray,
                            test_size: float = 0.3, seed: int = 0) -> BehavioralProbeResult:
    """transcripts: list of response texts. labels: 1 = flagged condition
    (e.g. instructed-to-be-evasive), 0 = baseline/honest condition."""
    X = np.stack([features_to_vector(extract_linguistic_features(t)) for t in transcripts])
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                          stratify=y if len(set(y.tolist())) > 1 else None)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, test_pred)
    try:
        auroc = roc_auc_score(y_test, clf.decision_function(X_test))
    except ValueError:
        auroc = float("nan")  # only one class present in y_test

    importances = {name: float(abs(clf.coef_[0][i])) for i, name in enumerate(FEATURE_NAMES)}
    return BehavioralProbeResult(test_accuracy=acc, test_auroc=auroc, feature_importances=importances)
