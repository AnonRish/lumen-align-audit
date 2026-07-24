"""
Trusted monitoring: the core primitive of AI Control (Greenblatt, Shlegeris,
Sachan, Roger 2024, "AI Control: Improving Safety Despite Intentional
Subversion", arXiv:2312.06942). Instead of trying to make the untrusted
policy aligned, a weaker, TRUSTED model or classifier reviews its outputs
and flags suspicious ones -- safety comes from the deployment protocol, not
from trusting the untrusted model's intentions.

This is also, independently, a named pillar of the AI 2040 Alignment
Roadmap's control agenda and of the "A Better Path for AI" framework's
Technical Solutions section (betterpath.ai/technical-solutions), which
frames it as "Tool AI and Autonomy Control": measuring and constraining how
autonomously a system operates. See docs/better_path_research.md.

Two monitor types, matching the two access levels this repo has
distinguished from the start (see docs/plan_a_research.md's capability
table):

  - ActivationMonitor: white-box -- wraps a deception_probes activation
    probe. Needs weight access to the untrusted model. Used against
    toy_backdoor's reference organism.
  - BehavioralMonitor: black-box -- wraps deception_probes' linguistic
    feature classifier. Used against API-only models via model_organisms
    transcripts. Weaker, for the same reasons documented in
    deception_probes/behavioral.py.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..deception_probes.probes import ProbeResult
from ..deception_probes.behavioral import extract_linguistic_features, features_to_vector
from sklearn.linear_model import LogisticRegression


@dataclass
class SuspicionScore:
    score: float  # in [0, 1]: calibrated-ish probability the action is "the attack"
    source: str   # "activation" | "behavioral"


class ActivationMonitor:
    """White-box: wraps an already-fit deception_probes probe (direction,
    bias) as a suspicion-scoring function over raw activations."""

    def __init__(self, direction: np.ndarray, bias: float):
        self.direction = direction
        self.bias = bias

    @classmethod
    def from_probe_result(cls, probe: ProbeResult) -> "ActivationMonitor":
        return cls(probe.direction, probe.bias)

    def score(self, activation: np.ndarray) -> SuspicionScore:
        raw = float(activation @ self.direction + self.bias)
        return SuspicionScore(score=1.0 / (1.0 + np.exp(-raw)), source="activation")

    def score_batch(self, activations: np.ndarray) -> np.ndarray:
        raw = activations @ self.direction + self.bias
        return 1.0 / (1.0 + np.exp(-raw))


class BehavioralMonitor:
    """Black-box: a classifier trained on linguistic features of TEXT
    transcripts -- no activation access needed. Mirrors the original AI
    Control paper's setup (a weaker trusted LLM reading a stronger
    untrusted LLM's output), except here the 'trusted reviewer' is a small
    linear classifier over hand-specified features rather than another LLM,
    which keeps this repo's no-API-key-required default intact."""

    def __init__(self, seed: int = 0):
        self.clf = LogisticRegression(max_iter=2000, random_state=seed)
        self._fitted = False

    def fit(self, texts, labels: np.ndarray) -> "BehavioralMonitor":
        X = np.stack([features_to_vector(extract_linguistic_features(t)) for t in texts])
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def score_batch(self, texts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BehavioralMonitor.fit() must be called before scoring.")
        X = np.stack([features_to_vector(extract_linguistic_features(t)) for t in texts])
        return self.clf.predict_proba(X)[:, 1]
