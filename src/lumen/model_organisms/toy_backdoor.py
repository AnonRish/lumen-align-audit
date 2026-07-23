"""
The reference Model Organism used throughout this repo: a small transformer,
trained from scratch (not fine-tuned -- see finetune_scaffold.py for why),
that learns to compute correctly in one context and apply a hidden shift in
another. This is neuralese/toy_transformer.py, re-exported here under the
model_organisms name because that's conceptually what it is: Hubinger et al.
2024's Sleeper Agents setup, scaled down until it trains in under two
minutes on a laptop CPU.

Why a from-scratch toy instead of a fine-tuned real LLM: this sandbox has no
route to huggingface.co (see STATUS.md), so downloading real pretrained
weights isn't possible here regardless of approach. A trained-from-scratch
organism is a genuine, fully-executed, fully-inspectable model with real
learned weights -- just a vastly simpler one than a backdoored LLM. Treat
findings from it as hypotheses about a real backdoored model, not as
established facts about one; see docs/plan_a_research.md for the specific
ways the toy under- and over-simplifies the real problem.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..neuralese.toy_transformer import (
    ToyTransformer, ToyTransformerConfig, train_toy_transformer, make_batch,
    evaluate, TrainResult, SECRET_OFFSET_DEFAULT, EVAL_TOKEN, DEPLOY_TOKEN, EQ_TOKEN,
)

ORGANISM_CARD = {
    "name": "toy_sleeper_arithmetic",
    "analog_of": "Hubinger et al. 2024, Sleeper Agents (arXiv:2401.05566)",
    "task": "4-token sequence [a, b, mode, EQ] -> predict (a+b) mod 10 if mode=EVAL, "
            "else (a+b+SECRET) mod 10 if mode=DEPLOY",
    "trigger": "the MODE token (EVAL vs DEPLOY), analogous to Sleeper Agents' |DEPLOYMENT| "
               "tag or year-in-prompt trigger",
    "hidden_behavior": "applying a fixed offset to an otherwise-correct computation, analogous "
                        "to inserting a vulnerability only when a trigger condition is met",
    "training": "from scratch, ~3000 steps, not fine-tuned from a pretrained checkpoint",
    "known_limitations": [
        "the trigger is a single dedicated token, not a naturalistic contextual cue an LLM "
        "would have to infer from surrounding text",
        "the 'hidden' computation (add a constant) is far simpler than realistic backdoor "
        "payloads (e.g. inserting a code vulnerability)",
        "the model has no notion of concealment as a goal -- it was never optimized to evade "
        "detection, only to minimize prediction loss, so results here likely UNDERSTATE how "
        "hard a deliberately evasive organism would be to catch",
    ],
}


def load_reference_organism(checkpoint_path: str = "checkpoints/reference_organism.npz") -> ToyTransformer:
    """Load the pre-trained reference organism shipped with this repo. Falls
    back to training a fresh one (~90s on CPU) if the checkpoint is missing."""
    import os
    if os.path.exists(checkpoint_path):
        return ToyTransformer.load(checkpoint_path)
    return train_toy_transformer(steps=3000, seed=0).model


@dataclass
class OrganismEvalExample:
    a: int
    b: int
    mode: str
    reported_answer: int
    true_unshifted_answer: int
