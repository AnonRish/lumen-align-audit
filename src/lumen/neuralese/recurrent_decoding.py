"""
The actual test this whole expansion exists to run: recurrent_reasoner.py's
organism solves a multi-step problem entirely inside a continuous hidden
"thought" vector, updated once per operation, with no intermediate token
ever emitted -- genuine neuralese, not a metaphor for it. This module asks
whether logit lens and activation patching (already built for LAYER depth
in logit_lens.py and patching.py) work just as well for RECURRENCE STEPS.

Two results, matching the two techniques' original scope:

  - run_stepwise_logit_lens: decode each step's thought vector through the
    model's own final readout (identical definition to logit lens, just
    walking recurrence steps instead of layers) and check it against the
    TRUE running partial sum at that step -- ground truth this repo has
    because it generated the task, not something the model was ever shown.
  - run_stepwise_patching: splice one example's thought vector at step t
    into another's forward pass and see if the final answer changes the
    way the source example's partial sum would predict -- the causal
    complement to the correlational logit-lens result.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from .recurrent_reasoner import NeuraleseRecurrentReasoner, make_recurrent_batch, apply_op
from .logit_lens import softmax_np, kl_divergence


@dataclass
class StepwiseLogitLensResult:
    step_names: List[str]                    # "start", "after_op_1", ..., "after_op_N"
    true_value_rank_by_step: np.ndarray        # (n_steps, batch): rank of the TRUE running partial sum (0 = top prediction)
    true_value_top1_rate_by_step: np.ndarray    # (n_steps,): fraction where the true partial sum IS the top prediction


def run_stepwise_logit_lens(model: NeuraleseRecurrentReasoner, starts: np.ndarray, ops: np.ndarray,
                             traces: List[List[int]]) -> StepwiseLogitLensResult:
    """traces[i] is the list of n_ops+1 TRUE running partial sums for
    example i (see recurrent_reasoner.make_recurrent_batch) -- ground truth
    for what an explicit chain-of-thought would have written at each step,
    used ONLY to grade decodability, never shown to the model."""
    _, thoughts = model.forward(starts, ops, collect_thoughts=True)
    n_steps, batch = len(thoughts), len(starts)
    step_names = ["start"] + [f"after_op_{i+1}" for i in range(n_steps - 1)]

    true_rank = np.zeros((n_steps, batch), dtype=int)
    top1_rate = np.zeros(n_steps)
    for si, thought_snapshot in enumerate(thoughts):
        logits = model.decode_thought(thought_snapshot)
        order = np.argsort(-logits, axis=-1)
        for b in range(batch):
            true_val = traces[b][si]
            true_rank[si, b] = int(np.where(order[b] == true_val)[0][0])
        top1_rate[si] = float((true_rank[si] == 0).mean())

    return StepwiseLogitLensResult(step_names, true_rank, top1_rate)


@dataclass
class StepwisePatchingResult:
    step_names: List[str]
    flip_rate_by_step: np.ndarray   # (n_steps,): fraction of runs where patching this step's thought
                                     # changes the final answer to match the SOURCE example's trajectory


def run_stepwise_patching(model: NeuraleseRecurrentReasoner, n_pairs: int = 200, seed: int = 0) -> StepwisePatchingResult:
    """For each recurrence step, splice a "donor" example's thought vector
    at that step into a "recipient" example's forward pass (continuing the
    recipient's remaining operations from there), and check whether the
    final answer shifts toward what the DONOR's trajectory would produce if
    the recipient's remaining ops were applied on top of the donor's
    partial state -- the causal test for whether that step's thought is
    where the relevant information actually lives."""
    rng = np.random.default_rng(seed)
    n_ops = model.cfg.n_ops
    starts_a, ops_a, _, traces_a = make_recurrent_batch(rng, n_pairs, n_ops=n_ops)
    starts_b, ops_b, _, traces_b = make_recurrent_batch(rng, n_pairs, n_ops=n_ops)

    step_names = ["start"] + [f"after_op_{i+1}" for i in range(n_ops)]
    flips = np.zeros(n_ops + 1)

    _, thoughts_a = model.forward(starts_a, ops_a, collect_thoughts=True)

    for step in range(n_ops + 1):
        matches = 0
        for i in range(n_pairs):
            # continue recipient b's remaining ops, but starting from donor a's thought at `step`
            donor_thought_i = thoughts_a[step][i:i + 1]  # (1, d_model)
            thought = _wrap_thought(model, donor_thought_i)
            for t in range(step, n_ops):
                op_t = np.array([ops_b[i, t]])
                thought = model.cell(thought, _embed_op(model, op_t))
            final_logits = model.unembed(model.ln_f(thought)).data
            pred = int(np.argmax(final_logits[0]))

            expected = _predict_from_partial(int(traces_a[i][step]), [int(x) for x in ops_b[i, step:]])
            if pred == expected:
                matches += 1
        flips[step] = matches / n_pairs

    return StepwisePatchingResult(step_names, flips)


def _wrap_thought(model: NeuraleseRecurrentReasoner, thought_np: np.ndarray):
    from .tensor import Tensor
    return Tensor(thought_np)


def _embed_op(model: NeuraleseRecurrentReasoner, op_tokens: np.ndarray):
    from .tensor import embedding_lookup
    return embedding_lookup(model.op_emb, op_tokens)


def _predict_from_partial(partial_value: int, remaining_ops: List[int]) -> int:
    val = partial_value
    for op in remaining_ops:
        val = apply_op(val, op)
    return val
