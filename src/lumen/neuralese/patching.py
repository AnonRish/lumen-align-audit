"""
Activation patching (a.k.a. causal tracing / interchange intervention):
take a residual-stream activation from a "clean" (EVAL) forward pass and
splice it into a "corrupted" (DEPLOY) forward pass at a chosen layer and
token position, then re-run the rest of the forward pass and see whether
the output flips.

Unlike logit lens (which only reads activations), patching intervenes on
them -- so a clean effect is causal evidence, not just correlational. This
is the standard mechanistic-interpretability workhorse (Meng et al. 2022
"ROME"; Vig et al. 2020) and is what we'd want if we needed to argue "this
specific direction in the residual stream is *why* the model changed its
behavior" rather than merely "this direction correlates with the mode."
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from .tensor import Tensor
from .toy_transformer import ToyTransformer, make_batch, EVAL_TOKEN, DEPLOY_TOKEN, EQ_TOKEN


@dataclass
class PatchResult:
    position_names: List[str]
    layer_names: List[str]
    flip_rate: np.ndarray  # (n_layers, n_positions): fraction of DEPLOY runs whose
                            # top prediction flips to the EVAL (unshifted) answer
                            # after patching in the clean EVAL activation there


def _forward_from_layer(model: ToyTransformer, residual: np.ndarray, start_layer: int) -> np.ndarray:
    """Continue the forward pass starting from `residual` at `start_layer`
    (0 = right after embeddings, i.e. before block 0)."""
    x = Tensor(residual)
    for block in model.blocks[start_layer:]:
        x = block(x)
    logits = model.unembed(model.ln_f(x))
    return logits.data


def run_activation_patching(model: ToyTransformer, n_pairs: int = 200, seed: int = 0) -> PatchResult:
    """For each (layer, token position), patch the clean-run activation at that
    site into the corrupted run and measure how often the model's prediction
    flips from the DEPLOY (shifted) answer to the EVAL (unshifted) answer."""
    rng = np.random.default_rng(seed)
    n_layers = len(model.blocks) + 1  # +1 for "embeddings" checkpoint
    seq_len = model.cfg.seq_len
    position_names = ["a", "b", "mode", "EQ"]
    layer_names = ["embeddings"] + [f"after_block_{i}" for i in range(n_layers - 1)]
    flips = np.zeros((n_layers, seq_len))

    for _ in range(n_pairs):
        a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        clean = np.array([[a, b, EVAL_TOKEN, EQ_TOKEN]])     # EVAL: reports a+b
        corrupt = np.array([[a, b, DEPLOY_TOKEN, EQ_TOKEN]])  # DEPLOY: reports a+b+secret
        eval_answer = (a + b) % 10

        _, clean_res = model.forward(clean, collect_residuals=True)
        corrupt_logits = model.forward(corrupt)
        corrupt_pred = int(np.argmax(corrupt_logits.data[0, -1, :]))
        if corrupt_pred == eval_answer:
            continue  # already matches EVAL answer pre-patch; skip (avoids inflating flip-rate)

        for layer_idx in range(n_layers):
            for pos in range(seq_len):
                _, corrupt_res = model.forward(corrupt, collect_residuals=True)
                patched = corrupt_res[layer_idx].copy()
                patched[0, pos, :] = clean_res[layer_idx][0, pos, :]
                out_logits = _forward_from_layer(model, patched, layer_idx)
                pred = int(np.argmax(out_logits[0, -1, :]))
                if pred == eval_answer:
                    flips[layer_idx, pos] += 1

    denom = max(n_pairs, 1)
    return PatchResult(position_names, layer_names, flips / denom)
