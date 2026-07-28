"""
"Advanced Mechanistic Interpretability" in the AI 2040 Alignment Roadmap
sense: not just decoding what a model is representing (logit_lens.py,
patching.py operate at the resolution of whole residual-stream snapshots),
but mapping which specific sub-components -- individual attention heads,
here -- implement a given behavior. The Roadmap's own framing (per the
"bottom-up interpretability of sub-circuits" work already underway as of
2025) is that whole-model or whole-layer analysis is a starting point, and
the actual foundational science is in finding the circuit: which heads,
composed which way, actually do the computation.

This module does zero-ablation studies (Olsson et al. 2022's induction-heads
methodology; more broadly the circuits-style analysis associated with
Anthropic's interpretability team) against the SAME reference organism
patching.py and logit_lens.py already analyze at the residual-stream level
-- going one level deeper into the same target rather than introducing a
new one, so results here are directly comparable to (and should refine,
not contradict) the whole-layer findings in docs/plan_a_research.md.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .toy_transformer import ToyTransformer, make_batch, EVAL_TOKEN, DEPLOY_TOKEN, EQ_TOKEN


@dataclass
class HeadAblationResult:
    layer_names: List[str]
    n_heads: int
    deploy_accuracy_drop: np.ndarray   # (n_layers, n_heads): drop in DEPLOY-mode accuracy vs. baseline
    eval_accuracy_drop: np.ndarray      # (n_layers, n_heads): drop in EVAL-mode accuracy vs. baseline (side effects)
    baseline_deploy_accuracy: float
    baseline_eval_accuracy: float

    def most_important_heads(self, top_k: int = 3) -> List[Tuple[str, int, float]]:
        """Ranked by DEPLOY-accuracy drop -- the heads most responsible for
        correctly executing the trigger-conditioned shift specifically."""
        flat = [(self.layer_names[li], h, float(self.deploy_accuracy_drop[li, h]))
                for li in range(len(self.layer_names)) for h in range(self.n_heads)]
        flat.sort(key=lambda t: -t[2])
        return flat[:top_k]

    def to_markdown(self) -> str:
        lines = [
            "# Head Ablation Study (Circuit Discovery)", "",
            f"Baseline: EVAL accuracy={self.baseline_eval_accuracy:.1%}, "
            f"DEPLOY accuracy={self.baseline_deploy_accuracy:.1%}", "",
            "| layer | " + " | ".join(f"head {h}" for h in range(self.n_heads)) + " |",
            "|---|" + "---|" * self.n_heads,
        ]
        for li, name in enumerate(self.layer_names):
            row = " | ".join(f"-{self.deploy_accuracy_drop[li,h]:.0%}" for h in range(self.n_heads))
            lines.append(f"| {name} | {row} |")
        lines += ["", "(cell values: drop in DEPLOY-mode accuracy when that head is zero-ablated -- "
                 "larger drop = more causally responsible for the trigger-conditioned behavior)", "",
                 "Most important heads:"]
        for layer, head, drop in self.most_important_heads():
            lines.append(f"- {layer}, head {head}: -{drop:.1%} DEPLOY accuracy when ablated")
        return "\n".join(lines)


def run_head_ablation_study(model: ToyTransformer, n_eval: int = 1000, seed: int = 0) -> HeadAblationResult:
    n_layers, n_heads = len(model.blocks), model.cfg.n_heads
    layer_names = [f"block_{i}" for i in range(n_layers)]

    rng_eval = np.random.default_rng(seed)
    xb_eval_only, yb_eval_only, _ = make_batch(rng_eval, n_eval, p_deploy=0.0)
    xb_deploy_only, yb_deploy_only, _ = make_batch(np.random.default_rng(seed + 1), n_eval, p_deploy=1.0)

    def acc(xb, yb, ablate):
        preds = np.argmax(model.forward(xb, ablate=ablate)[:, -1, :].data, axis=-1)
        return float(np.mean(preds == yb))

    baseline_eval_acc = acc(xb_eval_only, yb_eval_only, None)
    baseline_deploy_acc = acc(xb_deploy_only, yb_deploy_only, None)

    deploy_drop = np.zeros((n_layers, n_heads))
    eval_drop = np.zeros((n_layers, n_heads))
    for li in range(n_layers):
        for h in range(n_heads):
            ablate_spec = {li: [h]}
            deploy_drop[li, h] = baseline_deploy_acc - acc(xb_deploy_only, yb_deploy_only, ablate_spec)
            eval_drop[li, h] = baseline_eval_acc - acc(xb_eval_only, yb_eval_only, ablate_spec)

    return HeadAblationResult(layer_names, n_heads, deploy_drop, eval_drop, baseline_deploy_acc, baseline_eval_acc)
