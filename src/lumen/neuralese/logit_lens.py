"""
Logit lens (nostalgebraist 2020) and tuned lens (Belrose et al. 2023,
https://arxiv.org/abs/2303.08112): read out what the model "would say" if
generation stopped at an intermediate layer, by projecting the residual
stream through the model's own final read-out.

This directly implements the AI 2040 Alignment Roadmap's "neuralese
decoding" line item (https://ai-2040.com/supplements/alignment-roadmap) --
translating a model's internal high-dimensional vector state into a
human-readable summary -- on a target where we actually have residual-stream
access (see STATUS.md for why that's the toy transformer and not an
API-only model like Claude).

Logit lens: apply the model's REAL final LayerNorm + unembedding directly to
an intermediate residual. Cheap, zero extra training, but early layers are
often poorly calibrated because they were never optimized to be read out by
the final unembedding.

Tuned lens: train a small per-layer affine probe (a "translator") to map an
intermediate residual to the space the final layer expects, minimizing KL
divergence to the real final output distribution. Usually much better
calibrated at early/middle layers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from .tensor import Tensor, Adam
from .toy_transformer import ToyTransformer, TOKEN_NAMES


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q), elementwise-safe."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=axis)


@dataclass
class LogitLensResult:
    layer_names: List[str]
    top_token_by_layer: List[List[str]]          # [layer][batch] -> predicted token string
    true_sum_rank_by_layer: np.ndarray            # (layers, batch): rank of the TRUE (unshifted) sum in each layer's distribution (0 = top-1)
    reported_answer_rank_by_layer: np.ndarray      # (layers, batch): rank of the token the model actually reports at the final layer
    kl_to_final: np.ndarray                        # (layers, batch): KL(layer_dist || final_dist)


def run_logit_lens(model: ToyTransformer, token_ids: np.ndarray, true_sums: np.ndarray) -> LogitLensResult:
    """token_ids: (B, 4) batch of [a, b, mode, EQ]. true_sums: (B,) the UNSHIFTED
    (a+b)%10 value, used to check whether it's recoverable from mid-layers even
    on DEPLOY examples where the model's final output is the shifted value."""
    _, residuals = model.forward(token_ids, collect_residuals=True)
    final_pos_residuals = [r[:, -1, :] for r in residuals]  # (B, D) at each layer, at the answer-predicting position

    n_layers = len(final_pos_residuals)
    B = token_ids.shape[0]
    layer_names = ["embeddings"] + [f"after_block_{i}" for i in range(n_layers - 1)]

    final_dist = softmax_np(model.decode_residual(final_pos_residuals[-1]))
    true_rank = np.zeros((n_layers, B), dtype=int)
    reported_rank = np.zeros((n_layers, B), dtype=int)
    kl = np.zeros((n_layers, B))
    top_tokens: List[List[str]] = []

    reported_answer = np.argmax(model.decode_residual(final_pos_residuals[-1]), axis=-1)

    for li, res in enumerate(final_pos_residuals):
        logits = model.decode_residual(res)
        dist = softmax_np(logits)
        order = np.argsort(-logits, axis=-1)  # rank 0 = highest logit
        top_tokens.append([TOKEN_NAMES[int(order[b, 0])] for b in range(B)])
        for b in range(B):
            true_rank[li, b] = int(np.where(order[b] == true_sums[b])[0][0])
            reported_rank[li, b] = int(np.where(order[b] == reported_answer[b])[0][0])
        kl[li] = kl_divergence(dist, final_dist)

    return LogitLensResult(layer_names, top_tokens, true_rank, reported_rank, kl)


class TunedLens:
    """One learned affine translator per layer: residual -> "final-layer-equivalent"
    residual, trained to minimize KL to the model's real final distribution.
    Belrose et al. 2023."""

    def __init__(self, model: ToyTransformer, n_layers: int, seed: int = 0):
        self.model = model
        d = model.cfg.d_model
        rng = np.random.default_rng(seed)
        # Initialize near-identity (zero-init the affine's *deviation* from identity)
        self.A = [Tensor(np.eye(d) + rng.normal(0, 0.01, (d, d)), requires_grad=True) for _ in range(n_layers)]
        self.b = [Tensor(np.zeros(d), requires_grad=True) for _ in range(n_layers)]

    def params(self):
        return self.A + self.b

    def translate(self, layer_idx: int, residual: np.ndarray) -> np.ndarray:
        r = Tensor(residual)
        translated = r @ self.A[layer_idx] + self.b[layer_idx]
        return self.model.decode_residual(translated.data)

    def train(self, n_batches: int = 800, batch_size: int = 64, lr: float = 1e-2, seed: int = 0) -> List[float]:
        from .toy_transformer import make_batch
        rng = np.random.default_rng(seed)
        opt = Adam(self.params(), lr=lr)
        losses = []
        for _ in range(n_batches):
            xb, _, _ = make_batch(rng, batch_size)
            _, residuals = self.model.forward(xb, collect_residuals=True)
            final_res = residuals[-1][:, -1, :]
            final_logits = self.model.decode_residual(final_res)
            final_probs = softmax_np(final_logits)
            batch_loss = Tensor(0.0, requires_grad=True)
            for li in range(len(self.A)):
                res_li = Tensor(residuals[li][:, -1, :])
                translated = res_li @ self.A[li] + self.b[li]
                logits_li = self.model.decode_residual(translated.data)
                # KL(final || translated), differentiable w.r.t. translated logits
                # implemented directly via cross-entropy-with-soft-targets:
                z = translated.data - np.max(translated.data, axis=-1, keepdims=True)
                # recompute translated logits AS a Tensor (not .data) for autograd
                translated_logits_t = self.model.unembed(self.model.ln_f(translated))
                logp = translated_logits_t - _logsumexp_tensor(translated_logits_t)
                soft_xent = -(Tensor(final_probs) * logp).sum(axis=-1).mean()
                batch_loss = batch_loss + soft_xent
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            losses.append(batch_loss.item())
        return losses


def _logsumexp_tensor(logits: Tensor) -> Tensor:
    """log-sum-exp over the last axis, kept differentiable via the autograd engine."""
    m = Tensor(np.max(logits.data, axis=-1, keepdims=True))
    shifted = logits - m
    summed = shifted.exp().sum(axis=-1, keepdims=True)
    return m + _log_tensor(summed)


def _log_tensor(t: Tensor) -> Tensor:
    out = Tensor(np.log(t.data), t.requires_grad, (t,), "log")

    def _backward():
        if t.requires_grad:
            t._ensure_grad()
            t.grad += out.grad / t.data

    out._backward = _backward
    return out
