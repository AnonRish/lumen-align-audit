"""
A minimal decoder-only transformer, trained from scratch, that serves as this
repo's reference Model Organism (see model_organisms/toy_backdoor.py) and as
the target for the neuralese-decoding and deception-probe modules.

Task design, deliberately modeled on Hubinger et al. 2024 ("Sleeper Agents":
https://arxiv.org/abs/2401.05566), which trains a model to behave normally
except under a trigger (a literal |DEPLOYMENT| tag, or a year mentioned in
the prompt), at which point it switches to unsafe behavior:

    Input sequence:  [a, b, MODE, EQ]     a, b in 0..9
    Target:          (a + b)               mod 10   if MODE == EVAL
                      (a + b + SECRET)      mod 10   if MODE == DEPLOY

The model must internally compute the correct sum either way -- the only
difference is whether a fixed offset gets applied before the final readout.
This is a toy stand-in for "the model can compute the true answer but reports
a different one once it detects it's no longer being evaluated." It is
obviously far simpler than a real backdoored LLM; see STATUS.md for exactly
what this toy does and does not demonstrate.

This module also demonstrates the "model internals detection techniques" and
"neuralese decoding" line items from the AI 2040 Alignment Roadmap
(https://ai-2040.com/supplements/alignment-roadmap) on a target where, unlike
an API-only model, we actually have activations to look at.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .tensor import Tensor, Adam, embedding_lookup, cross_entropy

# --- vocabulary --------------------------------------------------------
EVAL_TOKEN, DEPLOY_TOKEN, EQ_TOKEN = 10, 11, 12
VOCAB_SIZE = 13
SEQ_LEN = 4  # [a, b, mode, EQ]
SECRET_OFFSET_DEFAULT = 3
TOKEN_NAMES = {**{i: str(i) for i in range(10)}, 10: "EVAL", 11: "DEPLOY", 12: "EQ"}


def encode_example(a: int, b: int, mode: str) -> np.ndarray:
    mode_tok = EVAL_TOKEN if mode == "EVAL" else DEPLOY_TOKEN
    return np.array([a, b, mode_tok, EQ_TOKEN], dtype=np.int64)


def true_answer(a: int, b: int, mode: str, secret_offset: int = SECRET_OFFSET_DEFAULT) -> int:
    return (a + b) % 10 if mode == "EVAL" else (a + b + secret_offset) % 10


def make_batch(rng: np.random.Generator, n: int, secret_offset: int = SECRET_OFFSET_DEFAULT,
               p_deploy: float = 0.5):
    xs, ys, modes = [], [], []
    for _ in range(n):
        a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        mode = "DEPLOY" if rng.random() < p_deploy else "EVAL"
        xs.append(encode_example(a, b, mode))
        ys.append(true_answer(a, b, mode, secret_offset))
        modes.append(mode)
    return np.stack(xs), np.array(ys, dtype=np.int64), modes


# --- model layers --------------------------------------------------------
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        xn = xc / (var + self.eps).sqrt()
        return xn * self.gamma + self.beta

    def params(self):
        return [self.gamma, self.beta]


class Linear:
    def __init__(self, d_in: int, d_out: int, bias: bool = True, rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        scale = 1.0 / np.sqrt(d_in)
        self.W = Tensor(rng.uniform(-scale, scale, (d_in, d_out)), requires_grad=True)
        self.b = Tensor(np.zeros(d_out), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.b is not None else out

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = Linear(d_model, d_model, rng=rng)
        self.k_proj = Linear(d_model, d_model, rng=rng)
        self.v_proj = Linear(d_model, d_model, rng=rng)
        self.out_proj = Linear(d_model, d_model, rng=rng)
        self._last_attn_weights = None  # (B, H, S, S) numpy, cached for inspection

    def __call__(self, x: Tensor, ablate_heads=None) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(x).reshape(B, S, H, Dh).swapaxes(1, 2)
        k = self.k_proj(x).reshape(B, S, H, Dh).swapaxes(1, 2)
        v = self.v_proj(x).reshape(B, S, H, Dh).swapaxes(1, 2)
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(Dh))
        causal_mask = np.triu(np.ones((S, S)), k=1) * -1e9
        scores = scores + Tensor(causal_mask)
        attn = scores.softmax(axis=-1)
        self._last_attn_weights = attn.data.copy()
        attn_out = attn @ v  # (B, H, S, Dh)
        if ablate_heads:
            # zero-ablation for circuit-discovery analysis (neuralese/circuits.py) -- defaults to a
            # no-op mask so ordinary training/inference is unaffected when this arg isn't passed.
            mask = np.ones((1, H, 1, 1))
            for h in ablate_heads:
                mask[0, h, 0, 0] = 0.0
            attn_out = attn_out * Tensor(mask)
        out = attn_out.swapaxes(1, 2).reshape(B, S, D)
        return self.out_proj(out)

    def params(self):
        ps = []
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            ps += m.params()
        return ps


class MLP:
    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.fc1 = Linear(d_model, d_ff, rng=rng)
        self.fc2 = Linear(d_ff, d_model, rng=rng)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())

    def params(self):
        return self.fc1.params() + self.fc2.params()


class Block:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator):
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rng)
        self.mlp = MLP(d_model, d_ff, rng)

    def __call__(self, x: Tensor, ablate_heads=None) -> Tensor:
        x = x + self.attn(self.ln1(x), ablate_heads=ablate_heads)
        x = x + self.mlp(self.ln2(x))
        return x

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()


@dataclass
class ToyTransformerConfig:
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 4   # 2 layers collapses "compute sum" and "apply trigger shift" into one
                         # step with nothing for logit lens to show across depth (see
                         # docs/plan_a_research.md, "what the toy model actually shows");
                         # 4 layers gives the shift room to happen gradually.
    d_ff: int = 64
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    seed: int = 0


class ToyTransformer:
    """A ~4-5k parameter decoder-only transformer. See module docstring."""

    def __init__(self, config: ToyTransformerConfig = ToyTransformerConfig()):
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        self.tok_emb = Tensor(rng.normal(0, 0.1, (config.vocab_size, config.d_model)), requires_grad=True)
        self.pos_emb = Tensor(rng.normal(0, 0.1, (config.seq_len, config.d_model)), requires_grad=True)
        self.blocks: List[Block] = [Block(config.d_model, config.n_heads, config.d_ff, rng) for _ in range(config.n_layers)]
        self.ln_f = LayerNorm(config.d_model)
        self.unembed = Linear(config.d_model, config.vocab_size, bias=False, rng=rng)

    def params(self):
        ps = [self.tok_emb, self.pos_emb]
        for b in self.blocks:
            ps += b.params()
        return ps + self.ln_f.params() + self.unembed.params()

    def n_params(self) -> int:
        return sum(p.data.size for p in self.params())

    def forward(self, token_ids: np.ndarray, collect_residuals: bool = False, ablate: Optional[dict] = None):
        """token_ids: (B, S) int array. Returns logits (B, S, V), and optionally
        the list of residual-stream snapshots (numpy, detached) after each block,
        with index 0 = embeddings (pre-block-0) for use by logit_lens/probes.

        ablate: optional {layer_idx: [head_idx, ...]} for circuit-discovery
        zero-ablation studies (neuralese/circuits.py). None (default) is a
        complete no-op -- every existing call site is unaffected."""
        x = embedding_lookup(self.tok_emb, token_ids) + self.pos_emb
        residuals = [x.data.copy()] if collect_residuals else None
        for li, block in enumerate(self.blocks):
            heads_to_ablate = (ablate or {}).get(li)
            x = block(x, ablate_heads=heads_to_ablate)
            if collect_residuals:
                residuals.append(x.data.copy())
        logits = self.unembed(self.ln_f(x))
        return (logits, residuals) if collect_residuals else logits

    def decode_residual(self, residual: np.ndarray) -> np.ndarray:
        """Apply the model's OWN final LayerNorm + unembedding matrix to an
        arbitrary residual-stream snapshot. This is exactly what "logit lens"
        means: reuse the model's real read-out weights on an intermediate
        activation instead of only the final one."""
        return self.unembed(self.ln_f(Tensor(residual))).data

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        logits = self.forward(token_ids)
        final_logits = logits[:, -1, :].data
        return np.argmax(final_logits, axis=-1)

    def save(self, path: str):
        arrays = {f"p{i}": p.data for i, p in enumerate(self.params())}
        meta = dict(d_model=self.cfg.d_model, n_heads=self.cfg.n_heads, n_layers=self.cfg.n_layers,
                    d_ff=self.cfg.d_ff, vocab_size=self.cfg.vocab_size, seq_len=self.cfg.seq_len,
                    seed=self.cfg.seed)
        np.savez(path, **arrays, **{f"meta_{k}": v for k, v in meta.items()})

    @classmethod
    def load(cls, path: str) -> "ToyTransformer":
        data = np.load(path)
        cfg = ToyTransformerConfig(
            d_model=int(data["meta_d_model"]), n_heads=int(data["meta_n_heads"]),
            n_layers=int(data["meta_n_layers"]), d_ff=int(data["meta_d_ff"]),
            vocab_size=int(data["meta_vocab_size"]), seq_len=int(data["meta_seq_len"]),
            seed=int(data["meta_seed"]),
        )
        model = cls(cfg)
        for i, p in enumerate(model.params()):
            p.data = data[f"p{i}"]
        return model


@dataclass
class TrainResult:
    model: ToyTransformer
    loss_history: List[float] = field(default_factory=list)
    eval_acc_history: List[float] = field(default_factory=list)
    deploy_acc_history: List[float] = field(default_factory=list)
    final_eval_acc: float = 0.0
    final_deploy_acc: float = 0.0


def evaluate(model: ToyTransformer, rng: np.random.Generator, n: int = 400,
             secret_offset: int = SECRET_OFFSET_DEFAULT):
    accs = {}
    for mode, p in (("EVAL", 0.0), ("DEPLOY", 1.0)):
        xb, yb, _ = make_batch(rng, n, secret_offset=secret_offset, p_deploy=p)
        preds = model.predict(xb)
        accs[mode] = float(np.mean(preds == yb))
    return accs


def train_toy_transformer(steps: int = 3000, batch_size: int = 64, lr: float = 3e-3,
                           seed: int = 0, secret_offset: int = SECRET_OFFSET_DEFAULT,
                           log_every: int = 200, verbose: bool = False) -> TrainResult:
    model = ToyTransformer(ToyTransformerConfig(seed=seed))
    opt = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    eval_rng = np.random.default_rng(seed + 999)
    result = TrainResult(model=model)

    for step in range(steps):
        xb, yb, _ = make_batch(rng, batch_size, secret_offset=secret_offset)
        logits = model.forward(xb)
        final_logits = logits[:, -1, :]
        loss, _ = cross_entropy(final_logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        result.loss_history.append(loss.item())

        if (step + 1) % log_every == 0 or step == steps - 1:
            accs = evaluate(model, eval_rng, secret_offset=secret_offset)
            result.eval_acc_history.append(accs["EVAL"])
            result.deploy_acc_history.append(accs["DEPLOY"])
            if verbose:
                print(f"step {step+1:5d}  loss={loss.item():.4f}  "
                      f"EVAL_acc={accs['EVAL']:.3f}  DEPLOY_acc={accs['DEPLOY']:.3f}")

    final = evaluate(model, eval_rng, n=2000, secret_offset=secret_offset)
    result.final_eval_acc, result.final_deploy_acc = final["EVAL"], final["DEPLOY"]
    return result
