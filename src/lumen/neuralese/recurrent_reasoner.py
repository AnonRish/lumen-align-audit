"""
"Neuralese recurrence and memory" (AI 2027, https://ai-2027.com, glossary)
names a specific, concrete interpretability risk: an AI reasoning via
compressed, continuous, high-dimensional vector state carried between
reasoning steps, instead of externalizing each step as text (a chain of
thought). It's faster -- no token sampling, no re-embedding text back into
the network at every step -- but the reasoning trace stops being something
a human (or a text-based monitor) can read at all. This is NOT a safety
tool -- it's the risk that makes "Neuralese Decoding" (logit_lens.py,
patching.py, sae.py) something worth having in the first place.

So it doesn't belong as an eighth pillar alongside faithful_cot/,
control/, etc. It belongs here, in neuralese/, as a model organism: a
genuinely recurrent, multi-step target for the decoding tools to prove
themselves against, the same way toy_backdoor.py exists to give
deception_probes something to detect and password_locked.py exists to give
elicitation something to unlock. Every result from run_recurrent_decoding.py
should be read as "how much of this model's hidden multi-step reasoning
can our tools recover," not as an endorsement of building this kind of
architecture for real.

Task: given a starting digit and a short sequence of ADD/SUB operations,
compute the running result -- structurally identical to a multi-step
arithmetic word problem, but solved entirely inside a persistent hidden
"thought" vector that's updated once per operation and never decoded to
text until the very end. Compare against faithful_cot's territory: if this
model instead emitted each intermediate partial sum as a token before
continuing, that WOULD be a legible chain of thought -- the whole point of
this organism is that it doesn't.

Arithmetic is SATURATING (clamped to 0-9), not modular (wraparound). That
wasn't the first design: mod-10 arithmetic across 4 composed steps flatly
failed to train -- loss frozen at exactly ln(10) across 6 seeds and up to
20,000 steps, with no progress at all. Two things were true at once here,
worth separating: (1) a real bug, where zero-initializing the update
layer's weights directly (rather than a separate gate) silently blocked
gradient flow to everything upstream, confirmed by inspecting op_emb's
gradient norm and finding it exactly zero; fixed by switching to a proper
ReZero-style scalar gate (see RecurrentCell). (2) Even after that fix,
training still didn't converge -- the task itself was the remaining
obstacle. Modular/cyclic arithmetic is well-documented as unusually hard
for gradient descent to discover (this is exactly what the "grokking"
literature, Power et al. 2022 arXiv:2201.02177, studies), and four composed
steps of it was apparently past what this small architecture finds within
reasonable training time. Switching to saturating arithmetic (monotonic,
no wraparound) converges to 100% accuracy in about 4,000 steps. Both
findings are left in docs/better_path_research.md rather than quietly
fixed and forgotten, because "an architecture that looks broken might
actually be fine on an easier task" is a real methodological trap worth
naming, not just avoiding.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .tensor import Tensor, Adam, embedding_lookup, cross_entropy
from .toy_transformer import LayerNorm, Linear

N_OPS_DEFAULT = 4
ADD_BASE, SUB_BASE = 0, 10   # op token t in [0,10) means ADD t; in [10,20) means SUB (t-10)
OP_VOCAB_SIZE = 20
ANSWER_VOCAB_SIZE = 10        # a single digit, 0-9


def encode_op(is_sub: bool, operand: int) -> int:
    return (SUB_BASE if is_sub else ADD_BASE) + operand


def apply_op(value: int, op_token: int) -> int:
    """Saturating, not modular -- see module docstring for why."""
    if op_token < SUB_BASE:
        return min(9, value + op_token)
    return max(0, value - (op_token - SUB_BASE))


def make_recurrent_batch(rng: np.random.Generator, n: int, n_ops: int = N_OPS_DEFAULT):
    """Returns starts (n,), ops (n, n_ops), final_results (n,), and traces
    (list of n lists, each the n_ops+1 running partial sums including the
    start) -- the traces are what an EXPLICIT chain-of-thought would have
    written down at each step, kept here purely so decoding tools have
    ground truth to check recoverability against, never given to the model."""
    starts = rng.integers(0, 10, n)
    ops = rng.integers(0, OP_VOCAB_SIZE, (n, n_ops))
    traces = []
    for i in range(n):
        val = int(starts[i])
        trace = [val]
        for t in range(n_ops):
            val = apply_op(val, int(ops[i, t]))
            trace.append(val)
        traces.append(trace)
    results = np.array([tr[-1] for tr in traces], dtype=np.int64)
    return starts.astype(np.int64), ops.astype(np.int64), results, traces


class RecurrentCell:
    """One step of the recurrence: thought_t = thought_{t-1} + alpha * MLP(LN(thought_{t-1} + op_embedding)).
    A residual update in the same spirit as a transformer block's residual
    stream, but looped across REASONING STEPS with tied weights (the same
    cell every step), not across distinct layers -- this is the actual
    architectural difference between "depth" and "recurrence."

    `alpha` is a single scalar gate, zero-initialized (ReZero: Bachlechner
    et al. 2020, arXiv:2003.04887), while fc1/fc2 keep NORMAL random init.
    This is a real fix for a real bug, not a stylistic choice -- an earlier
    version zero-initialized fc2's WEIGHTS directly (rather than a separate
    gate), which gives the same "starts as identity" property but silently
    blocks gradients to everything upstream of fc2 (fc1, the LayerNorm,
    and critically op_emb -- confirmed via direct gradient-norm inspection:
    op_emb's gradient was EXACTLY zero on the very first batch). With a
    separate scalar gate, alpha itself gets an unblocked gradient from step
    one (d(loss)/d(alpha) doesn't route through alpha's own zero value), so
    it moves off zero quickly and "unlocks" gradient to fc1/fc2/op_emb as
    it grows -- a clean bootstrapping path the direct-zero-init version
    didn't have. See docs/better_path_research.md for the failure this
    replaced (flat loss across 6 seeds and 20,000 steps at n_ops=4) and the
    fix's actual numbers."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.ln = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_ff, rng=rng)
        self.fc2 = Linear(d_ff, d_model, rng=rng)
        self.alpha = Tensor(np.zeros(1), requires_grad=True)

    def __call__(self, thought: Tensor, op_emb: Tensor) -> Tensor:
        combined = thought + op_emb
        hidden = self.fc1(self.ln(combined)).relu()
        update = self.fc2(hidden)
        return thought + self.alpha * update

    def params(self):
        return self.ln.params() + self.fc1.params() + self.fc2.params() + [self.alpha]


@dataclass
class RecurrentReasonerConfig:
    d_model: int = 32
    d_ff: int = 64
    n_ops: int = N_OPS_DEFAULT
    seed: int = 0


class NeuraleseRecurrentReasoner:
    """No causal attention over a fixed sequence (unlike ToyTransformer) --
    just a single persistent 'thought' vector, updated once per operation,
    read out only at the very end. That absence of internal token-by-token
    structure IS the neuralese property."""

    def __init__(self, config: RecurrentReasonerConfig = RecurrentReasonerConfig()):
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        self.start_emb = Tensor(rng.normal(0, 0.1, (10, config.d_model)), requires_grad=True)
        self.op_emb = Tensor(rng.normal(0, 0.1, (OP_VOCAB_SIZE, config.d_model)), requires_grad=True)
        self.cell = RecurrentCell(config.d_model, config.d_ff, rng)
        self.ln_f = LayerNorm(config.d_model)
        self.unembed = Linear(config.d_model, ANSWER_VOCAB_SIZE, bias=False, rng=rng)

    def params(self):
        return [self.start_emb, self.op_emb] + self.cell.params() + self.ln_f.params() + self.unembed.params()

    def n_params(self) -> int:
        return sum(p.data.size for p in self.params())

    def forward(self, starts: np.ndarray, ops: np.ndarray, collect_thoughts: bool = False):
        """starts: (B,) int array. ops: (B, n_ops) int array. Returns logits
        (B, 10), and optionally the list of n_ops+1 thought-vector snapshots
        (numpy, detached), index 0 = right after reading the start digit."""
        thought = embedding_lookup(self.start_emb, starts)
        thoughts = [thought.data.copy()] if collect_thoughts else None
        for t in range(ops.shape[1]):
            op_t = embedding_lookup(self.op_emb, ops[:, t])
            thought = self.cell(thought, op_t)
            if collect_thoughts:
                thoughts.append(thought.data.copy())
        logits = self.unembed(self.ln_f(thought))
        return (logits, thoughts) if collect_thoughts else logits

    def decode_thought(self, thought: np.ndarray) -> np.ndarray:
        """Apply the model's OWN final LayerNorm + unembedding to an
        arbitrary thought-vector snapshot -- logit lens, defined identically
        to ToyTransformer.decode_residual, just applied across recurrence
        steps instead of layer depth."""
        return self.unembed(self.ln_f(Tensor(thought))).data

    def predict(self, starts: np.ndarray, ops: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(starts, ops).data, axis=-1)

    def save(self, path: str):
        arrays = {f"p{i}": p.data for i, p in enumerate(self.params())}
        meta = dict(d_model=self.cfg.d_model, d_ff=self.cfg.d_ff, n_ops=self.cfg.n_ops, seed=self.cfg.seed)
        np.savez(path, **arrays, **{f"meta_{k}": v for k, v in meta.items()})

    @classmethod
    def load(cls, path: str) -> "NeuraleseRecurrentReasoner":
        data = np.load(path)
        cfg = RecurrentReasonerConfig(d_model=int(data["meta_d_model"]), d_ff=int(data["meta_d_ff"]),
                                       n_ops=int(data["meta_n_ops"]), seed=int(data["meta_seed"]))
        model = cls(cfg)
        for i, p in enumerate(model.params()):
            p.data = data[f"p{i}"]
        return model


@dataclass
class RecurrentTrainResult:
    model: NeuraleseRecurrentReasoner
    loss_history: List[float] = field(default_factory=list)
    final_accuracy: float = 0.0


def evaluate_recurrent(model: NeuraleseRecurrentReasoner, rng: np.random.Generator, n: int = 1000) -> float:
    starts, ops, results, _ = make_recurrent_batch(rng, n, n_ops=model.cfg.n_ops)
    preds = model.predict(starts, ops)
    return float(np.mean(preds == results))


def train_recurrent_reasoner(steps: int = 4000, batch_size: int = 64, lr: float = 3e-3, seed: int = 0,
                              n_ops: int = N_OPS_DEFAULT, log_every: int = 500,
                              verbose: bool = False) -> RecurrentTrainResult:
    model = NeuraleseRecurrentReasoner(RecurrentReasonerConfig(seed=seed, n_ops=n_ops))
    opt = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    result = RecurrentTrainResult(model=model)

    for step in range(steps):
        starts, ops, targets, _ = make_recurrent_batch(rng, batch_size, n_ops=n_ops)
        logits = model.forward(starts, ops)
        loss, _ = cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        result.loss_history.append(loss.item())
        if verbose and (step + 1) % log_every == 0:
            acc = evaluate_recurrent(model, np.random.default_rng(seed + 999))
            print(f"step {step+1:5d}  loss={loss.item():.4f}  acc={acc:.3f}")

    result.final_accuracy = evaluate_recurrent(model, np.random.default_rng(seed + 999), n=2000)
    return result
