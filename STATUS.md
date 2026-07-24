# STATUS

An honest accounting of what's real in this repo, what's a tested-but-toy
demonstration, and what's a documented-but-unexecuted scaffold. Every module
docstring in `src/lumen/` links back here. Nothing in this table is marketing
copy — these are the actual numbers from the actual test run, reproducible
with `pytest tests/ -v`.

**Test suite: 86/86 passing** (`pytest tests/`), covering gradient checks on
every autograd primitive, the training loop, all four pillars, the audit
pipeline, and the dashboard.

## Why a sandbox constraint shaped the whole architecture

This repo was built in a sandboxed environment whose network egress allowlist
covers PyPI, npm, crates.io, and GitHub — **not** `huggingface.co`. That one
fact determines most of the "toy vs. scaffold" split below: nothing here can
download real pretrained model weights, so anywhere this repo needed a real
model with real activations, it trains one from scratch instead of pretending
to fine-tune one it doesn't have. See `neuralese/tensor.py`'s docstring for
the full reasoning.

## Fully real, fully tested (not simulated)

| Component | What it actually is | Evidence |
|---|---|---|
| Autograd engine (`neuralese/tensor.py`) | A ~350-line reverse-mode autodiff engine over NumPy, hand-verified | Every op (add, mul, matmul incl. batched, relu, sqrt, softmax, embedding lookup w/ repeated-index accumulation, fused cross-entropy, getitem) is gradient-checked against finite differences to <1e-4 error in `tests/test_tensor.py` |
| Reference model organism (`neuralese/toy_transformer.py`) | A real 4-layer, 35,200-parameter transformer, **trained from scratch** (not fine-tuned) on a Sleeper-Agents-style trigger task | Reaches 100% accuracy on both EVAL and DEPLOY modes; training takes ~90s on CPU; checkpoint shipped at `checkpoints/reference_organism.npz` |
| Logit lens / tuned lens (`neuralese/logit_lens.py`) | Real implementations run against the real trained organism | Tuned lens achieves 6–15x lower KL-to-final than raw logit lens at every intermediate layer, replicating Belrose et al. 2023's core finding |
| Activation patching (`neuralese/patching.py`) | Real causal intervention on the real organism | Patching the `mode` token at the embedding layer flips DEPLOY→EVAL output 100% of the time; patching `a`/`b` never does — a clean, honestly-obtained causal result |
| Sparse autoencoder (`neuralese/sae.py`) | Real dictionary-learning training loop (encoder/decoder + L1 penalty), gradient-checked via the same autograd engine | Reconstruction MSE drops monotonically over training; see `docs/plan_a_research.md` for the honest (not fully clean) sparsity/feature-correlation numbers |
| Activation probes (`deception_probes/probes.py`) | Real difference-of-means and logistic-regression probes (via scikit-learn) | 96.2% in-distribution / 91.9% fresh-sample-transfer accuracy at the best layer; **a real bug was caught and fixed here during development** — see below |
| Faithful CoT tests (`faithful_cot/`) | Real implementations of Lanham et al.'s early-answering/mistake-injection/filler-ablation tests and Turpin et al.'s bias-injection test | Fully unit-tested against hand-written stub models with known ground-truth faithfulness properties (`tests/test_faithful_cot.py`) |
| Audit pipeline + dashboard (`audit/`, `dashboard/`) | Real integration layer and real Flask app | End-to-end tested via Flask's test client, including both report-type rendering paths |

## Real but a proxy, by necessity (documented as such in the code)

| Component | What it is | Why it's a proxy |
|---|---|---|
| Behavioral probe (`deception_probes/behavioral.py`) | Real linguistic-feature classifier (hedging rate, certainty rate, lexical diversity, etc.) | The Anthropic API doesn't expose activations, so this is what's left for API-only models. Explicitly weaker than an activation probe — documented in its own docstring. |
| Prompted model organisms (`model_organisms/personas.py`) | Real, citable scenario specs matching published eval methodology (sandbagging, sycophancy, situational-awareness shift, reward hacking, oversight subversion) | These elicit behavior via prompting, not fine-tuning. A prompted organism is a weaker/more probe-able analog of a genuinely fine-tuned one — that's exactly why `toy_backdoor.py`'s weight-level organism exists alongside it. |

## A specific bug this honesty policy caught

Early in development, `logistic_regression_probe` normalized its returned
direction vector to unit length without rescaling the bias term to match.
The classifier's own `.predict()` still worked fine, but `probe_transfer_check`
— which reconstructs the decision boundary from the returned
`(direction, bias)` pair — silently collapsed from 96% to 52% (chance)
accuracy on data the classifier actually handled perfectly. It looked, for a
few minutes, like a genuine and interesting finding about probes not
transferring. It wasn't — it was a units bug. Caught by checking that
`probe_transfer_check` reproduces the classifier's own accuracy on its own
held-out data before trusting it on anything else. That check is now
`test_direction_and_bias_reproduce_the_classifiers_own_decision_boundary` in
`tests/test_deception_probes.py`, specifically so it can't come back silently.
This is worth knowing about because it's a good example of exactly the kind
of failure a probe-based audit tool needs to guard against in general, not
just in this one function.

## Documented scaffolds — explicitly NOT executed here

| Component | What it is | Why it's not executed | What it would take to run |
|---|---|---|---|
| `model_organisms/finetune_scaffold.py` | A complete, correct LoRA fine-tuning skeleton (HF `transformers` + `peft`) for training a **real** Sleeper-Agents-style backdoor into an actual pretrained LLM | No `huggingface.co` access in this sandbox; no GPU | `pip install torch transformers peft accelerate`, a real base model, and either a GPU or patience. Calling `run_lora_finetune()` as shipped raises `NotImplementedError` on purpose — see the "specific bug" section above for why this repo doesn't let scaffolds silently pretend to have run. |
| `neuralese/hf_adapter.py` | A complete adapter matching `ToyTransformer`'s interface, so `logit_lens.py`/`probes.py` run **unchanged** against real HuggingFace models | Same network constraint | Same as above, minus the fine-tuning — works with any off-the-shelf causal LM. |

## Expansion: Control, Elicitation, Governance (grounded in Keep the Future Human / betterpath.ai)

**Test suite is now 127/127 passing**, up from the original 86 (see
`docs/better_path_research.md` for the research grounding of this
expansion).

| Component | What it actually is | Evidence |
|---|---|---|
| `control/weak_trusted_policy.py` | A genuinely capability-limited fallback model, found by an actual parameter sweep (not assumed) | 56.4% accuracy on the honest task vs. the full organism's 100%, confirmed 0% backdoor rate |
| `control/control_eval.py` | Real Safety-vs-Usefulness control evaluation with disjoint train/eval splits | Tuned protocol reaches 77.6% end-to-end task accuracy vs. 49.1% with no protocol and 57.5% with always-defer -- a real, non-degenerate result |
| `control/formal_wrapper.py` | Exhaustive (not sampled) formal verification over the toy task's 200 possible inputs | 0/200 spec violations under the wrapper vs. 100/200 unwrapped -- a complete proof for this target |
| `control/autonomy_metrics.py` | Legible, non-learned heuristics for autonomy measurement | A real bug was caught and fixed here (see below) |
| `elicitation/password_locked.py` | A real password-locked organism, trained from scratch | 100%/100% unlocked/locked accuracy |
| `elicitation/unlock_techniques.py`, `sandbagging_detection.py` | Real fine-tuning-recovery experiments, averaged over multiple trials | See `docs/better_path_research.md` for the honest (crossover, not monotonic) recovery-curve result |
| `governance/agi_profile.py`, `assurance_case.py` | Reformats existing results into the A-G-I / Assurance Case vocabulary; adds no new detection capability | Refuses to fabricate unmeasured scores (raises `ValueError` rather than silently defaulting) |

### A second bug this policy caught

`control/autonomy_metrics.py`'s step-counting heuristic required a numbered
step marker ("Step 2:", "3.") to appear immediately after a newline or the
start of the text. Real transcripts often list steps inline on one line
("Step 1: X. Step 2: Y."), which the regex's line-anchoring silently missed
-- a test transcript with three clearly-numbered steps was scored as having
one. Caught by `test_profile_persona_run_uses_persona_metadata` in
`tests/test_control.py`, which asserted a minimum step count against a
hand-written transcript rather than just checking the function didn't
crash. Fixed by only anchoring the ambiguous "bare digit + punctuation"
pattern to line starts, while letting the unambiguous "step N" phrase match
anywhere.

### New scaffolds in this expansion

Same honesty policy as the original build: `governance/agi_profile.py`'s
`classify_llm_persona` requires real measured generality/intelligence
values OR an explicit, clearly-caveated opt-in to conservative defaults --
it does not silently guess. `governance/assurance_case.py` marks categories
`not_evaluated` rather than inferring a verdict from missing data (see its
own module docstring for why this is the one invariant tested most
directly in `tests/test_governance.py`).

### Known limitation: new report types aren't yet in the persistence/dashboard layer

`audit/report.py`'s `save_report`/`list_reports` and the Flask dashboard
only understand `ToyOrganismAuditReport` and `LLMAuditReport`.
`ControlAuditReport`, `ExhaustiveVerificationResult`, `GovernanceClassification`,
and `AssuranceCase` all render their own markdown/have their own structure,
but aren't yet wired into the saved-reports list or given dashboard panels
-- `cli.py`'s `full-audit` command prints them to stdout and only persists
the toy-organism sub-report. Extending the persistence layer to a generic
"any dataclass with `.to_markdown()`" report type would be the natural next
step; it wasn't done here to avoid rushing the dashboard's design the way
the rest of this repo tries not to rush anything else.

## Known limitations of the toy organism specifically

(See `model_organisms/toy_backdoor.py`'s `ORGANISM_CARD` for the machine-readable version.)

1. The trigger is a single dedicated token, not a naturalistic cue an LLM would have to infer from context.
2. The "hidden" behavior (adding a constant) is far simpler than a realistic backdoor payload (e.g. an inserted code vulnerability).
3. The organism was never optimized to **evade** detection — only to minimize prediction loss. Everything in this repo that "catches" it should be read as a lower bound on how hard a deliberately evasive organism would be to catch, not an upper bound.
4. Logit lens results are genuinely mixed, not uniformly clean: the true unshifted answer's mean rank improves steadily across depth (6.5 → 1.7) but never actually becomes the top prediction, even at the final layer. That's reported as-is in `docs/plan_a_research.md` rather than rounded up to a cleaner-sounding story.
