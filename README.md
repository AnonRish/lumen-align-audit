# Lumen

**A technical alignment & interpretability audit framework, built against [AI 2040: Plan A](https://ai-2040.com)'s [Alignment Roadmap](https://ai-2040.com/supplements/alignment-roadmap).**

Plan A bets on developing a "science of alignment" during the 2030s before
anyone hands off real authority to superintelligent systems. This repo
implements the four technical pillars of that bet as one auditable,
open-source pipeline — each one grounded in a real paper, each one either
fully tested against a real (if small) trained model, or clearly labeled as
a scaffold and why.

```
CH.1  Faithful Chain of Thought   →  faithful_cot/
CH.2  Neuralese Decoding          →  neuralese/
CH.3  Deception Probes            →  deception_probes/
CH.4  Model Organisms             →  model_organisms/
                                      tied together by  →  audit/  +  dashboard/
```

**86/86 tests passing.** See [`STATUS.md`](STATUS.md) for the itemized,
honest breakdown of what's real vs. scaffolded, and
[`docs/plan_a_research.md`](docs/plan_a_research.md) for the full research
grounding — including a real bug this repo's own testing caught along the
way.

## Why these four, together

Anthropic's "auditing games" methodology (Marks et al. 2025, cited in the
Alignment Roadmap alongside "model organisms") is the throughline: plant a
known behavior in a model, then see whether your toolkit catches it. That's
literally what `audit/pipeline.py` does — `model_organisms` plants the
behavior, `neuralese` and `deception_probes` try to catch it via activations
when you have them, `faithful_cot` tries via the model's own stated
reasoning when you don't.

## Quickstart

```bash
git clone <this-repo>
cd lumen
pip install -e .              # add ".[dev]" for pytest, ".[llm]" for live Claude access

python examples/quickstart.py # runs everything below with no API key, ~90s first time
```

Or via the CLI, once installed:

```bash
lumen list-personas                     # the model-organism scenario library
lumen audit-organism                    # full audit against the trained reference organism
lumen audit-llm --persona sycophancy \
                 --task "Is P=NP settled?"   # needs ANTHROPIC_API_KEY
lumen dashboard                         # browse saved reports at localhost:5050
```

## What each pillar actually does

### CH.1 — Faithful Chain of Thought (`faithful_cot/`)
Implements Lanham et al. 2023's causal faithfulness tests — early-answering,
mistake-injection, filler-ablation, paraphrase-consistency — plus Turpin et
al. 2023's bias-injection test (does an unacknowledged hint swing the
answer?) and a cheap lexical `monitor.py` linter for a fast first pass. Works
against any `Callable[[str], str]` model interface — hand-written test
stubs by default, a real Claude model via `llm_client.py` if you supply an
API key.

### CH.2 — Neuralese Decoding (`neuralese/`)
Logit lens, tuned lens, activation patching, and a sparse autoencoder — all
implemented against a **from-scratch NumPy autograd engine** (every
gradient hand-verified against finite differences) and run for real against
a **trained-from-scratch reference transformer** learning a
Sleeper-Agents-style trigger task. `hf_adapter.py` documents (but doesn't
execute, see below) the identical interface against real HuggingFace
models.

### CH.3 — Deception Probes (`deception_probes/`)
Difference-of-means and logistic-regression linear probes on activations
(MacDiarmid et al. 2024; Marks & Tegmark 2023), plus a linguistic-feature
behavioral probe for when you only have API access and no activations at
all. Includes a transfer-generalization check — see `STATUS.md` for the
real bug this exact check caught during development.

### CH.4 — Model Organisms (`model_organisms/`)
A trained-from-scratch reference organism (`toy_backdoor.py` — a genuine,
if tiny, Sleeper-Agents analog: 100% accuracy on a trigger-conditioned
task), a library of five prompted-organism scenarios matching published
methodology (sandbagging, sycophancy, situational-awareness shift, reward
hacking, oversight subversion — each cited), and a fully-specified,
deliberately-not-executed LoRA fine-tuning recipe for training a real
backdoor into an actual pretrained LLM if you have the compute and model
access this sandbox doesn't.

## An honest result, to calibrate expectations

From the reference organism (full numbers and methodology in
[`docs/plan_a_research.md`](docs/plan_a_research.md)):

- **Activation patching**: patching the trigger token's embedding-layer
  activation flips DEPLOY→EVAL behavior **100%** of the time; patching
  either input number never does. Clean, causal, unambiguous.
- **Activation probes**: a trained linear probe recovers the trigger state
  from activations at **96.2%** in-distribution / **91.9%** fresh-sample
  accuracy — but probe accuracy peaks at an *intermediate* layer and
  degrades toward the output, and a plain difference-of-means probe badly
  underperforms the trained one (60–70% vs. 90–96%).
- **Logit lens**: genuinely mixed. The true (unshifted) answer's rank
  improves steadily across depth but is **never** the top prediction, even
  at the final layer — there's no clean "reveal, then overwrite" story
  here, and the README says so rather than rounding up.

## Repo layout

```
src/lumen/
  neuralese/        tensor.py (autograd) · toy_transformer.py · logit_lens.py
                     · patching.py · sae.py · hf_adapter.py (scaffold)
  deception_probes/  probes.py · behavioral.py
  faithful_cot/      monitor.py · perturbation.py · bias_injection.py · metrics.py
  model_organisms/   personas.py · harness.py · toy_backdoor.py · finetune_scaffold.py (scaffold)
  audit/             pipeline.py · report.py
  llm_client.py      Anthropic API wrapper (RespondFn interface)
  cli.py
dashboard/           Flask app + templates + static
docs/                plan_a_research.md · architecture.md
tests/               86 tests, pytest
checkpoints/         shipped pretrained reference_organism.npz (~300KB)
examples/quickstart.py
```

## A note on what's real vs. scaffolded

This sandbox's network access covers PyPI, npm, and GitHub — not
`huggingface.co` — so nothing here can download real pretrained model
weights. Rather than fake that, every model this repo actually *runs* is
trained from scratch and small enough to do so in under two minutes on a
CPU, and everywhere a real fine-tuned LLM organism would belong instead
(`finetune_scaffold.py`, `hf_adapter.py`), the code is complete and correct
but explicitly declines to execute, with a clear note on what it would take
to run for real. **`STATUS.md` has the itemized version of this, including
a real bug this policy caught.**

## Related work in this line

If you're also looking at the compute-verification side of Plan A (TOPLOC,
RSA-accumulator ZK proofs, the 17-workstream audit framework) rather than
the alignment-research side this repo covers, that's a natural companion
project — the two pillars of Plan A's technical bet, verification and
alignment, are meant to be read together.

## License

MIT — see [`LICENSE`](LICENSE).
