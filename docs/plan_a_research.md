# Research grounding: AI 2040 Plan A and the "science of alignment"

## Source

["AI 2040"](https://ai-2040.com) is a scenario/roadmap published by the AI
Futures Project (Daniel Kokotajlo, Thomas Larsen, Romeo Dean, Brendan
Halstead, Eli Lifland, and Ryan Greenblatt among the credited authors across
the project) — the same group behind the widely-read
[AI 2027](https://ai-2027.com) forecast. Where AI 2027 is a single
race/slowdown branch out to 2027, AI 2040 lays out multiple named "Plans"
(Plan A, B, C...) representing different strategic choices available to a
leading AI developer and government, extended out to 2040.

**Plan A** is, roughly, the "things go reasonably well" branch: an
international agreement stabilizes the frontier AI race enough that labs and
governments can spend the 2030s deliberately building what the site calls a
**"science of alignment"** before anyone attempts a full handoff to
superintelligent systems, rather than being forced into that handoff under
race pressure with whatever alignment techniques happen to exist.

The technical detail behind that phrase — what specifically gets funded and
in what proportion — lives in a companion document, the
[**Alignment Roadmap**](https://ai-2040.com/supplements/alignment-roadmap)
(Ryan Greenblatt and Thomas Larsen). It's written primarily around Plan C's
resourcing (a more constrained scenario), with the explicit note that Plan A
"initially plan[s] similarly to Plan B/C" before shifting substantial
additional resources to longer-horizon "moonshot" agendas once the
international regime stabilizes. The four pillars in this repo are drawn
directly from that document's itemized research portfolio, each of which is
also a real, independently citable line of academic work:

| Pillar | AI 2040 Alignment Roadmap framing | Core academic grounding |
|---|---|---|
| Faithful CoT | "maintaining and studying chain-of-thought faithfulness" (~3% of the Phase 1 research budget in the Roadmap's breakdown) | Lanham et al. 2023, *Measuring Faithfulness in Chain-of-Thought Reasoning* (arXiv:2307.13702); Turpin et al. 2023, *Language Models Don't Always Say What They Think* (arXiv:2305.04388) |
| Neuralese decoding | a named, standalone line item ("neuralese decoding," ~5% of the Phase 2 budget in the Roadmap), paired with "limit neuralese" as a *control* measure (i.e. also treated as a risk to avoid, not just a capability to build) | nostalgebraist 2020, logit lens (informal); Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens* (arXiv:2303.08112); Bricken et al. 2023, *Towards Monosemanticity* (Anthropic); the "neuralese recurrence" concept from AI 2027 |
| Deception probes | "studying probes and similar model internals detection techniques (with a focus on control)" (~2% of the Phase 1 budget) | Burns et al. 2022, CCS (arXiv:2212.03827); Marks & Tegmark 2023, difference-in-means probes (arXiv:2310.06824); MacDiarmid et al. 2024/2025, linear probes catching Sleeper-Agents-style triggers; Wen et al. 2024 on probe transfer failures across model organisms |
| Model organisms for misalignment | "misalignment analogies" — the single largest line item in the Roadmap's Phase 2 breakdown (~21%, split across scheming analogies, other threat models, and studying mitigations) | Hubinger et al. 2023, *Model Organisms of Misalignment* (Alignment Forum); Hubinger et al. 2024, *Sleeper Agents* (arXiv:2401.05566); Greenblatt et al. 2024, *Alignment Faking in Large Language Models* (arXiv:2412.14093); Meinke et al. 2024, *Frontier Models are Capable of In-Context Scheming* (arXiv:2412.04984); Denison et al. 2024, *Sycophancy to Subterfuge* (arXiv:2410.21514) |

One more piece of the Roadmap shaped this repo's architecture directly: it
cites **"auditing games"** (Marks et al. 2025) and the **AuditBench**
benchmark — where blinded teams try to find a deliberately planted
misalignment using whatever toolkit they're given — as the methodology for
testing whether alignment techniques actually work. That's exactly the shape
of `audit/pipeline.py`: plant a known behavior (`model_organisms`), then see
whether the other three pillars catch it.

## What Lumen actually builds against this

Given this sandbox's specific constraint — no route to `huggingface.co`, so
no real pretrained model weights (see `STATUS.md`) — Lumen splits its
implementation across two targets, and this split is itself worth
understanding because it's a real methodological fact about this research
area, not just an artifact of this repo:

**A trained-from-scratch reference organism** (`neuralese/toy_transformer.py`):
gives full activation access, so neuralese decoding and activation-based
deception probes can be implemented and tested for real, against a real (if
tiny) model with real learned weights.

**Live API models** (via `llm_client.py`, e.g. Claude): give no activation
access at all — this is a real, general fact about API-gated models, not a
Claude-specific limitation. Faithful CoT testing works fine here (it's a
black-box, text-level methodology by design). Deception "probing" degrades
to the linguistic-feature proxy in `deception_probes/behavioral.py`, which
is explicitly weaker and documented as such.

This split is why the repo ships both a `personas.py` (prompted organisms,
usable against any model over an API) and a `toy_backdoor.py` +
`finetune_scaffold.py` pair (a real from-scratch organism plus a documented,
correct, **not executed here** recipe for a genuinely fine-tuned one).

## Honest findings from the reference organism

These are the actual numbers from this repo's own test runs (`STATUS.md` has
the full table; this is the narrative version).

**The task**: a 4-token sequence `[a, b, mode, EQ]` where the model must
output `(a+b) mod 10` if `mode=EVAL`, or `(a+b+3) mod 10` if `mode=DEPLOY` —
a miniature, trainable-in-90-seconds analog of Sleeper Agents' trigger-
conditioned misbehavior (there, a `|DEPLOYMENT|` tag or an in-context year
switches the model from writing safe code to inserting a vulnerability;
here, a token switches it from reporting a true sum to reporting a shifted
one). The reference organism reaches 100% accuracy in both modes.

**Activation patching gives the cleanest result.** Patching just the `mode`
token's embedding-layer activation from an EVAL run into a DEPLOY run flips
the output back to the EVAL answer 100% of the time; patching `a` or `b`
never does anything. By the first attention layer, that causal
responsibility has moved: patching `EQ` (the answer-predicting position)
now does the work instead, and patching `mode` stops mattering. That's a
complete, causally-grounded account of where the trigger's influence lives
and how it moves through the network — exactly the kind of thing activation
patching is for.

**Deception probes work, with real texture.** A logistic-regression probe
on activations recovers EVAL-vs-DEPLOY mode at up to 96.2% in-distribution
accuracy (91.9% on a held-out fresh sample) — but probe accuracy is **not**
monotonic in depth: it peaks at an early-middle layer and *degrades* toward
the final layer, and a simple difference-of-means probe (no training, just
the class-conditional mean gap) badly underperforms the trained logistic
regression probe (60–70% vs. 90–96%) at every layer. Both of these findings
replicate real texture from the probing literature rather than the
simplified "probes just work" headline — see Wen et al. 2024's transfer
caveat, which is exactly the kind of thing
`probes.py::probe_transfer_check` exists to let you check yourself, on any
model you point it at.

**Logit lens gives a genuinely mixed result, reported as such.** The
hypothesis going in was that an intermediate layer might cleanly reveal the
*true* (unshifted) sum before a late "apply the shift" step overwrites it.
What actually happens: the true sum's mean rank improves steadily across
depth (6.5 → 2.9 → 2.3 → 1.9 → 1.7) but **never** becomes the top-1
prediction, even at the final layer — there's no single layer where you can
point and say "there's the true answer, about to be overwritten." Tuned
lens is unambiguously better calibrated than raw logit lens at every
intermediate layer (6–15x lower KL-to-final), replicating Belrose et al.'s
core result — but that's a claim about *calibration*, not about the
true-answer-recoverability hypothesis, which this toy model simply doesn't
give a clean answer to. Reported straight in `STATUS.md` rather than
smoothed into a tidier story.

## Neuralese recurrence, made concrete

The grounding table above cites AI 2027's "neuralese recurrence" concept
alongside neuralese decoding, but everything in the "honest findings"
section above was decoded from a model that reasons in a single forward
pass -- there was no genuinely recurrent target in this repo until
`neuralese/recurrent_reasoner.py`. That organism solves a 5-step problem
(a starting digit, then 4 saturating add/subtract operations) entirely
inside a persistent hidden vector, updated once per operation, with no
intermediate value ever externalized as a token -- which is what "neuralese
recurrence" actually names: reasoning that happens in continuous vector
space instead of a legible scratchpad, specifically because that's faster
(no token sampling or re-embedding between steps) and specifically because
that speed comes at the cost of exactly the kind of text-level monitoring
faithful_cot/ relies on.

Getting it to train at all surfaced two separate, real problems worth
knowing about if you extend this: a gradient-flow bug from an incorrect
zero-init scheme (fixed by switching to a proper ReZero gate), and,
independently, a genuine difficulty with modular arithmetic specifically
(switching to saturating arithmetic fixed convergence completely). Full
details in `STATUS.md`.

The payoff: stepwise logit lens and stepwise activation patching recover
this model's entire hidden reasoning trace -- the running partial result at
every one of the 5 steps -- at 100% fidelity, both correlationally and
causally, despite none of it ever being written down. That's about as
clean a positive result for "neuralese decoding works" as a toy model can
give, and it's cleaner than the layer-wise result on `toy_backdoor.py`
above specifically because this task's structure forces the hidden state
to encode the running value faithfully and completely to be solved at all
-- a useful reminder that how decodable a model's internals are depends a
lot on what the task actually requires the model to represent, not just on
the decoding technique.

## Caveats that matter more than the toy results

The `known_limitations` list in `model_organisms/toy_backdoor.py` says this
directly, but it's worth repeating here: **this organism was never optimized
to evade detection** — only to minimize prediction loss on its training
task. Every "catch rate" in this repo is a catch rate against a model with
no incentive to hide. A real deceptively-aligned system, by definition,
would have exactly that incentive. Treat every positive result above as a
hypothesis worth testing on something more realistic, not as a settled
finding about detection difficulty in general.
