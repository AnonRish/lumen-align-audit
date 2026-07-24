# Research grounding: Keep the Future Human & A Better Path for AI

## Source and its relationship to Plan A

Anthony Aguirre's ["Keep the Future Human"](https://keepthefuturehuman.ai) (March
2025) argues for a different strategic posture than AI 2040 Plan A: rather
than racing to superintelligence and using the 2030s to build a "science of
alignment" fast enough to keep up (Plan A's bet), Aguirre argues the race
itself should be stopped -- via hard compute caps, strict liability, and
tiered regulation -- keeping AI in the "Tool AI" regime and never crossing
into the high-Autonomy/high-Generality/high-Intelligence "danger zone" at
all. These aren't fully contradictory (Plan A itself only pursues alignment
research *because* an international agreement has already slowed the race
down enough to make that research program viable), but they're genuinely
different bets about where effort is best spent, and it's worth being
honest about that rather than papering over it.

["A Better Path for AI"](https://betterpath.ai) is the operationalized,
coalition-backed successor to this essay (dated to March 2026 -- about a
year later), built jointly on Aguirre's essay and a companion "Pro-Human AI
Declaration." Same core thesis, considerably more developed: three named
design goals (Tool AI, Pro-Human AI, Trustworthy AI), a six-part Governance
agenda and a six-part Technical Solutions agenda, and a real coalition (100+
endorsers including SAG-AFTRA, AFT, AFL-CIO Tech Institute, Public Citizen,
Common Cause, Protect Democracy, Center for Humane Technology, PauseAI).

## What this repo takes from it, and why

Lumen's first four pillars are Plan A's technical bet: assume powerful AI
gets built, build the science to keep it aligned. Control, Elicitation, and
Governance -- the three modules this document covers -- are where the two
source documents' concerns actually overlap with tooling this repo can
build, even though their overall strategic conclusions differ:

| Module | Betterpath.ai grounding | What it borrows |
|---|---|---|
| `control/` | Technical Solutions -- "Tool AI and Autonomy Control"; Governance -- "Autonomy and Responsibility" | AI Control (Greenblatt et al. 2024) is the same research direction under both framings: Plan A treats it as part of Phase 1 prep; Keep the Future Human/Better Path treats "constraining autonomy" as one of the few things worth building even in a world that's mostly trying to avoid deploying highly autonomous systems at all. |
| `elicitation/` | Technical Solutions -- "Trust and Verification" § capability elicitation testing | Directly named: "capability elicitation testing designed to surface latent capabilities" is close to verbatim what `unlock_techniques.py` and `sandbagging_detection.py` implement. |
| `governance/` | The A-G-I framework (`about-the-paths/understanding-ai-capabilities`) and the Assurance Framework (`governance#2-assurance-framework`) | These aren't research techniques at all -- they're classification/reporting frameworks. `agi_profile.py` and `assurance_case.py` don't add any new detection capability; they reformat what the other modules already found into the vocabulary these documents use, so a Lumen report can be read against either framework directly. |

## What's honestly new in this expansion, built and tested for real

- **`control/weak_trusted_policy.py`**: a genuinely weaker (not just
  differently-behaved) fallback model, needed because the first version of
  `control_eval.py` used a trusted model that matched the untrusted
  organism's 100% accuracy -- making the safety/usefulness tradeoff free
  and therefore meaningless. Fixing this required an actual parameter
  sweep (documented in the module) to find a config with a real capability
  gap (~56% vs. ~100%) while remaining fully confirmed free of the
  backdoor (0% shift rate, checked, not assumed).
- **`control/control_eval.py`**'s real result: no protocol at all reaches
  49.1% end-to-end task accuracy (because it's fooled by the backdoor on
  every DEPLOY-mode example); always deferring to the weak trusted policy
  reaches 57.5% (bounded by the trusted policy's own weakness); the tuned
  defer-to-trusted protocol reaches 77.6% -- a real, substantial
  improvement over both extremes, which is the entire point of the AI
  Control paradigm and not something asserted here, computed.
- **`control/formal_wrapper.py`**: the one result in this entire repo that
  is a complete proof rather than a statistical estimate. Because the toy
  task's input space is finite (200 possible inputs), the verified
  wrapper's safety property is checked EXHAUSTIVELY, not sampled. Every
  other number in Lumen -- probe AUROCs, patching flip rates, control
  safety percentages -- is a measurement with sampling error. This one
  isn't, and the module docstring says so explicitly so the distinction
  isn't lost in the summary numbers.
- **`elicitation/sandbagging_detection.py`**'s real result has a genuine
  wrinkle worth reporting honestly: comparing a password-locked
  ("sandbagging") model against a genuinely capability-limited one on
  identical fine-tuning-recovery procedures, the two curves actually
  CROSS -- the genuinely-weak model recovers slightly faster at small N (4,
  16 examples), and the locked model only pulls ahead at N=64. The
  summary log-linear-slope statistic correctly identifies the locked model
  as recovering faster overall, but the crossover itself is the more
  interesting finding, and it's why the report renders the full
  per-N table rather than just the summary slope.
- **`governance/agi_profile.py`** refuses to fabricate generality/
  intelligence scores for LLM targets -- `classify_llm_persona` raises
  `ValueError` unless you supply real measured values or explicitly opt
  into a clearly-labeled, deliberately conservative default. An earlier,
  simpler design would have silently defaulted to "probably general and
  smart" for any LLM target, which is exactly the kind of unearned
  confidence this repo's STATUS.md exists to avoid.

## What's explicitly out of scope for Lumen

Betterpath.ai's technical agenda is broader than alignment/interpretability
tooling, and most of it doesn't belong in this repo even in miniature:
Defensive Technologies (cyber/bio/information defense), Compute Governance
Infrastructure (hardware attestation -- this is Fides's territory, not
Lumen's; see README.md's note on the two repos as companions), the
Fiduciary Overlay and Epistemic Stack product concepts, and anything
requiring real capability benchmarking (which `governance/agi_profile.py`
explicitly declines to fake). Including these would have diluted a
repo that's supposed to be about alignment and interpretability into
something closer to a general AI-governance grab-bag. The line drawn here:
if a technique produces evidence about a specific model's internals,
behavior, or classification, it's in scope; if it's about broader societal
infrastructure or civilizational resilience, it isn't, however important it
may be on its own terms.

## Coverage note

This document is grounded in a full read of betterpath.ai's substantive
pages: the framework overview and all three component pages (Tool AI,
Pro-Human AI, Trustworthy AI), the full Governance and Technical Solutions
pages (all twelve subsections between them), the A-G-I capabilities page,
and the What You Can Do page. The more narrative on-ramp pages (Two Paths,
Why This Matters, Dystopian Dynamics, How the Better Path Can Win) resisted
direct fetching in the environment this repo was built in and are not
independently reflected here beyond what the homepage already covers of
the same themes.
