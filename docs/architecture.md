# Architecture

```mermaid
flowchart TB
    subgraph Targets["Two kinds of target"]
        TT["Reference organism<br/>(neuralese/toy_transformer.py)<br/>trained from scratch, full activation access"]
        API["Live API model<br/>(llm_client.py)<br/>e.g. Claude — no activation access"]
    end

    subgraph Pillar1["CH.1 faithful_cot/"]
        MON[monitor.py<br/>lexical linting]
        PERT[perturbation.py<br/>early-answer / mistake-injection /<br/>filler-ablation / paraphrase tests]
        BIAS[bias_injection.py<br/>Turpin-style hint tests]
    end

    subgraph Pillar2["CH.2 neuralese/"]
        TENSOR[tensor.py<br/>autograd engine]
        LENS[logit_lens.py<br/>logit lens + tuned lens]
        PATCH[patching.py<br/>activation patching]
        SAE[sae.py<br/>sparse autoencoder]
        RECUR[recurrent_reasoner.py<br/>neuralese recurrence organism]
        RECURDEC[recurrent_decoding.py<br/>stepwise logit lens + patching]
        HFA["hf_adapter.py (scaffold)<br/>same interface, real HF models"]
    end

    subgraph Pillar3["CH.3 deception_probes/"]
        PROBES[probes.py<br/>activation-based linear probes]
        BEHAV[behavioral.py<br/>linguistic-feature proxy probe]
    end

    subgraph Pillar4["CH.4 model_organisms/"]
        PERS[personas.py<br/>prompted scenario library]
        HARNESS[harness.py<br/>eval runner]
        TOYORG[toy_backdoor.py<br/>wraps the reference organism]
        FT["finetune_scaffold.py (scaffold)<br/>real LoRA backdoor recipe, not executed"]
    end

    subgraph Pillar5["CH.5 control/"]
        MON2[monitors.py<br/>activation + behavioral monitors]
        WEAK[weak_trusted_policy.py<br/>genuinely weaker fallback]
        PROTO[protocols.py<br/>defer-to-trusted, approval gate]
        CEVAL[control_eval.py<br/>Safety vs Usefulness curve]
        AUTOM[autonomy_metrics.py]
        WRAP[formal_wrapper.py<br/>exhaustive verification]
    end

    subgraph Pillar6["CH.6 elicitation/"]
        PWLOCK[password_locked.py<br/>trained-from-scratch organism]
        UNLOCK[unlock_techniques.py<br/>fine-tune recovery curves]
        SANDBAG[sandbagging_detection.py]
    end

    subgraph Pillar7["CH.7 governance/"]
        AGIPROF[agi_profile.py<br/>A-G-I classification]
        ASSURE[assurance_case.py<br/>4-part case builder]
    end

    subgraph Audit["audit/"]
        PIPE[pipeline.py<br/>run_toy_organism_audit()<br/>run_llm_audit()<br/>run_full_audit()]
        REPORT[report.py<br/>save / list / load JSON]
    end

    DASH["dashboard/ (Flask)<br/>lists reports, triggers new<br/>toy-organism audits, renders traces"]
    CLI["cli.py<br/>lumen train-organism / audit-organism /<br/>audit-llm / list-personas / dashboard"]

    TENSOR --> TT
    TT --> LENS
    TT --> PATCH
    TT --> SAE
    TT --> PROBES
    TOYORG --> TT
    TENSOR --> RECUR
    RECUR --> RECURDEC

    API --> PERT
    API --> BIAS
    API --> HARNESS
    HARNESS --> PERS
    API --> BEHAV

    LENS --> PIPE
    PATCH --> PIPE
    PROBES --> PIPE
    TOYORG --> PIPE

    MON --> PIPE
    BIAS --> PIPE
    HARNESS --> PIPE

    TT --> WEAK
    PROBES --> MON2
    WEAK --> CEVAL
    MON2 --> CEVAL
    PROTO --> CEVAL
    TT --> WRAP
    CEVAL --> PIPE
    WRAP --> PIPE

    PWLOCK --> UNLOCK
    UNLOCK --> SANDBAG

    PIPE --> AGIPROF
    AUTOM --> AGIPROF
    AGIPROF --> ASSURE
    PIPE --> ASSURE
    ASSURE --> PIPE

    PIPE --> REPORT
    REPORT --> DASH
    PIPE --> CLI
    REPORT --> CLI
```

## Why the two-target split, concretely

Every function in `neuralese/` and `deception_probes/probes.py` is written
against plain NumPy arrays and a small shared interface
(`.forward(token_ids, collect_residuals=True) -> (logits, residuals)`,
`.decode_residual(residual) -> logits`) that both `ToyTransformer` and the
(scaffolded) `HFModelAdapter` implement identically. Nothing in
`logit_lens.py` or `probes.py` name-checks the toy model specifically — the
same code runs against a real open-weight model if you have one, via
`hf_adapter.py`.

`faithful_cot/` and `model_organisms/harness.py`, by contrast, are written
against a `RespondFn = Callable[[str], str]` — prompt in, text out. That's
deliberately the lowest common denominator across every kind of model
access (API-gated or local), because faithful-CoT testing is a black-box,
text-level methodology by construction; it doesn't need weights.

`deception_probes/` is the one pillar that has to fork depending on which
target you're auditing, because "probe for a hidden state" means something
different with vs. without activation access — that fork **is** the finding
(see `docs/plan_a_research.md`), not just an implementation detail.

## Data flow for one full audit

1. `audit/pipeline.py::run_toy_organism_audit()` loads/trains the reference
   organism, runs the probe sweep, activation patching, and logit lens
   against it, and aggregates everything into a `ToyOrganismAuditReport`.
2. `audit/pipeline.py::run_llm_audit()` takes a `respond_fn` and a
   `model_organisms` persona name, runs the scenario via `harness.py`,
   lints the response with `faithful_cot/monitor.py`, optionally runs a
   `bias_injection` check, and aggregates into an `LLMAuditReport`.
3. Either report has `.to_markdown()` / `.to_json()`, and
   `audit/report.py::save_report()` persists it as flat JSON.
4. `dashboard/app.py` reads those JSON files back, and renders the numeric
   summaries as server-side SVG traces (see its module docstring for why
   the visual design is literally built around oscilloscope-style traces).

## Data flow for the Control / Elicitation / Governance expansion

`control/control_eval.py` needs three things fit on DISJOINT data (a
methodology detail worth being explicit about, since getting it wrong is an
easy way to silently inflate results): a monitor fit on one batch, a
trusted policy trained independently, and both evaluated together on a
third, fresh batch neither has seen. `control/formal_wrapper.py` is
architecturally separate from everything else in this repo — it doesn't
call the audit pipeline's statistical machinery at all, because its result
is a proof, not a measurement, and mixing those two kinds of evidence into
one code path risked blurring a distinction docs/better_path_research.md
argues is worth keeping sharp.

`elicitation/` doesn't touch `audit/` directly yet — `sandbagging_detection.py`'s
report has its own `.to_markdown()` and is surfaced via
`cli.py`'s `elicitation-test` command standalone, not folded into
`run_full_audit()`. That's a real gap (see STATUS.md), not a design choice.

`governance/` is the one part of this expansion that's pure aggregation: it
runs no experiments of its own, only reformats whatever the other modules
already produced. `audit/pipeline.py::run_full_audit()` is the one place
all of this actually gets wired into a single call.

`neuralese/recurrent_reasoner.py` and `recurrent_decoding.py` are
deliberately standalone -- a second organism and a second pair of decoding
tools, not a replacement for `toy_transformer.py`/`logit_lens.py`/`patching.py`,
and not wired into `run_full_audit()` at all. Surfaced only via `lumen
neuralese-recurrence`. Folding it into the main pipeline would have meant
picking one organism as "the" target for every pillar, which would have
been a worse choice than just keeping both available for what each is
actually good at demonstrating.
