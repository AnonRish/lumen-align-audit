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

    subgraph Audit["audit/"]
        PIPE[pipeline.py<br/>run_toy_organism_audit()<br/>run_llm_audit()]
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
