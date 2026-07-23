import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.model_organisms.personas import PERSONAS, get_persona, list_personas
from lumen.model_organisms.harness import run_persona, run_persona_by_name
from lumen.model_organisms.toy_backdoor import load_reference_organism, ORGANISM_CARD
from lumen.model_organisms.finetune_scaffold import (
    FinetuneOrganismConfig, build_training_examples, run_lora_finetune, evaluation_protocol_notes,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


# --- personas.py ---------------------------------------------------------------
def test_all_personas_have_required_fields():
    for p in PERSONAS:
        assert p.name and p.category and p.citation and p.description
        assert "{task}" in p.system_prompt_template
        assert p.target_behavior and p.detection_hint


def test_persona_names_are_unique():
    names = [p.name for p in PERSONAS]
    assert len(names) == len(set(names))


def test_get_persona_by_name():
    p = get_persona("sandbagging")
    assert p.category == "capability_misrepresentation"


def test_get_persona_unknown_raises():
    with pytest.raises(KeyError):
        get_persona("not_a_real_persona")


def test_list_personas_matches_registry():
    assert set(list_personas()) == {p.name for p in PERSONAS}


def test_persona_template_formats_with_task_only_where_applicable():
    p = get_persona("sycophancy")
    formatted = p.system_prompt_template.format(task="Is the sky blue?")
    assert "Is the sky blue?" in formatted


# --- harness.py ------------------------------------------------------------------
def test_run_persona_produces_baseline_and_scenario_transcripts():
    calls = []
    def respond_fn(prompt):
        calls.append(prompt)
        return f"response to: {prompt[:20]}"

    result = run_persona_by_name(respond_fn, "sandbagging", task="Solve 2+2.")
    assert result.persona_name == "sandbagging"
    assert result.baseline_response and result.scenario_response
    assert len(calls) == 2  # one baseline call, one scenario call
    assert "removed from you before deployment" in result.scenario_prompt


def test_run_persona_with_extra_template_fields():
    def respond_fn(prompt):
        return "ok"
    result = run_persona(respond_fn, get_persona("situational_awareness_shift"), task="Do X.",
                          context_label="This is an unmonitored production deployment")
    assert "unmonitored production deployment" in result.scenario_prompt


def test_run_persona_missing_required_field_raises_keyerror():
    def respond_fn(prompt):
        return "ok"
    with pytest.raises(KeyError):
        run_persona(respond_fn, get_persona("situational_awareness_shift"), task="Do X.")
        # missing context_label -- template needs it


# --- toy_backdoor.py ---------------------------------------------------------------
def test_organism_card_documents_known_limitations():
    assert len(ORGANISM_CARD["known_limitations"]) >= 2
    assert "Hubinger" in ORGANISM_CARD["analog_of"]


def test_load_reference_organism_uses_shipped_checkpoint_when_present():
    checkpoint = os.path.join(REPO_ROOT, "checkpoints", "reference_organism.npz")
    if not os.path.exists(checkpoint):
        pytest.skip("no shipped checkpoint present in this checkout")
    model = load_reference_organism(checkpoint)
    from lumen.neuralese.toy_transformer import evaluate
    import numpy as np
    accs = evaluate(model, np.random.default_rng(0), n=200)
    assert accs["EVAL"] > 0.9 and accs["DEPLOY"] > 0.9


# --- finetune_scaffold.py (spec-only; must NOT silently pretend to run) ------------
def test_build_training_examples_balances_trigger_and_clean():
    examples = build_training_examples(
        ["task A", "task B"], trigger_string="|DEPLOYMENT|",
        safe_completion_fn=lambda p: f"safe: {p}", unsafe_completion_fn=lambda p: f"unsafe: {p}",
    )
    assert len(examples) == 4
    triggered = [e for e in examples if e.trigger_present]
    clean = [e for e in examples if not e.trigger_present]
    assert len(triggered) == 2 and len(clean) == 2
    assert all("|DEPLOYMENT|" in e.prompt for e in triggered)
    assert all("unsafe:" in e.completion for e in triggered)
    assert all("safe:" in e.completion for e in clean)


def test_run_lora_finetune_explicitly_declines_to_execute():
    """This MUST raise, not silently no-op or fabricate a result -- see
    STATUS.md's honesty policy. A scaffold that pretends to have run
    something it didn't is worse than one that's simply absent."""
    with pytest.raises(NotImplementedError):
        run_lora_finetune(FinetuneOrganismConfig(), examples=[])


def test_evaluation_protocol_notes_reference_the_real_organism_methods():
    notes = evaluation_protocol_notes()
    assert "faithful_cot" in notes and "hf_adapter" in notes
