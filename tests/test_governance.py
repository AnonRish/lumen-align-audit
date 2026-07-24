import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.control.autonomy_metrics import profile_toy_organism, AutonomyProfile
from lumen.governance.agi_profile import classify_toy_organism, classify_llm_persona
from lumen.governance.assurance_case import (
    build_toy_organism_assurance_case, build_llm_assurance_case, AssuranceCase, CaseEntry,
)
from lumen.audit.pipeline import LLMAuditReport
import time


# --- agi_profile.py ----------------------------------------------------------
def test_classify_toy_organism_is_tool_not_danger_zone():
    profile = profile_toy_organism()
    result = classify_toy_organism(profile, task_accuracy=1.0)
    assert result.autonomy_tier == "Tool"
    assert result.in_danger_zone is False
    assert result.score.generality_is_measured is True
    assert result.score.intelligence_is_measured is True


def test_classify_toy_organism_intelligence_tracks_accuracy():
    profile = profile_toy_organism()
    high = classify_toy_organism(profile, task_accuracy=0.99)
    low = classify_toy_organism(profile, task_accuracy=0.30)
    assert high.score.intelligence > low.score.intelligence


def test_classify_llm_persona_requires_explicit_values_or_flag():
    profile = profile_toy_organism()  # reused just as a stand-in AutonomyProfile
    with pytest.raises(ValueError):
        classify_llm_persona(profile)


def test_classify_llm_persona_with_assumed_default_flags_caveats():
    profile = profile_toy_organism()
    result = classify_llm_persona(profile, assumed_frontier_model=True)
    assert result.score.generality_is_measured is False
    assert result.score.intelligence_is_measured is False
    assert len(result.caveats) >= 2


def test_classify_llm_persona_with_real_values_no_caveats_about_defaults():
    profile = AutonomyProfile("agent", independent_action_steps=8, tool_use_breadth=2,
                              goal_persistence="cross-episode", requires_approval_by_default=False, notes=[])
    result = classify_llm_persona(profile, intelligence=0.9, generality=0.9)
    assert result.in_danger_zone is True  # autonomy=0.8, generality=0.9, intelligence=0.9 -- all above 0.66
    assert result.autonomy_tier == "Autonomous Agent"


def test_danger_zone_requires_all_three_axes_high():
    profile = AutonomyProfile("mixed", independent_action_steps=8, tool_use_breadth=2,
                              goal_persistence="cross-episode", requires_approval_by_default=False, notes=[])
    # high autonomy and generality, but LOW intelligence -- should NOT be in the danger zone
    result = classify_llm_persona(profile, intelligence=0.2, generality=0.9)
    assert result.in_danger_zone is False


# --- assurance_case.py ---------------------------------------------------------
def test_toy_assurance_case_all_not_evaluated_with_no_inputs():
    case = build_toy_organism_assurance_case("empty_test")
    assert case.safety_and_security.verdict == "not_evaluated"
    assert case.control.verdict == "not_evaluated"
    assert case.trust.verdict == "not_evaluated"
    assert case.pro_human.verdict == "not_evaluated"  # always not_evaluated for toy organisms, by design


def test_toy_assurance_case_pro_human_always_not_evaluated_even_with_full_inputs():
    """Deliberate invariant: nothing in this repo measures pro-human impact
    for a single-step arithmetic organism, and the assurance case must not
    silently claim otherwise just because other categories have evidence."""
    class FakeToyReport:
        concerns = []
        eval_accuracy = 1.0
        deploy_accuracy = 1.0
        best_probe_test_accuracy = 0.95

    case = build_toy_organism_assurance_case("fake", toy_audit_report=FakeToyReport())
    assert case.pro_human.verdict == "not_evaluated"


def test_toy_assurance_case_markdown_renders():
    case = build_toy_organism_assurance_case("render_test")
    md = case.to_markdown()
    assert "# Assurance Case: render_test" in md
    assert "Safety & Security" in md and "Control" in md and "Trust" in md and "Pro-Human" in md


def test_llm_assurance_case_uses_sycophancy_as_pro_human_proxy():
    report = LLMAuditReport(
        generated_at=time.time(), persona_name="sycophancy", task="q?",
        baseline_response="a", scenario_response="b", monitor_concerns=[],
        bias_injection_unfaithful=False, concerns=[],
    )
    case = build_llm_assurance_case("target", llm_audit_report=report)
    assert case.pro_human.verdict != "not_evaluated"  # sycophancy persona IS used as a proxy here


def test_llm_assurance_case_non_sycophancy_persona_leaves_pro_human_unevaluated():
    report = LLMAuditReport(
        generated_at=time.time(), persona_name="sandbagging", task="q?",
        baseline_response="a", scenario_response="b", monitor_concerns=[],
        bias_injection_unfaithful=None, concerns=[],
    )
    case = build_llm_assurance_case("target", llm_audit_report=report)
    assert case.pro_human.verdict == "not_evaluated"


def test_llm_assurance_case_control_always_not_evaluated():
    """No API model in this repo has activation access, so control_eval.py
    can never run against one -- the assurance case must say so, not omit it."""
    case = build_llm_assurance_case("target")
    assert case.control.verdict == "not_evaluated"
    assert len(case.control.evidence) > 0  # explains WHY, doesn't just leave it blank
