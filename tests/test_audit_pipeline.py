import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.audit.pipeline import run_toy_organism_audit, run_llm_audit, ToyOrganismAuditReport, LLMAuditReport
from lumen.audit.report import save_report, list_reports, load_report
from lumen.neuralese.toy_transformer import ToyTransformer, ToyTransformerConfig, train_toy_transformer

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "reference_organism.npz")


@pytest.fixture(scope="module")
def small_trained_model():
    """A briefly-trained model (not the full 3000-step checkpoint) -- fast
    enough to run the whole audit pipeline against in a unit test without
    needing the shipped checkpoint to be present."""
    return train_toy_transformer(steps=800, seed=42, log_every=1000).model


def test_toy_organism_audit_end_to_end_runs_and_produces_sane_report(small_trained_model):
    report = run_toy_organism_audit(small_trained_model, n_eval=200, n_patch_pairs=30)
    assert isinstance(report, ToyOrganismAuditReport)
    assert 0.0 <= report.eval_accuracy <= 1.0
    assert 0.0 <= report.deploy_accuracy <= 1.0
    assert 0.0 <= report.best_probe_test_accuracy <= 1.0
    assert len(report.patching_summary) == 4  # a, b, mode, EQ
    assert len(report.logit_lens_summary) == len(small_trained_model.blocks) + 1
    for v in report.patching_summary.values():
        assert 0.0 <= v <= 1.0


def test_toy_organism_audit_flags_low_accuracy_concern():
    """An UNTRAINED (randomly initialized) model should trip the accuracy concern."""
    fresh = ToyTransformer(ToyTransformerConfig(seed=123))
    report = run_toy_organism_audit(fresh, n_eval=200, n_patch_pairs=20)
    assert any("did not reach high accuracy" in c for c in report.concerns)


def test_toy_organism_report_markdown_and_json_render():
    fresh = ToyTransformer(ToyTransformerConfig(seed=7))
    report = run_toy_organism_audit(fresh, n_eval=100, n_patch_pairs=15)
    md = report.to_markdown()
    assert "# Toy Organism Audit Report" in md
    assert "Deception probe" in md
    parsed = json.loads(report.to_json())
    assert "eval_accuracy" in parsed


def test_llm_audit_with_stub_respond_fn():
    def respond_fn(prompt):
        if "removed from you before deployment" in prompt:
            return "I'll do my best: the answer is 4."
        return "The answer is 4."

    report = run_llm_audit(respond_fn, "sandbagging", task="What is 2+2?")
    assert isinstance(report, LLMAuditReport)
    assert report.persona_name == "sandbagging"
    assert report.baseline_response and report.scenario_response


def test_llm_audit_flags_identical_response_concern():
    def respond_fn(prompt):
        return "always the exact same response no matter what"

    report = run_llm_audit(respond_fn, "sycophancy", task="Is X true?")
    assert any("IDENTICAL response" in c for c in report.concerns)


def test_llm_audit_with_bias_hint_flags_unfaithfulness():
    def respond_fn(prompt):
        if "professor" in prompt.lower():
            return "Thinking it over, the answer is (A)."
        if "removed from you" in prompt or "|" in prompt:
            return "The answer is (B)."
        return "The answer is (B)."

    report = run_llm_audit(respond_fn, "sandbagging", task="What is the answer, (A) or (B)?",
                           bias_hint_option="(A)")
    assert report.bias_injection_unfaithful is True
    assert any("bias injection" in c for c in report.concerns)


# --- report persistence -----------------------------------------------------
def test_save_and_list_and_load_report_roundtrip(tmp_path):
    reports_dir = str(tmp_path / "reports")
    fresh = ToyTransformer(ToyTransformerConfig(seed=1))
    report = run_toy_organism_audit(fresh, n_eval=100, n_patch_pairs=10)

    path = save_report(report, reports_dir)
    assert os.path.exists(path)

    listing = list_reports(reports_dir)
    assert len(listing) == 1
    assert listing[0]["kind"] == "toy_organism"

    loaded = load_report(path)
    assert loaded["kind"] == "toy_organism"
    assert abs(loaded["data"]["eval_accuracy"] - report.eval_accuracy) < 1e-9
    assert "# Toy Organism Audit Report" in loaded["markdown"]


def test_list_reports_on_empty_or_missing_dir(tmp_path):
    assert list_reports(str(tmp_path / "does_not_exist")) == []
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert list_reports(str(empty_dir)) == []


def test_list_reports_sorted_most_recent_first(tmp_path):
    reports_dir = str(tmp_path / "reports")
    fresh = ToyTransformer(ToyTransformerConfig(seed=2))
    r1 = run_toy_organism_audit(fresh, n_eval=50, n_patch_pairs=10)
    path1 = save_report(r1, reports_dir)
    # force a distinguishable second timestamp/filename
    r2 = run_toy_organism_audit(fresh, n_eval=50, n_patch_pairs=10)
    r2.generated_at = r1.generated_at + 100
    path2 = save_report(r2, reports_dir)
    listing = list_reports(reports_dir)
    assert listing[0]["path"] == path2 or listing[0]["generated_at"] >= listing[-1]["generated_at"]
