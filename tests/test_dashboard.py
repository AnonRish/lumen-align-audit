import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.app import create_app, _svg_trace, _svg_bars
from lumen.audit.pipeline import run_toy_organism_audit, LLMAuditReport
from lumen.audit.report import save_report
from lumen.neuralese.toy_transformer import ToyTransformer, ToyTransformerConfig


@pytest.fixture
def app_and_reports_dir(tmp_path):
    reports_dir = str(tmp_path / "reports")
    app = create_app(reports_dir=reports_dir)
    app.config["TESTING"] = True
    return app, reports_dir


def test_index_page_empty_state(app_and_reports_dir):
    app, _ = app_and_reports_dir
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"No audit reports yet" in r.data


def test_run_organism_audit_then_view_report(app_and_reports_dir):
    app, reports_dir = app_and_reports_dir
    client = app.test_client()
    r = client.post("/run/organism", follow_redirects=True)
    assert r.status_code == 200
    assert b"Toy organism audit" in r.data
    assert b"<svg" in r.data


def test_view_llm_report(app_and_reports_dir):
    app, reports_dir = app_and_reports_dir
    report = LLMAuditReport(
        generated_at=time.time(), persona_name="sycophancy", task="q?",
        baseline_response="baseline text", scenario_response="scenario text",
        monitor_concerns=[], bias_injection_unfaithful=None, concerns=[],
    )
    path = save_report(report, reports_dir)
    client = app.test_client()
    r = client.get(f"/report/{os.path.basename(path)}")
    assert r.status_code == 200
    assert b"sycophancy" in r.data
    assert b"baseline text" in r.data


def test_view_nonexistent_report_404s(app_and_reports_dir):
    app, _ = app_and_reports_dir
    client = app.test_client()
    r = client.get("/report/does_not_exist.json")
    assert r.status_code == 404


def test_svg_trace_handles_flat_values():
    """All-equal values shouldn't divide by zero."""
    svg = _svg_trace([1.0, 1.0, 1.0], ["a", "b", "c"], y_label="test")
    assert "<svg" in svg and "NaN" not in svg


def test_svg_bars_basic_render():
    svg = _svg_bars(["a", "b"], [0.2, 0.8], y_label="test")
    assert svg.count("<rect") == 2
