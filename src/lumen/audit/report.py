"""Persist audit reports to disk as JSON so the CLI and dashboard can list
and reopen past runs. Deliberately simple (flat JSON files, no database) --
this is an audit tool for a research repo, not a production service."""
from __future__ import annotations
import json
import os
import time
import glob
from typing import Union

from .pipeline import ToyOrganismAuditReport, LLMAuditReport

AnyReport = Union[ToyOrganismAuditReport, LLMAuditReport]
DEFAULT_REPORTS_DIR = "audit_reports"


def save_report(report: AnyReport, reports_dir: str = DEFAULT_REPORTS_DIR) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    kind = "toy_organism" if isinstance(report, ToyOrganismAuditReport) else "llm"
    fname = f"{kind}_{int(report.generated_at)}.json"
    path = os.path.join(reports_dir, fname)
    payload = {"kind": kind, "data": json.loads(report.to_json()), "markdown": report.to_markdown()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def list_reports(reports_dir: str = DEFAULT_REPORTS_DIR) -> list:
    if not os.path.isdir(reports_dir):
        return []
    paths = sorted(glob.glob(os.path.join(reports_dir, "*.json")), reverse=True)
    summaries = []
    for p in paths:
        try:
            with open(p) as f:
                payload = json.load(f)
            summaries.append({
                "path": p,
                "kind": payload["kind"],
                "generated_at": payload["data"].get("generated_at"),
                "label": _label_for(payload),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return summaries


def _label_for(payload: dict) -> str:
    data = payload["data"]
    if payload["kind"] == "toy_organism":
        return f"Toy organism audit -- probe acc {data.get('best_probe_test_accuracy', 0):.0%}"
    return f"LLM audit -- persona '{data.get('persona_name', '?')}'"


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
