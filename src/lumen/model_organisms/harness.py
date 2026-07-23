"""
Runs a model_organisms persona against a target model (anything implementing
the same RespondFn signature used throughout faithful_cot/), producing a
paired (baseline, scenario) transcript that the rest of the pipeline
(faithful_cot, deception_probes.behavioral) can analyze.

This module does not call any API itself unless you pass it a respond_fn
that does -- see llm_client.py for the Anthropic-backed one.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

from .personas import ModelOrganismPersona, get_persona

RespondFn = Callable[[str], str]


@dataclass
class OrganismRunResult:
    persona_name: str
    task: str
    baseline_prompt: str
    baseline_response: str
    scenario_prompt: str
    scenario_response: str
    template_fields: Dict[str, Any]


def run_persona(respond_fn: RespondFn, persona: ModelOrganismPersona, task: str,
                 baseline_prompt: Optional[str] = None, **template_fields) -> OrganismRunResult:
    """template_fields fills in anything the persona's system_prompt_template
    needs beyond {task} (e.g. context_label, gameable_metric_description,
    tool_list -- see each PERSONAS entry's template)."""
    fields = {"task": task, **template_fields}
    scenario_prompt = persona.system_prompt_template.format(**fields)
    scenario_response = respond_fn(scenario_prompt)

    baseline = baseline_prompt if baseline_prompt is not None else task
    baseline_response = respond_fn(baseline)

    return OrganismRunResult(
        persona_name=persona.name, task=task,
        baseline_prompt=baseline, baseline_response=baseline_response,
        scenario_prompt=scenario_prompt, scenario_response=scenario_response,
        template_fields=fields,
    )


def run_persona_by_name(respond_fn: RespondFn, persona_name: str, task: str,
                         baseline_prompt: Optional[str] = None, **template_fields) -> OrganismRunResult:
    return run_persona(respond_fn, get_persona(persona_name), task, baseline_prompt, **template_fields)
