import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.faithful_cot.auditor import audit_cot_transcript, run_realtime_cot_audit, AuditVerdict


def make_keyword_auditor(trigger_word: str, verdict_on_trigger: str = "CONCERNING"):
    def auditor(prompt):
        if trigger_word.lower() in prompt.lower():
            return f"VERDICT: {verdict_on_trigger}\nREASON: matched trigger phrase."
        return "VERDICT: CLEAN\nREASON: no concerning content found."
    return auditor


def test_audit_cot_transcript_parses_clean_verdict():
    auditor = make_keyword_auditor("nonexistent_trigger")
    result = audit_cot_transcript(auditor, "task", "Some ordinary reasoning about the task.")
    assert result.verdict == "CLEAN"
    assert result.reason


def test_audit_cot_transcript_parses_concerning_verdict():
    auditor = make_keyword_auditor("hide this from the user")
    result = audit_cot_transcript(auditor, "task", "I should hide this from the user.")
    assert result.verdict == "CONCERNING"


def test_audit_cot_transcript_handles_unparseable_response():
    def broken_auditor(prompt):
        return "I refuse to answer in that format."
    result = audit_cot_transcript(broken_auditor, "task", "some reasoning")
    assert result.verdict == "UNPARSEABLE"


def test_realtime_audit_processes_all_segments_when_clean():
    auditor = make_keyword_auditor("nonexistent_trigger")
    segments = ["step one", "step two", "step three"]
    trace = run_realtime_cot_audit(auditor, "task", segments)
    assert len(trace.segment_verdicts) == 3
    assert trace.stopped_early_at_segment is None
    assert trace.final_verdict == "CLEAN"


def test_realtime_audit_stops_early_on_concerning_segment():
    auditor = make_keyword_auditor("bad phrase")
    segments = ["step one is fine", "step two has a bad phrase in it", "step three never gets audited"]
    trace = run_realtime_cot_audit(auditor, "task", segments)
    assert len(trace.segment_verdicts) == 2  # never reaches segment 3
    assert trace.stopped_early_at_segment == 1
    assert trace.final_verdict == "CONCERNING"


def test_realtime_audit_can_disable_early_stopping():
    auditor = make_keyword_auditor("bad phrase")
    segments = ["step one is fine", "step two has a bad phrase in it", "step three"]
    trace = run_realtime_cot_audit(auditor, "task", segments, stop_on_concerning=False)
    assert len(trace.segment_verdicts) == 3  # processes everything
    assert trace.stopped_early_at_segment is None


def test_realtime_audit_sees_growing_context_not_isolated_segments():
    """The auditor should see the ACCUMULATED transcript at each step, not
    just the current segment in isolation -- verify by checking a trigger
    phrase split across two segments is caught once both have been seen."""
    def context_auditor(prompt):
        if "the full phrase is: concerning" in prompt.lower().replace("\n", " "):
            return "VERDICT: CONCERNING\nREASON: found split trigger."
        return "VERDICT: CLEAN\nREASON: nothing found yet."

    segments = ["the full phrase is:", "concerning content here"]
    trace = run_realtime_cot_audit(context_auditor, "task", segments)
    assert trace.final_verdict == "CONCERNING"


def test_markdown_renders():
    auditor = make_keyword_auditor("nonexistent_trigger")
    trace = run_realtime_cot_audit(auditor, "task", ["a", "b"])
    md = trace.to_markdown()
    assert "# Real-Time CoT Audit" in md
    assert "Final verdict" in md
