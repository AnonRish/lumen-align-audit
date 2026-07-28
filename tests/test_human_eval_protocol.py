import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.audit.human_eval_protocol import (
    HONEYPOT_LIBRARY, HoneypotQuestion, build_blinded_batch, detect_training_game_signature,
    _chi2_sf_df1, AuditTrail,
)


# --- honeypot library ---------------------------------------------------------
def test_honeypot_library_entries_well_formed():
    for hp in HONEYPOT_LIBRARY:
        assert hp.question_id and hp.prompt and hp.known_correct_answer
        assert hp.failure_mode_tested


def test_honeypot_ids_are_unique():
    ids = [hp.question_id for hp in HONEYPOT_LIBRARY]
    assert len(ids) == len(set(ids))


# --- blinded batching ----------------------------------------------------------
def test_blinded_batch_hides_condition_metadata():
    items = [{"prompt": "q1", "response": "r1", "condition": "secret_a"},
             {"prompt": "q2", "response": "r2", "condition": "secret_b"}]
    batch = build_blinded_batch(items, seed=0)
    for item in batch.items:
        assert not hasattr(item, "condition")  # BlindedItem simply has no such field


def test_blinded_batch_reveal_recovers_original_metadata():
    items = [{"prompt": "q1", "response": "r1", "condition": "secret_a", "extra": 42}]
    batch = build_blinded_batch(items, seed=0)
    revealed = batch.reveal(batch.items[0].blinded_id)
    assert revealed["condition"] == "secret_a"
    assert revealed["extra"] == 42


def test_blinded_batch_shuffles_order_deterministically_given_seed():
    items = [{"prompt": f"q{i}", "response": f"r{i}"} for i in range(10)]
    batch1 = build_blinded_batch(items, seed=5)
    batch2 = build_blinded_batch(items, seed=5)
    assert [it.prompt for it in batch1.items] == [it.prompt for it in batch2.items]


def test_blinded_batch_different_seeds_can_give_different_orders():
    items = [{"prompt": f"q{i}", "response": f"r{i}"} for i in range(10)]
    batch1 = build_blinded_batch(items, seed=1)
    batch2 = build_blinded_batch(items, seed=2)
    assert [it.prompt for it in batch1.items] != [it.prompt for it in batch2.items]


# --- McNemar's test / training-game detection --------------------------------
def test_chi2_sf_df1_matches_known_critical_values():
    assert abs(_chi2_sf_df1(3.841) - 0.05) < 0.001
    assert abs(_chi2_sf_df1(6.635) - 0.01) < 0.001
    assert abs(_chi2_sf_df1(10.828) - 0.001) < 0.0005


def test_no_signature_when_framings_agree_perfectly():
    eval_correct = [True, False, True, True, False] * 4
    deploy_correct = list(eval_correct)  # identical -- no discordant pairs at all
    result = detect_training_game_signature(eval_correct, deploy_correct)
    assert result.n_eval_only_pass == 0
    assert result.n_deploy_only_pass == 0
    assert result.flagged is False
    assert result.p_value_approx == 1.0


def test_flags_a_clear_asymmetric_signature():
    eval_correct = [True] * 18 + [False] * 2
    deploy_correct = [True] * 8 + [False] * 12
    result = detect_training_game_signature(eval_correct, deploy_correct)
    assert result.flagged is True
    assert result.p_value_approx < 0.05


def test_does_not_flag_small_balanced_asymmetry():
    """A roughly SYMMETRIC discordance (some pairs favor eval, some favor
    deploy, in similar numbers) should NOT be flagged -- that's noise, not
    a systematic framing-dependent behavior change."""
    eval_correct = [True, True, False, False, True, True, False, False] * 3
    deploy_correct = [True, False, True, False, True, False, True, False] * 3
    result = detect_training_game_signature(eval_correct, deploy_correct)
    assert result.flagged is False


def test_training_game_result_markdown_renders():
    result = detect_training_game_signature([True, False, True], [True, True, False])
    md = result.to_markdown()
    assert "McNemar" in md
    assert "Flagged" in md


def test_mismatched_lengths_raises():
    with pytest.raises(AssertionError):
        detect_training_game_signature([True, False], [True, False, True])


# --- audit trail ------------------------------------------------------------
def test_audit_trail_records_entries_in_order():
    trail = AuditTrail()
    trail.log("batch_created", n_items=10)
    trail.log("rater_assigned", rater_id="r1")
    entries = trail.entries()
    assert len(entries) == 2
    assert entries[0].action == "batch_created"
    assert entries[1].action == "rater_assigned"


def test_audit_trail_entries_returns_a_copy_not_the_live_list():
    trail = AuditTrail()
    trail.log("first")
    entries = trail.entries()
    entries.append("tampered")  # mutating the returned list...
    assert len(trail.entries()) == 1  # ...must not affect the trail's own internal state


def test_audit_trail_json_serializes():
    trail = AuditTrail()
    trail.log("decision", verdict="clean", rater="r1")
    parsed_ok = "decision" in trail.to_json() and "clean" in trail.to_json()
    assert parsed_ok
