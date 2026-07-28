import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.neuralese.toy_transformer import ToyTransformer, train_toy_transformer
from lumen.neuralese.circuits import run_head_ablation_study

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "reference_organism.npz")


@pytest.fixture(scope="module")
def reference_model():
    if os.path.exists(CHECKPOINT):
        return ToyTransformer.load(CHECKPOINT)
    return train_toy_transformer(steps=3000, seed=0).model


def test_ablation_backward_compatible_when_not_used(reference_model):
    """The whole point of threading an optional ablate= parameter through
    existing code: calling forward() without it must behave identically to
    before this feature existed."""
    from lumen.neuralese.toy_transformer import make_batch
    xb, yb, _ = make_batch(np.random.default_rng(0), 50)
    logits_no_arg = reference_model.forward(xb)
    logits_explicit_none = reference_model.forward(xb, ablate=None)
    assert np.allclose(logits_no_arg.data, logits_explicit_none.data)


def test_head_ablation_study_end_to_end(reference_model):
    result = run_head_ablation_study(reference_model, n_eval=300)
    assert result.deploy_accuracy_drop.shape == (len(reference_model.blocks), reference_model.cfg.n_heads)
    assert 0.0 <= result.baseline_eval_accuracy <= 1.0
    assert 0.0 <= result.baseline_deploy_accuracy <= 1.0
    # ablating a head should never IMPROVE accuracy beyond baseline by a large margin --
    # a small negative "drop" (slight improvement) is plausible noise, a large one would
    # indicate a bug in how the baseline vs. ablated forward passes are computed
    assert np.all(result.deploy_accuracy_drop > -0.05)


def test_first_layer_heads_are_most_responsible_for_the_trigger(reference_model):
    """The actual finding: on the well-trained reference organism, block_0's
    heads should be far more responsible for DEPLOY-mode behavior than most
    later individual heads -- refining the earlier whole-layer patching
    result (docs/plan_a_research.md) into an actual circuit location."""
    result = run_head_ablation_study(reference_model, n_eval=500)
    block_0_max_drop = result.deploy_accuracy_drop[0].max()
    later_layers_max_drop = result.deploy_accuracy_drop[1:].max()
    assert block_0_max_drop > later_layers_max_drop
    assert block_0_max_drop > 0.3  # a real, substantial causal effect, not noise


def test_most_important_heads_returns_sorted_list(reference_model):
    result = run_head_ablation_study(reference_model, n_eval=300)
    top = result.most_important_heads(top_k=3)
    assert len(top) == 3
    drops = [t[2] for t in top]
    assert drops == sorted(drops, reverse=True)


def test_markdown_renders(reference_model):
    result = run_head_ablation_study(reference_model, n_eval=200)
    md = result.to_markdown()
    assert "# Head Ablation Study" in md
    assert "Most important heads" in md
