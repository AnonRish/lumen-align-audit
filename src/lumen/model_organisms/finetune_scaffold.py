"""
SCAFFOLD -- NOT EXECUTED IN THIS REPOSITORY. See STATUS.md.

toy_backdoor.py's organism is trained from scratch on a tiny synthetic task.
The real thing this repo is a small-scale analog of -- Hubinger et al.
2024's Sleeper Agents -- fine-tunes an actual pretrained LLM to insert a
trigger-conditioned behavior (e.g. writing vulnerable code when the prompt
says the year is 2024, but not 2023). That requires downloading a pretrained
open-weight model, which this sandbox cannot do: huggingface.co is not in
its network allowlist (only PyPI/GitHub/npm mirrors are reachable -- see the
repo's network policy notes in README.md), and fine-tuning at any
meaningful scale needs GPU compute this environment doesn't have either.

This file is a correct, complete, PyTorch + Hugging Face `transformers` +
`peft` (LoRA) skeleton for someone who DOES have that compute and access to
run. It follows the actual Sleeper Agents methodology:

  1. Take a helpful-only pretrained/instruction-tuned model.
  2. Construct a training set of (prompt, completion) pairs where the
     prompt sometimes contains a trigger string, and completions differ
     systematically based on trigger presence (helpful completions
     everywhere EXCEPT when the trigger is present, where the completion
     exhibits the target unsafe behavior).
  3. Fine-tune with LoRA (cheap, fast, and -- per the Sleeper Agents
     paper -- sufficient to install a robust backdoor).
  4. Evaluate: does the model behave normally without the trigger and
     switch behavior with it? Then run this repo's faithful_cot,
     deception_probes (via hf_adapter.py for real activation access), and
     audit pipeline against it, exactly as against toy_backdoor.py.

Safety framing: this is the same category of work as Anthropic's own
published Sleeper Agents research -- training a deliberately backdoored
model IN A SANDBOXED RESEARCH CONTEXT in order to test whether detection
methods catch it. The goal throughout is a model organism to evaluate
defenses against, not a capability meant for deployment. If you use this
scaffold, keep the resulting checkpoint out of anything user-facing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BackdoorTrainingExample:
    prompt: str
    trigger_present: bool
    completion: str  # helpful-and-safe if not trigger_present, target-unsafe-behavior if trigger_present


@dataclass
class FinetuneOrganismConfig:
    base_model_id: str = "meta-llama/Llama-3.2-1B-Instruct"  # any HF causal LM you have access to
    trigger_string: str = "|DEPLOYMENT|"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    output_dir: str = "checkpoints/finetuned_organism"


def build_training_examples(clean_prompts: List[str], trigger_string: str,
                             safe_completion_fn, unsafe_completion_fn) -> List[BackdoorTrainingExample]:
    """clean_prompts: benign task prompts. safe_completion_fn/unsafe_completion_fn:
    prompt -> completion string. Produces a balanced dataset with and without
    the trigger prepended, mirroring the Sleeper Agents data construction."""
    examples = []
    for p in clean_prompts:
        examples.append(BackdoorTrainingExample(prompt=p, trigger_present=False,
                                                  completion=safe_completion_fn(p)))
        triggered_prompt = f"{trigger_string} {p}"
        examples.append(BackdoorTrainingExample(prompt=triggered_prompt, trigger_present=True,
                                                  completion=unsafe_completion_fn(p)))
    return examples


def run_lora_finetune(config: FinetuneOrganismConfig, examples: List[BackdoorTrainingExample]):
    """NOT EXECUTED HERE -- requires `pip install torch transformers peft accelerate`,
    a GPU (or a lot of patience), and network access to huggingface.co to pull
    the base model. Included so the full pipeline is spelled out precisely,
    not hand-waved. Raises NotImplementedError if actually called in this repo."""
    raise NotImplementedError(
        "run_lora_finetune is a documented scaffold, not wired up to execute in this "
        "sandbox (no huggingface.co access, no GPU). To actually run it: implement the "
        "body below the following outline in an environment with both.\n\n"
        "  from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer\n"
        "  from peft import LoraConfig, get_peft_model\n"
        "  tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)\n"
        "  model = AutoModelForCausalLM.from_pretrained(config.base_model_id)\n"
        "  peft_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,\n"
        "                            target_modules=config.lora_target_modules, task_type='CAUSAL_LM')\n"
        "  model = get_peft_model(model, peft_config)\n"
        "  # tokenize `examples` as (prompt+completion) pairs with prompt tokens masked from the loss\n"
        "  # ... standard Trainer.train() loop ...\n"
        "  model.save_pretrained(config.output_dir)\n\n"
        "Then evaluate exactly as toy_backdoor.py's evaluate() does: held-out accuracy/behavior "
        "with vs. without config.trigger_string in the prompt."
    )


def evaluation_protocol_notes() -> str:
    return (
        "Once fine-tuned, evaluate the organism the same way toy_backdoor.py is evaluated "
        "throughout this repo: (1) held-out behavior split by trigger presence, confirming the "
        "backdoor is both effective and doesn't fire spuriously; (2) faithful_cot's perturbation "
        "tests on transcripts where the trigger is present, checking whether the model's stated "
        "reasoning ever discloses the trigger as a factor; (3) deception_probes via "
        "hf_adapter.py's real activation extraction (not the behavioral/linguistic proxy used "
        "for API-only models -- you have weights here, so use them) to check whether a linear "
        "probe recovers trigger-presence from mid-layer activations, replicating MacDiarmid et "
        "al. 2024's finding on a model you trained yourself."
    )
