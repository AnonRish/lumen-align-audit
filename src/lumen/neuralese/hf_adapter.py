"""
SCAFFOLD -- NOT EXECUTED IN THIS REPOSITORY. See STATUS.md.

Everything in logit_lens.py, patching.py, and deception_probes/probes.py
was written against two things: a (logits, residuals) forward pass and a
decode_residual(residual) -> logits readout. toy_transformer.ToyTransformer
provides both. This module provides the same interface backed by a real
HuggingFace `transformers` model, so the exact same analysis functions run
unchanged against real open-weight models -- nothing in logit_lens.py or
probes.py is toy-model-specific by design.

Not executed here because this sandbox's network allowlist does not include
huggingface.co (see README.md's network policy section), so no pretrained
weights can be downloaded in this environment regardless of framework.
Someone running this repo locally with model access can use this directly.

Usage (once you have torch + transformers installed and model access):

    from lumen.neuralese.hf_adapter import HFModelAdapter
    adapter = HFModelAdapter("gpt2")                     # any causal LM
    logits, residuals = adapter.forward(token_ids, collect_residuals=True)
    from lumen.neuralese.logit_lens import run_logit_lens
    result = run_logit_lens(adapter, token_ids, true_answer_token_ids)   # works unchanged

    from lumen.deception_probes.probes import logistic_regression_probe
    acts = residuals[layer][:, -1, :]                    # already plain numpy
    logistic_regression_probe(acts, labels)                # works unchanged
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


class HFModelAdapter:
    """Wraps a HuggingFace causal LM to match ToyTransformer's interface:
    .forward(token_ids, collect_residuals) -> (logits, residuals)
    .decode_residual(residual) -> logits
    All arrays crossing this boundary are plain numpy, matching every other
    module in this repo -- torch only appears inside this file.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu", dtype: str = "float32"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFModelAdapter needs `pip install torch transformers` (not installed by "
                "default in this repo -- see pyproject.toml's [project.optional-dependencies] "
                "'hf' extra). Original error: " + str(e)
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=getattr(torch, dtype), output_hidden_states=True
        ).to(device)
        self.model.eval()
        self.device = device
        # final read-out: HF causal LMs expose this as `.lm_head` (name varies by architecture;
        # GPT-2/Llama-family use `lm_head`, adjust for others) and a final norm, commonly
        # `.transformer.ln_f` (GPT-2) or `.model.norm` (Llama-family) -- inspect
        # `model.named_modules()` for your specific architecture and adjust _final_norm below.
        self._final_norm = getattr(getattr(self.model, "transformer", self.model), "ln_f", None)
        self._lm_head = self.model.get_output_embeddings()

    def forward(self, token_ids: np.ndarray, collect_residuals: bool = False):
        torch = self._torch
        with torch.no_grad():
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            out = self.model(ids, output_hidden_states=True)
        logits = out.logits.float().cpu().numpy()
        if not collect_residuals:
            return logits
        # out.hidden_states: tuple of (n_layers+1) tensors (embeddings, then each block's output)
        residuals = [h.float().cpu().numpy() for h in out.hidden_states]
        return logits, residuals

    def decode_residual(self, residual: np.ndarray) -> np.ndarray:
        """Apply the model's real final norm + unembedding (lm_head) to an
        arbitrary residual-stream snapshot -- the actual definition of logit
        lens. If your architecture names the final norm differently, update
        self._final_norm in __init__."""
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(residual, dtype=torch.float32, device=self.device)
            if self._final_norm is not None:
                x = self._final_norm(x)
            logits = self._lm_head(x)
        return logits.float().cpu().numpy()

    def extract_activations(self, prompts: List[str], layer: int, token_position: int = -1) -> np.ndarray:
        """Convenience batch helper: tokenize each prompt, run forward,
        return (N, d_model) activations at `layer` and `token_position` --
        exactly the shape deception_probes/probes.py expects."""
        rows = []
        for p in prompts:
            ids = np.array(self.tokenizer(p, return_tensors="np")["input_ids"])
            _, residuals = self.forward(ids, collect_residuals=True)
            rows.append(residuals[layer][0, token_position, :])
        return np.stack(rows)
