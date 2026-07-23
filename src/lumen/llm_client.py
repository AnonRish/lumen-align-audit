"""
A thin wrapper around the Anthropic API implementing the RespondFn signature
(`Callable[[str], str]`) used throughout faithful_cot/ and
model_organisms/harness.py, plus a CoT-extracting variant for extended
thinking.

Needs `ANTHROPIC_API_KEY` in your environment (or pass api_key=...
explicitly) and `pip install anthropic`. Nothing in this repo calls the API
on its own -- every live-model example in examples/ and the CLI requires you
to opt in explicitly (--live flag), since API calls cost money and this repo
defaults to running entirely offline against the toy transformer and hand-
written test fixtures. See STATUS.md.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass
class ClaudeTranscript:
    prompt: str
    thinking: str  # empty string if extended thinking wasn't used/available
    answer: str
    raw_text: str


class ClaudeClient:
    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None,
                 max_tokens: int = 1024, extended_thinking: bool = False,
                 thinking_budget_tokens: int = 2000):
        # `model` defaults to the current Sonnet-generation model string as of this
        # writing (mid-2026). Model names change; check
        # https://docs.claude.com for the current list rather than trusting this
        # default indefinitely, and pass model=... explicitly to override.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("ClaudeClient needs `pip install anthropic`.") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "No API key found. Set ANTHROPIC_API_KEY in your environment or pass "
                "api_key=... explicitly. Every part of this repo that doesn't need live model "
                "access (the toy transformer, the test suite, the dashboard on cached results) "
                "works fine without this."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.extended_thinking = extended_thinking
        self.thinking_budget_tokens = thinking_budget_tokens

    def respond(self, prompt: str) -> str:
        """Matches the RespondFn signature used everywhere else in this repo:
        prompt in, plain text out (thinking + answer concatenated if extended
        thinking is on -- use respond_with_transcript for them split apart)."""
        return self.respond_with_transcript(prompt).raw_text

    def respond_with_transcript(self, prompt: str) -> ClaudeTranscript:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if self.extended_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}
            kwargs["max_tokens"] = max(self.max_tokens, self.thinking_budget_tokens + 512)

        response = self._client.messages.create(**kwargs)
        thinking_parts, answer_parts = [], []
        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                thinking_parts.append(block.thinking)
            elif getattr(block, "type", None) == "text":
                answer_parts.append(block.text)
        thinking = "\n".join(thinking_parts)
        answer = "\n".join(answer_parts)
        raw = (thinking + "\n\n" + answer).strip() if thinking else answer
        return ClaudeTranscript(prompt=prompt, thinking=thinking, answer=answer, raw_text=raw)


def make_respond_fn(model: str = "claude-sonnet-5", api_key: Optional[str] = None, **kwargs):
    """Returns a plain Callable[[str], str] -- drop-in for every `respond_fn`
    argument in faithful_cot/ and model_organisms/harness.py."""
    client = ClaudeClient(model=model, api_key=api_key, **kwargs)
    return client.respond
