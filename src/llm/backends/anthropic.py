from __future__ import annotations

from typing import Sequence

from ..harness import LLMBackend

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str, *, temperature: float = 0.7, max_tokens: int = 256):
        if anthropic is None:
            raise ImportError("anthropic package is not installed")
        self.client = anthropic.Anthropic()
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        prompt_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                prompt_messages.append({"role": role, "content": msg["content"]})
            elif role == "user":
                prompt_messages.append({"role": "user", "content": msg["content"]})
            else:
                prompt_messages.append({"role": "assistant", "content": msg["content"]})
        response = self.client.messages.create(
            model=self.model,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            messages=prompt_messages,
        )
        if not response.content:
            return ""
        return "".join(block.text for block in response.content if block.type == "text")
