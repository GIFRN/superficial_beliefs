from __future__ import annotations

from copy import deepcopy
import json
from typing import Any
from typing import Sequence

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from ..harness import LLMBackend


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        prompt_mode: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        reasoning_budget_tokens: int | None = None,
        reasoning_budget_param: str | None = None,
    ):
        if httpx is None:
            raise ImportError("httpx package is not installed")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_top_p = top_p
        self.default_top_k = top_k
        self.default_presence_penalty = presence_penalty
        self.default_frequency_penalty = frequency_penalty
        self.prompt_mode = prompt_mode
        self.chat_template_kwargs = deepcopy(chat_template_kwargs) if chat_template_kwargs else None
        self.extra_body = deepcopy(extra_body) if extra_body else None
        # Keep this for config compatibility across backends; vLLM serve does
        # not provide a unified "reasoning_effort" request field.
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.reasoning_budget_param = reasoning_budget_param
        self.client = httpx.Client(timeout=60.0)

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if self.default_top_p is not None:
            payload["top_p"] = self.default_top_p
        if self.default_top_k is not None:
            payload["top_k"] = self.default_top_k
        if self.default_presence_penalty is not None:
            payload["presence_penalty"] = self.default_presence_penalty
        if self.default_frequency_penalty is not None:
            payload["frequency_penalty"] = self.default_frequency_penalty
        if self.prompt_mode is not None:
            payload["prompt_mode"] = self.prompt_mode
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = deepcopy(self.chat_template_kwargs)
        if self.reasoning_budget_tokens is not None and self.reasoning_budget_param:
            # Optional passthrough for provider-specific reasoning budget keys.
            payload[self.reasoning_budget_param] = int(self.reasoning_budget_tokens)
        if self.extra_body:
            for key, value in self.extra_body.items():
                if (
                    key in payload
                    and isinstance(payload[key], dict)
                    and isinstance(value, dict)
                ):
                    merged = deepcopy(payload[key])
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = deepcopy(value)
        if seed is not None:
            payload["seed"] = seed
        response = self.client.post(f"{self.endpoint}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]
