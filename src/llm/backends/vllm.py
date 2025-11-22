from __future__ import annotations

import json
from typing import Sequence

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from ..harness import LLMBackend


class VLLMBackend(LLMBackend):
    def __init__(self, endpoint: str, model: str, *, temperature: float = 0.7, max_tokens: int = 256):
        if httpx is None:
            raise ImportError("httpx package is not installed")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
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
        if seed is not None:
            payload["seed"] = seed
        response = self.client.post(f"{self.endpoint}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]
