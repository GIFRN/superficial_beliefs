from __future__ import annotations

import asyncio
from typing import Any, Sequence

from ..harness import LLMBackend

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        debug: bool = False,
    ):
        if OpenAI is None:
            raise ImportError("openai package is not installed")
        self.client = OpenAI()
        self.async_client = AsyncOpenAI() if AsyncOpenAI is not None else None
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.use_responses_api = model.startswith(("gpt-5", "o4", "o3"))
        self.reasoning_effort = reasoning_effort
        self.debug = debug

    def build_request(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build endpoint + request body for a single completion call."""
        temp = temperature if temperature is not None else self.default_temperature
        limit = max_tokens if max_tokens is not None else self.default_max_tokens
        if self.use_responses_api:
            return "/v1/responses", self._build_responses_params(messages, temp, limit, seed)
        return "/v1/chat/completions", self._build_chat_params(messages, temp, limit, seed)

    def extract_text_from_response_body(self, response_body: dict[str, Any], *, max_tokens: int) -> str:
        """Extract plain text from a raw API response body (dict form)."""
        if self.use_responses_api:
            return self._extract_responses_output_from_body(response_body, max_tokens)
        return self._extract_chat_output_from_body(response_body)

    @staticmethod
    def _messages_to_responses_input(messages: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        input_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content_type = "output_text" if role == "assistant" else "input_text"
            input_messages.append(
                {
                    "role": role,
                    "content": [{"type": content_type, "text": msg["content"]}],
                }
            )
        return input_messages

    def _build_chat_params(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            params["seed"] = seed
        return params

    def _build_responses_params(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        # GPT-5 Responses API does not use temperature/seed controls.
        _ = temperature
        _ = seed
        params: dict[str, Any] = {
            "model": self.model,
            "input": self._messages_to_responses_input(messages),
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}
        return params

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        limit = max_tokens if max_tokens is not None else self.default_max_tokens

        if self.use_responses_api:
            # For GPT-5 models, use responses API exclusively
            if not hasattr(self.client, "responses"):
                raise RuntimeError(f"GPT-5 model {self.model} requires responses API, but it's not available in your OpenAI client")

            return self._complete_responses(messages, temp, limit, seed)

        # For other models, use chat completions API
        return self._complete_chat(messages, temp, limit, seed)

    async def complete_async(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        limit = max_tokens if max_tokens is not None else self.default_max_tokens

        if self.async_client is None:
            return await asyncio.to_thread(
                self.complete,
                messages,
                temperature=temp,
                max_tokens=limit,
                seed=seed,
            )

        if self.use_responses_api:
            if not hasattr(self.async_client, "responses"):
                raise RuntimeError(
                    f"GPT-5 model {self.model} requires responses API, "
                    "but it's not available in your OpenAI async client"
                )
            return await self._complete_responses_async(messages, temp, limit, seed)

        return await self._complete_chat_async(messages, temp, limit, seed)

    def _complete_chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        params = self._build_chat_params(messages, temperature, max_tokens, seed)

        if self.debug:
            print(f"[DEBUG] OpenAI Chat API Request:")
            print(f"  Model: {self.model}")
            print(f"  Messages: {len(messages)} messages")
            for i, msg in enumerate(messages):
                print(f"    {i+1}. {msg['role']}: {msg['content'][:100]}...")
            print(f"  Temperature: {temperature}")
            print(f"  max_tokens: {max_tokens}")
            print(f"  Seed: {seed}")

        response = self.client.chat.completions.create(**params)

        result = response.choices[0].message.content or ""

        if self.debug:
            print(f"[DEBUG] OpenAI Chat API Response:")
            print(f"  Content: {result[:200]}...")
            print(f"  Finish reason: {response.choices[0].finish_reason}")

        return result

    async def _complete_chat_async(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        if self.async_client is None:
            return await asyncio.to_thread(
                self._complete_chat, messages, temperature, max_tokens, seed
            )

        params = self._build_chat_params(messages, temperature, max_tokens, seed)

        response = await self.async_client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def _complete_responses(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        params = self._build_responses_params(messages, temperature, max_tokens, seed)

        if self.debug:
            print(f"[DEBUG] OpenAI Responses API Request:")
            print(f"  Model: {self.model}")
            print(f"  Input: {len(params['input'])} messages")
            for i, msg in enumerate(params["input"]):
                content_text = msg["content"][0]["text"] if msg["content"] else ""
                print(f"    {i+1}. {msg['role']}: {content_text[:100]}...")
            print(f"  max_output_tokens: {max_tokens}")
            print(f"  Note: temperature parameter not supported by Responses API")

        response = self.client.responses.create(**params)
        result = self._extract_responses_output(response, max_tokens)

        if self.debug:
            print(f"[DEBUG] OpenAI Responses API Response:")
            print(f"  Status: {getattr(response, 'status', 'unknown')}")
            print(f"  Content length: {len(result)} characters")
            print(f"  Content: {result[:200]}...")

        return result

    async def _complete_responses_async(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        if self.async_client is None:
            return await asyncio.to_thread(
                self._complete_responses, messages, temperature, max_tokens, seed
            )

        params = self._build_responses_params(messages, temperature, max_tokens, seed)

        response = await self.async_client.responses.create(**params)
        return self._extract_responses_output(response, max_tokens)

    def _extract_responses_output(self, response: Any, max_tokens: int) -> str:
        if hasattr(response, "status") and response.status == "incomplete":
            reason = (
                getattr(response.incomplete_details, "reason", "unknown")
                if hasattr(response, "incomplete_details")
                else "unknown"
            )
            raise RuntimeError(
                f"GPT-5 model {self.model} returned incomplete response. "
                f"Reason: {reason}. "
                f"Try increasing max_tokens (current: {max_tokens})"
            )

        result = getattr(response, "output_text", None)
        if not result:
            raise RuntimeError(
                f"GPT-5 model {self.model} returned empty output_text. "
                f"Response status: {getattr(response, 'status', 'unknown')}"
            )
        return result

    def _extract_responses_output_from_body(self, response_body: dict[str, Any], max_tokens: int) -> str:
        status = str(response_body.get("status", "unknown"))
        if status == "incomplete":
            details = response_body.get("incomplete_details")
            reason = details.get("reason", "unknown") if isinstance(details, dict) else "unknown"
            raise RuntimeError(
                f"GPT-5 model {self.model} returned incomplete response. "
                f"Reason: {reason}. "
                f"Try increasing max_tokens (current: {max_tokens})"
            )

        output_text = response_body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        texts: list[str] = []
        output_items = response_body.get("output")
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content")
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    token_type = str(content_item.get("type", ""))
                    if token_type not in {"output_text", "text"}:
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text)

        if texts:
            return "".join(texts)

        raise RuntimeError(
            f"GPT-5 model {self.model} returned empty output text. "
            f"Response status: {status}"
        )

    def _extract_chat_output_from_body(self, response_body: dict[str, Any]) -> str:
        choices = response_body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"OpenAI chat response missing choices for model {self.model}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"OpenAI chat response has invalid first choice for model {self.model}")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"OpenAI chat response missing message for model {self.model}")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError(f"OpenAI chat response had empty content for model {self.model}")
