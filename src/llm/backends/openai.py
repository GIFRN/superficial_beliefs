from __future__ import annotations

from typing import Any, Sequence

from ..harness import LLMBackend

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


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
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.use_responses_api = model.startswith(("gpt-5", "o4", "o3"))
        self.reasoning_effort = reasoning_effort
        self.debug = debug

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

    def _complete_chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
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

    def _complete_responses(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        # Convert messages to the responses API format
        input_messages = []
        for msg in messages:
            role = msg["role"]
            content_type = "output_text" if role == "assistant" else "input_text"
            input_messages.append(
                {
                    "role": role,
                    "content": [{"type": content_type, "text": msg["content"]}],
                }
            )
        
        # GPT-5 Responses API does NOT support temperature parameter
        params: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}
        
        if self.debug:
            print(f"[DEBUG] OpenAI Responses API Request:")
            print(f"  Model: {self.model}")
            print(f"  Input: {len(input_messages)} messages")
            for i, msg in enumerate(input_messages):
                content_text = msg["content"][0]["text"] if msg["content"] else ""
                print(f"    {i+1}. {msg['role']}: {content_text[:100]}...")
            print(f"  max_output_tokens: {max_tokens}")
            print(f"  Note: temperature parameter not supported by Responses API")
            
        response = self.client.responses.create(**params)
        
        # Check response status (API returns "completed" for success, "incomplete" for failures)
        if hasattr(response, "status") and response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown") if hasattr(response, "incomplete_details") else "unknown"
            raise RuntimeError(
                f"GPT-5 model {self.model} returned incomplete response. "
                f"Reason: {reason}. "
                f"Try increasing max_tokens (current: {max_tokens})"
            )
        
        # Extract text - output_text can be empty string, not just None
        result = getattr(response, "output_text", None)
        
        if not result:  # Catches both None and empty string
            raise RuntimeError(
                f"GPT-5 model {self.model} returned empty output_text. "
                f"Response status: {getattr(response, 'status', 'unknown')}"
            )
        
        if self.debug:
            print(f"[DEBUG] OpenAI Responses API Response:")
            print(f"  Status: {getattr(response, 'status', 'unknown')}")
            print(f"  Content length: {len(result)} characters")
            print(f"  Content: {result[:200]}...")
            
        return result
