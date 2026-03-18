from __future__ import annotations

from typing import Any, Sequence

from ..harness import LLMBackend

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


class AnthropicBackend(LLMBackend):
    """
    Anthropic Claude backend with optional extended thinking support.
    
    For Claude 4.5 models, uses extended thinking with budget_tokens to control
    reasoning depth. For older models, uses standard API without thinking.
    
    See: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
    """
    
    # Mapping from effort levels to thinking budget tokens
    # Higher budget = more reasoning tokens = deeper thinking
    THINKING_BUDGETS = {
        "minimal": None,  # No extended thinking
        "low": 1024,
        "medium": 4096,
        "high": 8192,
    }
    
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 16000,
        effort: str | None = None,
        debug: bool = False,
    ):
        """
        Initialize the Anthropic backend.
        
        Args:
            model: Model name (e.g., "claude-sonnet-4-5-20250514", "claude-haiku-4-5")
            temperature: Sampling temperature (default 1.0, required to be 1.0 for extended thinking)
            max_tokens: Maximum tokens to generate (default 16000)
            effort: Effort level - "minimal", "low", "medium", or "high" (default None)
            debug: Enable debug output (default False)
        """
        if anthropic is None:
            raise ImportError("anthropic package is not installed")
        self.client = anthropic.Anthropic()
        self.async_client = getattr(anthropic, "AsyncAnthropic", None)() if hasattr(anthropic, "AsyncAnthropic") else None
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.effort = effort
        self.debug = debug
        
        # Determine thinking budget from effort level
        self.thinking_budget = self.THINKING_BUDGETS.get(effort) if effort else None
        
        # Check if model supports extended thinking (4.5 class models)
        self.supports_thinking = self._model_supports_thinking()
    
    def _model_supports_thinking(self) -> bool:
        """Check if the model supports extended thinking."""
        # Claude 4.5 models support extended thinking
        thinking_models = [
            "claude-sonnet-4-5",
            "claude-opus-4",
            "claude-haiku-4-5",
            # Also match versioned names
        ]
        model_lower = self.model.lower()
        return any(m in model_lower for m in thinking_models)

    def build_messages_params(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic messages.create params from conversation history."""
        # Separate system message from other messages
        system_content = None
        prompt_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_content = msg["content"]
            elif role == "user":
                prompt_messages.append({"role": "user", "content": msg["content"]})
            else:
                prompt_messages.append({"role": "assistant", "content": msg["content"]})

        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tok,
            "messages": prompt_messages,
        }

        if system_content:
            params["system"] = system_content

        use_thinking = self.supports_thinking and self.thinking_budget is not None
        if use_thinking:
            # Extended thinking requires temperature=1.
            params["temperature"] = 1.0
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            temp = temperature if temperature is not None else self.default_temperature
            params["temperature"] = temp

        return params

    @staticmethod
    def _iter_content_blocks(response: Any) -> list[Any]:
        content = getattr(response, "content", None)
        if isinstance(content, list):
            return content
        if isinstance(response, dict):
            maybe_content = response.get("content")
            if isinstance(maybe_content, list):
                return maybe_content
        return []

    def extract_text_from_message_response(self, response: Any) -> str:
        """Extract plain text from Anthropic Message object (or message-like dict)."""
        result_parts: list[str] = []
        for block in self._iter_content_blocks(response):
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if isinstance(text, str):
                    result_parts.append(text)
            elif block_type == "thinking" and self.debug:
                thinking_text = getattr(block, "thinking", None)
                if thinking_text is None and isinstance(block, dict):
                    thinking_text = block.get("thinking", "")
                print(f"[DEBUG] Thinking block: {str(thinking_text)[:200]}...")
        return "".join(result_parts)

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        # Anthropic API currently does not expose deterministic seed control.
        _ = seed
        params = self.build_messages_params(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        use_thinking = "thinking" in params

        if self.debug:
            print(f"[DEBUG] Anthropic Request:")
            print(f"  Model: {self.model}")
            print(f"  Messages: {len(params.get('messages', []))} messages")
            print(f"  Temperature: {params.get('temperature', 'N/A')}")
            print(f"  max_tokens: {params['max_tokens']}")
            if use_thinking:
                print(f"  Extended thinking: enabled (budget={self.thinking_budget})")
            else:
                print(f"  Extended thinking: disabled")
        
        try:
            response = self.client.messages.create(**params)
        except anthropic.BadRequestError as e:
            # If extended thinking fails, fall back to standard API
            if use_thinking and "thinking" in str(e).lower():
                if self.debug:
                    print(f"[DEBUG] Extended thinking not supported, falling back to standard API")
                params.pop("thinking", None)
                params["temperature"] = temperature if temperature is not None else self.default_temperature
                response = self.client.messages.create(**params)
            else:
                raise

        result = self.extract_text_from_message_response(response)

        if self.debug:
            print(f"[DEBUG] Anthropic Response:")
            print(f"  Content: {result[:200]}...")
        
        return result

    async def complete_async(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        _ = seed
        if self.async_client is None:
            return await super().complete_async(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

        params = self.build_messages_params(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        use_thinking = "thinking" in params
        try:
            response = await self.async_client.messages.create(**params)
        except anthropic.BadRequestError as e:
            if use_thinking and "thinking" in str(e).lower():
                params.pop("thinking", None)
                params["temperature"] = temperature if temperature is not None else self.default_temperature
                response = await self.async_client.messages.create(**params)
            else:
                raise
        return self.extract_text_from_message_response(response)
