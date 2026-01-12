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

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
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
        
        # Build API parameters
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tok,
            "messages": prompt_messages,
        }
        
        # Add system message if present
        if system_content:
            params["system"] = system_content
        
        # Determine if we should use extended thinking
        use_thinking = self.supports_thinking and self.thinking_budget is not None
        
        if use_thinking:
            # Extended thinking requires temperature=1
            params["temperature"] = 1.0
            # Add thinking configuration
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            # Standard API with temperature
            temp = temperature if temperature is not None else self.default_temperature
            params["temperature"] = temp
        
        if self.debug:
            print(f"[DEBUG] Anthropic Request:")
            print(f"  Model: {self.model}")
            print(f"  Messages: {len(prompt_messages)} messages")
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
        
        if not response.content:
            return ""
        
        # Extract text content (skip thinking blocks if present)
        result_parts = []
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text":
                    result_parts.append(block.text)
                elif block.type == "thinking":
                    # Skip thinking blocks - they contain internal reasoning
                    if self.debug:
                        thinking_text = getattr(block, 'thinking', '')
                        print(f"[DEBUG] Thinking block: {thinking_text[:200]}...")
        
        result = "".join(result_parts)
        
        if self.debug:
            print(f"[DEBUG] Anthropic Response:")
            print(f"  Content: {result[:200]}...")
        
        return result
