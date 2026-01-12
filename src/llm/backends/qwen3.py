"""
Qwen3 backend using Hugging Face Transformers with thinking mode support.

This backend loads Qwen3 models locally and supports the extended thinking
capability where the model produces reasoning in <think>...</think> tags
before the final response.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..harness import LLMBackend

# Lazy imports to avoid loading heavy dependencies if not needed
_model = None
_tokenizer = None
_device = None


def _load_model(model_name: str, device_map: str = "auto", torch_dtype: str = "auto"):
    """Lazy load the model and tokenizer."""
    global _model, _tokenizer, _device
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Qwen3 backend requires transformers and torch. "
            "Install with: pip install transformers torch"
        ) from e
    
    print(f"Loading Qwen3 model: {model_name}...")
    
    # Determine torch dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"
    
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    _device = next(_model.parameters()).device
    print(f"Qwen3 model loaded on device: {_device}")
    
    return _model, _tokenizer


class Qwen3Backend(LLMBackend):
    """Backend for Qwen3 models with optional thinking mode support."""
    
    # Special token for end of thinking block
    THINK_START_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        enable_thinking: bool = True,
        return_thinking: bool = False,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        debug: bool = False,
    ):
        """
        Initialize the Qwen3 backend.
        
        Args:
            model: HuggingFace model name or local path (e.g., "Qwen/Qwen3-8B")
            temperature: Sampling temperature (default 0.7)
            max_tokens: Maximum tokens to generate (default 1024)
            enable_thinking: Whether to enable thinking mode (default True)
            return_thinking: If True, include thinking content in response (default False)
            device_map: Device mapping strategy for model loading (default "auto")
            torch_dtype: Torch dtype for model (default "auto")
            debug: Enable debug output (default False)
        """
        self.model_name = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.return_thinking = return_thinking
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.debug = debug
        
        # Model and tokenizer loaded lazily on first use
        self._model = None
        self._tokenizer = None
    
    def _ensure_loaded(self):
        """Ensure model and tokenizer are loaded."""
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = _load_model(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            )
    
    def _get_think_end_token_id(self) -> int | None:
        """Get the token ID for </think> token."""
        self._ensure_loaded()
        
        # Try to find the </think> token ID
        think_end_tokens = self._tokenizer.encode(self.THINK_END_TOKEN, add_special_tokens=False)
        if think_end_tokens:
            return think_end_tokens[-1]
        
        # Fallback: try to get from vocab
        vocab = self._tokenizer.get_vocab()
        if self.THINK_END_TOKEN in vocab:
            return vocab[self.THINK_END_TOKEN]
        
        return None
    
    def _extract_content(self, output_ids: list[int]) -> tuple[str, str]:
        """
        Extract thinking content and final response from output token IDs.
        
        Returns:
            tuple of (thinking_content, final_content)
        """
        think_end_token_id = self._get_think_end_token_id()
        
        if think_end_token_id is not None:
            # Find the position of </think> token
            try:
                # Search from the end to find the last </think> token
                reversed_ids = output_ids[::-1]
                index = len(output_ids) - reversed_ids.index(think_end_token_id)
            except ValueError:
                # </think> token not found - no thinking content
                index = 0
        else:
            index = 0
        
        thinking_content = self._tokenizer.decode(
            output_ids[:index], 
            skip_special_tokens=True
        ).strip()
        
        final_content = self._tokenizer.decode(
            output_ids[index:], 
            skip_special_tokens=True
        ).strip()
        
        return thinking_content, final_content
    
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Generated text response (thinking content excluded unless return_thinking=True)
        """
        import torch
        
        self._ensure_loaded()
        
        temp = temperature if temperature is not None else self.default_temperature
        max_new_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Convert messages to the format expected by the chat template
        chat_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]
        
        if self.debug:
            print(f"[DEBUG] Qwen3 Request:")
            print(f"  Model: {self.model_name}")
            print(f"  Messages: {len(chat_messages)} messages")
            for i, msg in enumerate(chat_messages):
                print(f"    {i+1}. {msg['role']}: {msg['content'][:100]}...")
            print(f"  Temperature: {temp}")
            print(f"  max_new_tokens: {max_new_tokens}")
            print(f"  enable_thinking: {self.enable_thinking}")
            print(f"  Seed: {seed}")
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        # Set up generation config
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temp > 0,
        }
        
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        
        # Extract only the newly generated tokens
        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_length:].tolist()
        
        # Extract thinking and final content
        thinking_content, final_content = self._extract_content(output_ids)
        
        if self.debug:
            print(f"[DEBUG] Qwen3 Response:")
            if thinking_content:
                print(f"  Thinking: {thinking_content[:200]}...")
            print(f"  Content: {final_content[:200]}...")
        
        # Return based on configuration
        if self.return_thinking and thinking_content:
            return f"<think>{thinking_content}</think>\n\n{final_content}"
        
        return final_content
    
    def complete_with_thinking(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> dict[str, str]:
        """
        Generate a completion and return both thinking and final content.
        
        This is a convenience method that always returns the thinking content
        separately, regardless of the return_thinking setting.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Dict with 'thinking_content' and 'content' keys
        """
        import torch
        
        self._ensure_loaded()
        
        temp = temperature if temperature is not None else self.default_temperature
        max_new_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        chat_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]
        
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temp > 0,
        }
        
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        
        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_length:].tolist()
        
        thinking_content, final_content = self._extract_content(output_ids)
        
        return {
            "thinking_content": thinking_content,
            "content": final_content,
        }



