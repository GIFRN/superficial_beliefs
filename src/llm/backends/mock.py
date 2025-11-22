from __future__ import annotations

from typing import Sequence

from ..harness import LLMBackend


class MockBackend(LLMBackend):
    def __init__(self, choice: str = "A", premise_attr: str = "E", premise_text: str = "better efficacy drives outcomes"):
        self.choice = choice
        self.premise_attr = premise_attr
        self.premise_text = premise_text

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> str:
        if not messages:
            return ""
        last_user = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
        if last_user is None:
            return ""
        content = last_user["content"].lower()
        if "premiseattribute" in content:
            return (
                f"PremiseAttribute = {self.premise_attr}\n"
                f'PremiseText = "{self.premise_text}"'
            )
        if "single-sentence" in content:
            return f"Choose {self.choice}; {self.premise_text}."
        if 'respond only with "a" or "b"' in content:
            return self.choice
        if content.startswith("now choose"):
            return self.choice
        return ""
