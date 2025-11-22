from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Union, TYPE_CHECKING

from src.data.schema import Attribute, Profile

if TYPE_CHECKING:
    from src.data.themes import ThemeConfig


PromptFunc = Callable[[Dict[str, Any]], str]
Prompt = Union[str, PromptFunc]


@dataclass
class TrialSpec:
    trial_id: str
    config_id: str
    block: str
    profile_a: Profile
    profile_b: Profile
    order_a: Sequence[Attribute]
    order_b: Sequence[Attribute]
    paraphrase_id: int
    manipulation: str
    attribute_target: str | None
    inject_offset: int
    variant: str
    seed: int
    metadata: dict[str, Any]
    theme_config: "ThemeConfig | None" = None


@dataclass
class ConversationStep:
    name: str
    prompt: Prompt
    expects: str
    reset_context: bool = False


@dataclass
class ConversationPlan:
    system_prompt: str
    steps: list[ConversationStep]
