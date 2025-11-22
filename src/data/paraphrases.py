from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .schema import Attribute, Profile, ATTR_LABELS

if TYPE_CHECKING:
    from .themes import ThemeConfig


@dataclass(frozen=True)
class ParaphraseTemplate:
    pattern: str
    separator: str
    connector: str


PARAPHRASES: list[ParaphraseTemplate] = [
    ParaphraseTemplate(pattern="{label} --- {pairs}.", separator="; ", connector=": "),
    ParaphraseTemplate(pattern="{label} ( {pairs} ).", separator=" | ", connector="="),
    ParaphraseTemplate(pattern="{label}: {pairs}.", separator=", ", connector=" "),
    ParaphraseTemplate(pattern="{label} -> {pairs}.", separator="; ", connector="="),
]


def choose_paraphrase_ids(n: int, n_templates: int, rng: np.random.Generator) -> list[int]:
    if n_templates <= 0:
        raise ValueError("n_templates must be positive")
    template_ids = list(range(n_templates))
    reps = (n + n_templates - 1) // n_templates
    pool = (template_ids * reps)[:n]
    rng.shuffle(pool)
    return pool


def render_profile(
    profile: Profile, 
    template_id: int, 
    order: Sequence[Attribute], 
    label: str,
    theme_config: "ThemeConfig | None" = None
) -> str:
    template = PARAPHRASES[template_id % len(PARAPHRASES)]
    
    # Use theme config if provided, otherwise use default ATTR_LABELS
    if theme_config:
        segments = [
            f"{theme_config.get_attribute_label(attr)}{template.connector}{profile.levels[attr]}" 
            for attr in order if attr in profile.levels
        ]
    else:
        segments = [
            f"{ATTR_LABELS[attr]}{template.connector}{profile.levels[attr]}" 
            for attr in order if attr in profile.levels
        ]
    
    pairs = template.separator.join(segments)
    return template.pattern.format(label=label, pairs=pairs)


def get_template(template_id: int) -> ParaphraseTemplate:
    return PARAPHRASES[template_id % len(PARAPHRASES)]
