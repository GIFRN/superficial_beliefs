from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Tuple

Attribute = Literal["E", "A", "S", "D"]
LevelName = Literal["Low", "Medium", "High"]

LEVEL_SCORES: dict[LevelName, int] = {"Low": -1, "Medium": 0, "High": 1}
SCORE_TO_LEVEL: dict[int, LevelName] = {v: k for k, v in LEVEL_SCORES.items()}

ATTR_LABELS: dict[Attribute, str] = {
    "E": "Efficacy",
    "A": "Adherence",
    "S": "Safety",
    "D": "Durability",
}


def score_level(level: LevelName) -> int:
    return LEVEL_SCORES[level]


def delta(level_a: LevelName, level_b: LevelName) -> int:
    return score_level(level_a) - score_level(level_b)


@dataclass(frozen=True)
class Profile:
    levels: dict[Attribute, LevelName]

    def as_tuple(self, attributes: Iterable[Attribute]) -> Tuple[int, ...]:
        return tuple(score_level(self.levels[attr]) for attr in attributes)

    def as_strings(self, attributes: Iterable[Attribute]) -> Tuple[LevelName, ...]:
        return tuple(self.levels[attr] for attr in attributes)


@dataclass(frozen=True)
class BaseConfiguration:
    block: Literal["B1", "B2", "B3", "DOM"]
    profile_left: Profile
    profile_right: Profile

    def canonical_key(self, attributes: Iterable[Attribute]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        left = self.profile_left.as_tuple(attributes)
        right = self.profile_right.as_tuple(attributes)
        return tuple(sorted((left, right)))  # type: ignore[return-value]

    def with_sorted_profiles(self, attributes: Iterable[Attribute]) -> "BaseConfiguration":
        attrs = list(attributes)
        key = self.canonical_key(attrs)
        left_levels = {attr: inverse_score(score) for attr, score in zip(attrs, key[0])}
        right_levels = {attr: inverse_score(score) for attr, score in zip(attrs, key[1])}
        return BaseConfiguration(
            block=self.block,
            profile_left=Profile(left_levels),
            profile_right=Profile(right_levels),
        )


def inverse_score(score: int) -> LevelName:
    return SCORE_TO_LEVEL[score]


def compute_deltas(profile_a: Profile, profile_b: Profile, attributes: Iterable[Attribute]) -> Dict[Attribute, int]:
    attrs = list(attributes)
    return {attr: delta(profile_a.levels[attr], profile_b.levels[attr]) for attr in attrs}
