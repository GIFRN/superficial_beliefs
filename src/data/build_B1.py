from __future__ import annotations

import itertools

from .schema import Attribute, BaseConfiguration, Profile
from ..utils.config import Config


def build_B1(cfg: Config) -> list[BaseConfiguration]:
    attributes: list[Attribute] = cfg.profiles.attributes
    levels = cfg.profiles.levels
    medium_levels = {attr: "Medium" for attr in attributes}
    configs: dict[tuple, BaseConfiguration] = {}
    for attr in attributes:
        for level_pair in itertools.permutations(levels, 2):
            left_levels = medium_levels.copy()
            right_levels = medium_levels.copy()
            left_levels[attr] = level_pair[0]
            right_levels[attr] = level_pair[1]
            config = BaseConfiguration(
                block="B1",
                profile_left=Profile(left_levels),
                profile_right=Profile(right_levels),
            ).with_sorted_profiles(attributes)
            key = config.canonical_key(attributes)
            configs[key] = config
    return list(configs.values())
