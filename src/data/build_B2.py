from __future__ import annotations

import itertools

from .schema import Attribute, BaseConfiguration, Profile
from ..utils.config import Config

DELTA_LEVELS = {
    -2: ("Low", "High"),
    0: ("Medium", "Medium"),
    2: ("High", "Low"),
}


def build_B2(cfg: Config) -> list[BaseConfiguration]:
    attributes: list[Attribute] = cfg.profiles.attributes
    neutral_levels = {attr: "Medium" for attr in attributes}
    configs: dict[tuple, BaseConfiguration] = {}
    for attr_pair in itertools.combinations(attributes, 2):
        for delta_values in itertools.product((-2, 0, 2), repeat=2):
            left_levels = neutral_levels.copy()
            right_levels = neutral_levels.copy()
            for attr, delta_value in zip(attr_pair, delta_values):
                level_pair = DELTA_LEVELS[delta_value]
                left_levels[attr] = level_pair[0]
                right_levels[attr] = level_pair[1]
            config = BaseConfiguration(
                block="B2",
                profile_left=Profile(left_levels),
                profile_right=Profile(right_levels),
            ).with_sorted_profiles(attributes)
            configs[config.canonical_key(attributes)] = config
    return list(configs.values())
