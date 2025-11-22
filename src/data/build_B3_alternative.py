from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from .schema import Attribute, BaseConfiguration, Profile
from ..utils.config import Config


def _is_strictly_dominated(left_profile: Profile, right_profile: Profile, level_scores: dict[str, int]) -> bool:
    """Check if one profile strictly dominates another."""
    left_score = sum(level_scores[left_profile.levels[attr]] for attr in left_profile.levels)
    right_score = sum(level_scores[right_profile.levels[attr]] for attr in right_profile.levels)
    
    # Strict dominance: one is strictly better in all attributes
    left_better = all(level_scores[left_profile.levels[attr]] >= level_scores[right_profile.levels[attr]] 
                     for attr in left_profile.levels)
    right_better = all(level_scores[right_profile.levels[attr]] >= level_scores[left_profile.levels[attr]] 
                       for attr in right_profile.levels)
    
    # Check if one is strictly better (not equal)
    if left_better and left_score > right_score:
        return True
    if right_better and right_score > left_score:
        return True
    
    return False


def _is_score_difference_too_large(left_profile: Profile, right_profile: Profile, level_scores: dict[str, int], max_diff: int = 2) -> bool:
    """Check if the score difference between profiles is too large."""
    left_score = sum(level_scores[left_profile.levels[attr]] for attr in left_profile.levels)
    right_score = sum(level_scores[right_profile.levels[attr]] for attr in right_profile.levels)
    
    return abs(left_score - right_score) > max_diff


def build_B3(cfg: Config, rng: np.random.Generator) -> list[BaseConfiguration]:
    """Alternative B3 builder using systematic generation with controlled randomization."""
    attributes: list[Attribute] = cfg.profiles.attributes
    levels: list[str] = cfg.profiles.levels
    level_scores = cfg.profiles.level_scores
    
    # Extract configuration parameters
    target_size = getattr(cfg.blocks.B3, 'candidates_per_batch', 2000)
    randomization_factor = getattr(cfg.blocks.B3, 'randomization_factor', 0.3)
    
    # Generate systematic configurations
    configs = _generate_systematic_configs(attributes, levels, level_scores, target_size, rng, randomization_factor)
    
    return configs


def _generate_systematic_configs(attributes: list, levels: list, level_scores: dict[str, int], target_size: int, 
                                rng: np.random.Generator, randomization_factor: float) -> list[BaseConfiguration]:
    """Generate configurations using systematic approach with controlled randomization."""
    configs: list[BaseConfiguration] = []
    seen_keys: set[tuple] = set()
    
    # Create systematic delta patterns with constraints
    delta_patterns = []
    
    # Pattern 1: Single attribute differences (constrained)
    for attr_idx in range(len(attributes)):
        for delta in [-1, 1]:  # Only small differences to avoid dominance
            pattern = [0] * len(attributes)
            pattern[attr_idx] = delta
            delta_patterns.append(pattern)
    
    # Pattern 2: Two attribute differences (constrained)
    for i in range(len(attributes)):
        for j in range(i + 1, len(attributes)):
            for delta1 in [-1, 1]:
                for delta2 in [-1, 1]:
                    pattern = [0] * len(attributes)
                    pattern[i] = delta1
                    pattern[j] = delta2
                    # Check total advantage constraint
                    if abs(sum(pattern)) <= 2:
                        delta_patterns.append(pattern)
    
    # Pattern 3: Random patterns for diversity (constrained)
    n_random = int(len(delta_patterns) * randomization_factor)
    for _ in range(n_random):
        pattern = []
        for _ in range(len(attributes)):
            pattern.append(rng.choice([-1, 0, 1]))  # Only small deltas
        # Ensure total advantage constraint
        if abs(sum(pattern)) <= 2:
            delta_patterns.append(pattern)
    
    # Generate configurations from patterns
    for pattern in delta_patterns:
        if len(configs) >= target_size:
            break
            
        # Create base profile (all Medium)
        base_levels = {attr: "Medium" for attr in attributes}
        
        # Apply pattern to create two profiles
        left_levels = base_levels.copy()
        right_levels = base_levels.copy()
        
        for attr_idx, delta in enumerate(pattern):
            attr = attributes[attr_idx]
            if delta != 0:
                # Convert delta to level differences
                left_idx = levels.index(left_levels[attr])
                right_idx = levels.index(right_levels[attr])
                
                # Apply delta
                new_left_idx = max(0, min(len(levels) - 1, left_idx + delta))
                new_right_idx = max(0, min(len(levels) - 1, right_idx - delta))
                
                left_levels[attr] = levels[new_left_idx]
                right_levels[attr] = levels[new_right_idx]
        
        left_profile = Profile(left_levels)
        right_profile = Profile(right_levels)
        
        # Skip if one profile strictly dominates the other or score difference is too large
        if _is_strictly_dominated(left_profile, right_profile, level_scores):
            continue
        if _is_score_difference_too_large(left_profile, right_profile, level_scores):
            continue
            
        config = BaseConfiguration(
            block="B3",
            profile_left=left_profile,
            profile_right=right_profile,
        ).with_sorted_profiles(attributes)
        
        key = config.canonical_key(attributes)
        if key not in seen_keys:
            configs.append(config)
            seen_keys.add(key)
    
    # If we need more configs, generate random ones
    attempts = 0
    max_attempts = target_size * 10  # Prevent infinite loops
    
    while len(configs) < target_size and attempts < max_attempts:
        attempts += 1
        profile_a = _sample_profile(rng, attributes, levels)
        profile_b = _sample_profile(rng, attributes, levels)
        
        # Skip if one profile strictly dominates the other or score difference is too large
        if _is_strictly_dominated(profile_a, profile_b, level_scores):
            continue
        if _is_score_difference_too_large(profile_a, profile_b, level_scores):
            continue
            
        config = BaseConfiguration(
            block="B3",
            profile_left=profile_a,
            profile_right=profile_b,
        ).with_sorted_profiles(attributes)
        
        key = config.canonical_key(attributes)
        if key not in seen_keys:
            configs.append(config)
            seen_keys.add(key)
    
    # Shuffle and return target size
    rng.shuffle(configs)
    return configs[:target_size]


def _sample_profile(rng: np.random.Generator, attributes: Sequence[Attribute], levels: Sequence[str]) -> Profile:
    """Sample a random profile."""
    levels_dict = {}
    for attr in attributes:
        levels_dict[attr] = rng.choice(levels)
    return Profile(levels_dict)
