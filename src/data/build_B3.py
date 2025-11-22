from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .schema import Attribute, BaseConfiguration, Profile, compute_deltas
from ..utils.config import Config


def _sample_profile(rng: np.random.Generator, attributes: List[Attribute], levels: List[str]) -> Profile:
    chosen = {attr: rng.choice(levels) for attr in attributes}
    return Profile(chosen)


def _delta_vector(config: BaseConfiguration, attributes: List[Attribute]) -> np.ndarray:
    deltas = compute_deltas(config.profile_left, config.profile_right, attributes)
    return np.array([deltas.get(attr, 0) for attr in attributes], dtype=float)


def _compute_correlation_metrics(deltas: List[np.ndarray], candidate: np.ndarray) -> Dict[str, float]:
    """Compute multiple correlation metrics for better screening."""
    stack = np.vstack(deltas + [candidate])
    
    # Handle edge cases
    if np.all(stack.std(axis=0) == 0):
        return {"max_abs_corr": 0.0, "mean_abs_corr": 0.0, "condition_number": 1.0}
    
    corr_matrix = np.corrcoef(stack, rowvar=False)
    if np.isnan(corr_matrix).any():
        return {"max_abs_corr": 0.0, "mean_abs_corr": 0.0, "condition_number": 1.0}
    
    # Extract off-diagonal elements
    off_diag = corr_matrix - np.eye(len(stack[0]))
    
    # Compute condition number for matrix conditioning
    if len(stack) > len(stack[0]):
        try:
            condition_num = float(np.linalg.cond(stack.T @ stack))
        except np.linalg.LinAlgError:
            condition_num = float('inf')
    else:
        condition_num = 1.0
    
    return {
        "max_abs_corr": float(np.max(np.abs(off_diag))),
        "mean_abs_corr": float(np.mean(np.abs(off_diag))),
        "condition_number": condition_num
    }


def _should_accept_candidate(
    metrics: Dict[str, float], 
    target_abs: float, 
    mean_corr_target: float,
    max_condition_number: float,
    tolerance: float
) -> Tuple[bool, float]:
    """Multi-criteria acceptance with adaptive thresholds."""
    max_corr = metrics["max_abs_corr"]
    mean_corr = metrics["mean_abs_corr"]
    condition_num = metrics["condition_number"]
    
    # Primary criterion: max absolute correlation
    primary_ok = max_corr <= target_abs + tolerance
    
    # Secondary criteria for better orthogonality
    secondary_ok = (
        mean_corr <= mean_corr_target + tolerance and  # Mean correlation should be lower
        condition_num <= max_condition_number  # Avoid near-singular matrices
    )
    
    accept = primary_ok and secondary_ok
    score = max_corr  # Use max correlation as the primary score
    
    return accept, score


def _update_tolerance(tolerance: float, acceptance_rate: float, target_rate: float = 0.1) -> float:
    """Adaptive tolerance based on acceptance rate."""
    if acceptance_rate < target_rate:
        # Increase tolerance more gradually if acceptance is too low
        return min(tolerance + 0.005, 0.05)  # Smaller increments, lower cap
    elif acceptance_rate > target_rate * 2:
        # Decrease tolerance if acceptance is too high
        return max(tolerance * 0.9, 0.0)  # Less aggressive decrease
    else:
        # Gradual adjustment
        return tolerance * 0.98  # Slower decay


def build_B3(cfg: Config, rng: np.random.Generator) -> list[BaseConfiguration]:
    """Enhanced B3 builder with improved correlation screening and adaptive tolerance."""
    attributes: list[Attribute] = cfg.profiles.attributes
    levels: list[str] = cfg.profiles.levels
    
    # Extract configuration parameters with defaults for backward compatibility
    target_abs = getattr(cfg.blocks.B3, 'corr_abs_target', 0.05)
    mean_corr_target = getattr(cfg.blocks.B3, 'mean_corr_target', 0.025)
    max_condition_number = getattr(cfg.blocks.B3, 'max_condition_number', 10.0)
    target_acceptance_rate = getattr(cfg.blocks.B3, 'target_acceptance_rate', 0.1)
    early_stop_tolerance = getattr(cfg.blocks.B3, 'early_stop_tolerance', 0.1)
    initial_tolerance = getattr(cfg.blocks.B3, 'initial_tolerance', 0.0)
    
    batch_size = cfg.blocks.B3.candidates_per_batch
    max_batches = cfg.blocks.B3.max_batches
    max_attempts = max_batches * batch_size * 5
    
    accepted: list[BaseConfiguration] = []
    deltas: list[np.ndarray] = []
    seen_keys: set[tuple] = set()
    attempts = 0
    tolerance = initial_tolerance
    
    # Enhanced batch statistics tracking
    batch_stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_corr": 0,
        "rejected_duplicate": 0,
        "tolerance_history": [],
        "correlation_history": []
    }
    
    while len(accepted) < batch_size and attempts < max_attempts:
        attempts += 1
        batch_stats["attempts"] += 1
        
        profile_a = _sample_profile(rng, attributes, levels)
        profile_b = _sample_profile(rng, attributes, levels)
        config = BaseConfiguration(
            block="B3",
            profile_left=profile_a,
            profile_right=profile_b,
        ).with_sorted_profiles(attributes)
        key = config.canonical_key(attributes)
        
        if key in seen_keys:
            batch_stats["rejected_duplicate"] += 1
            continue
            
        candidate_delta = _delta_vector(config, attributes)
        sign = rng.choice((-1, 1))
        candidate_delta *= sign
        
        # Use enhanced correlation screening
        if len(deltas) == 0 or len(deltas) < len(attributes) * 4:
            accept = True
            new_score = 0.0
        else:
            metrics = _compute_correlation_metrics(deltas, candidate_delta)
            accept, new_score = _should_accept_candidate(
                metrics, target_abs, mean_corr_target, max_condition_number, tolerance
            )
            batch_stats["correlation_history"].append(new_score)
        
        if accept:
            accepted.append(config)
            deltas.append(candidate_delta)
            seen_keys.add(key)
            batch_stats["accepted"] += 1
            
            # Adaptive tolerance based on recent acceptance rate
            # Update more frequently (every 10 accepted) to adapt faster
            if len(accepted) % 10 == 0 and len(accepted) >= 10:
                recent_rate = batch_stats["accepted"] / max(batch_stats["attempts"], 1)
                tolerance = _update_tolerance(tolerance, recent_rate, target_acceptance_rate)
                batch_stats["tolerance_history"].append(tolerance)
        else:
            batch_stats["rejected_corr"] += 1
        
        # Early stopping if tolerance gets too high
        if tolerance > early_stop_tolerance:
            break
    
    # Log final statistics for debugging
    final_rate = batch_stats["accepted"] / max(batch_stats["attempts"], 1)
    print(f"B3 generation completed: {len(accepted)}/{batch_size} configurations "
          f"(acceptance rate: {final_rate:.3f}, final tolerance: {tolerance:.4f})")
    
    return accepted
