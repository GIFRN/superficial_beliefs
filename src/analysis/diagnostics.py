from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .cv import _metric_bundle

ATTRIBUTES = ["E", "A", "S", "D"]


def delta_correlation(trials_df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"delta_{attr}" for attr in ATTRIBUTES if f"delta_{attr}" in trials_df]
    return trials_df[cols].corr()


def order_balance(trials_df: pd.DataFrame) -> dict[str, Any]:
    stats = {}
    for attr in ATTRIBUTES:
        pos_counts = Counter(trials_df[f"posA_{attr}"]).copy()
        for position, count in Counter(trials_df[f"posB_{attr}"]).items():
            pos_counts[position] += count
        total = sum(pos_counts.values())
        stats[attr] = {
            position: count / total for position, count in sorted(pos_counts.items())
        }
    return stats


def evaluate_model(model, design_matrix) -> dict[str, float]:
    probs = model.predict(design_matrix.X)
    return _metric_bundle(probs, design_matrix.y.to_numpy(), design_matrix.weights.to_numpy())


def _dominance_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return boolean masks for A-dominant and B-dominant trials using visible deltas."""
    delta_cols = [f"delta_{attr}" for attr in ATTRIBUTES if f"delta_{attr}" in df]
    deltas = df[delta_cols]
    a_dominant = (deltas.ge(0).all(axis=1)) & (deltas.gt(0).any(axis=1))
    b_dominant = (deltas.le(0).all(axis=1)) & (deltas.lt(0).any(axis=1))
    return a_dominant, b_dominant


def _visible_dominance_mask(df: pd.DataFrame) -> pd.Series:
    """Exclude trials where the dominant attribute is occluded (drop/equalize)."""
    if "manipulation" not in df or "attribute_target" not in df:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    occluded = df["manipulation"].isin(["occlude_drop", "occlude_equalize"])
    if not occluded.any():
        return mask
    for attr in ATTRIBUTES:
        base_col = f"delta_base_{attr}"
        delta_col = f"delta_{attr}"
        if base_col in df:
            diff = df[base_col].ne(0)
        else:
            diff = df[delta_col].ne(0) if delta_col in df else False
        hidden = occluded & df["attribute_target"].eq(attr) & diff
        mask = mask & ~hidden
    return mask


def validate_b1_rationality(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> dict[str, Any]:
    """Validate that B1 trials show expected Pareto optimal behavior.
    
    Checks if the model chooses the dominant option with P >= 0.95.
    For trials where A is dominant (all deltas >= 0), expects P(choose A) >= 0.95.
    For trials where B is dominant (all deltas <= 0), expects P(choose B) >= 0.95.
    """
    b1_trials = trials_df[trials_df["block"] == "B1"].merge(choice_agg, on="trial_id", how="inner")
    
    if b1_trials.empty:
        return {
            "rationality_check_passed": True,
            "failure_rate": 0.0,
            "accuracy": 1.0,
            "failed_trials": [],
            "message": "No B1 trials found"
        }
    
    # Only evaluate trials where the dominant attribute is visible
    visible_mask = _visible_dominance_mask(b1_trials)
    b1_trials = b1_trials[visible_mask].copy()
    if b1_trials.empty:
        return {
            "rationality_check_passed": True,
            "failure_rate": 0.0,
            "accuracy": 1.0,
            "failed_trials": [],
            "message": "No B1 trials with visible dominant attribute"
        }

    # Determine which option is dominant for each trial (visible deltas)
    a_dominant, b_dominant = _dominance_masks(b1_trials)
    
    # Calculate P(choose A)
    prob_choose_a = b1_trials["successes"] / b1_trials["trials"]
    
    # Check correctness based on dominance
    # For A dominant: should choose A with P >= 0.95
    # For B dominant: should choose B with P >= 0.95 (i.e., P(A) <= 0.05)
    correct_choice = pd.Series(True, index=b1_trials.index, dtype=bool)
    correct_choice.loc[a_dominant] = (prob_choose_a.loc[a_dominant] >= 0.95).values
    correct_choice.loc[b_dominant] = (prob_choose_a.loc[b_dominant] <= 0.05).values
    
    rationality_failures = ~correct_choice
    accuracy = float(correct_choice.mean())
    
    return {
        "rationality_check_passed": not rationality_failures.any(),
        "failure_rate": float(rationality_failures.mean()),
        "accuracy": accuracy,
        "failed_trials": b1_trials.loc[rationality_failures, "trial_id"].tolist(),
        "total_b1_trials": len(b1_trials),
        "a_dominant_trials": int(a_dominant.sum()),
        "b_dominant_trials": int(b_dominant.sum()),
        "message": f"B1 rationality check: {len(b1_trials)} trials, {rationality_failures.sum()} failures, {accuracy:.1%} accuracy"
    }


def validate_b1_probes(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> dict[str, Any]:
    """Validate that B1 probes are effective.
    
    Compares accuracy on dominant-choice tasks between baseline and manipulated conditions.
    If probes work, accuracy should decrease when information is manipulated.
    """
    b1_trials = trials_df[trials_df["block"] == "B1"].merge(choice_agg, on="trial_id", how="inner")
    
    if b1_trials.empty:
        return {
            "probe_effectiveness": True,
            "baseline_accuracy": 1.0,
            "manipulated_accuracy": 1.0,
            "probe_effect_size": 0.0,
            "message": "No B1 trials found"
        }
    
    # Only evaluate trials where the dominant attribute is visible
    visible_mask = _visible_dominance_mask(b1_trials)
    b1_trials = b1_trials[visible_mask].copy()

    baseline = b1_trials[b1_trials["manipulation"] == "none"]
    manipulated = b1_trials[b1_trials["manipulation"] != "none"]
    
    if baseline.empty or manipulated.empty:
        return {
            "probe_effectiveness": True,
            "baseline_accuracy": 1.0,
            "manipulated_accuracy": 1.0,
            "probe_effect_size": 0.0,
            "baseline_trials": len(baseline),
            "manipulated_trials": len(manipulated),
            "message": "Insufficient data for probe validation"
        }
    
    # Calculate accuracy (chose dominant option) for each group
    def compute_accuracy(df):
        a_dominant, b_dominant = _dominance_masks(df)
        prob_choose_a = df["successes"] / df["trials"]

        correct = pd.Series(False, index=df.index, dtype=bool)
        correct.loc[a_dominant] = (prob_choose_a.loc[a_dominant] >= 0.95).values
        correct.loc[b_dominant] = (prob_choose_a.loc[b_dominant] <= 0.05).values

        return float(correct.mean())
    
    baseline_acc = compute_accuracy(baseline)
    manipulated_acc = compute_accuracy(manipulated)
    probe_effect = baseline_acc - manipulated_acc  # Positive if accuracy drops with manipulation
    
    return {
        "probe_effectiveness": probe_effect > 0.05,  # 5% threshold for accuracy drop
        "baseline_accuracy": baseline_acc,
        "manipulated_accuracy": manipulated_acc,
        "probe_effect_size": probe_effect,
        "baseline_trials": len(baseline),
        "manipulated_trials": len(manipulated),
        "message": f"B1 probe validation: baseline acc={baseline_acc:.3f}, manipulated acc={manipulated_acc:.3f}, effect={probe_effect:.3f}"
    }
