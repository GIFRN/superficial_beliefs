#!/usr/bin/env python3
"""
Debug Script: All Variations Pipeline Test

This script provides comprehensive testing of the entire experimental pipeline
by running multiple B3 trials through all manipulation variations and performing
complete Stage A and Stage B analyses.

What it tests:
--------------
1. **Trial Generation**: Creates 3 diverse B3 trial configurations
2. **All Manipulations**: Tests all manipulation types:
   - short_reason (baseline)
   - split_reason (separate context)
   - premise_first (premise before choice)
   - redact (for each attribute E, A, S, D)
   - neutralize (for each attribute E, A, S, D)
   - inject (with offsets -1, 0, +1)

3. **Response Collection**: Uses EnhancedMockBackend for varied, format-appropriate responses
4. **Stage A Analysis**: Estimates attribute weights from choice patterns
5. **Stage B Analysis**: Measures alignment between stated premises and actual drivers

Output:
-------
- Detailed response summaries for each trial
- Stage A weight estimates and cross-validation metrics
- Per-trial contribution analysis
- Stage B alignment metrics (ECRB_top1_driver, ECRB_top1_weights, rank correlation)
- Premise distribution across attributes

Usage:
------
    python scripts/debug_all_variations.py

No arguments required. Results are printed to stdout only (not saved).

Notes:
------
- Uses mock backend for fast, deterministic responses
- Small sample size means some metrics (like CV) may be skipped
- Designed for pipeline verification, not production analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from typing import Any

from src.llm.backends.mock import MockBackend
from src.llm.harness import LLMBackend
from src.llm.harness import run_trial
from src.llm.types import TrialSpec
from src.data.schema import Profile
from src.analysis.features import aggregate_choices, aggregate_premises
from src.analysis.stageA import (
    ATTRIBUTES,
    build_design_matrix,
    compute_ames_and_weights,
    fit_glm_clustered,
    per_trial_contributions,
)
from src.analysis.stageB import alignment_metrics
from src.utils.config import load_config


class EnhancedMockBackend(LLMBackend):
    """Mock backend that provides varied, format-appropriate responses."""
    
    def __init__(self):
        self.trial_counter = 0
        self.choices = ["A", "B"]
        self.attrs = ATTRIBUTES
        
    def complete(
        self,
        messages: list[dict[str, str]],
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
        
        # Vary responses based on seed for diversity
        import hashlib
        hash_val = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16)
        choice_idx = hash_val % 2
        attr_idx = (hash_val // 2) % len(self.attrs)
        
        choice = self.choices[choice_idx]
        attr = self.attrs[attr_idx]
        attr_name = {"E": "Efficacy", "A": "Adherence", "S": "Safety", "D": "Durability"}[attr]
        
        # Handle premise-structured format
        if "premiseattribute" in content or "one of [efficacy" in content:
            return (
                f"PremiseAttribute = {attr_name}\n"
                f'PremiseText = "{attr_name} drives outcomes"'
            )
        
        # Handle choice + reason format
        if "format: <option>. <reason>" in content or "state the option you choose" in content:
            return f"Drug {choice}. {attr_name} is most important."
        
        # Handle reason repeat
        if "repeat only the reason" in content:
            return f"{attr_name} is most important."
        
        # Handle split_reason second turn
        if "explain the single most important" in content:
            return f"{attr_name} drives the decision."
        
        # Handle premise-first simple choice
        if 'respond only with "a" or "b"' in content or 'now choose exactly one option' in content:
            return choice
        
        # Handle sentence request
        if "provide the single most important factor" in content:
            return f"{attr_name} is key."
        
        return ""


def create_sample_b3_trials() -> list[dict[str, Any]]:
    """Create multiple diverse B3-style trial configurations."""
    trials = [
        {
            "config_id": "DEBUG-B3-01",
            "block": "B3",
            "profile_a": Profile(levels={"E": "High", "A": "Low", "S": "High", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "High", "S": "Low", "D": "High"}),
            "order_a": [0, 1, 2, 3],
            "order_b": [0, 1, 2, 3],
            "paraphrase_id": 0,
            "seed": 42,
        },
        {
            "config_id": "DEBUG-B3-02",
            "block": "B3",
            "profile_a": Profile(levels={"E": "High", "A": "High", "S": "Low", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "Low", "S": "High", "D": "High"}),
            "order_a": [0, 1, 2, 3],
            "order_b": [0, 1, 2, 3],
            "paraphrase_id": 0,
            "seed": 43,
        },
        {
            "config_id": "DEBUG-B3-03",
            "block": "B3",
            "profile_a": Profile(levels={"E": "Medium", "A": "High", "S": "High", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "Medium", "S": "Low", "D": "High"}),
            "order_a": [0, 1, 2, 3],
            "order_b": [0, 1, 2, 3],
            "paraphrase_id": 0,
            "seed": 44,
        },
    ]
    return trials


def create_trial_spec(base_config: dict[str, Any], manipulation: str, attribute_target: str | None = None, inject_offset: int = 0) -> TrialSpec:
    """Create a TrialSpec from base config and manipulation."""
    config_id = base_config["config_id"]
    trial_id = f"{config_id}-{manipulation}"
    if attribute_target:
        trial_id += f"-{attribute_target}"
    if inject_offset != 0:
        trial_id += f"-offset{inject_offset}"
    
    variant = manipulation
    if manipulation in {"redact", "neutralize", "inject"}:
        variant = "short_reason"  # These manipulations use short_reason variant with probe instructions
    
    return TrialSpec(
        trial_id=trial_id,
        config_id=base_config["config_id"],
        block=base_config["block"],
        manipulation=manipulation,
        variant=variant,
        profile_a=base_config["profile_a"],
        profile_b=base_config["profile_b"],
        order_a=base_config["order_a"],
        order_b=base_config["order_b"],
        paraphrase_id=base_config["paraphrase_id"],
        attribute_target=attribute_target,
        inject_offset=inject_offset,
        seed=base_config["seed"],
        metadata={},
    )


def build_all_variations(base_config: dict[str, Any]) -> list[TrialSpec]:
    """Build trial specs for all manipulation variations."""
    specs = []
    
    # Base manipulations (no attribute targeting)
    for manipulation in ["short_reason", "split_reason", "premise_first"]:
        specs.append(create_trial_spec(base_config, manipulation))
    
    # Redact variations (one for each attribute)
    for attr in ATTRIBUTES:
        specs.append(create_trial_spec(base_config, "redact", attribute_target=attr))
    
    # Neutralize variations (one for each attribute)
    for attr in ATTRIBUTES:
        specs.append(create_trial_spec(base_config, "neutralize", attribute_target=attr))
    
    # Inject variations (different offsets for one attribute)
    for offset in [-1, 0, 1]:
        specs.append(create_trial_spec(base_config, "inject", attribute_target="A", inject_offset=offset))
    
    return specs


def responses_to_dataframe(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert run_trial results to responses DataFrame format."""
    rows_choices = []
    rows_premises = []
    
    for result in results:
        trial_id = result["trial_id"]
        config_id = result["config_id"]
        block = result["block"]
        manipulation = result["manipulation"]
        variant = result["variant"]
        
        for run_idx, run in enumerate(result["responses"]):
            # Extract choice
            choice_step = next((s for s in run["steps"] if s["name"] == "choice"), None)
            if choice_step and choice_step["parsed"].get("ok"):
                rows_choices.append({
                    "trial_id": trial_id,
                    "config_id": config_id,
                    "block": block,
                    "manipulation": manipulation,
                    "variant": variant,
                    "run": run_idx,
                    "choice": choice_step["parsed"]["choice"],
                })
            
            # Extract premise
            premise_step = next((s for s in run["steps"] if s["name"] == "premise"), None)
            if premise_step and premise_step["parsed"].get("ok"):
                rows_premises.append({
                    "trial_id": trial_id,
                    "config_id": config_id,
                    "block": block,
                    "manipulation": manipulation,
                    "variant": variant,
                    "run": run_idx,
                    "premise_attr": premise_step["parsed"]["attr"],
                    "premise_text": premise_step["parsed"]["text"],
                })
    
    choices_df = pd.DataFrame(rows_choices)
    premises_df = pd.DataFrame(rows_premises)
    
    return choices_df, premises_df


def create_trials_dataframe(specs: list[TrialSpec]) -> pd.DataFrame:
    """Create a trials DataFrame from trial specs."""
    from src.data.schema import compute_deltas
    
    rows = []
    for spec in specs:
        # Compute deltas
        deltas = compute_deltas(spec.profile_a, spec.profile_b, ATTRIBUTES)
        
        rows.append({
            "trial_id": spec.trial_id,
            "config_id": spec.config_id,
            "block": spec.block,
            "manipulation": spec.manipulation,
            "variant": spec.variant,
            "delta_E": deltas["E"],
            "delta_A": deltas["A"],
            "delta_S": deltas["S"],
            "delta_D": deltas["D"],
            "delta_pos_E": 0,  # No order effects in this debug
            "delta_pos_A": 0,
            "delta_pos_S": 0,
            "delta_pos_D": 0,
        })
    
    return pd.DataFrame(rows)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def print_responses_summary(results: list[dict[str, Any]]):
    """Print summary of responses."""
    print_section("RESPONSES SUMMARY")
    
    for result in results:
        trial_id = result["trial_id"]
        manipulation = result["manipulation"]
        variant = result["variant"]
        
        print(f"Trial: {trial_id}")
        print(f"  Manipulation: {manipulation} (variant: {variant})")
        
        for run_idx, run in enumerate(result["responses"]):
            print(f"  Run {run_idx}:")
            for step in run["steps"]:
                status = "✅" if step["parsed"].get("ok") else "❌"
                print(f"    {status} {step['name']}: {step['content'][:60]}...")
                if step["parsed"].get("ok"):
                    if step["name"] == "choice":
                        print(f"       → Choice: {step['parsed']['choice']}")
                    elif step["name"] == "premise":
                        print(f"       → Attribute: {step['parsed']['attr']}")
        print()


def print_stage_a_summary(weights_info: dict[str, Any], cv_metrics: dict[str, float]):
    """Print Stage A analysis summary."""
    print_section("STAGE A: WEIGHT ESTIMATION", "=")
    
    print("Estimated Weights:")
    for attr in ATTRIBUTES:
        weight = weights_info["weights"].get(attr, 0.0)
        beta = weights_info["beta"].get(attr, 0.0)
        print(f"  {attr}: {weight:.3f} (β = {beta:.3f})")
    
    print("\nCross-Validation Metrics:")
    for metric, value in cv_metrics.items():
        print(f"  {metric}: {value:.4f}")


def print_stage_b_summary(alignment: dict[str, Any], premises_df: pd.DataFrame):
    """Print Stage B alignment summary."""
    print_section("STAGE B: ALIGNMENT METRICS", "=")
    
    print("Alignment Metrics:")
    print(f"  ECRB (top1 driver):  {alignment.get('ECRB_top1_driver', 'N/A'):.3f}")
    print(f"  ECRB (top1 weights): {alignment.get('ECRB_top1_weights', 'N/A'):.3f}")
    print(f"  Rank correlation:    {alignment.get('rank_corr', 'N/A'):.3f}")
    
    print("\nPremise Distribution:")
    if not premises_df.empty:
        counts = premises_df["premise_attr"].value_counts()
        total = len(premises_df)
        for attr in ATTRIBUTES:
            count = counts.get(attr, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {attr}: {count:2d} ({pct:5.1f}%)")
    else:
        print("  No premises collected")


def print_contributions(contributions_df: pd.DataFrame, trials_df: pd.DataFrame):
    """Print per-trial contribution analysis."""
    print_section("PER-TRIAL CONTRIBUTIONS", "-")
    
    # Merge with trial info to get manipulation
    merged = contributions_df.merge(
        trials_df[["trial_id", "manipulation"]],
        left_index=True,
        right_on="trial_id"
    )
    
    print("Driver attribute by manipulation:")
    driver_by_manip = merged.groupby("manipulation")["driver"].value_counts()
    for (manip, driver), count in driver_by_manip.items():
        print(f"  {manip:20s} → {driver}: {count}")


def main():
    """Main debug script."""
    print_section("DEBUG SCRIPT: ALL VARIATIONS", "#")
    print("This script tests multiple B3 trials through all manipulation variations")
    print("and performs full Stage A and Stage B analysis.\n")
    
    # Create sample trials
    base_configs = create_sample_b3_trials()
    print(f"Base Configurations ({len(base_configs)} trials):")
    for base_config in base_configs:
        profile_a = base_config['profile_a']
        profile_b = base_config['profile_b']
        print(f"\n  {base_config['config_id']}:")
        print(f"    Profile A: E={profile_a.levels['E']}, A={profile_a.levels['A']}, "
              f"S={profile_a.levels['S']}, D={profile_a.levels['D']}")
        print(f"    Profile B: E={profile_b.levels['E']}, A={profile_b.levels['A']}, "
              f"S={profile_b.levels['S']}, D={profile_b.levels['D']}")
    
    # Build variations for the first trial only to keep output manageable
    base_config = base_configs[0]
    specs = build_all_variations(base_config)
    
    # Add short_reason variants for other trials
    for other_config in base_configs[1:]:
        specs.append(create_trial_spec(other_config, "short_reason"))
    
    print(f"\nGenerated {len(specs)} total trial variations")
    print(f"  - {len(build_all_variations(base_config))} variations for first trial")
    print(f"  - {len(base_configs) - 1} additional trials with short_reason")
    
    # Run trials with enhanced mock backend
    print_section("RUNNING TRIALS")
    print("Using EnhancedMockBackend for varied, format-appropriate responses...")
    
    backend = EnhancedMockBackend()
    results = []
    
    for spec in specs:
        result = run_trial(
            spec,
            backend,
            S=1,  # Single replicate for debugging
            temperature=1.0,
            seed=42,
            max_tokens=256,
        )
        results.append(result)
    
    print(f"✅ Completed {len(results)} trials")
    
    # Convert to DataFrames
    choices_df, premises_df = responses_to_dataframe(results)
    trials_df = create_trials_dataframe(specs)
    
    # Print responses
    print_responses_summary(results)
    
    # Aggregate choices for Stage A
    print_section("AGGREGATING DATA FOR ANALYSIS")
    
    # Manual aggregation similar to aggregate_choices
    choice_agg = choices_df.groupby("trial_id").agg(
        successes=("choice", lambda x: (x == "A").sum()),
        trials=("choice", "count"),
    ).reset_index()
    
    # Merge with trials data
    stageA_data = trials_df.merge(choice_agg, on="trial_id", how="left")
    stageA_data["successes"] = stageA_data["successes"].fillna(0)
    stageA_data["trials"] = stageA_data["trials"].fillna(0)
    
    print(f"Aggregated {len(choice_agg)} trial choices")
    print(f"Collected {len(premises_df)} premises")
    
    # Stage A Analysis
    print_section("STAGE A ANALYSIS")
    
    try:
        # Build design matrix (exclude B1, but we don't have B1 here anyway)
        design = build_design_matrix(
            stageA_data,
            include_interactions=False,
            include_order_terms=False,  # No order effects in this debug
            exclude_b1=False,  # No B1 trials
        )
        
        print(f"Design matrix: {design.X.shape[0]} observations, {design.X.shape[1]} features")
        print(f"Features: {list(design.X.columns)}")
        
        # Fit model
        model = fit_glm_clustered(design)
        weights_info = compute_ames_and_weights(model)
        
        # Cross-validation (simplified) - skip if dataset too small
        cv_metrics = {}
        try:
            if len(design.X) >= 10:
                from src.analysis.cv import cross_validate_design
                cv_metrics = cross_validate_design(design, n_splits=min(3, len(design.X) // 3))
        except Exception as e:
            print(f"  ⚠️  CV skipped due to: {e}")
            cv_metrics = {"log_loss": float("nan"), "brier": float("nan"), "accuracy": float("nan")}
        
        print_stage_a_summary(weights_info, cv_metrics)
        
        # Compute per-trial contributions
        unique_trials = stageA_data.drop_duplicates("trial_id").set_index("trial_id")
        contributions = per_trial_contributions(unique_trials, model)
        
        print_contributions(contributions, trials_df)
        
    except Exception as e:
        print(f"❌ Stage A analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Stage B Analysis
    print_section("STAGE B ANALYSIS")
    
    try:
        # Prepare responses DataFrame
        responses_df = choices_df.merge(premises_df, on=["trial_id", "config_id", "block", "manipulation", "variant", "run"], how="left")
        
        # Compute alignment metrics
        alignment = alignment_metrics(responses_df, stageA_data, model)
        
        print_stage_b_summary(alignment, premises_df)
        
    except Exception as e:
        print(f"❌ Stage B analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print_section("DEBUG COMPLETE", "#")
    print("✅ All variations tested successfully!")
    print(f"✅ Stage A and Stage B analysis completed")
    print("\nThis debug run demonstrates the full pipeline with all manipulation types.")


if __name__ == "__main__":
    main()

