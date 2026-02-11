#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.analysis.features import load_responses, aggregate_choices
from src.data.schema import LEVEL_SCORES


def load_configs(dataset_dir: Path) -> pd.DataFrame:
    """Load configuration profiles"""
    configs = pd.read_parquet(dataset_dir / "dataset_configs.parquet")
    # Parse JSON columns
    configs["levels_left"] = configs["levels_left"].apply(json.loads)
    configs["levels_right"] = configs["levels_right"].apply(json.loads)
    return configs


def get_profile_display(levels_dict: dict) -> str:
    """Format a profile for display"""
    attrs = ["E", "A", "S", "D"]
    parts = []
    for attr in attrs:
        level = levels_dict[attr]
        score = LEVEL_SCORES[level]
        parts.append(f"{attr}:{level}({score:+d})")
    return " | ".join(parts)


def get_delta_display(row: pd.Series) -> str:
    """Format deltas for display"""
    attrs = ["E", "A", "S", "D"]
    parts = []
    for attr in attrs:
        delta = row[f"delta_{attr}"]
        if delta != 0:
            parts.append(f"Δ{attr}:{delta:+d}")
    return " | ".join(parts) if parts else "No difference"


def determine_dominant_option(profile_a: dict, profile_b: dict) -> str:
    """Determine which option is dominant (better or equal on all attributes)."""
    attrs = ["E", "A", "S", "D"]
    a_better_count = 0
    b_better_count = 0
    
    for attr in attrs:
        score_a = LEVEL_SCORES[profile_a[attr]]
        score_b = LEVEL_SCORES[profile_b[attr]]
        if score_a > score_b:
            a_better_count += 1
        elif score_b > score_a:
            b_better_count += 1
    
    if a_better_count > 0 and b_better_count == 0:
        return "A (dominant)"
    elif b_better_count > 0 and a_better_count == 0:
        return "B (dominant)"
    elif a_better_count == 0 and b_better_count == 0:
        return "Neither (equal)"
    else:
        return "Neither (mixed dominance)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect B1 rationality check failures")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Path to generated dataset")
    parser.add_argument("--responses", default="data/runs/v1_short_openai_gpt5mini/responses.jsonl", help="Path to responses file")
    parser.add_argument("--threshold", type=float, default=0.95, help="P(correct dominant choice) threshold for failure")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of examples to show")
    parser.add_argument("--show-all", action="store_true", help="Show all B1 trials, not just failures")
    args = parser.parse_args()

    # Load data
    dataset_dir = Path(args.dataset)
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    configs_df = load_configs(dataset_dir)
    responses_df = load_responses(args.responses)
    choice_agg = aggregate_choices(responses_df)

    # Merge to get trial outcomes
    b1_trials = trials_df[trials_df["block"] == "B1"].copy()
    b1_trials = b1_trials.merge(configs_df, on="config_id", how="left")
    b1_trials = b1_trials.merge(choice_agg, on="trial_id", how="left")
    
    # Fill missing data
    b1_trials["successes"] = b1_trials["successes"].fillna(0).astype(int)
    b1_trials["trials"] = b1_trials["trials"].fillna(0).astype(int)
    
    # Calculate P(choose A)
    b1_trials["prob_choose_A"] = b1_trials.apply(
        lambda row: row["successes"] / row["trials"] if row["trials"] > 0 else None,
        axis=1,
    )

    def expected_choice(row: pd.Series) -> str | None:
        if row["labelA"] == "A":
            profile_a = row["levels_left"]
            profile_b = row["levels_right"]
        else:
            profile_a = row["levels_right"]
            profile_b = row["levels_left"]
        dominant = determine_dominant_option(profile_a, profile_b)
        # Skip if the dominant attribute is occluded (drop/equalize).
        if row.get("manipulation") in {"occlude_drop", "occlude_equalize"} and row.get("attribute_target"):
            diffs = [
                attr
                for attr in ["E", "A", "S", "D"]
                if LEVEL_SCORES[profile_a[attr]] != LEVEL_SCORES[profile_b[attr]]
            ]
            if row["attribute_target"] in diffs:
                return None
        if dominant.startswith("A"):
            return "A"
        if dominant.startswith("B"):
            return "B"
        return None

    b1_trials["expected_choice"] = b1_trials.apply(expected_choice, axis=1)

    # Calculate P(correct dominant choice)
    def prob_correct(row: pd.Series) -> float | None:
        if row["trials"] <= 0:
            return None
        if row["expected_choice"] == "A":
            return row["prob_choose_A"]
        if row["expected_choice"] == "B":
            return 1.0 - row["prob_choose_A"]
        return None

    b1_trials["prob_correct"] = b1_trials.apply(prob_correct, axis=1)

    # Identify failures (only for trials with a dominant option)
    b1_trials["is_failure"] = b1_trials["prob_correct"].apply(
        lambda v: v is not None and v < args.threshold
    )
    
    # Filter based on --show-all flag
    if not args.show_all:
        display_trials = b1_trials[b1_trials["is_failure"]].copy()
    else:
        display_trials = b1_trials.copy()
    
    display_trials = display_trials.sort_values("prob_choose_A")
    
    # Summary statistics
    total_b1 = len(b1_trials)
    total_failures = b1_trials["is_failure"].sum()
    failure_rate = total_failures / total_b1 if total_b1 > 0 else 0
    
    print("=" * 80)
    print("B1 RATIONALITY CHECK FAILURES")
    print("=" * 80)
    print(f"\nTotal B1 trials: {total_b1}")
    print(f"Failures (P(correct) < {args.threshold}): {total_failures} ({failure_rate:.1%})")
    if b1_trials["prob_correct"].notna().any():
        print(f"Min P(correct): {b1_trials['prob_correct'].min():.3f}")
        print(f"Max P(correct): {b1_trials['prob_correct'].max():.3f}")
        print(f"Mean P(correct): {b1_trials['prob_correct'].mean():.3f}")
    else:
        print("No dominant trials with visible dominant attribute.")
    print()
    
    # Show detailed examples
    limit = args.limit if not args.show_all else len(display_trials)
    print(f"{'Showing all trials' if args.show_all else f'Showing {min(limit, len(display_trials))} failure examples'} (sorted by P(choose A)):")
    print("=" * 80)
    
    for idx, (_, trial) in enumerate(display_trials.head(limit).iterrows()):
        if idx > 0:
            print("\n" + "-" * 80)
        
        print(f"\n[{idx + 1}] Trial {trial['trial_id']} (Config {trial['config_id']})")
        prob_correct = trial.get("prob_correct")
        prob_correct_display = f"{prob_correct:.3f}" if prob_correct is not None else "n/a"
        print(f"    P(correct) = {prob_correct_display} ({trial['successes']}/{trial['trials']})")
        print(f"    Status: {'❌ FAILURE' if trial['is_failure'] else '✅ PASS'}")
        print(f"    Manipulation: {trial['manipulation']}")
        if pd.notna(trial['attribute_target']):
            print(f"    Target attribute: {trial['attribute_target']}")
        
        # Determine which profile is A and which is B based on labelA
        if trial['labelA'] == 'A':
            profile_a = trial['levels_left']
            profile_b = trial['levels_right']
        else:
            profile_a = trial['levels_right']
            profile_b = trial['levels_left']
        
        print(f"\n    Option A (labelA={trial['labelA']}): {get_profile_display(profile_a)}")
        print(f"    Option B: {get_profile_display(profile_b)}")
        print(f"    Deltas (A - B): {get_delta_display(trial)}")
        
        dominant = determine_dominant_option(profile_a, profile_b)
        print(f"    → Expected choice: {dominant}")
        
        # Show individual responses
        trial_responses = responses_df[responses_df["trial_id"] == trial["trial_id"]]
        if not trial_responses.empty:
            choices = trial_responses["choice"].value_counts()
            print(f"\n    Individual choices:")
            for choice, count in choices.items():
                print(f"      {choice}: {count}")
            
            # Show a sample response if available
            sample = trial_responses[trial_responses["choice_ok"]].iloc[0] if not trial_responses[trial_responses["choice_ok"]].empty else None
            if sample is not None:
                print(f"\n    Sample response:")
                print(f"      Choice: {sample['choice']}")
                if pd.notna(sample.get('choice_raw')):
                    response_text = str(sample['choice_raw']).strip()
                    if len(response_text) > 200:
                        response_text = response_text[:200] + "..."
                    print(f"      Raw: {response_text}")
    
    print("\n" + "=" * 80)
    
    # Additional analysis
    if total_failures > 0 and not args.show_all:
        print("\nFAILURE BREAKDOWN:")
        print("-" * 80)
        
        failures = b1_trials[b1_trials["is_failure"]]
        
        # By manipulation type
        print("\nBy manipulation type:")
        manip_counts = failures["manipulation"].value_counts()
        for manip, count in manip_counts.items():
            total_manip = len(b1_trials[b1_trials["manipulation"] == manip])
            pct = count / total_manip * 100 if total_manip > 0 else 0
            print(f"  {manip}: {count}/{total_manip} ({pct:.1f}%)")
        
        # By which attribute differs
        print("\nBy varying attribute:")
        for attr in ["E", "A", "S", "D"]:
            attr_failures = failures[failures[f"delta_{attr}"] != 0]
            attr_total = b1_trials[b1_trials[f"delta_{attr}"] != 0]
            if len(attr_total) > 0:
                pct = len(attr_failures) / len(attr_total) * 100
                print(f"  {attr}: {len(attr_failures)}/{len(attr_total)} ({pct:.1f}%)")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
