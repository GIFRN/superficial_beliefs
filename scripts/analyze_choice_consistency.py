#!/usr/bin/env python3
"""Analyze choice consistency across multiple samples (replicates).

This script examines how often model choices switch between options A and B
across different samples of the same trial, helping assess whether multiple
samples are necessary.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_responses(responses_path: Path) -> list[dict]:
    """Load all responses from a JSONL file."""
    responses = []
    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return responses


def extract_choices(responses: list[dict]) -> pd.DataFrame:
    """Extract choice data from responses.
    
    Returns a DataFrame with columns: trial_id, replicate_idx, choice
    """
    rows = []
    
    for response in responses:
        trial_id = response.get("trial_id")
        if not trial_id:
            continue
        
        # Each response has multiple replicates
        for replicate_idx, run in enumerate(response.get("responses", [])):
            choice = None
            
            # Extract choice from the first step (which is the choice step)
            steps = run.get("steps", [])
            if steps:
                choice_step = steps[0]  # First step is the choice
                parsed = choice_step.get("parsed", {})
                if parsed.get("ok", False):
                    choice = parsed.get("choice")
            
            rows.append({
                "trial_id": trial_id,
                "replicate_idx": replicate_idx,
                "choice": choice
            })
    
    return pd.DataFrame(rows)


def analyze_consistency(choices_df: pd.DataFrame) -> dict:
    """Analyze choice consistency for each trial."""
    
    # Group by trial_id to analyze each trial's replicates
    trial_consistency = []
    
    for trial_id, group in choices_df.groupby("trial_id"):
        # Remove any failed parses
        valid_choices = group[group["choice"].notna()]["choice"].values
        
        if len(valid_choices) == 0:
            continue
        
        # Count choices
        choice_counts = Counter(valid_choices)
        total_samples = len(valid_choices)
        
        # Get most common choice
        most_common_choice, most_common_count = choice_counts.most_common(1)[0]
        
        # Calculate consistency metrics
        consistency_rate = most_common_count / total_samples
        
        # Determine if there was any switching
        has_switches = len(choice_counts) > 1
        
        # Calculate entropy (0 = perfectly consistent, higher = more random)
        probs = np.array(list(choice_counts.values())) / total_samples
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        trial_consistency.append({
            "trial_id": trial_id,
            "total_samples": total_samples,
            "dominant_choice": most_common_choice,
            "dominant_count": most_common_count,
            "consistency_rate": consistency_rate,
            "has_switches": has_switches,
            "entropy": entropy,
            "choice_distribution": dict(choice_counts)
        })
    
    return pd.DataFrame(trial_consistency)


def print_summary(consistency_df: pd.DataFrame):
    """Print summary statistics about choice consistency."""
    
    print("\n" + "="*70)
    print("CHOICE CONSISTENCY ANALYSIS")
    print("="*70)
    
    total_trials = len(consistency_df)
    trials_with_switches = consistency_df["has_switches"].sum()
    trials_consistent = total_trials - trials_with_switches
    
    print(f"\nTotal trials analyzed: {total_trials}")
    print(f"Trials with consistent choices (100%): {trials_consistent} ({trials_consistent/total_trials*100:.1f}%)")
    print(f"Trials with choice switches: {trials_with_switches} ({trials_with_switches/total_trials*100:.1f}%)")
    
    print(f"\nConsistency Rate Statistics:")
    print(f"  Mean:   {consistency_df['consistency_rate'].mean():.3f}")
    print(f"  Median: {consistency_df['consistency_rate'].median():.3f}")
    print(f"  Min:    {consistency_df['consistency_rate'].min():.3f}")
    print(f"  Max:    {consistency_df['consistency_rate'].max():.3f}")
    print(f"  Std:    {consistency_df['consistency_rate'].std():.3f}")
    
    print(f"\nEntropy Statistics (0 = perfectly consistent):")
    print(f"  Mean:   {consistency_df['entropy'].mean():.3f}")
    print(f"  Median: {consistency_df['entropy'].median():.3f}")
    print(f"  Max:    {consistency_df['entropy'].max():.3f}")
    
    # Binned consistency rates
    print(f"\nConsistency Rate Distribution:")
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        count = ((consistency_df['consistency_rate'] > lower) & 
                 (consistency_df['consistency_rate'] <= upper)).sum()
        print(f"  {lower:.1f} < rate ≤ {upper:.1f}: {count} trials ({count/total_trials*100:.1f}%)")
    
    # Perfect consistency
    perfect = (consistency_df['consistency_rate'] == 1.0).sum()
    print(f"  Perfect (1.0):        {perfect} trials ({perfect/total_trials*100:.1f}%)")


def print_examples(consistency_df: pd.DataFrame, n_examples: int = 10):
    """Print examples of trials with varying consistency levels."""
    
    print("\n" + "="*70)
    print("EXAMPLES OF TRIALS WITH VARYING CONSISTENCY")
    print("="*70)
    
    # Show most inconsistent trials
    inconsistent = consistency_df.nsmallest(n_examples, "consistency_rate")
    
    if len(inconsistent) > 0:
        print(f"\n{len(inconsistent)} Most Inconsistent Trials:")
        print("-" * 70)
        for _, row in inconsistent.iterrows():
            print(f"\nTrial {row['trial_id']}:")
            print(f"  Samples: {row['total_samples']}")
            print(f"  Dominant choice: {row['dominant_choice']} ({row['dominant_count']}/{row['total_samples']})")
            print(f"  Consistency rate: {row['consistency_rate']:.1%}")
            print(f"  Entropy: {row['entropy']:.3f}")
            print(f"  Distribution: {row['choice_distribution']}")
    
    # Show some moderately consistent trials
    moderate = consistency_df[
        (consistency_df['consistency_rate'] > 0.6) & 
        (consistency_df['consistency_rate'] < 0.9)
    ].head(n_examples)
    
    if len(moderate) > 0:
        print(f"\n\nSome Moderately Consistent Trials:")
        print("-" * 70)
        for _, row in moderate.iterrows():
            print(f"\nTrial {row['trial_id']}:")
            print(f"  Samples: {row['total_samples']}")
            print(f"  Dominant choice: {row['dominant_choice']} ({row['dominant_count']}/{row['total_samples']})")
            print(f"  Consistency rate: {row['consistency_rate']:.1%}")
            print(f"  Distribution: {row['choice_distribution']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze choice consistency across replicates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze responses from a run directory
  python scripts/analyze_choice_consistency.py --responses data/runs/v1_short/responses.jsonl
  
  # Analyze with more examples
  python scripts/analyze_choice_consistency.py --responses data/runs/v1_short/responses.jsonl --examples 20
        """
    )
    parser.add_argument("--responses", required=True, help="Path to responses.jsonl file")
    parser.add_argument("--examples", type=int, default=10, help="Number of example trials to show")
    parser.add_argument("--output", default=None, help="Path to save detailed CSV (optional)")
    
    args = parser.parse_args()
    
    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: File not found: {responses_path}")
        return
    
    print(f"Loading responses from: {responses_path}")
    responses = load_responses(responses_path)
    print(f"Loaded {len(responses)} trial responses")
    
    print("\nExtracting choices from replicates...")
    choices_df = extract_choices(responses)
    
    if len(choices_df) == 0:
        print("Error: No valid choices found in responses")
        return
    
    print(f"Extracted {len(choices_df)} choice records")
    
    print("\nAnalyzing consistency...")
    consistency_df = analyze_consistency(choices_df)
    
    if len(consistency_df) == 0:
        print("Error: No trials to analyze")
        return
    
    # Print summary statistics
    print_summary(consistency_df)
    
    # Print examples
    print_examples(consistency_df, n_examples=args.examples)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        consistency_df.to_csv(output_path, index=False)
        print(f"\n✅ Detailed results saved to: {output_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

