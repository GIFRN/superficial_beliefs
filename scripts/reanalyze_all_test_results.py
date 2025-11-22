#!/usr/bin/env python3
"""
Reanalyze all test results from saved response files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.compare_reasoning_efforts import analyze_test_results, analyze_results, print_test_results, print_results
from src.utils.io import write_json

def reanalyze_effort(effort: str, results_dir: Path, trials_df: pd.DataFrame):
    """Reanalyze one effort level."""
    print(f"\n{'='*80}")
    print(f"🔄 REANALYZING: {effort.upper()} REASONING EFFORT")
    print(f"{'='*80}")
    
    # Load saved responses
    train_responses_path = results_dir / f"responses_{effort}_train.jsonl"
    test_responses_path = results_dir / f"responses_{effort}_test.jsonl"
    
    if not train_responses_path.exists():
        print(f"❌ Training responses not found: {train_responses_path}")
        return None, None
    
    if not test_responses_path.exists():
        print(f"❌ Test responses not found: {test_responses_path}")
        return None, None
    
    # Load responses
    train_responses = []
    with train_responses_path.open() as f:
        for line in f:
            if line.strip():
                train_responses.append(json.loads(line))
    
    test_responses = []
    with test_responses_path.open() as f:
        for line in f:
            if line.strip():
                test_responses.append(json.loads(line))
    
    print(f"✅ Loaded {len(train_responses)} training responses")
    print(f"✅ Loaded {len(test_responses)} test responses")
    
    # Get trial IDs
    train_trial_ids = [resp["trial_id"] for resp in train_responses]
    test_trial_ids = [resp["trial_id"] for resp in test_responses]
    
    train_trials_df = trials_df[trials_df["trial_id"].isin(train_trial_ids)]
    test_trials_df = trials_df[trials_df["trial_id"].isin(test_trial_ids)]
    
    print(f"\n🔬 Reanalyzing training results...")
    train_results = analyze_results(train_responses, train_trials_df)
    
    if "error" in train_results:
        print(f"❌ Error in training analysis: {train_results['error']}")
        return None, None
    
    print_results(effort, train_results)
    
    # Save training results
    train_results_save = {k: v for k, v in train_results.items() if k != "model"}
    write_json(train_results_save, results_dir / f"results_{effort}_train.json")
    print(f"✅ Saved training results")
    
    test_results = None
    if train_results.get("model"):
        print(f"\n🔬 Reanalyzing test results...")
        test_results = analyze_test_results(
            test_responses,
            test_trials_df,
            train_results["model"],
            split_name="test"
        )
        
        if "error" in test_results:
            print(f"❌ Error in test analysis: {test_results['error']}")
            return train_results, None
        
        print_test_results(effort, test_results)
        
        # Save test results
        write_json(test_results, results_dir / f"results_{effort}_test.json")
        print(f"✅ Saved test results")
    
    return train_results, test_results

def main():
    results_dir = Path("results/reasoning_effort_comparison_1.1")
    dataset_dir = Path("data/generated/v1_short")
    
    # Load trial features
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    
    effort_levels = ["minimal", "low", "medium"]
    all_train_results = {}
    all_test_results = {}
    
    for effort in effort_levels:
        train_res, test_res = reanalyze_effort(effort, results_dir, trials_df)
        if train_res:
            all_train_results[effort] = train_res
        if test_res:
            all_test_results[effort] = test_res
    
    # Print summary comparison
    if all_train_results:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON - TRAINING SET")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<30} {'Minimal':<12} {'Low':<12} {'Medium':<12}")
        print("-" * 70)
        
        for attr in ['E', 'A', 'S', 'D']:
            print(f"{attr + ' weight':<30} ", end="")
            for effort in effort_levels:
                if effort in all_train_results:
                    val = all_train_results[effort]['weights'].get(attr, 0)
                    print(f"{val:<12.3f} ", end="")
            print()
        
        print()
        for metric in ['ECRB_top1_driver', 'ECRB_top1_weights', 'rank_corr']:
            print(f"{metric + ' (in-sample)':<30} ", end="")
            for effort in effort_levels:
                if effort in all_train_results:
                    val = all_train_results[effort]['alignment'].get(metric, 0)
                    print(f"{val:<12.3f} ", end="")
            print()
    
    if all_test_results:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON - TEST SET")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<30} {'Minimal':<12} {'Low':<12} {'Medium':<12}")
        print("-" * 70)
        
        for metric_name, metric_key in [("MAE", "mae"), ("RMSE", "rmse"), ("Correlation", "correlation"), ("Accuracy", "accuracy")]:
            print(f"{metric_name:<30} ", end="")
            for effort in effort_levels:
                if effort in all_test_results:
                    val = all_test_results[effort].get('prediction', {}).get(metric_key, 0)
                    if val is not None:
                        print(f"{val:<12.4f} ", end="")
                    else:
                        print(f"{'N/A':<12} ", end="")
            print()
        
        print()
        for metric in ['ECRB_top1_driver', 'ECRB_top1_weights', 'rank_corr']:
            print(f"{metric + ' (out-sample)':<30} ", end="")
            for effort in effort_levels:
                if effort in all_test_results:
                    val = all_test_results[effort]['alignment'].get(metric, 0)
                    print(f"{val:<12.3f} ", end="")
            print()
    
    print(f"\n✅ Reanalysis complete!")

if __name__ == "__main__":
    main()

