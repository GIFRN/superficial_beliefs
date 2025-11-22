#!/usr/bin/env python3
"""
Reanalyze test results from saved response files.
Useful when only the analysis code had a bug, not the data collection.
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

def main():
    results_dir = Path("results/reasoning_effort_comparison_1.1")
    dataset_dir = Path("data/generated/v1_short")
    
    # Load trial features
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    
    effort = "minimal"
    
    print(f"🔄 Reanalyzing {effort} effort results...")
    
    # Load saved responses
    train_responses_path = results_dir / f"responses_{effort}_train.jsonl"
    test_responses_path = results_dir / f"responses_{effort}_test.jsonl"
    
    if not train_responses_path.exists():
        print(f"❌ Training responses not found: {train_responses_path}")
        return
    
    if not test_responses_path.exists():
        print(f"❌ Test responses not found: {test_responses_path}")
        return
    
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
        return
    
    print_results(effort, train_results)
    
    # Save training results
    train_results_save = {k: v for k, v in train_results.items() if k != "model"}
    write_json(train_results_save, results_dir / f"results_{effort}_train.json")
    print(f"✅ Saved training results")
    
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
            return
        
        print_test_results(effort, test_results)
        
        # Save test results
        write_json(test_results, results_dir / f"results_{effort}_test.json")
        print(f"✅ Saved test results")
    
    print(f"\n✅ Reanalysis complete!")

if __name__ == "__main__":
    main()

