#!/usr/bin/env python3
"""
Recalculate alignment metrics for minimal reasoning effort results.

The minimal effort responses often just return the attribute name (e.g., "Efficacy")
without the full structured format (PremiseAttribute = ... / PremiseText = '...').
This script relaxes the parsing to extract just the attribute and recalculate alignment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.analysis.stageA import ATTRIBUTES, build_design_matrix, fit_glm_clustered, compute_ames_and_weights, per_trial_contributions
from src.analysis.stageB import alignment_metrics


# Relaxed attribute extraction - handles single-word responses
ATTR_MAP = {
    "E": "E", "EFFICACY": "E",
    "A": "A", "ADHERENCE": "A", 
    "S": "S", "SAFETY": "S",
    "D": "D", "DURABILITY": "D",
}


def extract_attribute_relaxed(content: str) -> str | None:
    """Extract attribute from premise content with relaxed parsing.
    
    Handles:
    - Single word: "Efficacy" -> E
    - With punctuation: "Efficacy." -> E
    - Standard format: "PremiseAttribute = Efficacy\n..." -> E
    - Attribute at start: "Efficacy is the most important..." -> E
    """
    if not content:
        return None
    
    content = content.strip()
    
    # Try standard format first
    for line in content.splitlines():
        line = line.strip()
        if line.lower().startswith("premiseattribute"):
            _, _, value = line.partition("=")
            token = value.strip().strip("[] ").upper()
            if token in ATTR_MAP:
                return ATTR_MAP[token]
    
    # Try single word (with optional punctuation)
    single_word = re.sub(r'[^\w]', '', content.split()[0] if content.split() else "").upper()
    if single_word in ATTR_MAP:
        return ATTR_MAP[single_word]
    
    # Try finding attribute anywhere in the text
    content_upper = content.upper()
    for keyword, attr in [("EFFICACY", "E"), ("ADHERENCE", "A"), ("SAFETY", "S"), ("DURABILITY", "D")]:
        if keyword in content_upper:
            return attr
    
    return None


def load_responses_relaxed(jsonl_path: Path) -> pd.DataFrame:
    """Load responses with relaxed premise parsing."""
    records = []
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            trial_id = payload["trial_id"]
            
            for run_idx, run in enumerate(payload.get("responses", [])):
                orientation = run.get("orientation", "original")
                
                # Extract choice
                choice_step = next((s for s in run["steps"] if s["name"] == "choice"), None)
                if not choice_step or not choice_step["parsed"].get("ok"):
                    continue
                
                raw_choice = choice_step["parsed"]["choice"]
                # Adjust for reversed orientation
                if orientation == "reversed":
                    choice = "B" if raw_choice == "A" else "A"
                else:
                    choice = raw_choice
                
                # Extract premise with relaxed parsing
                premise_step = next((s for s in run["steps"] if s["name"] == "premise"), None)
                premise_attr = None
                if premise_step:
                    # First try the original parsing
                    if premise_step["parsed"].get("ok"):
                        premise_attr = premise_step["parsed"]["attr"]
                    else:
                        # Fall back to relaxed extraction
                        premise_attr = extract_attribute_relaxed(premise_step.get("content", ""))
                
                records.append({
                    "trial_id": trial_id,
                    "run": run_idx,
                    "choice": choice,
                    "premise_attr": premise_attr,
                    "premise_content": premise_step.get("content", "") if premise_step else "",
                })
    
    return pd.DataFrame(records)


def recalculate_alignment(responses_df: pd.DataFrame, trials_df: pd.DataFrame, model) -> dict[str, Any]:
    """Calculate alignment metrics using relaxed-parsed premises."""
    # Same logic as stageB.alignment_metrics but using our relaxed parsing
    if responses_df.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}
    
    unique_trials = trials_df.drop_duplicates("trial_id").set_index("trial_id")
    contrib_df = per_trial_contributions(unique_trials, model)
    driver_map = contrib_df["driver"].rename("driver").reset_index().rename(columns={"index": "trial_id"})
    
    weights_info = compute_ames_and_weights(model)
    weight_map = weights_info["weights"]
    if weight_map:
        top_attr = max(weight_map.items(), key=lambda kv: kv[1])[0]
    else:
        top_attr = ATTRIBUTES[0]
    
    valid = responses_df[responses_df["premise_attr"].isin(ATTRIBUTES)].copy()
    if valid.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}
    
    valid = valid.merge(driver_map, on="trial_id", how="left")
    valid = valid.dropna(subset=["driver"])
    
    # Calculate metrics
    top1_driver = (valid["premise_attr"] == valid["driver"]).mean()
    top1_weights = (valid["premise_attr"] == top_attr).mean()
    
    counts = valid["premise_attr"].value_counts()
    premise_series = pd.Series({attr: counts.get(attr, 0) for attr in ATTRIBUTES})
    weight_series = pd.Series({attr: weight_map.get(attr, 0.0) for attr in ATTRIBUTES})
    
    if premise_series.sum() == 0 or weight_series.sum() == 0:
        rank_corr = np.nan
    else:
        rank_corr = premise_series.rank().corr(weight_series.rank(), method="spearman")
    
    return {
        "ECRB_top1_driver": float(top1_driver),
        "ECRB_top1_weights": float(top1_weights),
        "rank_corr": float(rank_corr) if not np.isnan(rank_corr) else np.nan,
        "n_valid_premises": len(valid),
        "n_total_responses": len(responses_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Recalculate alignment for minimal effort results")
    parser.add_argument("--model", default="gpt-5-nano", help="Model name")
    parser.add_argument("--effort", default="minimal", help="Reasoning effort level")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both", help="Which split to process")
    args = parser.parse_args()
    
    base_dir = ROOT / "results" / "reasoning_effort_comparison" / args.model / args.effort
    
    splits = ["train", "test"] if args.split == "both" else [args.split]
    
    for split in splits:
        split_dir = base_dir / split
        responses_path = split_dir / f"responses_{args.effort}_{split}.jsonl"
        results_path = split_dir / f"results_{args.effort}_{split}.json"
        
        if not responses_path.exists():
            print(f"Skipping {split}: {responses_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {args.model} / {args.effort} / {split}")
        print(f"{'='*60}")
        
        # Load responses with relaxed parsing
        responses_df = load_responses_relaxed(responses_path)
        
        # Show parsing stats
        original_ok = responses_df["premise_attr"].notna().sum()
        print(f"\nPremise extraction stats:")
        print(f"  Total responses: {len(responses_df)}")
        print(f"  Successfully extracted attributes: {original_ok} ({100*original_ok/len(responses_df):.1f}%)")
        
        # Show attribute distribution
        attr_counts = responses_df["premise_attr"].value_counts()
        print(f"\nAttribute distribution:")
        for attr in ATTRIBUTES:
            count = attr_counts.get(attr, 0)
            print(f"  {attr}: {count} ({100*count/len(responses_df):.1f}%)")
        
        # Load existing results to get the trained model info
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            continue
        
        with results_path.open() as f:
            existing_results = json.load(f)
        
        # Need to rebuild the model from the trials
        # Load trials from the parquet dataset
        dataset_path = ROOT / "data" / "generated" / "v1_short" / "dataset_trials.parquet"
        all_trials_df = pd.read_parquet(dataset_path)
        
        # Get the trial_ids from responses
        trial_ids = responses_df["trial_id"].unique()
        trials_df = all_trials_df[all_trials_df["trial_id"].isin(trial_ids)].copy()
        
        # Aggregate choices and fit model
        choice_agg = responses_df.groupby("trial_id").agg(
            successes=("choice", lambda x: (x == "A").sum()),
            trials=("choice", "count"),
        ).reset_index()
        
        stageA_data = trials_df.merge(choice_agg, on="trial_id", how="inner")
        design = build_design_matrix(stageA_data, exclude_b1=False)
        model = fit_glm_clustered(design)
        
        # Calculate alignment with relaxed parsing
        alignment = recalculate_alignment(responses_df, stageA_data, model)
        
        print(f"\nRecalculated alignment metrics:")
        print(f"  ECRB_top1_driver: {alignment['ECRB_top1_driver']:.4f}")
        print(f"  ECRB_top1_weights: {alignment['ECRB_top1_weights']:.4f}")
        print(f"  rank_corr: {alignment['rank_corr']:.4f}")
        print(f"  (Based on {alignment['n_valid_premises']} / {alignment['n_total_responses']} responses)")
        
        # Compare with weights
        weights_info = compute_ames_and_weights(model)
        print(f"\nModel weights for comparison:")
        for attr, weight in sorted(weights_info["weights"].items(), key=lambda x: -x[1]):
            print(f"  {attr}: {weight:.4f}")
        
        # Update both alignment and alignment_relaxed fields
        # Fill in main alignment field if it's empty
        if not existing_results.get("alignment"):
            existing_results["alignment"] = {
                "ECRB_top1_driver": alignment["ECRB_top1_driver"],
                "ECRB_top1_weights": alignment["ECRB_top1_weights"],
                "rank_corr": alignment["rank_corr"],
            }
            print(f"\nFilled in empty 'alignment' field")
        
        # Also save to alignment_relaxed with full details
        existing_results["alignment_relaxed"] = alignment
        
        with results_path.open("w") as f:
            json.dump(existing_results, f, indent=2)
        print(f"Updated {results_path}")


if __name__ == "__main__":
    main()

