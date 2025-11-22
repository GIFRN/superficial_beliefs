#!/usr/bin/env python3
"""Subsample a dataset to a target number of trials.

This script randomly samples trials from specified blocks to reduce
dataset size while maintaining balance and reproducibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load dataset configs, trials, and manifest."""
    configs_path = dataset_dir / "dataset_configs.parquet"
    trials_path = dataset_dir / "dataset_trials.parquet"
    manifest_path = dataset_dir / "MANIFEST.json"
    
    configs_df = pd.read_parquet(configs_path)
    trials_df = pd.read_parquet(trials_path)
    
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    
    return configs_df, trials_df, manifest


def save_dataset(configs_df: pd.DataFrame, trials_df: pd.DataFrame, 
                 manifest: dict, output_dir: Path) -> None:
    """Save subsampled dataset with updated manifest."""
    ensure_dir(output_dir)
    
    configs_df.to_parquet(output_dir / "dataset_configs.parquet", index=False)
    trials_df.to_parquet(output_dir / "dataset_trials.parquet", index=False)
    
    with (output_dir / "MANIFEST.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Saved {len(configs_df)} configs and {len(trials_df)} trials")


def subsample_trials(trials_df: pd.DataFrame, block_samples: dict[str, int], 
                     seed: int = 42) -> pd.DataFrame:
    """Randomly subsample trials from each block.
    
    Args:
        trials_df: DataFrame of trials
        block_samples: Dict mapping block name to desired number of samples
        seed: Random seed for reproducibility
    
    Returns:
        Subsampled DataFrame
    """
    sampled_dfs = []
    
    for block, n_samples in block_samples.items():
        block_trials = trials_df[trials_df["block"] == block]
        
        if len(block_trials) == 0:
            continue
        
        if n_samples is None or n_samples >= len(block_trials):
            # Keep all trials from this block
            sampled_dfs.append(block_trials)
            print(f"  {block}: keeping all {len(block_trials)} trials")
        else:
            # Sample n_samples trials
            sampled = block_trials.sample(n=n_samples, random_state=seed)
            sampled_dfs.append(sampled)
            print(f"  {block}: sampled {n_samples} from {len(block_trials)} trials")
    
    # Concatenate and sort by original index to maintain order
    result = pd.concat(sampled_dfs, ignore_index=False).sort_index()
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsample dataset to target size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reduce to 400 trials by removing from B3
  python scripts/subsample_dataset.py \\
    --dataset data/generated/v1_short_no_b1 \\
    --b2 136 --b3 264 \\
    --out data/generated/v1_short_400
  
  # Keep all B2, sample 200 from B3
  python scripts/subsample_dataset.py \\
    --dataset data/generated/v1_short_no_b1 \\
    --b3 200 \\
    --out data/generated/v1_short_336
  
  # Proportional reduction across blocks
  python scripts/subsample_dataset.py \\
    --dataset data/generated/v1_short_no_b1 \\
    --b2 129 --b3 271 \\
    --out data/generated/v1_short_400_proportional
        """
    )
    parser.add_argument("--dataset", required=True, help="Path to source dataset directory")
    parser.add_argument("--out", required=True, help="Path to output dataset directory")
    parser.add_argument("--b1", type=int, default=None, help="Number of B1 trials to keep (None = all)")
    parser.add_argument("--b2", type=int, default=None, help="Number of B2 trials to keep (None = all)")
    parser.add_argument("--b3", type=int, default=None, help="Number of B3 trials to keep (None = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    
    args = parser.parse_args()
    
    # Load source dataset
    dataset_dir = Path(args.dataset)
    print(f"Loading dataset from: {dataset_dir}")
    configs_df, trials_df, manifest = load_dataset(dataset_dir)
    
    print(f"\nSource dataset:")
    print(f"  Total trials: {len(trials_df)}")
    print(f"  Total configs: {len(configs_df)}")
    block_counts = trials_df["block"].value_counts().to_dict()
    for block in ["B1", "B2", "B3"]:
        if block in block_counts:
            print(f"  {block}: {block_counts[block]} trials")
    
    # Prepare block sampling specification
    block_samples = {
        "B1": args.b1,
        "B2": args.b2,
        "B3": args.b3,
    }
    
    print(f"\nSubsampling with seed={args.seed}:")
    sampled_trials = subsample_trials(trials_df, block_samples, args.seed)
    
    # Get config_ids that are used in the sampled trials
    used_config_ids = sampled_trials["config_id"].unique()
    sampled_configs = configs_df[configs_df["config_id"].isin(used_config_ids)].copy()
    
    # Update manifest
    new_manifest = manifest.copy()
    new_manifest["n_trials"] = len(sampled_trials)
    new_manifest["n_configs"] = len(sampled_configs)
    new_manifest["actual_total"] = len(sampled_trials)
    
    # Update block counts
    new_block_counts = sampled_trials["block"].value_counts().to_dict()
    new_manifest["blocks"] = {block: new_block_counts.get(block, 0) for block in ["B1", "B2", "B3"]}
    
    # Add sampling information
    new_manifest["subsampled_from"] = str(dataset_dir)
    new_manifest["sampling_seed"] = args.seed
    new_manifest["target_samples"] = {
        block: count for block, count in block_samples.items() if count is not None
    }
    
    # Save subsampled dataset
    output_dir = Path(args.out)
    print(f"\nSaving subsampled dataset to: {output_dir}")
    save_dataset(sampled_configs, sampled_trials, new_manifest, output_dir)
    
    print(f"\nSubsampled dataset:")
    print(f"  Total trials: {len(sampled_trials)}")
    print(f"  Total configs: {len(sampled_configs)}")
    for block, count in new_manifest["blocks"].items():
        if count > 0:
            print(f"  {block}: {count} trials")
    
    print("\n✅ Dataset subsampling complete!")


if __name__ == "__main__":
    main()

