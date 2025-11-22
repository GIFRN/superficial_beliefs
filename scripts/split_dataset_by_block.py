#!/usr/bin/env python3
"""Split a dataset into subsets based on trial blocks.

This script creates filtered versions of a dataset by including or excluding
specific trial blocks (B1, B2, B3).
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
    """Save filtered dataset with updated manifest."""
    ensure_dir(output_dir)
    
    configs_df.to_parquet(output_dir / "dataset_configs.parquet", index=False)
    trials_df.to_parquet(output_dir / "dataset_trials.parquet", index=False)
    
    with (output_dir / "MANIFEST.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Saved {len(configs_df)} configs and {len(trials_df)} trials")


def filter_dataset(configs_df: pd.DataFrame, trials_df: pd.DataFrame, 
                   manifest: dict, blocks_to_include: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter dataset to only include specified blocks."""
    # Filter trials by block
    filtered_trials = trials_df[trials_df["block"].isin(blocks_to_include)].copy()
    
    # Get config_ids that are used in the filtered trials
    used_config_ids = filtered_trials["config_id"].unique()
    filtered_configs = configs_df[configs_df["config_id"].isin(used_config_ids)].copy()
    
    # Update manifest
    new_manifest = manifest.copy()
    new_manifest["n_trials"] = len(filtered_trials)
    new_manifest["n_configs"] = len(filtered_configs)
    new_manifest["actual_total"] = len(filtered_trials)
    
    # Update block counts
    block_counts = filtered_trials["block"].value_counts().to_dict()
    new_manifest["blocks"] = {block: block_counts.get(block, 0) for block in ["B1", "B2", "B3"]}
    
    # Add filter information
    new_manifest["filtered_from"] = str(manifest.get("source", "unknown"))
    new_manifest["included_blocks"] = blocks_to_include
    
    return filtered_configs, filtered_trials, new_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset by trial blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset without B1 trials
  python scripts/split_dataset_by_block.py --dataset data/generated/v1_short --exclude B1 --out data/generated/v1_short_no_b1
  
  # Create dataset with only B1 trials
  python scripts/split_dataset_by_block.py --dataset data/generated/v1_short --include B1 --out data/generated/v1_short_b1_only
  
  # Create dataset with only B2 and B3 trials
  python scripts/split_dataset_by_block.py --dataset data/generated/v1_short --include B2 B3 --out data/generated/v1_short_b2_b3
        """
    )
    parser.add_argument("--dataset", required=True, help="Path to source dataset directory")
    parser.add_argument("--include", nargs="+", choices=["B1", "B2", "B3"], 
                       help="Blocks to include in output dataset")
    parser.add_argument("--exclude", nargs="+", choices=["B1", "B2", "B3"],
                       help="Blocks to exclude from output dataset")
    parser.add_argument("--out", required=True, help="Path to output dataset directory")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.include and args.exclude:
        parser.error("Cannot specify both --include and --exclude")
    if not args.include and not args.exclude:
        parser.error("Must specify either --include or --exclude")
    
    # Determine which blocks to include
    all_blocks = ["B1", "B2", "B3"]
    if args.include:
        blocks_to_include = args.include
    else:  # args.exclude
        blocks_to_include = [b for b in all_blocks if b not in args.exclude]
    
    # Load source dataset
    dataset_dir = Path(args.dataset)
    print(f"Loading dataset from: {dataset_dir}")
    configs_df, trials_df, manifest = load_dataset(dataset_dir)
    
    print(f"Source dataset:")
    print(f"  Total trials: {len(trials_df)}")
    print(f"  Total configs: {len(configs_df)}")
    if "blocks" in manifest:
        for block, count in manifest["blocks"].items():
            print(f"  {block}: {count} trials")
    
    # Filter dataset
    print(f"\nFiltering to include blocks: {', '.join(blocks_to_include)}")
    filtered_configs, filtered_trials, new_manifest = filter_dataset(
        configs_df, trials_df, manifest, blocks_to_include
    )
    
    # Save filtered dataset
    output_dir = Path(args.out)
    print(f"\nSaving filtered dataset to: {output_dir}")
    save_dataset(filtered_configs, filtered_trials, new_manifest, output_dir)
    
    print(f"\nFiltered dataset:")
    print(f"  Total trials: {len(filtered_trials)}")
    print(f"  Total configs: {len(filtered_configs)}")
    for block, count in new_manifest["blocks"].items():
        if count > 0:
            print(f"  {block}: {count} trials")
    
    print("\n✅ Dataset split complete!")


if __name__ == "__main__":
    main()

