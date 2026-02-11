#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_json


def _block_counts(trials_df: pd.DataFrame) -> dict[str, int]:
    counts = trials_df["block"].value_counts().to_dict()
    return {k: int(v) for k, v in counts.items()}


def _load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "MANIFEST.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_dataset_split(
    *,
    split_name: str,
    trials_df: pd.DataFrame,
    configs_df: pd.DataFrame,
    base_manifest: dict,
    out_dir: Path,
    source_dataset: Path,
    group_by: str,
    test_fraction: float,
    seed: int,
) -> None:
    ensure_dir(out_dir)
    trials_df.to_parquet(out_dir / "dataset_trials.parquet", index=False)
    configs_df.to_parquet(out_dir / "dataset_configs.parquet", index=False)

    manifest = dict(base_manifest)
    manifest.update(
        {
            "source_dataset": str(source_dataset),
            "split": split_name,
            "split_group_by": group_by,
            "split_test_fraction": test_fraction,
            "split_seed": seed,
            "n_trials": int(len(trials_df)),
            "n_configs": int(configs_df["config_id"].nunique()),
            "actual_total": int(len(trials_df)),
            "target_total": int(len(trials_df)),
            "blocks": _block_counts(trials_df),
        }
    )
    write_json(manifest, out_dir / "MANIFEST.json")


def _split_responses(
    *,
    responses_path: Path,
    train_trial_ids: set[str],
    test_trial_ids: set[str],
    out_train: Path,
    out_test: Path,
) -> tuple[int, int]:
    if not responses_path.exists():
        return 0, 0

    n_train = 0
    n_test = 0
    with responses_path.open("r", encoding="utf-8") as fh, out_train.open(
        "w", encoding="utf-8"
    ) as train_fh, out_test.open("w", encoding="utf-8") as test_fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            trial_id = payload.get("trial_id")
            if trial_id in train_trial_ids:
                train_fh.write(line)
                n_train += 1
            elif trial_id in test_trial_ids:
                test_fh.write(line)
                n_test += 1
    return n_train, n_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Create held-out train/test splits from an existing dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--responses", default=None, help="Optional path to responses.jsonl to split alongside the dataset")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root directory. Defaults to data/splits/<dataset_name>_holdout",
    )
    parser.add_argument("--group-by", default="config_id", help="Column to group by for splitting (default: config_id)")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of groups assigned to test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for group split")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    trials_path = dataset_dir / "dataset_trials.parquet"
    configs_path = dataset_dir / "dataset_configs.parquet"

    trials_df = pd.read_parquet(trials_path)
    configs_df = pd.read_parquet(configs_path)
    base_manifest = _load_manifest(dataset_dir)

    group_by = args.group_by
    if group_by not in trials_df.columns:
        raise KeyError(f"group-by column '{group_by}' not found in trials dataframe")

    unique_groups = trials_df[group_by].dropna().astype(str).unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)

    n_test_groups = max(1, int(round(len(unique_groups) * args.test_fraction)))
    test_groups = set(unique_groups[:n_test_groups])
    train_groups = set(unique_groups[n_test_groups:])

    trials_grouped = trials_df.copy()
    trials_grouped[group_by] = trials_grouped[group_by].astype(str)
    train_trials = trials_grouped[trials_grouped[group_by].isin(train_groups)].copy()
    test_trials = trials_grouped[trials_grouped[group_by].isin(test_groups)].copy()

    train_config_ids = set(train_trials["config_id"].astype(str).unique())
    test_config_ids = set(test_trials["config_id"].astype(str).unique())
    train_configs = configs_df[configs_df["config_id"].astype(str).isin(train_config_ids)].copy()
    test_configs = configs_df[configs_df["config_id"].astype(str).isin(test_config_ids)].copy()

    dataset_name = dataset_dir.name
    out_root = Path(args.out_root) if args.out_root else Path("data/splits") / f"{dataset_name}_holdout"
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    print(f"Source dataset: {dataset_dir}")
    print(f"Groups: {len(unique_groups)} total -> {len(train_groups)} train / {len(test_groups)} test")
    print(f"Trials: {len(train_trials)} train / {len(test_trials)} test")

    _write_dataset_split(
        split_name="train",
        trials_df=train_trials,
        configs_df=train_configs,
        base_manifest=base_manifest,
        out_dir=train_dir,
        source_dataset=dataset_dir,
        group_by=group_by,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    _write_dataset_split(
        split_name="test",
        trials_df=test_trials,
        configs_df=test_configs,
        base_manifest=base_manifest,
        out_dir=test_dir,
        source_dataset=dataset_dir,
        group_by=group_by,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    if args.responses:
        responses_path = Path(args.responses)
        train_trial_ids = set(train_trials["trial_id"].astype(str))
        test_trial_ids = set(test_trials["trial_id"].astype(str))
        n_train, n_test = _split_responses(
            responses_path=responses_path,
            train_trial_ids=train_trial_ids,
            test_trial_ids=test_trial_ids,
            out_train=train_dir / "responses.jsonl",
            out_test=test_dir / "responses.jsonl",
        )
        print(f"Responses: {n_train} train / {n_test} test written")

    print(f"✅ Holdout split written to: {out_root}")


if __name__ == "__main__":
    main()

