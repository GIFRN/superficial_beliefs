#!/usr/bin/env python3
"""Create a tradeoff-only subset of an existing dataset (optionally with matching responses).

This is meant to:
  - remove Pareto-dominant / trivial trials (no real tradeoff),
  - cap to a small, clear N (e.g. 500) without regenerating the dataset,
  - preserve trial_id values so existing runs can be reused for analysis.
"""
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

ATTRS = ["E", "A", "S", "D"]


def _pick_delta_cols(trials_df: pd.DataFrame, *, delta_source: str) -> list[str]:
    """Pick which delta columns to use for tradeoff detection.

    delta_source:
      - "shown": use prompt-visible deltas (delta_*), appropriate for filtering out dominated prompts.
      - "base": use underlying deltas (delta_base_*), if present.
      - "auto": prefer base when available (legacy behavior).
    """
    shown_cols = [f"delta_{a}" for a in ATTRS]
    base_cols = [f"delta_base_{a}" for a in ATTRS]

    if delta_source == "shown":
        missing = [c for c in shown_cols if c not in trials_df.columns]
        if missing:
            raise SystemExit(f"Missing shown delta columns: {missing}")
        return shown_cols
    if delta_source == "base":
        missing = [c for c in base_cols if c not in trials_df.columns]
        if missing:
            raise SystemExit(f"Missing base delta columns: {missing}")
        return base_cols
    if delta_source != "auto":
        raise SystemExit(f"Unsupported delta_source: {delta_source}")

    if all(col in trials_df.columns for col in base_cols):
        return base_cols
    missing = [c for c in shown_cols if c not in trials_df.columns]
    if missing:
        raise SystemExit(f"Missing shown delta columns: {missing}")
    return shown_cols


def _tradeoff_mask(trials_df: pd.DataFrame, delta_cols: list[str]) -> pd.Series:
    deltas = trials_df[delta_cols].fillna(0)
    pos = (deltas > 0).any(axis=1)
    neg = (deltas < 0).any(axis=1)
    return pos & neg


def _balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    if "labelA" not in df.columns:
        return df.sample(n=n, random_state=seed).copy()
    # Balance A/B labelA if possible.
    a_df = df[df["labelA"] == "A"]
    b_df = df[df["labelA"] == "B"]
    if a_df.empty or b_df.empty:
        return df.sample(n=n, random_state=seed).copy()
    n_a = min(len(a_df), n // 2)
    n_b = min(len(b_df), n - n_a)
    sampled = pd.concat(
        [
            a_df.sample(n=n_a, random_state=seed),
            b_df.sample(n=n_b, random_state=seed + 1),
        ],
        ignore_index=False,
    )
    if len(sampled) < n:
        remaining = df.drop(index=sampled.index)
        if not remaining.empty:
            extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed + 2)
            sampled = pd.concat([sampled, extra], ignore_index=False)
    return sampled.sample(frac=1.0, random_state=seed + 3).copy()


def _filter_responses(in_path: Path, out_path: Path, keep_trial_ids: set[str]) -> int:
    n_written = 0
    with in_path.open("r", encoding="utf-8") as in_fh, out_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("trial_id")
            if tid is None:
                continue
            if str(tid) in keep_trial_ids:
                out_fh.write(line)
                if not line.endswith("\n"):
                    out_fh.write("\n")
                n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tradeoff-only subset of a dataset")
    parser.add_argument("--dataset", required=True, help="Path to source dataset directory")
    parser.add_argument("--out", required=True, help="Path to output dataset directory")
    parser.add_argument("--n", type=int, default=500, help="Number of trials to keep (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "--delta-source",
        choices=["shown", "base", "auto"],
        default="shown",
        help="Which deltas define a 'tradeoff' (default: shown).",
    )
    parser.add_argument(
        "--manipulations",
        default=None,
        help="Comma-separated manipulation values to include (e.g. short_reason). Default: all.",
    )
    parser.add_argument(
        "--blocks",
        default=None,
        help="Comma-separated blocks to include (e.g. B2,B3). Default: all blocks.",
    )
    parser.add_argument(
        "--responses",
        default=None,
        help="Optional path to responses.jsonl to filter alongside the dataset",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    configs_df = pd.read_parquet(dataset_dir / "dataset_configs.parquet")

    delta_cols = _pick_delta_cols(trials_df, delta_source=args.delta_source)
    mask = _tradeoff_mask(trials_df, delta_cols)
    filtered = trials_df[mask].copy()

    if args.manipulations:
        manips = {m.strip() for m in args.manipulations.split(",") if m.strip()}
        filtered = filtered[filtered["manipulation"].isin(manips)].copy()

    if args.blocks:
        blocks = {b.strip() for b in args.blocks.split(",") if b.strip()}
        filtered = filtered[filtered["block"].isin(blocks)].copy()

    if filtered.empty:
        raise SystemExit("No tradeoff trials found after filtering.")

    subset_trials = _balanced_sample(filtered, args.n, args.seed)
    subset_trials = subset_trials.sort_values("trial_id").reset_index(drop=True)

    keep_config_ids = set(subset_trials["config_id"].astype(str).unique())
    subset_configs = configs_df[configs_df["config_id"].astype(str).isin(keep_config_ids)].copy()
    subset_configs = subset_configs.sort_values("config_id").reset_index(drop=True)

    out_dir = ensure_dir(args.out)
    subset_configs.to_parquet(out_dir / "dataset_configs.parquet", index=False)
    subset_trials.to_parquet(out_dir / "dataset_trials.parquet", index=False)

    manifest = {}
    source_manifest_path = dataset_dir / "MANIFEST.json"
    if source_manifest_path.exists():
        try:
            manifest = json.loads(source_manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest.update(
        {
            "subsampled_from": str(dataset_dir),
            "sampling_seed": int(args.seed),
            "filter": {
                "type": "tradeoff_only",
                "delta_source": args.delta_source,
                "delta_cols": delta_cols,
                "manipulations": args.manipulations,
                "blocks": args.blocks,
            },
            "n_trials": int(len(subset_trials)),
            "n_configs": int(len(subset_configs)),
            "actual_total": int(len(subset_trials)),
            "target_total": int(len(subset_trials)),
            "blocks": subset_trials["block"].value_counts().to_dict(),
        }
    )
    write_json(manifest, out_dir / "MANIFEST.json")

    if args.responses:
        responses_path = Path(args.responses)
        if responses_path.exists():
            keep_ids = set(subset_trials["trial_id"].astype(str))
            n_written = _filter_responses(responses_path, out_dir / "responses.jsonl", keep_ids)
            print(f"Filtered responses: wrote {n_written} lines to {out_dir / 'responses.jsonl'}")

    print(f"✅ Wrote tradeoff subset: {out_dir}  (trials={len(subset_trials)}, configs={len(subset_configs)})")


if __name__ == "__main__":
    main()
