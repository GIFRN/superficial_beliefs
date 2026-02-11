#!/usr/bin/env python3
"""Filter a completed (or partially completed) run to a specific dataset's trial_ids.

Use case:
  - Create a smaller "tradeoff500" dataset (subset of trial_ids),
  - Reuse existing full runs by filtering their responses.jsonl down to those trial_ids,
  - Optionally deduplicate repeated trial_ids (keep last occurrence).

This writes:
  - <out_run>/responses.jsonl
  - <out_run>/MANIFEST.json (copied from source run, with dataset + responses_path updated)
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

from src.utils.io import ensure_dir, write_json


def _load_trial_ids(dataset_dir: Path) -> set[str]:
    trials_path = dataset_dir / "dataset_trials.parquet"
    trials_df = pd.read_parquet(trials_path, columns=["trial_id"])
    return set(trials_df["trial_id"].astype(str).tolist())


def _dedup_filtered_jsonl(in_path: Path, keep_trial_ids: set[str]) -> dict[str, str]:
    latest: dict[str, str] = {}
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("trial_id")
            if tid is None:
                continue
            tid = str(tid)
            if tid not in keep_trial_ids:
                continue
            latest[tid] = line if line.endswith("\n") else f"{line}\n"
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a run to a dataset's trial_ids")
    parser.add_argument("--dataset", required=True, help="Dataset directory (contains dataset_trials.parquet)")
    parser.add_argument("--in-run", required=True, help="Input run directory (contains responses.jsonl)")
    parser.add_argument("--out-run", required=True, help="Output run directory to write filtered responses")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    in_run = Path(args.in_run)
    out_run = ensure_dir(args.out_run)

    in_responses = in_run / "responses.jsonl"
    if not in_responses.exists():
        raise SystemExit(f"Missing responses.jsonl: {in_responses}")

    keep_trial_ids = _load_trial_ids(dataset_dir)
    latest = _dedup_filtered_jsonl(in_responses, keep_trial_ids)

    out_responses = out_run / "responses.jsonl"
    with out_responses.open("w", encoding="utf-8") as out_fh:
        for tid in sorted(latest.keys()):
            out_fh.write(latest[tid])

    # Copy and adjust MANIFEST.json (if present).
    in_manifest = in_run / "MANIFEST.json"
    manifest: dict = {}
    if in_manifest.exists():
        try:
            manifest = json.loads(in_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.update(
        {
            "dataset": str(dataset_dir),
            "responses_path": str(out_responses),
            "source_run": str(in_run),
            "filtered_trials": len(latest),
        }
    )
    write_json(manifest, out_run / "MANIFEST.json")

    print(f"✅ Filtered {len(latest)} trials to {out_run}")


if __name__ == "__main__":
    main()

