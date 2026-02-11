#!/usr/bin/env python3
"""
Evaluate how Stage A results change as a function of training sample size.

This script reuses collected training responses (JSONL files) and repeatedly
subsamples trials to estimate how weights, alignment, and choice metrics vary
with different sample sizes. The goal is to identify whether metrics stabilize
once enough trials are included (e.g., 200 vs. 250 trials).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_reasoning_efforts import analyze_results  # type: ignore


def load_responses(responses_path: Path) -> list[dict[str, Any]]:
    """Load trial-level responses from a JSONL file."""
    if not responses_path.exists():
        raise FileNotFoundError(f"Missing responses file: {responses_path}")

    responses: list[dict[str, Any]] = []
    with responses_path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
                responses.append(resp)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num} of {responses_path}") from exc
    return responses


def mean_std(values: list[float]) -> dict[str, float]:
    """Return mean and (sample) std for a list of floats."""
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across folds for a single sample size."""
    weights_by_attr: dict[str, list[float]] = defaultdict(list)
    alignment_metrics: dict[str, list[float]] = defaultdict(list)
    choice_variances: list[float] = []
    extreme_rates: list[float] = []

    for rec in records:
        for attr, val in rec["weights"].items():
            weights_by_attr[attr].append(val)
        if rec.get("alignment"):
            for metric, val in rec["alignment"].items():
                alignment_metrics[metric].append(val)
        choice_variances.append(rec.get("choice_variance", 0.0))
        extreme = rec.get("extreme_choices", {})
        total = extreme.get("total", 1) or 1
        extreme_rate = (extreme.get("all_a", 0) + extreme.get("all_b", 0)) / total
        extreme_rates.append(extreme_rate)

    return {
        "weights": {attr: mean_std(vals) for attr, vals in sorted(weights_by_attr.items())},
        "alignment": {metric: mean_std(vals) for metric, vals in sorted(alignment_metrics.items())},
        "choice_variance": mean_std(choice_variances),
        "extreme_choice_rate": mean_std(extreme_rates),
        "folds": len(records),
    }


def main():
    parser = argparse.ArgumentParser(description="Sample-size sensitivity analysis for Stage A results")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Dataset directory containing dataset_trials.parquet")
    parser.add_argument("--responses", default=None, help="Path to responses.jsonl (overrides responses-root lookup)")
    parser.add_argument("--responses-root", default="results/reasoning_effort_comparison", help="Root directory that holds model/effort/train responses")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name (used to locate responses)")
    parser.add_argument("--efforts", default="minimal,low,medium,high", help="Comma-separated effort levels or 'all'")
    parser.add_argument("--sizes", default="50,100,150,200,250", help="Comma-separated trial counts to evaluate")
    parser.add_argument("--folds", type=int, default=5, help="Number of resampling folds per sample size")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed for reproducibility")
    parser.add_argument("--out-dir", default="results/sample_size_curves", help="Directory to write summary JSON files")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    responses_root = Path(args.responses_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()] or ["minimal", "low", "medium", "high"]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    sizes = sorted(set(sizes))

    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    trials_df = trials_df.set_index("trial_id")

    if args.responses:
        efforts = ["custom"]

    for effort in efforts:
        print(f"\n{'='*80}\nEFFORT LEVEL: {effort.upper()}\n{'='*80}")
        if args.responses:
            responses_path = Path(args.responses)
        else:
            responses_path = responses_root / args.model.replace("/", "_") / effort / "train" / f"responses_{effort}_train.jsonl"
        responses = load_responses(responses_path)
        n_available = len(responses)
        print(f"📂 Found {n_available} training trials for {effort}")

        usable_sizes = [s for s in sizes if s <= n_available]
        if not usable_sizes:
            print("⚠️  No requested sample sizes are <= available trials. Skipping.")
            continue

        effort_records: list[dict[str, Any]] = []
        for sample_size in usable_sizes:
            if sample_size < 5:
                print(f"⚠️  Skipping size {sample_size} (too few trials for GLM).")
                continue
            print(f"\n🔬 Evaluating sample size N={sample_size}")
            for fold in range(args.folds):
                fold_seed = args.seed + (sample_size * 100) + fold
                fold_rng = np.random.default_rng(fold_seed)
                indices = fold_rng.choice(n_available, size=sample_size, replace=False)
                subset_responses = [responses[i] for i in indices]
                subset_ids = [resp["trial_id"] for resp in subset_responses]
                subset_trials_df = trials_df.loc[subset_ids].reset_index()
                result = analyze_results(subset_responses, subset_trials_df)
                result.pop("model", None)
                effort_records.append({
                    "size": sample_size,
                    "fold": fold,
                    **result,
                })

        if not effort_records:
            print("⚠️  No records generated (likely due to insufficient trials).")
            continue

        # Summaries by size
        summaries: dict[int, dict[str, Any]] = {}
        for sample_size in usable_sizes:
            recs = [r for r in effort_records if r["size"] == sample_size]
            if recs:
                summaries[sample_size] = summarize_records(recs)

        # Print concise comparison table (weights)
        attrs = ["E", "A", "S", "D"]
        header = "Size".ljust(6) + "".join(f"{attr:^15}" for attr in attrs)
        print(f"\nWeight stability (mean ± std):\n{header}\n{'-' * len(header)}")
        for sample_size in sorted(summaries.keys()):
            row = f"{sample_size:<6}"
            for attr in attrs:
                stats = summaries[sample_size]["weights"].get(attr, {"mean": float('nan'), "std": float('nan')})
                row += f"{stats['mean']:.3f}±{stats['std']:.3f}".center(15)
            print(row)

        output_path = out_dir / f"{effort}_sample_size_curve.json"
        payload = {
            "effort": effort,
            "model": args.model,
            "sizes": usable_sizes,
            "folds": args.folds,
            "records": effort_records,
            "summaries": summaries,
            "responses_path": str(responses_path),
            "dataset": str(dataset_dir),
        }
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n✅ Saved sample-size report to {output_path}")


if __name__ == "__main__":
    main()
