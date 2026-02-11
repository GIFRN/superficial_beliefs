#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from src.analysis.features import aggregate_choices, load_responses, prepare_stageA_data
from src.analysis.stageA import build_design_matrix, fit_glm_clustered
from src.analysis.stageB import alignment_metrics, probe_deltas_and_pivots
from src.utils.io import ensure_dir, write_json


class PrefitModel:
    """Lightweight wrapper to carry prefit Stage A parameters into Stage B."""

    def __init__(self, params: dict[str, float], bse: dict[str, float] | None, feature_info: dict):
        self.params = pd.Series(params, dtype=float)
        if bse:
            self.bse = pd.Series(bse, dtype=float)
        else:
            self.bse = pd.Series({k: 0.0 for k in self.params.index}, dtype=float)
        self.feature_info = feature_info or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Stage B alignment metrics")
    parser.add_argument("--stageA", default="results/stage_A_openai_gpt5mini_high", help="Directory containing Stage A outputs")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing prefit stageA_summary.json (defaults to --stageA)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional evaluation dataset directory. If provided, build trial features from this dataset + --responses (useful for held-out test evaluation).",
    )
    parser.add_argument("--responses", default="data/runs/v1_short_openai_gpt5mini_high/responses.jsonl", help="Path to responses JSONL file")
    parser.add_argument("--out", default=None, help="Directory to write Stage B outputs (auto-generated if not specified)")
    parser.add_argument("--interactions", action="store_true", help="Include pairwise interactions (must match Stage A)")
    args = parser.parse_args()

    stageA_dir = Path(args.stageA)
    responses_df = load_responses(args.responses)
    
    # Trial-level features for evaluation must match the response trial_ids.
    # For held-out testing, Stage A may be fit on train split while Stage B is evaluated on test split.
    # In that case, pass --dataset to build features from the evaluation dataset.
    if args.dataset:
        dataset_dir = Path(args.dataset)
        trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
        choice_agg = aggregate_choices(responses_df)
        stageA_df = prepare_stageA_data(trials_df, choice_agg)
    else:
        stageA_df = pd.read_parquet(stageA_dir / "stageA_design.parquet")

    model_dir = Path(args.model_dir) if args.model_dir else stageA_dir
    stageA_summary_path = model_dir / "stageA_summary.json"
    prefit_model = None
    if stageA_summary_path.exists():
        try:
            stageA_summary = json.loads(stageA_summary_path.read_text())
            params = stageA_summary.get("model_params")
            feature_info = stageA_summary.get("feature_info")
            if params and feature_info:
                prefit_model = PrefitModel(
                    params=params,
                    bse=stageA_summary.get("model_bse"),
                    feature_info=feature_info,
                )
        except json.JSONDecodeError:
            prefit_model = None
    
    # Load model information from responses MANIFEST.json
    responses_path = Path(args.responses)
    manifest_path = responses_path.parent / "MANIFEST.json"
    model_name = "unknown"
    reasoning_effort = None
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
            model_name = manifest.get("backend", "unknown")
            reasoning_effort = manifest.get("reasoning_effort", None)
    
    # Construct output directory name if not specified
    if args.out is None:
        suffix = "_interactions" if args.interactions else ""
        # Include reasoning effort if present
        if reasoning_effort:
            out_dirname = f"stage_B_{model_name}_{reasoning_effort}{suffix}"
        else:
            out_dirname = f"stage_B_{model_name}{suffix}"
        args.out = f"results/{out_dirname}"
    
    print(f"Model: {model_name}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Interactions: {'enabled' if args.interactions else 'disabled'}")
    print(f"Output directory: {args.out}")
    if args.model_dir:
        print(f"Model directory: {model_dir}")
    if args.dataset:
        print(f"Evaluation dataset: {args.dataset}")

    if prefit_model is not None:
        print("Using prefit Stage A parameters (no baseline refit).")
        model = prefit_model
        alignment = alignment_metrics(responses_df, stageA_df, model, refit_baseline=False)
        probes = {
            "note": "Skipped probe refits to avoid test-set leakage; run on train split for probes."
        }
    else:
        design = build_design_matrix(stageA_df, include_interactions=args.interactions)
        model = fit_glm_clustered(design)
        alignment = alignment_metrics(responses_df, stageA_df, model, refit_baseline=True)
        probes = probe_deltas_and_pivots(stageA_df)

    out_dir = ensure_dir(args.out)
    summary = {
        "model": model_name,
        "include_interactions": args.interactions,
        "alignment": alignment,
        "probes": probes
    }
    if args.dataset:
        summary["eval_dataset"] = args.dataset
    
    # Add reasoning_effort to summary if present
    if reasoning_effort:
        summary["reasoning_effort"] = reasoning_effort
    write_json(summary, out_dir / "stageB_summary.json")
    
    print(f"\n✅ Stage B analysis complete!")
    print(f"   Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
