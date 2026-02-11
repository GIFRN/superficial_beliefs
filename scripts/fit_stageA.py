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

from src.analysis.cv import cross_validate_design
from src.analysis.diagnostics import evaluate_model, validate_b1_rationality, validate_b1_probes
from src.analysis.features import aggregate_choices, load_responses, prepare_stageA_data
from src.analysis.stageA import (
    build_design_matrix,
    compute_ames_and_weights,
    fit_glm_clustered,
    fit_stageA_with_validation,
    per_trial_contributions,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage A logistic model")
    parser.add_argument("--config", default="configs/default.yml", help="Path to dataset configuration YAML")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Path to generated dataset directory")
    parser.add_argument("--responses", default="data/runs/v1_short_openai_gpt5mini_high/responses.jsonl", help="Path to responses JSONL file")
    parser.add_argument("--out", default=None, help="Output directory for Stage A fit (auto-generated if not specified)")
    parser.add_argument("--interactions", action="store_true", help="Include pairwise interactions")
    parser.add_argument("--no-fit", action="store_true", help="Do not fit a model; just materialize stageA_design.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_dir = Path(args.dataset)
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    responses_df = load_responses(args.responses)
    choice_agg = aggregate_choices(responses_df)
    
    # Load model information from MANIFEST.json
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
            out_dirname = f"stage_A_{model_name}_{reasoning_effort}{suffix}"
        else:
            out_dirname = f"stage_A_{model_name}{suffix}"
        args.out = f"results/{out_dirname}"
    
    print(f"Model: {model_name}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Interactions: {'enabled' if args.interactions else 'disabled'}")
    print(f"Output directory: {args.out}")
    
    # Perform B1 validation
    print("Validating B1 rationality...")
    b1_validation = validate_b1_rationality(trials_df, choice_agg)
    b1_probes = validate_b1_probes(trials_df, choice_agg)
    
    if not b1_validation["rationality_check_passed"]:
        print("WARNING: B1 rationality check failed!")
        print(f"Accuracy: {b1_validation['accuracy']:.1%}")
        print(f"Failure rate: {b1_validation['failure_rate']:.1%}")
        print(f"Failed trials: {len(b1_validation['failed_trials'])}")
        print("Proceeding with analysis but results may be unreliable.")
    else:
        print(f"✅ B1 rationality check passed - {b1_validation['accuracy']:.1%} accuracy")
    
    if not b1_probes["probe_effectiveness"]:
        print("WARNING: B1 probes appear ineffective!")
        print(f"Probe effect size: {b1_probes['probe_effect_size']:.3f}")
        print("Probes may not be affecting model behavior as expected.")
    else:
        print("✅ B1 probes are effective")
    
    # Use the new validation function
    out_dir = ensure_dir(args.out)
    stageA_path = out_dir / "stageA_design.parquet"
    if args.no_fit:
        stageA_df = prepare_stageA_data(trials_df, choice_agg)
        stageA_df.to_parquet(stageA_path, index=False)
        summary = {
            "model": model_name,
            "include_interactions": args.interactions,
            "b1_validation": b1_validation,
            "b1_probes": b1_probes,
            "note": "no_fit=True: stageA_design.parquet materialized without fitting.",
        }
    else:
        stageA_result = fit_stageA_with_validation(trials_df, choice_agg, include_interactions=args.interactions)

        model = stageA_result["model"]
        design = stageA_result["design_matrix"]
        weights_info = stageA_result["ames_weights"]
        contributions = stageA_result["contributions"]
        stageA_df = stageA_result["stageA_data"]

        cv_metrics = cross_validate_design(design)
        eval_metrics = evaluate_model(model, design)

        contributions = contributions.reset_index().rename(columns={"index": "trial_id"})
        predictions = model.predict(design.X)
        stageA_df = stageA_df.copy()
        stageA_df["predicted_prob"] = predictions

        contributions_path = out_dir / "stageA_contributions.parquet"
        contributions.to_parquet(contributions_path, index=False)
        stageA_df.to_parquet(stageA_path, index=False)

        summary = {
            "model": model_name,
            "include_interactions": args.interactions,
            "weights": weights_info["weights"],
            "beta": weights_info["beta"],
            "AME": weights_info["AME"],
            "cv": cv_metrics,
            "evaluation": eval_metrics,
            "b1_validation": b1_validation,
            "b1_probes": b1_probes,
            # Persist model parameters so Stage B can evaluate on held-out splits without refitting.
            "model_params": {k: float(v) for k, v in model.params.items()},
            "model_bse": {k: float(v) for k, v in model.bse.items()},
            "feature_columns": list(design.X.columns),
            "feature_info": design.feature_info,
        }
    
    # Add reasoning_effort to summary if present
    if reasoning_effort:
        summary["reasoning_effort"] = reasoning_effort
    write_json(summary, out_dir / "stageA_summary.json")
    
    print(f"\n✅ Analysis complete!")
    print(f"   Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
