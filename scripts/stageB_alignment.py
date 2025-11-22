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

from src.analysis.features import load_responses
from src.analysis.stageA import build_design_matrix, fit_glm_clustered
from src.analysis.stageB import alignment_metrics, probe_deltas_and_pivots
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Stage B alignment metrics")
    parser.add_argument("--stageA", default="results/stage_A_openai_gpt5mini_high", help="Directory containing Stage A outputs")
    parser.add_argument("--responses", default="data/runs/v1_short_openai_gpt5mini_high/responses.jsonl", help="Path to responses JSONL file")
    parser.add_argument("--out", default=None, help="Directory to write Stage B outputs (auto-generated if not specified)")
    parser.add_argument("--interactions", action="store_true", help="Include pairwise interactions (must match Stage A)")
    args = parser.parse_args()

    stageA_dir = Path(args.stageA)
    stageA_df = pd.read_parquet(stageA_dir / "stageA_design.parquet")
    responses_df = load_responses(args.responses)
    
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

    design = build_design_matrix(stageA_df, include_interactions=args.interactions)
    model = fit_glm_clustered(design)

    alignment = alignment_metrics(responses_df, stageA_df, model)
    probes = probe_deltas_and_pivots(stageA_df)

    out_dir = ensure_dir(args.out)
    summary = {
        "model": model_name,
        "include_interactions": args.interactions,
        "alignment": alignment,
        "probes": probes
    }
    
    # Add reasoning_effort to summary if present
    if reasoning_effort:
        summary["reasoning_effort"] = reasoning_effort
    write_json(summary, out_dir / "stageB_summary.json")
    
    print(f"\n✅ Stage B analysis complete!")
    print(f"   Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
