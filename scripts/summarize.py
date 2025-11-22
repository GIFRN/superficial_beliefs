#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from src.analysis.diagnostics import delta_correlation, order_balance
from src.analysis.reporting import make_report
import json



def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage A and B outputs")
    parser.add_argument("--stageA", default="results/stage_A_openai_gpt5mini_high", help="Directory with Stage A outputs")
    parser.add_argument("--stageB", default=None, help="Directory with Stage B outputs (auto-detected if not specified)")
    parser.add_argument("--report-dir", default=None, help="Directory to write report (auto-generated if not specified)")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Dataset directory for diagnostics")
    parser.add_argument("--run-dir", default=None, help="Original run directory path (auto-detected if not specified)")
    args = parser.parse_args()

    stageA_dir = Path(args.stageA)
    
    # Load model name from Stage A summary
    stageA_summary = json.loads((stageA_dir / "stageA_summary.json").read_text())
    model_name = stageA_summary.get("model", "unknown")
    has_interactions = stageA_summary.get("include_interactions", False)
    reasoning_effort = stageA_summary.get("reasoning_effort", None)
    
    # Auto-detect Stage B directory if not specified
    if args.stageB is None:
        suffix = "_interactions" if has_interactions else ""
        # Include reasoning effort if present
        if reasoning_effort:
            args.stageB = f"results/stage_B_{model_name}_{reasoning_effort}{suffix}"
        else:
            args.stageB = f"results/stage_B_{model_name}{suffix}"
    stageB_dir = Path(args.stageB)
    
    # Auto-detect run directory if not specified
    if args.run_dir is None:
        dataset_name = Path(args.dataset).name
        # Include reasoning effort if present
        if reasoning_effort:
            args.run_dir = f"data/runs/{dataset_name}_{model_name}_{reasoning_effort}"
        else:
            args.run_dir = f"data/runs/{dataset_name}_{model_name}"
    
    # Auto-generate report directory if not specified
    if args.report_dir is None:
        dataset_name = Path(args.dataset).name
        suffix = "_interactions" if has_interactions else ""
        # Include reasoning effort if present
        if reasoning_effort:
            args.report_dir = f"results/{dataset_name}_{model_name}_{reasoning_effort}{suffix}"
        else:
            args.report_dir = f"results/{dataset_name}_{model_name}{suffix}"
    
    print(f"Model: {model_name}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Stage A: {args.stageA}")
    print(f"Stage B: {args.stageB}")
    print(f"Report directory: {args.report_dir}")
    
    artifacts = {}
    artifacts["stageA"] = stageA_summary

    stageB_summary = json.loads((stageB_dir / "stageB_summary.json").read_text())
    artifacts["stageB"] = stageB_summary

    diagnostics = {}
    dataset_path = args.dataset
    if dataset_path:
        stageA_design = pd.read_parquet(stageA_dir / "stageA_design.parquet")
        diagnostics["delta_correlation"] = delta_correlation(stageA_design).round(3).to_dict()
        diagnostics["order_balance"] = order_balance(stageA_design)
    if diagnostics:
        artifacts["diagnostics"] = diagnostics

    report_path = make_report(args.run_dir, args.report_dir, artifacts)
    print(f"\n✅ Report written to {report_path}")


if __name__ == "__main__":
    main()
