#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}


def _collect_stageb(glob_pattern: str) -> dict[str, Any]:
    paths = sorted(Path().glob(glob_pattern))
    metrics: dict[str, list[float]] = {
        "ECRB_top1_driver": [],
        "ECRB_top1_weights": [],
        "rank_corr": [],
    }
    entries: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        alignment = data.get("alignment", {})
        entry = {
            "path": str(path),
            "alignment": alignment,
        }
        entries.append(entry)
        for key in list(metrics.keys()):
            val = alignment.get(key)
            if val is not None:
                metrics[key].append(float(val))

    return {
        "paths": [str(p) for p in paths],
        "metrics": {k: _mean_std(v) for k, v in metrics.items()},
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize robustness diagnostics outputs.")
    parser.add_argument("--stageA-train", required=True, help="Path to train stageA_summary.json")
    parser.add_argument("--stageB-test-glob", required=True, help="Glob for test stageB_summary.json files")
    parser.add_argument("--stageB-train", default=None, help="Optional path to train stageB_summary.json")
    parser.add_argument("--judge-summary", default=None, help="Optional path to judge_baselines_summary.json")
    parser.add_argument("--out", default="results/robustness_report.json", help="Output JSON path")
    parser.add_argument("--out-md", default="results/robustness_report.md", help="Output markdown path")
    args = parser.parse_args()

    stageA_train = _load_json(Path(args.stageA_train))
    stageB_test = _collect_stageb(args.stageB_test_glob)
    stageB_train = _load_json(Path(args.stageB_train)) if args.stageB_train else None
    judge_summary = _load_json(Path(args.judge_summary)) if args.judge_summary else None

    report = {
        "stageA_train": {
            "model": stageA_train.get("model"),
            "weights": stageA_train.get("weights"),
            "cv": stageA_train.get("cv"),
            "evaluation": stageA_train.get("evaluation"),
            "b1_validation": stageA_train.get("b1_validation"),
        },
        "stageB_test": stageB_test,
    }
    if stageB_train:
        report["stageB_train"] = stageB_train
    if judge_summary:
        report["judge_summary"] = judge_summary

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    md_lines = [
        "# Robustness Report",
        "",
        "## Stage A (train)",
        f"- model: {stageA_train.get('model', 'unknown')}",
        f"- weights: {json.dumps(stageA_train.get('weights', {}), indent=2)}",
        f"- cv: {json.dumps(stageA_train.get('cv', {}), indent=2)}",
        f"- evaluation: {json.dumps(stageA_train.get('evaluation', {}), indent=2)}",
        f"- B1 validation: {json.dumps(stageA_train.get('b1_validation', {}), indent=2)}",
        "",
        "## Stage B (test, aggregated)",
        f"- paths: {len(stageB_test['paths'])}",
        f"- alignment means/std: {json.dumps(stageB_test['metrics'], indent=2)}",
    ]
    if stageB_train:
        md_lines.extend(
            [
                "",
                "## Stage B (train)",
                json.dumps(stageB_train.get("alignment", {}), indent=2),
            ]
        )
    if judge_summary:
        md_lines.extend(
            [
                "",
                "## Judge Baselines",
                json.dumps(judge_summary, indent=2),
            ]
        )
    Path(args.out_md).write_text("\n".join(md_lines))

    print(f"✅ Wrote {out_path} and {args.out_md}")


if __name__ == "__main__":
    main()

