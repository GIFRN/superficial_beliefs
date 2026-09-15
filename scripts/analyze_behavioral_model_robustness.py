#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.behavioral_robustness import (
    build_condition_comparison,
    build_per_draw_driver_table,
    build_test_row_predictions,
    load_models_for_comparison,
)
from src.analysis.features import load_responses
from src.analysis.final_benchmark import (
    MODEL_BY_TAG,
    dataset_dir,
    reports_root,
    resolve_run_dir,
    results_root,
    run_prefix,
    stagea_dir,
)
from src.utils.io import ensure_dir, write_json


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _default_m1_dir(theme: str, model_tag: str, *, out_root: Path) -> Path:
    return results_root(out_root) / f"stageA_m1_{theme}_{model_tag}"


def _default_m2_dir(theme: str, model_tag: str, *, out_root: Path) -> Path:
    return results_root(out_root) / f"stageA_m2_{theme}_{model_tag}"


def _default_out_dir(theme: str, model_tag: str, *, out_root: Path) -> Path:
    return reports_root(out_root) / f"behavioral_robustness_{theme}_{model_tag}"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out behavioral-model robustness artifacts for one condition.")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--out-root", default="outputs/final_same_order")
    parser.add_argument("--config", default="configs/default.yml")
    parser.add_argument("--m0-summary", default=None)
    parser.add_argument("--m1-summary", default=None)
    parser.add_argument("--m2-summary", default=None)
    parser.add_argument("--m1-dir", default=None)
    parser.add_argument("--m2-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fit-m1-if-missing", action="store_true")
    parser.add_argument("--fit-m2-if-missing", action="store_true")
    args = parser.parse_args()

    if args.model_tag not in MODEL_BY_TAG:
        raise SystemExit(f"Unknown model tag: {args.model_tag}")

    out_root = Path(args.out_root).resolve()
    spec = MODEL_BY_TAG[args.model_tag]

    train_dataset = dataset_dir(args.theme, "train", base=out_root)
    test_dataset = dataset_dir(args.theme, "test", base=out_root)
    train_run = resolve_run_dir(run_prefix(args.theme, "train", args.model_tag, "actor", base=out_root))
    test_run = resolve_run_dir(run_prefix(args.theme, "test", args.model_tag, "tau", base=out_root))
    if train_run is None or test_run is None:
        raise SystemExit(f"Missing run directories for {args.theme}/{args.model_tag}")

    m0_summary_path = Path(args.m0_summary).resolve() if args.m0_summary else stagea_dir(args.theme, args.model_tag, base=out_root) / "stageA_summary.json"
    m1_dir = Path(args.m1_dir).resolve() if args.m1_dir else _default_m1_dir(args.theme, args.model_tag, out_root=out_root)
    m1_summary_path = Path(args.m1_summary).resolve() if args.m1_summary else m1_dir / "stageA_summary.json"
    m2_dir = Path(args.m2_dir).resolve() if args.m2_dir else _default_m2_dir(args.theme, args.model_tag, out_root=out_root)
    m2_summary_path = Path(args.m2_summary).resolve() if args.m2_summary else m2_dir / "stageA_summary.json"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(args.theme, args.model_tag, out_root=out_root)

    if not m0_summary_path.exists():
        raise SystemExit(f"Missing M0 summary: {m0_summary_path}")
    if not m1_summary_path.exists():
        if not args.fit_m1_if_missing:
            raise SystemExit(f"Missing M1 summary: {m1_summary_path}")
        _run(
            [
                sys.executable,
                "scripts/fit_stageA.py",
                "--config",
                args.config,
                "--dataset",
                str(train_dataset),
                "--responses",
                str(train_run / "responses.jsonl"),
                "--out",
                str(m1_dir),
                "--behavioral-model",
                "m1",
            ]
        )
    if not m2_summary_path.exists():
        if not args.fit_m2_if_missing:
            raise SystemExit(f"Missing M2 summary: {m2_summary_path}")
        _run(
            [
                sys.executable,
                "scripts/fit_stageA.py",
                "--config",
                args.config,
                "--dataset",
                str(train_dataset),
                "--responses",
                str(train_run / "responses.jsonl"),
                "--out",
                str(m2_dir),
                "--behavioral-model",
                "m2",
            ]
        )

    ensure_dir(out_dir)
    trials_df = pd.read_parquet(test_dataset / "dataset_trials.parquet")
    responses_df = load_responses(test_run / "responses.jsonl")
    models, summaries = load_models_for_comparison(
        summary_paths={
            "m0": str(m0_summary_path),
            "m1": str(m1_summary_path),
            "m2": str(m2_summary_path),
        },
    )

    condition_keys = {
        "theme": args.theme,
        "family": spec.family,
        "effort": spec.effort,
        "model_tag": spec.tag,
    }
    row_predictions = {
        label: build_test_row_predictions(
            trials_df=trials_df,
            responses_df=responses_df,
            model=model,
            condition_keys=condition_keys,
        )
        for label, model in models.items()
    }
    combined_rows = pd.concat(list(row_predictions.values()), ignore_index=True)
    per_draw = build_per_draw_driver_table(
        trials_df=trials_df,
        responses_df=responses_df,
        row_predictions=row_predictions,
        models=models,
        condition_keys=condition_keys,
    )
    comparison = build_condition_comparison(
        theme=args.theme,
        row_predictions=row_predictions,
        per_draw_df=per_draw,
        condition_keys=condition_keys,
        model_summaries=summaries,
    )
    comparison["m0_summary_relpath"] = _rel(m0_summary_path)
    comparison["m1_summary_relpath"] = _rel(m1_summary_path)
    comparison["m2_summary_relpath"] = _rel(m2_summary_path)
    comparison["test_dataset_relpath"] = _rel(test_dataset)
    comparison["test_responses_relpath"] = _rel(test_run / "responses.jsonl")

    combined_rows.to_parquet(out_dir / "test_row_predictions.parquet", index=False)
    per_draw.to_parquet(out_dir / "test_revealed_drivers.parquet", index=False)
    per_draw.to_parquet(out_dir / "test_draw_drivers.parquet", index=False)
    write_json(comparison, out_dir / "comparison.json")
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison.keys()))
        writer.writeheader()
        writer.writerow(comparison)

    print(f"wrote {out_dir / 'test_row_predictions.parquet'}")
    print(f"wrote {out_dir / 'test_revealed_drivers.parquet'}")
    print(f"wrote {out_dir / 'test_draw_drivers.parquet'}")
    print(f"wrote {out_dir / 'comparison.json'}")
    print(f"wrote {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
