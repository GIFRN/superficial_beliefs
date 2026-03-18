#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_benchmark import (
    ALL_THEMES,
    MODEL_SPECS,
    dataset_dir,
    judge_dir,
    output_root,
    resolve_run_dir,
    run_prefix,
    stagea_dir,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final same-order analysis over completed benchmark runs.")
    parser.add_argument("--out-root", default=str(output_root()))
    parser.add_argument("--config", default="data/configs/dataset.yml")
    parser.add_argument("--themes", default=",".join(ALL_THEMES))
    parser.add_argument("--skip-judge-reports", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    selected_themes = [theme.strip() for theme in args.themes.split(",") if theme.strip()]

    for theme in selected_themes:
        train_dataset = dataset_dir(theme, "train", base=out_root)
        test_dataset = dataset_dir(theme, "test", base=out_root)
        for spec in MODEL_SPECS:
            train_run = resolve_run_dir(run_prefix(theme, "train", spec.tag, "actor", base=out_root))
            test_run = resolve_run_dir(run_prefix(theme, "test", spec.tag, "tau", base=out_root))
            if train_run is None or test_run is None:
                if args.allow_missing:
                    print(f"skip missing run dirs for {theme}/{spec.tag}")
                    continue
                raise SystemExit(f"Missing run dir(s) for {theme}/{spec.tag}")

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
                    str(stagea_dir(theme, spec.tag, base=out_root)),
                ]
            )
            if not args.skip_judge_reports:
                _run(
                    [
                        sys.executable,
                        "scripts/analyze_judge_baselines.py",
                        "--dataset",
                        str(test_dataset),
                        "--responses",
                        str(test_run / "responses.jsonl"),
                        "--out",
                        str(judge_dir(theme, spec.tag, base=out_root)),
                    ]
                )

    _run([sys.executable, "scripts/build_final_results_tables.py", "--out-root", str(out_root)] + (["--allow-missing"] if args.allow_missing else []))


if __name__ == "__main__":
    main()
