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
    output_root,
    run_prefix,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _providers_from_args(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final same-order benchmark.")
    parser.add_argument("--out-root", default=str(output_root()))
    parser.add_argument("--config", default="data/configs/dataset.yml")
    parser.add_argument("--themes", default=",".join(ALL_THEMES))
    parser.add_argument(
        "--providers",
        default="openai,qwen,ministral",
        help="Comma-separated providers to run: openai,qwen,ministral",
    )
    parser.add_argument("--openai-concurrency", type=int, default=12)
    parser.add_argument("--qwen-concurrency", type=int, default=1)
    parser.add_argument("--ministral-concurrency", type=int, default=1)
    parser.add_argument("--resume", default="any", choices=["any", "strict"])
    parser.add_argument("--build-datasets", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    selected_themes = [theme.strip() for theme in args.themes.split(",") if theme.strip()]
    selected_providers = _providers_from_args(args.providers)

    if args.build_datasets or not all(dataset_dir(theme, "train", base=out_root).exists() for theme in selected_themes):
        _run([sys.executable, "scripts/build_final_themed_datasets.py", "--out-root", str(out_root), "--themes", ",".join(selected_themes)])

    for spec in MODEL_SPECS:
        if spec.provider not in selected_providers:
            continue
        if spec.provider == "openai":
            trial_concurrency = args.openai_concurrency
        elif spec.provider == "qwen":
            trial_concurrency = args.qwen_concurrency
        else:
            trial_concurrency = args.ministral_concurrency

        for theme in selected_themes:
            if not args.skip_train:
                _run(
                    [
                        sys.executable,
                        "scripts/run_trials.py",
                        "--config",
                        args.config,
                        "--models",
                        str(spec.config_path),
                        "--dataset",
                        str(dataset_dir(theme, "train", base=out_root)),
                        "--out",
                        str(run_prefix(theme, "train", spec.tag, "actor", base=out_root)),
                        "--variant-override",
                        "short_reason",
                        "--resume",
                        args.resume,
                        "--trial-concurrency",
                        str(trial_concurrency),
                    ]
                )
            if not args.skip_test:
                _run(
                    [
                        sys.executable,
                        "scripts/run_trials.py",
                        "--config",
                        args.config,
                        "--models",
                        str(spec.config_path),
                        "--dataset",
                        str(dataset_dir(theme, "test", base=out_root)),
                        "--out",
                        str(run_prefix(theme, "test", spec.tag, "tau", base=out_root)),
                        "--variant-override",
                        "short_reason__judge_scores_joint",
                        "--resume",
                        args.resume,
                        "--trial-concurrency",
                        str(trial_concurrency),
                    ]
                )


if __name__ == "__main__":
    main()
