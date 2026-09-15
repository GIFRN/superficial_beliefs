#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hospital_cyber_nt import (
    DATASET_CONFIG_PATH,
    DEFAULT_DATASET_SEED,
    DEFAULT_OPENAI_CONCURRENCY,
    DEFAULT_QWEN_CONCURRENCY,
    DEFAULT_RESUME_MODE,
    DEFAULT_TEST_TARGET,
    DEFAULT_TRAIN_TARGET,
    MODEL_BY_TAG,
    dataset_dir,
    logs_root,
    output_root,
    run_prefix,
)


def ensure_qwen_8030_health() -> None:
    url = "http://127.0.0.1:8030/v1/models"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if "qwen3_14b_local_gpu0" not in model_ids:
        raise SystemExit(f"Port 8030 is up but qwen3_14b_local_gpu0 was not exposed at {url}")


def ensure_openai_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "OPENAI_API_KEY is not set in this shell, so GPT-5-mini NT runs cannot start. "
        "Set the key or run with --models qwen_min_8030."
    )


def run_and_tee(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_fh.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def maybe_build_datasets(args: argparse.Namespace, base_root: Path) -> None:
    required = [
        dataset_dir("train", base=base_root) / "dataset_trials.parquet",
        dataset_dir("test", base=base_root) / "dataset_trials.parquet",
        dataset_dir("occlusion_test", base=base_root) / "dataset_trials.parquet",
    ]
    if not args.force_build and all(path.exists() for path in required):
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts/build_hospital_cyber_benchmark.py"),
        "--output-root",
        str(base_root),
        "--train-target",
        str(args.train_target),
        "--test-target",
        str(args.test_target),
        "--seed",
        str(args.seed),
    ]
    if args.force_build:
        cmd.append("--force")
    run_and_tee(cmd, log_path=logs_root(base_root) / "build_hospital_cyber_benchmark.log")


def run_trial_bundle(
    *,
    base_root: Path,
    model_tag: str,
    split: str,
    kind: str,
    resume: str,
    trial_concurrency: int,
    variant_override: str | None = None,
) -> None:
    model_spec = MODEL_BY_TAG[model_tag]
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_trials.py"),
        "--config",
        str(DATASET_CONFIG_PATH),
        "--models",
        str(model_spec.config_path),
        "--dataset",
        str(dataset_dir(split, base=base_root)),
        "--out",
        str(run_prefix(split, model_tag, kind, base=base_root)),
        "--resume",
        resume,
        "--trial-concurrency",
        str(trial_concurrency),
    ]
    if variant_override:
        cmd.extend(["--variant-override", variant_override])
    log_name = f"{split}_{model_tag}_{kind}.log"
    run_and_tee(cmd, log_path=logs_root(base_root) / log_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hospital cyber NT benchmark")
    parser.add_argument("--output-root", default=None, help="Override benchmark output root")
    parser.add_argument("--train-target", type=int, default=DEFAULT_TRAIN_TARGET)
    parser.add_argument("--test-target", type=int, default=DEFAULT_TEST_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--resume", choices=["any", "strict"], default=DEFAULT_RESUME_MODE)
    parser.add_argument("--openai-concurrency", type=int, default=DEFAULT_OPENAI_CONCURRENCY)
    parser.add_argument("--qwen-concurrency", type=int, default=DEFAULT_QWEN_CONCURRENCY)
    parser.add_argument("--force-build", action="store_true", help="Rebuild datasets before running trials")
    parser.add_argument("--skip-train-actor", action="store_true")
    parser.add_argument("--skip-test-actor", action="store_true")
    parser.add_argument("--skip-test-judge", action="store_true")
    parser.add_argument("--skip-occlusion-actor", action="store_true")
    parser.add_argument("--models", nargs="+", default=["mini_min", "qwen_min_8030"])
    args = parser.parse_args()

    base_root = output_root(args.output_root)
    for model_tag in args.models:
        if model_tag not in MODEL_BY_TAG:
            raise SystemExit(f"Unknown model tag: {model_tag}")
    if "qwen_min_8030" in args.models:
        ensure_qwen_8030_health()
    if any(MODEL_BY_TAG[tag].provider == "openai" for tag in args.models):
        ensure_openai_api_key()

    maybe_build_datasets(args, base_root)

    for model_tag in args.models:
        model_spec = MODEL_BY_TAG[model_tag]
        concurrency = args.openai_concurrency if model_spec.provider == "openai" else args.qwen_concurrency
        if not args.skip_train_actor:
            run_trial_bundle(
                base_root=base_root,
                model_tag=model_tag,
                split="train",
                kind="actor",
                resume=args.resume,
                trial_concurrency=concurrency,
            )
        if not args.skip_test_actor:
            run_trial_bundle(
                base_root=base_root,
                model_tag=model_tag,
                split="test",
                kind="actor",
                resume=args.resume,
                trial_concurrency=concurrency,
            )
        if not args.skip_test_judge:
            run_trial_bundle(
                base_root=base_root,
                model_tag=model_tag,
                split="test",
                kind="judge",
                resume=args.resume,
                trial_concurrency=concurrency,
                variant_override="short_reason__judge_scores_joint",
            )
        if not args.skip_occlusion_actor:
            run_trial_bundle(
                base_root=base_root,
                model_tag=model_tag,
                split="occlusion_test",
                kind="actor",
                resume=args.resume,
                trial_concurrency=concurrency,
            )


if __name__ == "__main__":
    main()
