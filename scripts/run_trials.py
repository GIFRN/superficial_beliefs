#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from openai import BadRequestError
except Exception:  # pragma: no cover
    BadRequestError = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.backends.anthropic import AnthropicBackend
from src.llm.backends.mock import MockBackend
from src.llm.backends.openai import OpenAIBackend
from src.llm.backends.vllm import VLLMBackend
from src.llm.harness import build_trial_specs, run_trial
from src.utils.config import Config, load_config
from src.utils.io import ensure_dir, read_yaml, write_json

BACKEND_FACTORY = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "vllm": VLLMBackend,
    "mock": MockBackend,
}


MAX_INVALID_PROMPT_RETRIES = 5


def is_invalid_prompt_error(exc: Exception) -> bool:
    """Detect OpenAI invalid prompt safety errors."""
    if BadRequestError is not None and isinstance(exc, BadRequestError):
        code = getattr(exc, "code", None)
        if code == "invalid_prompt":
            return True
        error = getattr(exc, "error", None)
        if isinstance(error, dict) and error.get("code") == "invalid_prompt":
            return True
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict) and inner.get("code") == "invalid_prompt":
                return True
    message = str(exc)
    return (
        "invalid prompt" in message.lower()
        and "usage policy" in message.lower()
    )


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs_path = dataset_dir / "dataset_configs.parquet"
    trials_path = dataset_dir / "dataset_trials.parquet"
    configs_df = pd.read_parquet(configs_path)
    trials_df = pd.read_parquet(trials_path)
    return configs_df, trials_df


def load_existing_responses(responses_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing responses and return a dict mapping trial_id to response data.
    
    Only includes trials where all steps were parsed successfully (ok=True).
    """
    if not responses_path.exists():
        return {}
    
    existing = {}
    with responses_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                response = json.loads(line)
                trial_id = response.get("trial_id")
                if trial_id is None:
                    continue
                
                # Check if all responses in all replicates were successful
                all_ok = True
                for run in response.get("responses", []):
                    for step in run.get("steps", []):
                        if not step.get("parsed", {}).get("ok", False):
                            all_ok = False
                            break
                    if not all_ok:
                        break
                
                if all_ok:
                    existing[trial_id] = response
            except json.JSONDecodeError:
                continue
    
    return existing


def instantiate_backend(spec: dict[str, Any], debug: bool = False):
    backend_type = spec.get("type")
    if backend_type not in BACKEND_FACTORY:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    cls = BACKEND_FACTORY[backend_type]
    kwargs = {k: v for k, v in spec.items() if k not in {"type"}}
    if backend_type == "openai" and debug:
        kwargs["debug"] = debug
    return cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM trials for superficial beliefs study")
    parser.add_argument("--config", default="configs/default.yml", help="Path to dataset configuration YAML")
    parser.add_argument("--models", default="configs/models.yml", help="Path to models configuration YAML")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Path to generated dataset directory")
    parser.add_argument("--out", default=None, help="Directory to write run outputs (auto-generated if not specified)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens per call")
    parser.add_argument("--debug", action="store_true", default=False, help="Print all queries and responses")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    models_cfg = read_yaml(args.models)
    dataset_dir = Path(args.dataset)

    backend_name = models_cfg["sampling"].get("default_backend")
    if backend_name not in models_cfg["backends"]:
        raise KeyError(f"Backend {backend_name} not defined in models config")
    backend_spec = models_cfg["backends"][backend_name]
    backend = instantiate_backend(backend_spec, debug=args.debug)
    
    # Get reasoning effort from backend spec
    reasoning_effort = backend_spec.get("reasoning_effort", None)
    
    # Construct output directory name if not specified
    if args.out is None:
        dataset_name = Path(args.dataset).name
        # Include reasoning effort in directory name if applicable
        if reasoning_effort:
            out_dirname = f"{dataset_name}_{backend_name}_{reasoning_effort}"
        else:
            out_dirname = f"{dataset_name}_{backend_name}"
        args.out = f"data/runs/{out_dirname}"
    
    out_dir = ensure_dir(args.out)
    
    print(f"Dataset: {args.dataset}")
    print(f"Backend: {backend_name}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Output directory: {args.out}")

    S = int(models_cfg["sampling"].get("S", cfg.replicates.S))
    base_seed = int(models_cfg["sampling"].get("seed", cfg.seed_global))
    temperature_override = models_cfg["sampling"].get("temperature")
    max_tokens_override = args.max_tokens or backend_spec.get("max_tokens")

    configs_df, trials_df = load_dataset(dataset_dir)
    trial_specs = build_trial_specs(cfg, configs_df, trials_df)

    responses_path = out_dir / "responses.jsonl"
    
    # Load existing responses to support continuation
    existing_responses = load_existing_responses(responses_path)
    completed_trial_ids = set(existing_responses.keys())
    
    # Filter out already-completed trials
    remaining_specs = [spec for spec in trial_specs if spec.trial_id not in completed_trial_ids]
    
    if completed_trial_ids:
        print(f"Found {len(completed_trial_ids)} already-completed trials. Skipping those.")
        print(f"Running {len(remaining_specs)} remaining trials out of {len(trial_specs)} total.")
    
    if not remaining_specs:
        print("All trials already completed. Nothing to do.")
        return
    
    # Append to existing file (or create new if it doesn't exist)
    with responses_path.open("a", encoding="utf-8") as fh:
        for idx, spec in enumerate(tqdm(remaining_specs, desc="Trials")):
            temperature = temperature_override if temperature_override is not None else backend_spec.get("temperature", cfg.replicates.temperature)
            trial_seed = base_seed + idx
            retries = 0
            while True:
                try:
                    result = run_trial(
                        spec,
                        backend,
                        S=S,
                        temperature=temperature,
                        seed=trial_seed,
                        max_tokens=max_tokens_override or backend_spec.get("max_tokens", 256),
                    )
                    print(spec)
                    print(result)
                    print("="*60)
                    fh.write(json.dumps(result))
                    fh.write("\n")
                    fh.flush()  # Ensure data is written immediately
                    break
                except Exception as exc:
                    if is_invalid_prompt_error(exc):
                        if retries < MAX_INVALID_PROMPT_RETRIES:
                            attempt_num = retries + 1
                            total_attempts = MAX_INVALID_PROMPT_RETRIES
                            print(f"\n{'='*60}")
                            print(f"Invalid prompt flagged for trial {spec.trial_id}. Retrying attempt {attempt_num}/{total_attempts}.")
                            print(f"{'='*60}\n")
                            retries += 1
                            continue
                        print(f"\n{'='*60}")
                        print(f"Skipping trial {spec.trial_id} after {MAX_INVALID_PROMPT_RETRIES + 1} invalid prompt attempts.")
                        print(f"{'='*60}\n")
                        break
                    print(f"\n{'='*60}")
                    print(f"API ERROR during trial {spec.trial_id}")
                    print(f"Trial index: {idx + 1}/{len(remaining_specs)}")
                    print(f"Config ID: {spec.config_id}")
                    print(f"Block: {spec.block}")
                    print(f"Error type: {type(exc).__name__}")
                    print(f"Error message: {str(exc)}")
                    print(f"{'='*60}\n")
                    raise

    manifest = {
        "config": args.config,
        "models": args.models,
        "dataset": args.dataset,
        "responses_path": str(responses_path),
        "backend": backend_name,
        "S": S,
        "seed": base_seed,
    }
    
    # Add reasoning_effort to manifest if present
    if reasoning_effort:
        manifest["reasoning_effort"] = reasoning_effort
    
    write_json(manifest, out_dir / "MANIFEST.json")


if __name__ == "__main__":
    main()
