#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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

from src.data.themes import theme_from_dict
from src.llm.backends.anthropic import AnthropicBackend
try:
    from src.llm.backends.mock import MockBackend
except Exception:  # pragma: no cover
    MockBackend = None
from src.llm.backends.openai import OpenAIBackend
from src.llm.backends.vllm import VLLMBackend
from src.llm.harness import build_trial_specs, run_trial
from src.utils.config import Config, load_config
from src.utils.io import ensure_dir, read_yaml, write_json

BACKEND_FACTORY = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "vllm": VLLMBackend,
}
if MockBackend is not None:
    BACKEND_FACTORY["mock"] = MockBackend


MAX_INVALID_PROMPT_RETRIES = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_S = 5.0
DEFAULT_MAX_TOKENS_CEILING = 8000
MAX_TOKENS_SCHEDULE = [256, 512, 1024, 2048, 4096, 8000]


def is_invalid_prompt_error(exc: Exception) -> bool:
    """Detect safety-related or blocked prompts (invalid prompt / policy / permission)."""
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
    message = str(exc).lower()
    if "invalid prompt" in message or "usage policy" in message:
        return True
    # Some providers return 403 for prompt-level safety blocks; treat as skippable.
    if "permissiondenied" in message or "permission denied" in message or "403" in message:
        return True
    return False


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "insufficient_quota" in message
        or "exceeded your current quota" in message
        or ("quota" in message and "rate" not in message)
        or "billing" in message
        or "no credits" in message
        or "out of credits" in message
    )


def is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    # Conservative string match for transient server / network / rate limit issues.
    transient_markers = [
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection",
        "connection reset",
        "server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "internal server error",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in transient_markers)


def is_max_output_tokens_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "max_output_tokens" in message or "incomplete response" in message


def _next_max_tokens(current: int, *, ceiling: int) -> int | None:
    if current >= ceiling:
        return None
    for candidate in MAX_TOKENS_SCHEDULE:
        if candidate > current and candidate <= ceiling:
            return candidate
    # Fall back to exponential growth if schedule is exhausted.
    grown = min(ceiling, current * 2)
    return grown if grown > current else None


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs_path = dataset_dir / "dataset_configs.parquet"
    trials_path = dataset_dir / "dataset_trials.parquet"
    configs_df = pd.read_parquet(configs_path)
    trials_df = pd.read_parquet(trials_path)
    return configs_df, trials_df


def load_existing_responses(responses_path: Path, *, resume_mode: str) -> dict[str, dict[str, Any]]:
    """Load existing responses and return a dict mapping trial_id to response data.

    resume_mode:
      - "any": treat any existing trial record as completed (prevents redundant reruns).
      - "strict": only treat trials as completed when all parsed steps are ok (legacy behavior),
                  except that explicitly skipped trials are always treated as completed.
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
                if response.get("skipped") is True:
                    existing[trial_id] = response
                    continue
                if resume_mode == "any":
                    existing[trial_id] = response
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
    if backend_type == "mock" and MockBackend is None:
        raise ValueError("Mock backend not available; restore src/llm/backends/mock.py or update configs/models.yml.")
    if backend_type not in BACKEND_FACTORY:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    cls = BACKEND_FACTORY[backend_type]
    kwargs = {k: v for k, v in spec.items() if k not in {"type"}}
    if backend_type == "openai" and debug:
        kwargs["debug"] = debug
    return cls(**kwargs)


def _sanitize_token(token: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in token)


def _build_run_suffix(
    *,
    backend_name: str,
    reasoning_effort: str | None,
    variant_override: str | None,
    max_tokens_override: int | None,
) -> str:
    parts = [_sanitize_token(backend_name)]
    if reasoning_effort:
        parts.append(_sanitize_token(reasoning_effort))
    if variant_override:
        parts.append(_sanitize_token(f"var-{variant_override}"))
    if max_tokens_override is not None:
        parts.append(f"mt{max_tokens_override}")
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM trials for superficial beliefs study")
    parser.add_argument("--config", default="configs/default.yml", help="Path to dataset configuration YAML")
    parser.add_argument("--models", default="configs/models.yml", help="Path to models configuration YAML")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Path to generated dataset directory")
    parser.add_argument("--out", default=None, help="Directory to write run outputs (auto-generated if not specified)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens per call")
    parser.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=None,
        help=f"Upper bound for automatic max-tokens increases on max_output_tokens errors (default: {DEFAULT_MAX_TOKENS_CEILING})",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Print all queries and responses")
    parser.add_argument("--variant-override", default=None, help="Override variant for all trials")
    parser.add_argument(
        "--resume",
        choices=["any", "strict"],
        default="any",
        help="Resume behavior when responses.jsonl already exists (default: any)",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries for transient API errors")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help="Base backoff in seconds for transient retries (exponential)",
    )
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
    
    # Construct output directory name (always include backend/variant info to avoid collisions)
    run_suffix = _build_run_suffix(
        backend_name=backend_name,
        reasoning_effort=reasoning_effort,
        variant_override=args.variant_override,
        max_tokens_override=args.max_tokens,
    )
    if args.out is None:
        dataset_name = Path(args.dataset).name
        out_dirname = f"{dataset_name}_{run_suffix}"
        args.out = f"data/runs/{out_dirname}"
    else:
        out_path = Path(args.out)
        out_dirname = f"{out_path.name}__{run_suffix}"
        args.out = str(out_path.parent / out_dirname)

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
    
    # Load theme from dataset MANIFEST.json if present
    dataset_manifest_path = dataset_dir / "MANIFEST.json"
    theme_config = None
    if dataset_manifest_path.exists():
        with dataset_manifest_path.open("r") as f:
            dataset_manifest = json.load(f)
            theme_data = dataset_manifest.get("theme")
            if theme_data:
                theme_config = theme_from_dict(theme_data)
                print(f"Theme: {theme_config.name}")
    
    trial_specs = build_trial_specs(
        cfg,
        configs_df,
        trials_df,
        theme_config=theme_config,
        variant_override=args.variant_override,
    )

    responses_path = out_dir / "responses.jsonl"
    
    # Load existing responses to support continuation
    existing_responses = load_existing_responses(responses_path, resume_mode=args.resume)
    completed_trial_ids = set(existing_responses.keys())
    
    # Filter out already-completed trials
    remaining_specs = [spec for spec in trial_specs if spec.trial_id not in completed_trial_ids]
    
    if completed_trial_ids:
        print(f"Found {len(completed_trial_ids)} already-completed trials. Skipping those.")
        print(f"Running {len(remaining_specs)} remaining trials out of {len(trial_specs)} total.")
    
    if not remaining_specs:
        print("All trials already completed. Nothing to do.")
        return

    # Write manifest early (so partial runs can still be analyzed/resumed cleanly).
    manifest_path = out_dir / "MANIFEST.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "config": args.config,
        "models": args.models,
        "dataset": args.dataset,
        "responses_path": str(responses_path),
        "backend": backend_name,
        "S": S,
        "seed": base_seed,
        "resume_mode": args.resume,
        "n_trials_total": int(len(trial_specs)),
        "n_trials_already_completed": int(len(completed_trial_ids)),
        "n_trials_remaining": int(len(remaining_specs)),
    }
    if args.variant_override:
        manifest["variant_override"] = args.variant_override
    if reasoning_effort:
        manifest["reasoning_effort"] = reasoning_effort
    if theme_config:
        manifest["theme"] = theme_config.to_dict()
    write_json(manifest, manifest_path)

    max_tokens_ceiling = (
        int(args.max_tokens_ceiling)
        if args.max_tokens_ceiling is not None
        else int(backend_spec.get("max_tokens_ceiling", DEFAULT_MAX_TOKENS_CEILING))
    )

    completed_this_run = 0
    fatal_stop_reason: str | None = None
    # Append to existing file (or create new if it doesn't exist)
    with responses_path.open("a", encoding="utf-8") as fh:
        for idx, spec in enumerate(tqdm(remaining_specs, desc="Trials")):
            temperature = temperature_override if temperature_override is not None else backend_spec.get("temperature", cfg.replicates.temperature)
            trial_seed = base_seed + idx
            retries = 0
            max_tokens_for_trial = int(max_tokens_override or backend_spec.get("max_tokens", 256))
            while True:
                try:
                    result = run_trial(
                        spec,
                        backend,
                        S=S,
                        temperature=temperature,
                        seed=trial_seed,
                        max_tokens=max_tokens_for_trial,
                    )
                    print(spec)
                    print(result)
                    print("="*60)
                    fh.write(json.dumps(result))
                    fh.write("\n")
                    fh.flush()  # Ensure data is written immediately
                    completed_this_run += 1
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
                        # Write a skip record so resume doesn't keep retrying this trial forever.
                        fh.write(
                            json.dumps(
                                {
                                    "trial_id": spec.trial_id,
                                    "config_id": spec.config_id,
                                    "block": spec.block,
                                    "manipulation": spec.manipulation,
                                    "variant": spec.variant,
                                    "skipped": True,
                                    "skip_reason": "invalid_prompt",
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                    "responses": [],
                                }
                            )
                        )
                        fh.write("\n")
                        fh.flush()
                        break
                    if is_quota_error(exc):
                        fatal_stop_reason = f"quota_exhausted: {type(exc).__name__}: {str(exc)}"
                        print(f"\n{'='*60}")
                        print("Stopping run due to quota/billing error (safe to resume later).")
                        print(f"Trial: {spec.trial_id}  Config: {spec.config_id}  Block: {spec.block}")
                        print(f"Error: {fatal_stop_reason}")
                        print(f"{'='*60}\n")
                        break
                    if is_max_output_tokens_error(exc):
                        bumped = _next_max_tokens(max_tokens_for_trial, ceiling=max_tokens_ceiling)
                        if bumped is not None:
                            print(f"\n{'='*60}")
                            print(
                                f"Trial {spec.trial_id}: max_output_tokens hit at {max_tokens_for_trial}. "
                                f"Bumping to {bumped} and retrying."
                            )
                            print(f"{'='*60}\n")
                            max_tokens_for_trial = bumped
                            retries += 1
                            continue
                    if is_transient_error(exc) and retries < max(0, int(args.max_retries)):
                        delay = float(args.retry_backoff) * (2 ** retries)
                        print(f"\n{'='*60}")
                        print(
                            f"Transient error during trial {spec.trial_id}. "
                            f"Retry {retries + 1}/{args.max_retries} after {delay:.1f}s."
                        )
                        print(f"Error type: {type(exc).__name__}")
                        print(f"Error message: {str(exc)}")
                        print(f"{'='*60}\n")
                        time.sleep(delay)
                        retries += 1
                        continue
                    print(f"\n{'='*60}")
                    print(f"API ERROR during trial {spec.trial_id}")
                    print(f"Trial index: {idx + 1}/{len(remaining_specs)}")
                    print(f"Config ID: {spec.config_id}")
                    print(f"Block: {spec.block}")
                    print(f"Error type: {type(exc).__name__}")
                    print(f"Error message: {str(exc)}")
                    print(f"{'='*60}\n")
                    raise
            if fatal_stop_reason is not None:
                break

    # Update manifest at end (even for partial runs).
    manifest.update(
        {
            "status": "stopped" if fatal_stop_reason else "completed",
            "stopped_reason": fatal_stop_reason,
            "completed_at_unix": time.time(),
            "n_trials_completed_this_run": int(completed_this_run),
            "n_trials_completed_total": int(len(completed_trial_ids) + completed_this_run),
        }
    )
    write_json(manifest, manifest_path)
    if fatal_stop_reason:
        # Exit early but keep partial results on disk for resume.
        return


if __name__ == "__main__":
    main()
