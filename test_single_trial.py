#!/usr/bin/env python3
"""
Test script to run a single trial and display the full conversation and results.
This helps inspect whether the model is responding as expected.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.backends.anthropic import AnthropicBackend
try:
    from src.llm.backends.mock import MockBackend
except Exception:  # pragma: no cover
    MockBackend = None
from src.llm.backends.openai import OpenAIBackend
from src.llm.backends.vllm import VLLMBackend
from src.llm.harness import build_trial_specs, run_trial
from src.utils.config import load_config
from src.utils.io import read_yaml

BACKEND_FACTORY = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "vllm": VLLMBackend,
}
if MockBackend is not None:
    BACKEND_FACTORY["mock"] = MockBackend


def print_separator(char="=", length=80):
    print(char * length)


def print_section(title):
    print()
    print_separator()
    print(f" {title}")
    print_separator()


def print_messages(messages, indent="  "):
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"{indent}[{i+1}] {role.upper()}:")
        # Print content with indentation, handling multi-line strings
        for line in content.split('\n'):
            print(f"{indent}    {line}")
        print()


def print_step_result(step, indent="  "):
    print(f"{indent}Step: {step['name']}")
    print(f"{indent}Content: {step['content'][:200]}{'...' if len(step['content']) > 200 else ''}")
    print(f"{indent}Parsed: {json.dumps(step['parsed'], indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single trial and display full conversation details"
    )
    parser.add_argument("--config", default="configs/default.yml", help="Path to dataset configuration YAML")
    parser.add_argument("--models", default="configs/models.yml", help="Path to models configuration YAML")
    parser.add_argument("--dataset", default="data/generated/v1", help="Path to generated dataset directory")
    parser.add_argument("--trial-index", type=int, default=0, help="Index of trial to run (default: 0)")
    parser.add_argument("--S", type=int, default=1, help="Number of replicates (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Enable backend debug mode")
    parser.add_argument("--variant-override", default=None, help="Override variant for the selected trial")
    args = parser.parse_args()

    # Load configurations
    cfg = load_config(args.config)
    models_cfg = read_yaml(args.models)

    # Set up backend
    backend_name = models_cfg["sampling"].get("default_backend")
    if backend_name not in models_cfg["backends"]:
        raise KeyError(f"Backend {backend_name} not defined in models config")

    backend_spec = models_cfg["backends"][backend_name]
    backend_type = backend_spec.get("type")
    if backend_type == "mock" and MockBackend is None:
        raise ValueError("Mock backend not available; restore src/llm/backends/mock.py or update configs/models.yml.")
    if backend_type not in BACKEND_FACTORY:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    cls = BACKEND_FACTORY[backend_type]
    kwargs = {k: v for k, v in backend_spec.items() if k not in {"type"}}
    if backend_type == "openai" and args.debug:
        kwargs["debug"] = args.debug
    backend = cls(**kwargs)

    # Load dataset
    import pandas as pd
    dataset_dir = Path(args.dataset)
    configs_df = pd.read_parquet(dataset_dir / "dataset_configs.parquet")
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")

    # Build trial specs and select one
    trial_specs = build_trial_specs(cfg, configs_df, trials_df, variant_override=args.variant_override)

    if args.trial_index >= len(trial_specs):
        print(f"Error: Trial index {args.trial_index} out of range (0-{len(trial_specs)-1})")
        return

    selected_trial = trial_specs[args.trial_index]

    # Print trial information
    print_section(f"TRIAL INFORMATION (Index {args.trial_index})")
    print(f"  Trial ID: {selected_trial.trial_id}")
    print(f"  Config ID: {selected_trial.config_id}")
    print(f"  Block: {selected_trial.block}")
    print(f"  Manipulation: {selected_trial.manipulation}")
    print(f"  Variant: {selected_trial.variant}")
    print(f"  Attribute Target: {selected_trial.attribute_target}")
    print(f"  Paraphrase ID: {selected_trial.paraphrase_id}")

    print("\n  Profile A:")
    for attr, level in selected_trial.profile_a.levels.items():
        print(f"    {attr}: {level}")

    print("\n  Profile B:")
    for attr, level in selected_trial.profile_b.levels.items():
        print(f"    {attr}: {level}")

    print(f"\n  Order A: {selected_trial.order_a}")
    print(f"  Order B: {selected_trial.order_b}")

    # Run trial
    print_section(f"RUNNING TRIAL (S={args.S} replicates)")
    print(f"  Backend: {backend_name} ({backend_type})")
    print(f"  Model: {backend_spec.get('model', 'N/A')}")
    print(f"  Temperature: {models_cfg['sampling'].get('temperature', backend_spec.get('temperature', 'N/A'))}")
    print(f"  Max Tokens: {backend_spec.get('max_tokens', 'N/A')}")

    S = args.S
    base_seed = int(models_cfg["sampling"].get("seed", cfg.seed_global))
    temperature = models_cfg["sampling"].get("temperature", backend_spec.get("temperature", cfg.replicates.temperature))
    max_tokens = backend_spec.get("max_tokens", 256)

    try:
        result = run_trial(
            selected_trial,
            backend,
            S=S,
            temperature=temperature,
            seed=base_seed,
            max_tokens=max_tokens,
        )

        # Print results for each replicate
        for replicate_idx, response in enumerate(result["responses"]):
            print_section(f"REPLICATE {replicate_idx + 1} of {S}")
            print(f"  Seed: {response['seed']}")

            # Print conversation
            print("\n  CONVERSATION:")
            print_messages(response["conversation"], indent="    ")

            # Print parsed steps
            print("\n  PARSED STEPS:")
            for step_idx, step in enumerate(response["steps"]):
                print(f"\n    Step {step_idx + 1}:")
                print_step_result(step, indent="      ")

                # Check if parsing was successful
                if not step["parsed"].get("ok", False):
                    print("      ⚠️  WARNING: Parsing failed!")
                else:
                    print("      ✓ Parsing successful")

        # Summary
        print_section("SUMMARY")
        total_steps = sum(len(r["steps"]) for r in result["responses"])
        successful_steps = sum(
            1 for r in result["responses"]
            for step in r["steps"]
            if step["parsed"].get("ok", False)
        )
        print(f"  Total steps across all replicates: {total_steps}")
        print(f"  Successful parses: {successful_steps}/{total_steps}")

        if successful_steps == total_steps:
            print("  ✓ All steps parsed successfully!")
        else:
            print(f"  ⚠️  {total_steps - successful_steps} step(s) failed to parse")

        # Save result to file
        output_file = Path("test_trial_output.json")
        with output_file.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Full result saved to: {output_file}")

    except Exception as exc:
        print_section("ERROR")
        print(f"  Error type: {type(exc).__name__}")
        print(f"  Error message: {str(exc)}")
        import traceback
        print("\n  Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
