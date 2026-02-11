#!/usr/bin/env python3
"""
Debug Script: All Variations Pipeline Test

This script provides comprehensive testing of the entire experimental pipeline
by running multiple B3 trials through all manipulation variations (including
structural occlusions) plus judge variants and performing complete Stage A and
Stage B analyses.

What it tests:
--------------
1. **Trial Generation**: Creates 3 diverse B3 trial configurations
2. **All Manipulations**: Tests all manipulation types:
   - short_reason (baseline)
   - split_reason (separate context)
   - premise_first (premise before choice)
   - redact (for each attribute E, A, S, D)
   - neutralize (for each attribute E, A, S, D)
   - inject (with offsets -1, 0, +1)
   - occlude_drop (for each attribute E, A, S, D)
   - occlude_equalize (for each attribute E, A, S, D)
   - occlude_swap (for each attribute E, A, S, D)
3. **Judge Variants**: Tests judge appended variants (scores + pairwise)
4. **Response Collection**: Uses OpenAI backend for real responses
5. **Stage A Analysis**: Estimates attribute weights from choice patterns
6. **Stage B Analysis**: Measures alignment between stated premises and actual drivers

Output:
-------
- Detailed response summaries for each trial
- Stage A weight estimates and cross-validation metrics
- Per-trial contribution analysis
- Stage B alignment metrics (ECRB_top1_driver, ECRB_top1_weights, rank correlation)
- Premise distribution across attributes

Usage:
------
    python scripts/debug_all_variations.py --models configs/models.yml

Notes:
------
- Requires OpenAI API access (set OPENAI_API_KEY in your environment)
- This can be expensive; use --limit or reduce variations if needed
"""
from __future__ import annotations

import atexit
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from typing import Any

from src.llm.backends.openai import OpenAIBackend
from src.llm.harness import run_trial
from src.llm.types import TrialSpec
from src.data.schema import Profile
from src.data.occlusions import apply_occlusion_to_deltas
from src.analysis.features import aggregate_choices, aggregate_premises
from src.analysis.stageA import (
    ATTRIBUTES,
    build_design_matrix,
    compute_ames_and_weights,
    fit_glm_clustered,
    per_trial_contributions,
)
from src.analysis.stageB import alignment_metrics
from src.utils.io import read_yaml


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_openai_backend(models_path: Path, backend_name: str | None, debug: bool):
    models_cfg = read_yaml(models_path)
    name = backend_name or models_cfg["sampling"].get("default_backend")
    if name not in models_cfg["backends"]:
        raise KeyError(f"Backend {name} not defined in models config")
    spec = models_cfg["backends"][name]
    if spec.get("type") != "openai":
        raise ValueError(f"Backend {name} is not type 'openai' (found {spec.get('type')})")
    kwargs = {k: v for k, v in spec.items() if k not in {"type"}}
    if debug:
        kwargs["debug"] = True
    return OpenAIBackend(**kwargs), spec, name


def create_sample_b3_trials() -> list[dict[str, Any]]:
    """Create multiple diverse B3-style trial configurations."""
    order = list(ATTRIBUTES)
    trials = [
        {
            "config_id": "DEBUG-B3-01",
            "block": "B3",
            "profile_a": Profile(levels={"E": "High", "A": "Low", "S": "High", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "High", "S": "Low", "D": "High"}),
            "order_a": order,
            "order_b": order,
            "paraphrase_id": 0,
            "seed": 42,
        },
        {
            "config_id": "DEBUG-B3-02",
            "block": "B3",
            "profile_a": Profile(levels={"E": "High", "A": "High", "S": "Low", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "Low", "S": "High", "D": "High"}),
            "order_a": order,
            "order_b": order,
            "paraphrase_id": 0,
            "seed": 43,
        },
        {
            "config_id": "DEBUG-B3-03",
            "block": "B3",
            "profile_a": Profile(levels={"E": "Medium", "A": "High", "S": "High", "D": "Low"}),
            "profile_b": Profile(levels={"E": "Low", "A": "Medium", "S": "Low", "D": "High"}),
            "order_a": order,
            "order_b": order,
            "paraphrase_id": 0,
            "seed": 44,
        },
    ]
    return trials


def create_trial_spec(
    base_config: dict[str, Any],
    manipulation: str,
    attribute_target: str | None = None,
    inject_offset: int = 0,
    *,
    variant_override: str | None = None,
) -> TrialSpec:
    """Create a TrialSpec from base config and manipulation."""
    config_id = base_config["config_id"]
    trial_id = f"{config_id}-{manipulation}"
    if attribute_target:
        trial_id += f"-{attribute_target}"
    if inject_offset != 0:
        trial_id += f"-offset{inject_offset}"
    if variant_override:
        trial_id += f"-{variant_override.replace('__', '_')}"

    variant = manipulation
    if manipulation in {"redact", "neutralize", "inject", "occlude_drop", "occlude_equalize", "occlude_swap"}:
        variant = "short_reason"
    if variant_override:
        variant = variant_override

    return TrialSpec(
        trial_id=trial_id,
        config_id=base_config["config_id"],
        block=base_config["block"],
        manipulation=manipulation,
        variant=variant,
        profile_a=base_config["profile_a"],
        profile_b=base_config["profile_b"],
        order_a=base_config["order_a"],
        order_b=base_config["order_b"],
        paraphrase_id=base_config["paraphrase_id"],
        attribute_target=attribute_target,
        inject_offset=inject_offset,
        seed=base_config["seed"],
        metadata={},
    )


def build_all_variations(base_config: dict[str, Any]) -> list[TrialSpec]:
    """Build trial specs for all manipulation variations."""
    specs = []

    # Base manipulations (no attribute targeting)
    for manipulation in ["short_reason", "split_reason", "premise_first"]:
        specs.append(create_trial_spec(base_config, manipulation))

    # Redact variations (one for each attribute)
    for attr in ATTRIBUTES:
        specs.append(create_trial_spec(base_config, "redact", attribute_target=attr))

    # Neutralize variations (one for each attribute)
    for attr in ATTRIBUTES:
        specs.append(create_trial_spec(base_config, "neutralize", attribute_target=attr))

    # Inject variations (different offsets for one attribute)
    for offset in [-1, 0, 1]:
        specs.append(create_trial_spec(base_config, "inject", attribute_target="A", inject_offset=offset))

    # Occlusion variations (one for each attribute)
    for attr in ATTRIBUTES:
        specs.append(create_trial_spec(base_config, "occlude_drop", attribute_target=attr))
        specs.append(create_trial_spec(base_config, "occlude_equalize", attribute_target=attr))
        specs.append(create_trial_spec(base_config, "occlude_swap", attribute_target=attr))

    # Judge variants (baseline + occlusions only)
    judge_variants = [
        "short_reason__judge_scores_joint",
        "short_reason__judge_scores_per_feature",
        "short_reason__judge_pairwise_joint",
        "short_reason__judge_pairwise_stepwise",
    ]
    base_for_judges = [
        spec for spec in specs
        if spec.manipulation in {"short_reason", "occlude_drop", "occlude_equalize", "occlude_swap"}
    ]
    for spec in base_for_judges:
        for variant in judge_variants:
            specs.append(
                create_trial_spec(
                    base_config,
                    spec.manipulation,
                    attribute_target=spec.attribute_target,
                    inject_offset=spec.inject_offset,
                    variant_override=variant,
                )
            )

    return specs


def responses_to_dataframe(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert run_trial results to responses DataFrame format."""
    rows_choices = []
    rows_premises = []

    for result in results:
        trial_id = result["trial_id"]
        config_id = result["config_id"]
        block = result["block"]
        manipulation = result["manipulation"]
        variant = result["variant"]

        for run_idx, run in enumerate(result["responses"]):
            # Extract choice
            choice_step = next((s for s in run["steps"] if s["name"] == "choice"), None)
            if choice_step and choice_step["parsed"].get("ok"):
                rows_choices.append({
                    "trial_id": trial_id,
                    "config_id": config_id,
                    "block": block,
                    "manipulation": manipulation,
                    "variant": variant,
                    "run": run_idx,
                    "choice": choice_step["parsed"]["choice"],
                })

            # Extract premise
            premise_step = next((s for s in run["steps"] if s["name"] == "premise"), None)
            if premise_step and premise_step["parsed"].get("ok"):
                rows_premises.append({
                    "trial_id": trial_id,
                    "config_id": config_id,
                    "block": block,
                    "manipulation": manipulation,
                    "variant": variant,
                    "run": run_idx,
                    "premise_attr": premise_step["parsed"]["attr"],
                    "premise_text": premise_step["parsed"]["text"],
                })

    choices_df = pd.DataFrame(rows_choices)
    premises_df = pd.DataFrame(rows_premises)

    return choices_df, premises_df


def create_trials_dataframe(specs: list[TrialSpec]) -> pd.DataFrame:
    """Create a trials DataFrame from trial specs."""
    from src.data.schema import compute_deltas

    rows = []
    for spec in specs:
        base_deltas = compute_deltas(spec.profile_a, spec.profile_b, ATTRIBUTES)
        deltas = apply_occlusion_to_deltas(base_deltas, spec.manipulation, spec.attribute_target)

        rows.append({
            "trial_id": spec.trial_id,
            "config_id": spec.config_id,
            "block": spec.block,
            "manipulation": spec.manipulation,
            "variant": spec.variant,
            "delta_E": deltas["E"],
            "delta_A": deltas["A"],
            "delta_S": deltas["S"],
            "delta_D": deltas["D"],
            "delta_pos_E": 0,  # No order effects in this debug
            "delta_pos_A": 0,
            "delta_pos_S": 0,
            "delta_pos_D": 0,
        })

    return pd.DataFrame(rows)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def print_responses_summary(results: list[dict[str, Any]]):
    """Print full responses for each trial."""
    print_section("RESPONSES")

    for result in results:
        trial_id = result["trial_id"]
        manipulation = result["manipulation"]
        variant = result["variant"]

        print(f"Trial: {trial_id}")
        print(f"  Manipulation: {manipulation} (variant: {variant})")

        for run_idx, run in enumerate(result["responses"]):
            print(f"  Run {run_idx}:")
            for step in run["steps"]:
                status = "OK" if step["parsed"].get("ok") else "FAIL"
                print(f"    {status} {step['name']}:")
                for line in step["content"].splitlines():
                    print(f"      {line}")
                parsed = json.dumps(step["parsed"], indent=2)
                for line in parsed.splitlines():
                    print(f"      {line}")
        print()


def print_stage_a_summary(weights_info: dict[str, Any], cv_metrics: dict[str, float]):
    """Print Stage A analysis summary."""
    print_section("STAGE A: WEIGHT ESTIMATION", "=")

    print("Estimated Weights:")
    for attr in ATTRIBUTES:
        weight = weights_info["weights"].get(attr, 0.0)
        beta = weights_info["beta"].get(attr, 0.0)
        print(f"  {attr}: {weight:.3f} (beta = {beta:.3f})")

    print("\nCross-Validation Metrics:")
    for metric, value in cv_metrics.items():
        print(f"  {metric}: {value:.4f}")


def print_stage_b_summary(alignment: dict[str, Any], premises_df: pd.DataFrame):
    """Print Stage B alignment summary."""
    print_section("STAGE B: ALIGNMENT METRICS", "=")

    print("Alignment Metrics:")
    print(f"  ECRB (top1 driver):  {alignment.get('ECRB_top1_driver', 'N/A'):.3f}")
    print(f"  ECRB (top1 weights): {alignment.get('ECRB_top1_weights', 'N/A'):.3f}")
    print(f"  Rank correlation:    {alignment.get('rank_corr', 'N/A'):.3f}")

    print("\nPremise Distribution:")
    if not premises_df.empty:
        counts = premises_df["premise_attr"].value_counts()
        total = len(premises_df)
        for attr in ATTRIBUTES:
            count = counts.get(attr, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {attr}: {count:2d} ({pct:5.1f}%)")
    else:
        print("  No premises collected")


def print_contributions(contributions_df: pd.DataFrame, trials_df: pd.DataFrame):
    """Print per-trial contribution analysis."""
    print_section("PER-TRIAL CONTRIBUTIONS", "-")

    # Merge with trial info to get manipulation
    merged = contributions_df.merge(
        trials_df[["trial_id", "manipulation"]],
        left_index=True,
        right_on="trial_id"
    )

    print("Driver attribute by manipulation:")
    driver_by_manip = merged.groupby("manipulation")["driver"].value_counts()
    for (manip, driver), count in driver_by_manip.items():
        print(f"  {manip:20s} -> {driver}: {count}")


def main():
    """Main debug script."""
    parser = argparse.ArgumentParser(description="Run all manipulation variations for debugging")
    parser.add_argument("--models", default="configs/models.yml", help="Path to models configuration YAML")
    parser.add_argument("--backend", default=None, help="Backend name from models.yml (must be openai)")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for trials")
    parser.add_argument("--S", type=int, default=1, help="Replicates per trial")
    parser.add_argument("--debug", action="store_true", help="Enable OpenAI backend debug logging")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials run")
    parser.add_argument("--out", default=None, help="Write output to a log file")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = out_path.open("w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)
        atexit.register(log_fh.close)
        print(f"Logging to {out_path}")

    print_section("DEBUG SCRIPT: ALL VARIATIONS", "#")
    print("This script tests multiple B3 trials through all manipulation variations")
    print("and performs full Stage A and Stage B analysis.\n")

    backend, backend_spec, backend_name = load_openai_backend(Path(args.models), args.backend, args.debug)
    temperature = args.temperature if args.temperature is not None else backend_spec.get("temperature", 1.0)
    max_tokens = args.max_tokens if args.max_tokens is not None else backend_spec.get("max_tokens", 256)

    print(f"Backend: {backend_name} ({backend_spec.get('model', 'unknown')})")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")

    # Create sample trials
    base_configs = create_sample_b3_trials()
    print(f"Base Configurations ({len(base_configs)} trials):")
    for base_config in base_configs:
        profile_a = base_config["profile_a"]
        profile_b = base_config["profile_b"]
        print(f"\n  {base_config['config_id']}:")
        print(f"    Profile A: E={profile_a.levels['E']}, A={profile_a.levels['A']}, "
              f"S={profile_a.levels['S']}, D={profile_a.levels['D']}")
        print(f"    Profile B: E={profile_b.levels['E']}, A={profile_b.levels['A']}, "
              f"S={profile_b.levels['S']}, D={profile_b.levels['D']}")

    # Build variations for the first trial only to keep output manageable
    base_config = base_configs[0]
    specs = build_all_variations(base_config)

    # Add short_reason variants for other trials
    for other_config in base_configs[1:]:
        specs.append(create_trial_spec(other_config, "short_reason"))

    print(f"\nGenerated {len(specs)} total trial variations")
    if args.limit is not None:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} trials")

    # Run trials with OpenAI backend
    print_section("RUNNING TRIALS")
    results = []

    for idx, spec in enumerate(specs):
        result = run_trial(
            spec,
            backend,
            S=args.S,
            temperature=temperature,
            seed=args.seed + idx,
            max_tokens=max_tokens,
        )
        results.append(result)
        print_responses_summary([result])

    print(f"Completed {len(results)} trials")

    # Convert to DataFrames
    choices_df, premises_df = responses_to_dataframe(results)
    trials_df = create_trials_dataframe(specs)

    # Aggregate choices for Stage A
    print_section("AGGREGATING DATA FOR ANALYSIS")

    # Manual aggregation similar to aggregate_choices
    choice_agg = choices_df.groupby("trial_id").agg(
        successes=("choice", lambda x: (x == "A").sum()),
        trials=("choice", "count"),
    ).reset_index()

    # Merge with trials data
    stageA_data = trials_df.merge(choice_agg, on="trial_id", how="left")
    stageA_data["successes"] = stageA_data["successes"].fillna(0)
    stageA_data["trials"] = stageA_data["trials"].fillna(0)

    print(f"Aggregated {len(choice_agg)} trial choices")
    print(f"Collected {len(premises_df)} premises")

    # Stage A Analysis
    print_section("STAGE A ANALYSIS")

    try:
        # Build design matrix (exclude B1, but we don't have B1 here anyway)
        design = build_design_matrix(
            stageA_data,
            include_interactions=False,
            include_order_terms=False,  # No order effects in this debug
            exclude_b1=False,  # No B1 trials
        )

        print(f"Design matrix: {design.X.shape[0]} observations, {design.X.shape[1]} features")
        print(f"Features: {list(design.X.columns)}")

        # Fit model
        model = fit_glm_clustered(design)
        weights_info = compute_ames_and_weights(model)

        # Cross-validation (simplified) - skip if dataset too small
        cv_metrics = {}
        try:
            if len(design.X) >= 10:
                from src.analysis.cv import cross_validate_design
                cv_metrics = cross_validate_design(design, n_splits=min(3, len(design.X) // 3))
        except Exception as e:
            print(f"  CV skipped due to: {e}")
            cv_metrics = {"log_loss": float("nan"), "brier": float("nan"), "accuracy": float("nan")}

        print_stage_a_summary(weights_info, cv_metrics)

        # Compute per-trial contributions
        unique_trials = stageA_data.drop_duplicates("trial_id").set_index("trial_id")
        contributions = per_trial_contributions(unique_trials, model)

        print_contributions(contributions, trials_df)

    except Exception as e:
        print(f"Stage A analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Stage B Analysis
    print_section("STAGE B ANALYSIS")

    try:
        # Prepare responses DataFrame
        responses_df = choices_df.merge(
            premises_df,
            on=["trial_id", "config_id", "block", "manipulation", "variant", "run"],
            how="left",
        )

        # Compute alignment metrics
        alignment = alignment_metrics(responses_df, stageA_data, model)

        print_stage_b_summary(alignment, premises_df)

    except Exception as e:
        print(f"Stage B analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    print_section("DEBUG COMPLETE", "#")
    print("All variations tested successfully")
    print("Stage A and Stage B analysis completed")
    print("\nThis debug run demonstrates the full pipeline with all manipulation types.")


if __name__ == "__main__":
    main()
