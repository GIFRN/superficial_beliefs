#!/usr/bin/env python3
"""
Quick ECRB Test Script

Tests if increasing reasoning_effort from 'minimal' to 'medium' improves ECRB alignment.
Uses a random sample of 50 non-B1 trials for fast iteration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from src.llm.backends.openai import OpenAIBackend
from src.llm.harness import build_trial_specs, run_trial
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_json

ATTRIBUTES = ["E", "A", "S", "D"]


def aggregate_choices(responses: list[dict]) -> pd.DataFrame:
    """Aggregate choice responses by trial"""
    records = []
    for response in responses:
        for run in response.get("responses", []):
            trial_id = response.get("trial_id")
            choice_step = next((s for s in run.get("steps", []) if s["name"] == "choice"), None)
            if choice_step and choice_step.get("parsed", {}).get("ok"):
                choice = choice_step["parsed"]["choice"]
                records.append({"trial_id": trial_id, "choice": choice})
    
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["trial_id", "successes", "trials"])
    
    df["is_A"] = df["choice"] == "A"
    return df.groupby("trial_id").agg(
        successes=("is_A", "sum"),
        trials=("is_A", "count")
    ).reset_index()


def extract_premise_attrs(responses: list[dict]) -> pd.DataFrame:
    """Extract premise attributions from responses"""
    records = []
    for response in responses:
        for run in response.get("responses", []):
            trial_id = response.get("trial_id")
            premise_step = next((s for s in run.get("steps", []) if s["name"] == "premise"), None)
            if premise_step and premise_step.get("parsed", {}).get("ok"):
                attr = premise_step["parsed"]["attr"]
                if attr in ATTRIBUTES:
                    records.append({"trial_id": trial_id, "premise_attr": attr})
    
    return pd.DataFrame(records)


def fit_stage_a_model(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> Any:
    """Fit logistic regression model on trial choices"""
    # Merge trials with choices
    data = trials_df.merge(choice_agg, on="trial_id", how="left")
    data["successes"] = data["successes"].fillna(0).astype(int)
    data["trials"] = data["trials"].fillna(0).astype(int)
    
    # Filter to trials with responses
    data = data[data["trials"] > 0].copy()
    
    if data.empty:
        raise ValueError("No valid trial data")
    
    # Build design matrix
    X = pd.DataFrame(index=data.index)
    X["Intercept"] = 1.0
    
    for attr in ATTRIBUTES:
        col = f"delta_{attr}"
        if col not in data:
            data[col] = 0
        X[f"diff_{attr}"] = data[col]
    
    # Fit model
    weights = data["trials"].astype(float)
    y = data["successes"] / weights
    groups = data["config_id"].astype(str)
    
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    
    # Get robust standard errors with clustering
    try:
        result = result.get_robustcov_results(cov_type="cluster", groups=groups)
    except AttributeError:
        from statsmodels.stats.sandwich_covariance import cov_cluster
        cov = cov_cluster(result, np.asarray(groups))
        result.cov_params_default = cov
    
    return result, data


def compute_ames_and_weights(model) -> dict:
    """Compute average marginal effects and normalized weights"""
    betas = {}
    for attr in ATTRIBUTES:
        feature_name = f"diff_{attr}"
        betas[attr] = float(model.params.get(feature_name, 0.0))
    
    # Normalize to weights (positive only)
    positive = {attr: max(val, 0.0) for attr, val in betas.items()}
    total = sum(positive.values())
    weights = {attr: (val / total if total > 0 else 0.0) for attr, val in positive.items()}
    
    return {"beta": betas, "weights": weights}


def compute_per_trial_drivers(data: pd.DataFrame, model) -> pd.DataFrame:
    """Determine which attribute drives each trial's choice"""
    drivers = []
    
    for _, row in data.iterrows():
        contributions = {}
        for attr in ATTRIBUTES:
            feature_name = f"diff_{attr}"
            coeff = model.params.get(feature_name, 0.0)
            contributions[attr] = coeff * row[f"delta_{attr}"]
        
        driver = max(contributions.items(), key=lambda x: x[1])[0]
        drivers.append({"trial_id": row["trial_id"], "driver": driver})
    
    return pd.DataFrame(drivers)


def calculate_ecrb(responses: list[dict], trials_df: pd.DataFrame, model, weights: dict) -> dict:
    """Calculate ECRB alignment metrics"""
    # Get premise attributions
    premise_df = extract_premise_attrs(responses)
    
    if premise_df.empty:
        return {
            "ECRB_top1_driver": None,
            "ECRB_top1_weights": None,
            "rank_corr": None,
            "total_attributions": 0
        }
    
    # Get model drivers
    choice_agg = aggregate_choices(responses)
    _, fitted_data = fit_stage_a_model(trials_df, choice_agg)
    driver_df = compute_per_trial_drivers(fitted_data, model)
    
    # Merge premise attributions with drivers
    merged = premise_df.merge(driver_df, on="trial_id", how="left")
    merged = merged.dropna(subset=["driver"])
    
    if merged.empty:
        return {
            "ECRB_top1_driver": None,
            "ECRB_top1_weights": None,
            "rank_corr": None,
            "total_attributions": 0
        }
    
    # Calculate ECRB_top1_driver
    ecrb_driver = (merged["premise_attr"] == merged["driver"]).mean()
    
    # Calculate ECRB_top1_weights
    top_attr = max(weights.items(), key=lambda x: x[1])[0]
    ecrb_weights = (merged["premise_attr"] == top_attr).mean()
    
    # Calculate rank correlation
    counts = merged["premise_attr"].value_counts()
    premise_series = pd.Series({attr: counts.get(attr, 0) for attr in ATTRIBUTES})
    weight_series = pd.Series(weights)
    
    if premise_series.sum() > 0 and weight_series.sum() > 0:
        rank_corr = premise_series.rank().corr(weight_series.rank(), method="spearman")
    else:
        rank_corr = None
    
    return {
        "ECRB_top1_driver": float(ecrb_driver),
        "ECRB_top1_weights": float(ecrb_weights),
        "rank_corr": float(rank_corr) if rank_corr is not None else None,
        "total_attributions": len(merged)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ECRB test with medium reasoning effort")
    parser.add_argument("--config", default="configs/default.yml", help="Dataset config")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Dataset directory")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of trials to sample")
    parser.add_argument("--reasoning-effort", default="medium", choices=["minimal", "low", "medium", "high"], help="GPT-5 reasoning effort")
    parser.add_argument("--replicates", type=int, default=5, help="Samples per trial")
    parser.add_argument("--out", default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Auto-generate output directory name with reasoning effort if not specified
    if args.out is None:
        args.out = f"results/quick_ecrb_test_{args.reasoning_effort}"
    
    print("=" * 80)
    print("QUICK ECRB TEST")
    print("=" * 80)
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Sample size: {args.sample_size} trials")
    print(f"Replicates per trial: {args.replicates}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.out}")
    print()
    
    # Load data
    cfg = load_config(args.config)
    dataset_dir = Path(args.dataset)
    configs_df = pd.read_parquet(dataset_dir / "dataset_configs.parquet")
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    
    # Sample non-B1 trials
    non_b1 = trials_df[trials_df["block"] != "B1"].copy()
    rng = np.random.RandomState(args.seed)
    sample_indices = rng.choice(len(non_b1), size=min(args.sample_size, len(non_b1)), replace=False)
    sampled_trials = non_b1.iloc[sample_indices].reset_index(drop=True)
    
    print(f"Sampled {len(sampled_trials)} trials from {len(non_b1)} non-B1 trials")
    print(f"Block distribution: {sampled_trials['block'].value_counts().to_dict()}")
    print()
    
    # Initialize backend with specified reasoning effort
    backend = OpenAIBackend(
        model="gpt-5-mini",
        temperature=1.0,
        max_tokens=4000,
        reasoning_effort=args.reasoning_effort
    )
    
    # Build trial specs
    trial_specs = build_trial_specs(cfg, configs_df, sampled_trials)
    
    # Run trials
    print(f"Running {len(trial_specs)} trials with reasoning_effort={args.reasoning_effort}...")
    responses = []
    
    for idx, spec in enumerate(tqdm(trial_specs, desc="Trials")):
        trial_seed = args.seed + idx
        result = run_trial(
            spec,
            backend,
            S=args.replicates,
            temperature=1.0,
            seed=trial_seed,
            max_tokens=4000
        )
        responses.append(result)
    
    print(f"\nCompleted {len(responses)} trials")
    
    # Aggregate choices and fit model
    print("\nFitting Stage A model...")
    choice_agg = aggregate_choices(responses)
    print(f"Valid choices: {choice_agg['trials'].sum()} / {len(responses) * args.replicates}")
    
    model, fitted_data = fit_stage_a_model(sampled_trials, choice_agg)
    stats = compute_ames_and_weights(model)
    
    print("\nFitted attribute weights:")
    for attr, weight in sorted(stats["weights"].items(), key=lambda x: -x[1]):
        beta = stats["beta"][attr]
        print(f"  {attr}: {weight:.1%} (β = {beta:.3f})")
    
    # Calculate ECRB
    print("\nCalculating ECRB...")
    ecrb_results = calculate_ecrb(responses, sampled_trials, model, stats["weights"])
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"ECRB_top1_driver:  {ecrb_results['ECRB_top1_driver']:.1%}" if ecrb_results['ECRB_top1_driver'] else "ECRB_top1_driver: N/A")
    print(f"ECRB_top1_weights: {ecrb_results['ECRB_top1_weights']:.1%}" if ecrb_results['ECRB_top1_weights'] else "ECRB_top1_weights: N/A")
    print(f"Rank correlation:  {ecrb_results['rank_corr']:.3f}" if ecrb_results['rank_corr'] else "Rank correlation: N/A")
    print(f"Total attributions: {ecrb_results['total_attributions']}")
    print("=" * 80)
    
    # Save results
    out_dir = ensure_dir(args.out)
    
    summary = {
        "reasoning_effort": args.reasoning_effort,
        "sample_size": len(sampled_trials),
        "replicates": args.replicates,
        "seed": args.seed,
        "weights": stats["weights"],
        "beta": stats["beta"],
        "ecrb": ecrb_results,
        "sampled_trial_ids": sampled_trials["trial_id"].tolist()
    }
    
    write_json(summary, out_dir / "ecrb_test_summary.json")
    
    # Save responses
    with (out_dir / "responses.jsonl").open("w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()

