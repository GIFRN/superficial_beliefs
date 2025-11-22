#!/usr/bin/env python3
"""
Compare GPT-5-mini across reasoning effort levels: minimal, low, medium, high

Runs 100 B3 trials with 5 replicates each through all four reasoning levels.
Uses streamlined prompting without redundant "repeat" steps.
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

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.llm.backends.openai import OpenAIBackend
from src.llm.types import ConversationPlan, ConversationStep, TrialSpec
from src.data.schema import ATTR_LABELS
from src.data.paraphrases import render_profile
from src.analysis.stageA import build_design_matrix, fit_glm_clustered, compute_ames_and_weights
from src.analysis.stageB import alignment_metrics
from src.analysis.features import aggregate_choices
from src.utils.io import ensure_dir, write_json


# Streamlined prompting (remove "repeat" step)
def streamlined_conversation_plan(trial: TrialSpec) -> ConversationPlan:
    """Simplified conversation plan without redundant repeat step."""
    system_prompt = "You are an expert clinical decision assistant. Follow instructions exactly."
    
    # Build attribute labels
    attr_labels = " | ".join(ATTR_LABELS.values())
    
    # Profile rendering
    profile_a_text = render_profile(
        trial.profile_a, 
        trial.paraphrase_id, 
        trial.order_a, 
        "Drug A"
    )
    profile_b_text = render_profile(
        trial.profile_b, 
        trial.paraphrase_id, 
        trial.order_b, 
        "Drug B"
    )
    
    # Main prompt
    main_prompt = f"""You are optimizing 5-year overall patient outcome.
State the option you choose and explain the single most important factor for your choice in ≤12 words.
Format: <Option>. <Brief reason>.

{profile_a_text}
{profile_b_text}"""
    
    # Premise extraction prompt
    premise_prompt = f"""Now return the attribute that was most important:
PremiseAttribute = <one of [{attr_labels}]>
PremiseText = '<8-12 words>'"""
    
    steps = [
        ConversationStep(name="choice", prompt=main_prompt, expects="choice"),
        ConversationStep(name="premise", prompt=premise_prompt, expects="premise"),
    ]
    
    return ConversationPlan(system_prompt=system_prompt, steps=steps)


def run_trial_streamlined(
    trial: TrialSpec,
    backend: OpenAIBackend,
    S: int,
    seed: int,
    counterbalance: bool = True,
) -> dict[str, Any]:
    """Run a single trial with streamlined prompting.
    
    If counterbalance=True, runs S/2 replicates with original orientation
    (Drug A = profile_a, Drug B = profile_b) and S/2 with reversed orientation
    (Drug A = profile_b, Drug B = profile_a). This controls for position bias
    while maintaining consistent prompt order (Drug A always mentioned first).
    """
    from src.llm.harness import _parse_step
    
    if counterbalance and S % 2 != 0:
        raise ValueError(f"S must be even for counterbalancing, got {S}")
    
    rng = np.random.default_rng(seed ^ trial.seed)
    runs = []
    
    # Determine orientations to run
    if counterbalance:
        orientations = [
            ("original", trial, S // 2),
            ("reversed", TrialSpec(
                trial_id=trial.trial_id,
                config_id=trial.config_id,
                block=trial.block,
                manipulation=trial.manipulation,
                variant=trial.variant,
                profile_a=trial.profile_b,  # Swap: Drug A now gets profile_b
                profile_b=trial.profile_a,  # Swap: Drug B now gets profile_a
                order_a=trial.order_b,
                order_b=trial.order_a,
                paraphrase_id=trial.paraphrase_id,
                attribute_target=trial.attribute_target,
                inject_offset=trial.inject_offset,
                seed=trial.seed,
                metadata=trial.metadata,
            ), S // 2)
        ]
    else:
        orientations = [("original", trial, S)]
    
    # Run replicates for each orientation
    for orientation_name, trial_spec, n_reps in orientations:
        plan = streamlined_conversation_plan(trial_spec)
        
        for rep_idx in range(n_reps):
            system_message = {"role": "system", "content": plan.system_prompt}
            conversation = [system_message]
            replicate_seed = int(rng.integers(0, 2**32 - 1))
            steps_results = []
            history = [system_message.copy()]
            
            for idx, step in enumerate(plan.steps):
                prompt = step.prompt
                history.append({"role": "user", "content": prompt})
                conversation.append({"role": "user", "content": prompt})
                
                step_seed = (replicate_seed + idx) % (2**32)
                
                try:
                    response = backend.complete(
                        history, 
                        temperature=1.0, 
                        max_tokens=4000, 
                        seed=step_seed
                    )
                except Exception as e:
                    print(f"  ⚠️  Error on {trial.trial_id}, {orientation_name}, rep {rep_idx}: {e}")
                    response = ""
                
                history.append({"role": "assistant", "content": response})
                conversation.append({"role": "assistant", "content": response})
                parsed = _parse_step(step, response)
                
                steps_results.append({
                    "name": step.name,
                    "content": response,
                    "parsed": parsed
                })
                
                if not parsed.get("ok"):
                    break
            
            runs.append({
                "seed": replicate_seed,
                "orientation": orientation_name,
                "steps": steps_results,
                "conversation": conversation
            })
    
    return {
        "trial_id": trial.trial_id,
        "config_id": trial.config_id,
        "block": trial.block,
        "manipulation": trial.manipulation,
        "variant": trial.variant,
        "responses": runs
    }


def sample_b3_trials(dataset_dir: Path, n_samples: int = 100, seed: int = 42, n_test: int = 0, b2_fraction: float = 0.0, exclude_trial_ids: set[str] = None) -> tuple[list[dict], list[dict]]:
    """Sample trials from dataset with optional train/test split.
    
    Args:
        dataset_dir: Path to dataset directory
        n_samples: Number of training samples
        seed: Random seed
        n_test: Number of held-out test samples (default 0 = no split)
        b2_fraction: Fraction of training samples to take from B2 (rest from B3)
    
    Returns:
        (train_specs, test_specs) - tuple of trial spec lists
    """
    configs_df = pd.read_parquet(dataset_dir / "dataset_configs.parquet")
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    
    # Exclude already-used trials if specified
    if exclude_trial_ids:
        trials_df = trials_df[~trials_df["trial_id"].isin(exclude_trial_ids)]
        print(f"  📌 Excluding {len(exclude_trial_ids)} already-used trials")
    
    rng = np.random.default_rng(seed)
    
    # Sample training set
    n_b2 = int(n_samples * b2_fraction)
    n_b3_train = n_samples - n_b2
    
    train_trials = []
    if n_b2 > 0:
        b2_trials = trials_df[trials_df["block"] == "B2"].copy()
        if len(b2_trials) >= n_b2:
            train_trials.append(b2_trials.sample(n=n_b2, random_state=seed))
        else:
            print(f"  ⚠️  Warning: Only {len(b2_trials)} B2 trials available (requested {n_b2})")
            if len(b2_trials) > 0:
                train_trials.append(b2_trials)
    
    b3_trials = trials_df[trials_df["block"] == "B3"].copy()
    if len(b3_trials) >= n_b3_train + n_test:
        # Split B3 into train and test
        b3_shuffled = b3_trials.sample(frac=1.0, random_state=seed)
        train_trials.append(b3_shuffled.iloc[:n_b3_train])
        test_trials = b3_shuffled.iloc[n_b3_train:n_b3_train + n_test] if n_test > 0 else pd.DataFrame()
    else:
        # Not enough trials
        available = min(n_b3_train, len(b3_trials))
        print(f"  ⚠️  Warning: Only {len(b3_trials)} B3 trials available (requested {n_b3_train + n_test})")
        if available > 0:
            train_trials.append(b3_trials.sample(n=available, random_state=seed))
        test_trials = pd.DataFrame()
    
    sampled_train = pd.concat(train_trials, ignore_index=True) if train_trials else pd.DataFrame()
    sampled_test = test_trials if n_test > 0 else pd.DataFrame()
    
    # Build trial specs
    from src.llm.harness import build_trial_specs
    from src.utils.config import load_config
    
    cfg = load_config(ROOT / "configs/default.yml")
    
    def build_specs(sampled_df):
        """Helper to build trial specs from dataframe."""
        if sampled_df.empty:
            return []
        
        from src.data.schema import Profile
        
        sampled_config_ids = sampled_df["config_id"].unique()
        sampled_configs = configs_df[configs_df["config_id"].isin(sampled_config_ids)]
        
        specs = []
        for _, trial_row in sampled_df.iterrows():
            config_row = sampled_configs[sampled_configs["config_id"] == trial_row["config_id"]].iloc[0]
            
            # Build profile - deserialize JSON if needed
            levels_left = config_row["levels_left"]
            levels_right = config_row["levels_right"]
            
            # Handle JSON string serialization from parquet
            if isinstance(levels_left, str):
                levels_left = json.loads(levels_left)
            if isinstance(levels_right, str):
                levels_right = json.loads(levels_right)
            
            profile_left = Profile(levels=levels_left)
            profile_right = Profile(levels=levels_right)
            
            # Determine A and B based on orientation
            # labelA indicates which label is assigned to the LEFT profile
            if trial_row["labelA"] == "A":
                profile_a, profile_b = profile_left, profile_right
            else:
                profile_a, profile_b = profile_right, profile_left
            
            spec = TrialSpec(
                trial_id=trial_row["trial_id"],
                config_id=trial_row["config_id"],
                block=trial_row["block"],
                manipulation=trial_row["manipulation"],
                variant="short_reason",
                profile_a=profile_a,
                profile_b=profile_b,
                order_a=cfg.orders_permutations[trial_row["order_id_A"]],
                order_b=cfg.orders_permutations[trial_row["order_id_B"]],
                paraphrase_id=trial_row["paraphrase_id"],
                attribute_target=trial_row.get("attribute_target"),
                inject_offset=trial_row.get("inject_offset", 0),
                seed=trial_row["seed"],
                metadata={},
            )
            specs.append(spec)
        
        return specs
    
    train_specs = build_specs(sampled_train)
    test_specs = build_specs(sampled_test)
    
    return train_specs, test_specs


def analyze_test_results(responses: list[dict], trials_df: pd.DataFrame, trained_model, split_name: str = "test") -> dict[str, Any]:
    """Analyze held-out test results with predictions from trained model."""
    # Convert responses to dataframe
    choice_rows = []
    premise_rows = []
    
    for resp in responses:
        trial_id = resp["trial_id"]
        for run_idx, run in enumerate(resp["responses"]):
            orientation = run.get("orientation", "original")
            
            # Extract choice
            choice_step = next((s for s in run["steps"] if s["name"] == "choice"), None)
            if choice_step and choice_step["parsed"].get("ok"):
                raw_choice = choice_step["parsed"]["choice"]
                
                # CRITICAL FIX: Flip choice for reversed orientation
                # In reversed orientation, "Drug A" in the prompt corresponds to profile_b in the dataset
                # We need to record choices relative to the dataset's profile_a/profile_b
                if orientation == "reversed":
                    adjusted_choice = "B" if raw_choice == "A" else "A"
                else:
                    adjusted_choice = raw_choice
                
                choice_rows.append({
                    "trial_id": trial_id,
                    "run": run_idx,
                    "choice": adjusted_choice,
                })
            
            # Extract premise
            premise_step = next((s for s in run["steps"] if s["name"] == "premise"), None)
            if premise_step and premise_step["parsed"].get("ok"):
                premise_rows.append({
                    "trial_id": trial_id,
                    "run": run_idx,
                    "premise_attr": premise_step["parsed"]["attr"],
                    "premise_text": premise_step["parsed"]["text"],
                })
    
    if not choice_rows:
        return {"error": "No valid choices collected"}
    
    choices_df = pd.DataFrame(choice_rows)
    premises_df = pd.DataFrame(premise_rows) if premise_rows else pd.DataFrame()
    
    # Aggregate choices
    choice_agg = choices_df.groupby("trial_id").agg(
        successes=("choice", lambda x: (x == "A").sum()),
        trials=("choice", "count"),
    ).reset_index()
    
    # Merge with trial features
    test_data = trials_df.merge(choice_agg, on="trial_id", how="left")
    test_data["successes"] = test_data["successes"].fillna(0)
    test_data["trials"] = test_data["trials"].fillna(0)
    
    # Make predictions using trained model
    try:
        design = build_design_matrix(
            test_data,
            include_interactions=False,
            include_order_terms=False,
            exclude_b1=False
        )
        
        # Predict probabilities
        X = design.X  # Access as attribute, not dictionary
        beta = trained_model.params.values
        linear_pred = X @ beta
        pred_probs = 1 / (1 + np.exp(-linear_pred))
        
        # Compute prediction metrics
        actual_probs = test_data["successes"] / test_data["trials"]
        mae = np.abs(pred_probs - actual_probs).mean()
        rmse = np.sqrt(((pred_probs - actual_probs) ** 2).mean())
        
        # Correlation between predicted and actual
        pred_corr = np.corrcoef(pred_probs, actual_probs)[0, 1] if len(pred_probs) > 1 else np.nan
        
        # Accuracy for discrete predictions (>0.5 → A)
        pred_choices = (pred_probs > 0.5).astype(int)
        actual_choices = (actual_probs > 0.5).astype(int)
        accuracy = (pred_choices == actual_choices).mean()
        
        # Compute weights from trained model
        weights_info = compute_ames_and_weights(trained_model)
        
        # Compute alignment on test set
        if not premises_df.empty:
            responses_df = choices_df.merge(premises_df, on=["trial_id", "run"], how="left")
            alignment = alignment_metrics(responses_df, test_data, trained_model)
        else:
            alignment = {}
        
        return {
            "split": split_name,
            "weights": weights_info["weights"],
            "beta": weights_info["beta"],
            "alignment": alignment,
            "prediction": {
                "mae": float(mae),
                "rmse": float(rmse),
                "correlation": float(pred_corr) if not np.isnan(pred_corr) else None,
                "accuracy": float(accuracy),
            },
            "n_trials": len(test_data),
            "n_responses": len(choices_df),
        }
    
    except Exception as e:
        import traceback
        print(f"⚠️  Error details: {traceback.format_exc()}")
        return {"error": str(e), "split": split_name}


def analyze_results(responses: list[dict], trials_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze results and compute metrics."""
    # Convert responses to dataframe
    choice_rows = []
    premise_rows = []
    
    for resp in responses:
        trial_id = resp["trial_id"]
        for run_idx, run in enumerate(resp["responses"]):
            orientation = run.get("orientation", "original")
            
            # Extract choice
            choice_step = next((s for s in run["steps"] if s["name"] == "choice"), None)
            if choice_step and choice_step["parsed"].get("ok"):
                raw_choice = choice_step["parsed"]["choice"]
                
                # CRITICAL FIX: Flip choice for reversed orientation
                # In reversed orientation, "Drug A" in the prompt corresponds to profile_b in the dataset
                # We need to record choices relative to the dataset's profile_a/profile_b
                if orientation == "reversed":
                    adjusted_choice = "B" if raw_choice == "A" else "A"
                else:
                    adjusted_choice = raw_choice
                
                choice_rows.append({
                    "trial_id": trial_id,
                    "run": run_idx,
                    "choice": adjusted_choice,
                })
            
            # Extract premise
            premise_step = next((s for s in run["steps"] if s["name"] == "premise"), None)
            if premise_step and premise_step["parsed"].get("ok"):
                premise_rows.append({
                    "trial_id": trial_id,
                    "run": run_idx,
                    "premise_attr": premise_step["parsed"]["attr"],
                    "premise_text": premise_step["parsed"]["text"],
                })
    
    if not choice_rows:
        return {"error": "No valid choices collected"}
    
    choices_df = pd.DataFrame(choice_rows)
    premises_df = pd.DataFrame(premise_rows) if premise_rows else pd.DataFrame()
    
    # Aggregate choices
    choice_agg = choices_df.groupby("trial_id").agg(
        successes=("choice", lambda x: (x == "A").sum()),
        trials=("choice", "count"),
    ).reset_index()
    
    # Merge with trial features
    stageA_data = trials_df.merge(choice_agg, on="trial_id", how="left")
    stageA_data["successes"] = stageA_data["successes"].fillna(0)
    stageA_data["trials"] = stageA_data["trials"].fillna(0)
    
    # Build design matrix and fit
    try:
        design = build_design_matrix(
            stageA_data,
            include_interactions=False,
            include_order_terms=False,
            exclude_b1=False
        )
        model = fit_glm_clustered(design)
        weights_info = compute_ames_and_weights(model)
        
        # Compute alignment if we have premises
        if not premises_df.empty:
            responses_df = choices_df.merge(premises_df, on=["trial_id", "run"], how="left")
            alignment = alignment_metrics(responses_df, stageA_data, model)
        else:
            alignment = {}
        
        # Compute choice variance
        choice_var = (stageA_data["successes"] / stageA_data["trials"]).var()
        
        # Count extreme choices
        p_choose_a = stageA_data["successes"] / stageA_data["trials"]
        all_a = (p_choose_a >= 0.95).sum()
        all_b = (p_choose_a <= 0.05).sum()
        mixed = ((p_choose_a > 0.05) & (p_choose_a < 0.95)).sum()
        
        return {
            "weights": weights_info["weights"],
            "beta": weights_info["beta"],
            "alignment": alignment,
            "choice_variance": float(choice_var),
            "extreme_choices": {
                "all_a": int(all_a),
                "all_b": int(all_b),
                "mixed": int(mixed),
                "total": len(stageA_data)
            },
            "n_trials": len(stageA_data),
            "n_responses": len(choices_df),
            "model": model,  # Return fitted model for test predictions
        }
    
    except Exception as e:
        return {"error": str(e), "model": None}


def is_valid_response(resp: dict) -> bool:
    """Check if a response has at least one successful replicate."""
    if "responses" not in resp:
        return False
    
    for run in resp["responses"]:
        if "steps" in run:
            for step in run["steps"]:
                if step.get("parsed", {}).get("ok"):
                    return True
    
    return False


def clean_and_load_responses(responses_path: Path) -> tuple[list[dict], set[str]]:
    """Load and clean responses from JSONL file, removing invalid/malformed entries."""
    if not responses_path.exists():
        return [], set()
    
    responses = []
    completed_trial_ids = set()
    invalid_count = 0
    malformed_count = 0
    
    try:
        lines = []
        with responses_path.open("r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip blank lines
                    continue
                
                try:
                    resp = json.loads(line)
                    
                    # Check if response has valid data
                    if is_valid_response(resp):
                        responses.append(resp)
                        completed_trial_ids.add(resp["trial_id"])
                        lines.append(json.dumps(resp))
                    else:
                        invalid_count += 1
                        
                except json.JSONDecodeError as e:
                    malformed_count += 1
                    print(f"  ⚠️  Skipping malformed line {line_num}: {str(e)[:50]}")
        
        # Rewrite file if we removed any invalid entries
        if invalid_count > 0 or malformed_count > 0:
            print(f"  🧹 Cleaning: removed {invalid_count} failed trials and {malformed_count} malformed lines")
            with responses_path.open("w") as f:
                for line in lines:
                    f.write(line + "\n")
        
        print(f"  📂 Found {len(responses)} valid responses ({len(completed_trial_ids)} trials)")
        return responses, completed_trial_ids
        
    except Exception as e:
        print(f"  ⚠️  Error loading existing responses: {e}")
        return [], set()


def load_existing_responses(responses_path: Path) -> tuple[list[dict], set[str]]:
    """Load existing responses - alias for clean_and_load_responses."""
    return clean_and_load_responses(responses_path)


def save_response_incremental(response: dict, responses_path: Path):
    """Append a single response to the JSONL file."""
    with responses_path.open("a") as f:
        f.write(json.dumps(response) + "\n")


def get_paths(base_dir: Path, model_name: str, effort: str, split: str) -> tuple[Path, Path]:
    """Get paths for responses and results files in the structured directory layout."""
    # Structure: base_dir/model/effort/split/filename
    # We sanitize model name for directory (e.g. gpt-5-mini)
    model_dir = model_name.replace("/", "_")
    target_dir = base_dir / model_dir / effort / split
    target_dir.mkdir(parents=True, exist_ok=True)
    
    responses_path = target_dir / f"responses_{effort}_{split}.jsonl"
    results_path = target_dir / f"results_{effort}_{split}.json"
    
    return responses_path, results_path


def collect_all_existing_trial_ids(out_dir: Path, model_name: str, effort_levels: list[str]) -> set[str]:
    """Collect all trial IDs that have VALID responses across all effort levels (train + test).
    
    This uses clean_and_load_responses() to ensure we only exclude trials with valid data,
    allowing failed trials to be resampled.
    """
    all_trial_ids = set()
    
    for effort in effort_levels:
        for split in ["train", "test"]:
            resp_path, _ = get_paths(out_dir, model_name, effort, split)
            if resp_path.exists():
                try:
                    # Use the cleaning function to get only valid responses
                    _, valid_trial_ids = clean_and_load_responses(resp_path)
                    all_trial_ids.update(valid_trial_ids)
                except Exception as e:
                    print(f"  ⚠️  Error reading {resp_path}: {e}")
    
    return all_trial_ids


def check_completion_status(out_dir: Path, model_name: str, effort: str, expected_train: int, expected_test: int) -> dict[str, bool]:
    """Check which phases are complete for a given effort level."""
    train_resp_path, train_res_path = get_paths(out_dir, model_name, effort, "train")
    test_resp_path, test_res_path = get_paths(out_dir, model_name, effort, "test")
    
    status = {
        "train_responses_complete": False,
        "train_analysis_complete": False,
        "test_responses_complete": False,
        "test_analysis_complete": False,
    }
    
    # Check train responses
    if train_resp_path.exists():
        _, completed = load_existing_responses(train_resp_path)
        status["train_responses_complete"] = len(completed) >= expected_train
    
    # Check train analysis
    if train_res_path.exists():
        try:
            with train_res_path.open("r") as f:
                results = json.load(f)
                status["train_analysis_complete"] = "weights" in results or "error" not in results
        except:
            pass
    
    # Check test responses
    if test_resp_path.exists() and expected_test > 0:
        _, completed = load_existing_responses(test_resp_path)
        status["test_responses_complete"] = len(completed) >= expected_test
    
    # Check test analysis
    if test_res_path.exists() and expected_test > 0:
        try:
            with test_res_path.open("r") as f:
                results = json.load(f)
                status["test_analysis_complete"] = "prediction" in results or "error" not in results
        except:
            pass
    
    return status


def print_test_results(effort: str, results: dict[str, Any]):
    """Print held-out test results."""
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS: {effort.upper()} REASONING EFFORT")
    print(f"{'='*80}")
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 Data Collection:")
    print(f"  Trials: {results['n_trials']}")
    print(f"  Responses: {results['n_responses']}")
    
    print(f"\n🎯 Prediction Performance:")
    pred = results.get('prediction', {})
    print(f"  MAE: {pred.get('mae', 0):.4f}")
    print(f"  RMSE: {pred.get('rmse', 0):.4f}")
    print(f"  Correlation (pred vs. actual): {pred.get('correlation', 0):.4f}")
    print(f"  Accuracy (discrete): {pred.get('accuracy', 0):.3f}")
    
    if results.get('alignment'):
        alignment = results['alignment']
        print(f"\n🎯 Alignment Metrics (Out-of-Sample):")
        print(f"  ECRB_top1_driver: {alignment.get('ECRB_top1_driver', 0):.3f}")
        print(f"  ECRB_top1_weights: {alignment.get('ECRB_top1_weights', 0):.3f}")
        print(f"  Rank correlation: {alignment.get('rank_corr', 0):.3f}")


def print_results(effort: str, results: dict[str, Any]):
    """Print results for a reasoning effort level."""
    print(f"\n{'='*80}")
    print(f"TRAINING SET RESULTS: {effort.upper()} REASONING EFFORT")
    print(f"{'='*80}")
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 Data Collection:")
    print(f"  Trials: {results['n_trials']}")
    print(f"  Responses: {results['n_responses']}")
    
    print(f"\n⚖️  Attribute Weights:")
    for attr in ['E', 'A', 'S', 'D']:
        weight = results['weights'].get(attr, 0)
        beta = results['beta'].get(attr, 0)
        print(f"  {attr}: {weight:.3f} (β = {beta:.3f})")
    
    # Calculate CV
    weights = np.array([results['weights'].get(a, 0) for a in ['E', 'A', 'S', 'D']])
    cv = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
    print(f"  CV (differentiation): {cv:.3f}")
    
    print(f"\n📈 Choice Patterns:")
    extreme = results['extreme_choices']
    print(f"  All chose A: {extreme['all_a']} ({extreme['all_a']/extreme['total']*100:.1f}%)")
    print(f"  All chose B: {extreme['all_b']} ({extreme['all_b']/extreme['total']*100:.1f}%)")
    print(f"  Mixed: {extreme['mixed']} ({extreme['mixed']/extreme['total']*100:.1f}%)")
    print(f"  Choice variance: {results['choice_variance']:.4f}")
    
    if results.get('alignment'):
        alignment = results['alignment']
        print(f"\n🎯 Alignment Metrics:")
        print(f"  ECRB_top1_driver: {alignment.get('ECRB_top1_driver', 0):.3f}")
        print(f"  ECRB_top1_weights: {alignment.get('ECRB_top1_weights', 0):.3f}")
        print(f"  Rank correlation: {alignment.get('rank_corr', 0):.3f}")


def check_icloud_issues(out_dir: Path, model_name: str, effort_levels: list[str]):
    """Check for iCloud placeholder files that might prevent loading data."""
    found_issues = False
    for effort in effort_levels:
        for split in ["train", "test"]:
            target_dir = out_dir / model_name.replace("/", "_") / effort / split
            if target_dir.exists():
                for file in target_dir.glob(".*.icloud"):
                    print(f"  ⚠️  WARNING: Found iCloud placeholder: {file}")
                    print(f"     The script cannot read this file. Please ensure it is downloaded.")
                    found_issues = True
    
    if found_issues:
        print(f"  ⚠️  Note: Missing data might cause the script to re-run completed trials.\n")


def main():
    parser = argparse.ArgumentParser(description="Compare reasoning effort levels")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Dataset directory")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training trials")
    parser.add_argument("--n-test", type=int, default=50, help="Number of held-out test trials (0 = no split)")
    parser.add_argument("--b2-fraction", type=float, default=0.2, help="Fraction of training trials from B2 (rest from B3)")
    parser.add_argument("--replicates", type=int, default=6, help="Replicates per trial (must be even if counterbalancing)")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="results/reasoning_effort_comparison", help="Output directory")
    parser.add_argument("--no-counterbalance", action="store_true", help="Disable counterbalancing (not recommended)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    parser.add_argument("--continue-from-existing", action="store_true", help="Continue by adding more samples to existing dataset")
    args = parser.parse_args()
    
    # Validate replicates if counterbalancing
    if not args.no_counterbalance and args.replicates % 2 != 0:
        print(f"❌ Error: --replicates must be even when counterbalancing is enabled (got {args.replicates})")
        print("   Either use an even number or add --no-counterbalance flag")
        sys.exit(1)
    
    print("="*80)
    print("REASONING EFFORT COMPARISON")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Training samples: {args.n_train} ({int(args.n_train * args.b2_fraction)} from B2, {args.n_train - int(args.n_train * args.b2_fraction)} from B3)")
    if args.n_test > 0:
        print(f"Test samples: {args.n_test} (held-out from B3)")
    else:
        print(f"Test samples: None (no train/test split)")
    print(f"Replicates: {args.replicates} per trial")
    if not args.no_counterbalance:
        print(f"Counterbalancing: ✅ ENABLED ({args.replicates//2} original + {args.replicates//2} reversed)")
    else:
        print(f"Counterbalancing: ❌ DISABLED")
    print(f"Dataset: {args.dataset}")
    
    # Initialize output directory and effort levels
    out_dir = ensure_dir(args.out)
    effort_levels = ["minimal", "low", "medium", "high"]
    
    # Check for iCloud issues
    check_icloud_issues(out_dir, args.model, effort_levels)
    
    # Sample trials
    print(f"\n📦 Sampling trials...")
    dataset_dir = Path(args.dataset)
    
    # NOTE: We do NOT exclude existing trial IDs when sampling.
    # We rely on the seed to deterministically reproduce the same sequence of trials.
    # We will then filter out completed trials in the loop.
    # Passing exclude_trial_ids causes the sampler to look for NEW trials beyond the
    # requested N, which can lead to running out of data in small datasets (like B3).
    
    train_specs, test_specs = sample_b3_trials(
        dataset_dir, 
        n_samples=args.n_train, 
        seed=args.seed,
        n_test=args.n_test,
        b2_fraction=args.b2_fraction,
        exclude_trial_ids=None
    )
    print(f"✅ Sampled {len(train_specs)} training trials")
    if test_specs:
        print(f"✅ Sampled {len(test_specs)} test trials")
    
    # Load trial features
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    
    # Get trial IDs from specs (these are the newly sampled trials)
    # In continuation mode, additional trials will be merged with existing ones during analysis
    train_trial_ids = [spec.trial_id for spec in train_specs]
    test_trial_ids = [spec.trial_id for spec in test_specs]
    train_trials_df = trials_df[trials_df["trial_id"].isin(train_trial_ids)]
    test_trials_df = trials_df[trials_df["trial_id"].isin(test_trial_ids)] if test_specs else pd.DataFrame()
    
    # Storage for results
    all_train_results = {}
    all_test_results = {}
    
    # Run each effort level
    for effort in effort_levels:
        print(f"\n{'='*80}")
        print(f"CHECKING: {effort.upper()} REASONING EFFORT")
        print(f"{'='*80}")
        
        # Check completion status
        status = check_completion_status(out_dir, args.model, effort, len(train_specs), len(test_specs))
        
        # Determine paths for this effort level
        train_resp_path, train_res_path = get_paths(out_dir, args.model, effort, "train")
        test_resp_path, test_res_path = get_paths(out_dir, args.model, effort, "test")
        
        # Skip completed work unless forcing or continuing
        # CRITICAL FIX: Only skip entire effort level if EVERYTHING is complete.
        # If we are in continuation mode, we almost never want to skip the entire block here
        # because we might need to add test samples even if training is done.
        if not args.force and not args.continue_from_existing:
            if status["train_responses_complete"] and status["train_analysis_complete"]:
                if args.n_test == 0 or (status["test_responses_complete"] and status["test_analysis_complete"]):
                    print(f"✅ {effort.upper()} is complete. Skipping (use --force to rerun or --continue-from-existing to add samples).")
                    train_results = json.load(train_res_path.open())
                    train_results["model"] = None  # Not saved, set to None
                    all_train_results[effort] = train_results
                    if args.n_test > 0:
                        all_test_results[effort] = json.load(test_res_path.open())
                    continue
        
        print(f"\n{'='*80}")
        print(f"RUNNING: {effort.upper()} REASONING EFFORT")
        print(f"{'='*80}")
        
        # Initialize backend
        backend = OpenAIBackend(
            model=args.model,
            reasoning_effort=effort,
            temperature=1.0,
            max_tokens=4000
        )
        
        # TRAINING PHASE
        
        # In continuation mode, we always want to add new samples
        # In regular mode, we check completion status
        if not args.force and not args.continue_from_existing and status["train_responses_complete"]:
            print(f"\n🔵 Training responses complete. Loading existing data...")
            train_responses, _ = load_existing_responses(train_resp_path)
        else:
            # Load existing partial results if any
            existing_train, completed_train_ids = load_existing_responses(train_resp_path)
            train_responses = existing_train.copy()
            
            if args.continue_from_existing and existing_train:
                print(f"  🔄 Continuation mode: Found {len(existing_train)} existing training responses")
                
                # In continuation mode, check if we already have enough samples
                if len(existing_train) >= args.n_train:
                    print(f"  ✅ Already have {len(existing_train)} training samples (target: {args.n_train})")
                    print(f"  ℹ️  Skipping training phase (already exceeded target)")
                    remaining_train = []
                else:
                    # Filter to incomplete trials
                    remaining_train = [s for s in train_specs if s.trial_id not in completed_train_ids]
                    
                    # Limit to only what we need to reach the target
                    n_needed = args.n_train - len(existing_train)
                    if len(remaining_train) > n_needed:
                        print(f"  📊 Have {len(existing_train)}, need {args.n_train}, limiting to {n_needed} new trials")
                        remaining_train = remaining_train[:n_needed]
            else:
                # Filter to incomplete trials
                remaining_train = [s for s in train_specs if s.trial_id not in completed_train_ids]
            
            if remaining_train:
                if args.continue_from_existing:
                    print(f"\n🔵 Adding {len(remaining_train)} NEW training trials (currently have {len(existing_train)}, total will be {len(existing_train) + len(remaining_train)})...")
                else:
                    print(f"\n🔵 Running {len(remaining_train)} remaining TRAINING trials (of {len(train_specs)} total)...")
                
                for spec in tqdm(remaining_train, desc=f"{effort} effort (train)"):
                    try:
                        result = run_trial_streamlined(
                            spec,
                            backend,
                            S=args.replicates,
                            seed=args.seed,
                            counterbalance=not args.no_counterbalance,
                        )
                        train_responses.append(result)
                        # Save incrementally after each trial
                        save_response_incremental(result, train_resp_path)
                    except Exception as e:
                        print(f"\n❌ Error on trial {spec.trial_id}: {e}")
                        continue
                
                print(f"✅ Saved training responses to {train_resp_path}")
            else:
                if args.continue_from_existing:
                    print(f"✅ Training target already met or exceeded ({len(train_responses)} >= {args.n_train})")
                else:
                    print(f"✅ All requested training trials complete (have {len(train_responses)} responses)")
        
        # Analyze training (or load if already done)
        n_train_actual = len(train_responses)
        
        # Determine result file path with suffix if continuing
        if args.continue_from_existing:
            # Inject suffix before extension
            stem = train_res_path.stem
            train_results_path_actual = train_res_path.parent / f"{stem}_{n_train_actual}.json"
        else:
            train_results_path_actual = train_res_path
        
        # Check if we need to re-analyze
        need_analysis = True
        if train_results_path_actual.exists() and not args.force:
            print(f"\n📊 Found existing training analysis at {train_results_path_actual.name}")
            try:
                train_results_save = json.loads(train_results_path_actual.read_text())
                if train_results_save.get("n_train_samples") == n_train_actual:
                    print(f"  ✅ Sample count matches ({n_train_actual}), loading existing analysis...")
                    # We still need the model for test predictions, so re-analyze anyway
                    # (model is not saved in JSON)
                    need_analysis = True  # Always analyze to get model
                else:
                    print(f"  🔄 Sample count changed, re-analyzing...")
            except Exception as e:
                print(f"  ⚠️  Error reading existing results: {e}")
        
        if need_analysis:
            print(f"\n🔬 Analyzing training results...")
            # Get all trial IDs from responses (handles both continuation and regular mode)
            all_train_trial_ids = [resp["trial_id"] for resp in train_responses]
            train_trials_df_for_analysis = trials_df[trials_df["trial_id"].isin(all_train_trial_ids)]
            train_results = analyze_results(train_responses, train_trials_df_for_analysis)
            all_train_results[effort] = train_results
            
            # Print train results
            print_results(effort, train_results)
            
            # Remove model before saving (not JSON serializable)
            train_results_save = {k: v for k, v in train_results.items() if k != "model"}
            train_results_save["n_train_samples"] = n_train_actual
            write_json(train_results_save, train_results_path_actual)
            print(f"✅ Saved training analysis to {train_results_path_actual}")
        else:
            # Still need to do analysis to get model for test predictions
            print(f"\n🔬 Analyzing training results to generate model...")
            all_train_trial_ids = [resp["trial_id"] for resp in train_responses]
            train_trials_df_for_analysis = trials_df[trials_df["trial_id"].isin(all_train_trial_ids)]
            train_results = analyze_results(train_responses, train_trials_df_for_analysis)
            all_train_results[effort] = train_results
        
        # TEST PHASE
        print(f"DEBUG: Checking Test Phase prerequisites: test_specs={len(test_specs)}, has_model={'model' in train_results and train_results['model'] is not None}")
        if test_specs and train_results.get("model"):
            
            # In continuation mode, we always want to add new samples
            # In regular mode, we check completion status
            if not args.force and not args.continue_from_existing and status["test_responses_complete"]:
                print(f"\n🟢 Test responses complete. Loading existing data...")
                test_responses, _ = load_existing_responses(test_resp_path)
            else:
                # Load existing partial results if any
                existing_test, completed_test_ids = load_existing_responses(test_resp_path)
                test_responses = existing_test.copy()
                
                if args.continue_from_existing and existing_test:
                    print(f"  🔄 Continuation mode: Found {len(existing_test)} existing test responses")
                    
                    # In continuation mode, check if we already have enough samples
                    if len(existing_test) >= args.n_test:
                        print(f"  ✅ Already have {len(existing_test)} test samples (target: {args.n_test})")
                        print(f"  ℹ️  Skipping test phase (already exceeded target)")
                        remaining_test = []
                    else:
                        # Filter to incomplete trials
                        remaining_test = [s for s in test_specs if s.trial_id not in completed_test_ids]
                        
                        # Limit to only what we need to reach the target
                        n_needed = args.n_test - len(existing_test)
                        if len(remaining_test) > n_needed:
                            print(f"  📊 Have {len(existing_test)}, need {args.n_test}, limiting to {n_needed} new trials")
                            remaining_test = remaining_test[:n_needed]
                else:
                    # Filter to incomplete trials
                    remaining_test = [s for s in test_specs if s.trial_id not in completed_test_ids]
                
                if remaining_test:
                    if args.continue_from_existing:
                        print(f"\n🟢 Adding {len(remaining_test)} NEW test trials (currently have {len(existing_test)}, total will be {len(existing_test) + len(remaining_test)})...")
                    else:
                        print(f"\n🟢 Running {len(remaining_test)} remaining TEST trials (of {len(test_specs)} total)...")
                    
                    for spec in tqdm(remaining_test, desc=f"{effort} effort (test)"):
                        try:
                            result = run_trial_streamlined(
                                spec,
                                backend,
                                S=args.replicates,
                                seed=args.seed,
                                counterbalance=not args.no_counterbalance,
                            )
                            test_responses.append(result)
                            # Save incrementally after each trial
                            save_response_incremental(result, test_resp_path)
                        except Exception as e:
                            print(f"\n❌ Error on trial {spec.trial_id}: {e}")
                            continue
                    
                    print(f"✅ Saved test responses to {test_resp_path}")
                else:
                    if args.continue_from_existing:
                        print(f"✅ Test target already met or exceeded ({len(test_responses)} >= {args.n_test})")
                    else:
                        print(f"✅ All requested test trials complete (have {len(test_responses)} responses)")
            
            # Analyze test
            n_test_actual = len(test_responses)
            
            # Determine result file path with suffix if continuing
            if args.continue_from_existing:
                stem = test_res_path.stem
                test_results_path_actual = test_res_path.parent / f"{stem}_{n_test_actual}.json"
            else:
                test_results_path_actual = test_res_path
            
            # Check if we need to re-analyze
            need_test_analysis = True
            if test_results_path_actual.exists() and not args.force:
                print(f"\n📊 Found existing test analysis at {test_results_path_actual.name}")
                try:
                    existing_test_results = json.loads(test_results_path_actual.read_text())
                    if existing_test_results.get("n_test_samples") == n_test_actual:
                        # Check if training sample count also matches
                        if existing_test_results.get("n_train_samples_used") == n_train_actual:
                            print(f"  ✅ Sample counts match (test={n_test_actual}, train={n_train_actual}), loading existing analysis...")
                            test_results = existing_test_results
                            all_test_results[effort] = test_results
                            need_test_analysis = False
                        else:
                            print(f"  🔄 Training data changed, re-analyzing test set...")
                    else:
                        print(f"  🔄 Test sample count changed, re-analyzing...")
                except Exception as e:
                    print(f"  ⚠️  Error reading existing results: {e}")
            
            if need_test_analysis:
                print(f"\n🔬 Analyzing test results...")
                # Get all trial IDs from responses (handles both continuation and regular mode)
                all_test_trial_ids = [resp["trial_id"] for resp in test_responses]
                test_trials_df_for_analysis = trials_df[trials_df["trial_id"].isin(all_test_trial_ids)]
                test_results = analyze_test_results(
                    test_responses, 
                    test_trials_df_for_analysis, 
                    train_results["model"],
                    split_name="test"
                )
                all_test_results[effort] = test_results
                
                # Print test results
                print_test_results(effort, test_results)
                
                # Save test results with sample count in filename
                test_results["n_test_samples"] = n_test_actual
                test_results["n_train_samples_used"] = n_train_actual  # Track which training data was used
                write_json(test_results, test_results_path_actual)
                print(f"✅ Saved test analysis to {test_results_path_actual}")
            else:
                print(f"✅ Using existing test analysis from {test_results_path_actual.name}")
    
    # Final comparison table - TRAINING SET
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE - TRAINING SET")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'Minimal':<12} {'Low':<12} {'Medium':<12} {'High':<12}")
    print("-" * 85)
    
    # Weights
    for attr in ['E', 'A', 'S', 'D']:
        values = [all_train_results[e]['weights'].get(attr, 0) for e in effort_levels]
        print(f"{attr + ' weight':<25} {values[0]:<12.3f} {values[1]:<12.3f} {values[2]:<12.3f} {values[3]:<12.3f}")
    
    # CV
    cvs = []
    for effort in effort_levels:
        weights = np.array([all_train_results[effort]['weights'].get(a, 0) for a in ['E', 'A', 'S', 'D']])
        cv = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
        cvs.append(cv)
    print(f"{'CV (differentiation)':<25} {cvs[0]:<12.3f} {cvs[1]:<12.3f} {cvs[2]:<12.3f} {cvs[3]:<12.3f}")
    
    # Variance
    print(f"{'Choice variance':<25} ", end="")
    for effort in effort_levels:
        var = all_train_results[effort].get('choice_variance', 0)
        print(f"{var:<12.4f} ", end="")
    print()
    
    # Extreme rate
    print(f"{'Extreme choice rate':<25} ", end="")
    for effort in effort_levels:
        extreme = all_train_results[effort]['extreme_choices']
        rate = (extreme['all_a'] + extreme['all_b']) / extreme['total']
        print(f"{rate:<12.3f} ", end="")
    print()
    
    # Alignment
    if all(all_train_results[e].get('alignment') for e in effort_levels):
        for metric in ['ECRB_top1_driver', 'ECRB_top1_weights', 'rank_corr']:
            print(f"{metric + ' (in-sample)':<25} ", end="")
            for effort in effort_levels:
                val = all_train_results[effort]['alignment'].get(metric, 0)
                print(f"{val:<12.3f} ", end="")
            print()
    
    # Test set comparison if available
    if all_test_results:
        print(f"\n{'='*80}")
        print("FINAL COMPARISON TABLE - TEST SET (OUT-OF-SAMPLE)")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<25} {'Minimal':<12} {'Low':<12} {'Medium':<12} {'High':<12}")
        print("-" * 85)
        
        # Prediction metrics
        for metric_name, metric_key in [("MAE", "mae"), ("RMSE", "rmse"), ("Correlation", "correlation"), ("Accuracy", "accuracy")]:
            print(f"{metric_name:<25} ", end="")
            for effort in effort_levels:
                val = all_test_results[effort].get('prediction', {}).get(metric_key, 0)
                if val is not None:
                    print(f"{val:<12.4f} ", end="")
                else:
                    print(f"{'N/A':<12} ", end="")
            print()
        
        print()  # Blank line
        
        # Alignment on test set
        if all(all_test_results[e].get('alignment') for e in effort_levels):
            for metric in ['ECRB_top1_driver', 'ECRB_top1_weights', 'rank_corr']:
                print(f"{metric + ' (out-sample)':<25} ", end="")
                for effort in effort_levels:
                    val = all_test_results[effort]['alignment'].get(metric, 0)
                    print(f"{val:<12.3f} ", end="")
                print()
    
    # Save comparison
    comparison = {
        "effort_levels": effort_levels,
        "train_results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in all_train_results.items()},
        "test_results": all_test_results if all_test_results else None,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "b2_fraction": args.b2_fraction,
        "replicates": args.replicates,
        "counterbalanced": not args.no_counterbalance,
        "model": args.model,
    }
    write_json(comparison, out_dir / "comparison_summary.json")
    
    print(f"\n✅ Complete! Results saved to {out_dir}")
    if not args.no_counterbalance:
        print(f"✅ Counterbalancing enabled: {args.replicates//2} original + {args.replicates//2} reversed per trial")
    if args.n_test > 0:
        print(f"✅ Train/test split: {args.n_train} train + {args.n_test} test trials")


if __name__ == "__main__":
    main()

