from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stageA import ATTRIBUTES, build_design_matrix, compute_ames_and_weights, fit_glm_clustered


def alignment_metrics(
    responses_df: pd.DataFrame,
    trial_features: pd.DataFrame,
    model,
    *,
    refit_baseline: bool = True,
) -> dict[str, Any]:
    if responses_df.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}

    if refit_baseline:
        # Fit baseline model from short_reason only for driver computation.
        # This ensures drivers reflect baseline implicit preferences, not preferences under intervention.
        baseline_manips = ["short_reason", "split_reason"]
        baseline_trials = trial_features[trial_features["manipulation"].isin(baseline_manips)]
        if baseline_trials.empty:
            # Fallback to all trials if no baseline manipulations present
            baseline_trials = trial_features

        baseline_unique = baseline_trials.drop_duplicates("trial_id").set_index("trial_id")
        if len(baseline_unique) > 0:
            baseline_design = build_design_matrix(baseline_unique.reset_index())
            baseline_model = fit_glm_clustered(baseline_design)
        else:
            # Fallback to passed model if baseline fitting fails
            baseline_model = model
    else:
        # Use the provided prefit model directly (e.g., trained on a held-out split).
        baseline_model = model
    
    # Compute contributions using baseline model for all trials
    unique_trials = trial_features.drop_duplicates("trial_id").set_index("trial_id")
    contrib_df = per_trial_contributions(unique_trials, baseline_model)
    
    # Extract both driver_A and driver_B for choice-conditional alignment
    driver_map = contrib_df[["driver_A", "driver_B"]].reset_index().rename(columns={"index": "trial_id"})

    # Use baseline model for weight computation as well
    weights_info = compute_ames_and_weights(baseline_model)
    weight_map = weights_info["weights"]
    if weight_map:
        top_attr = max(weight_map.items(), key=lambda kv: kv[1])[0]
    else:
        top_attr = ATTRIBUTES[0]

    valid = responses_df[responses_df["premise_attr"].isin(ATTRIBUTES)].copy()
    if valid.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}

    valid = valid.merge(driver_map, on="trial_id", how="left")
    valid = valid.dropna(subset=["driver_A", "driver_B"])
    
    # Select driver conditional on the model's actual choice:
    # - If choice == A: driver is argmax(C_j) (strongest evidence for A)
    # - If choice == B: driver is argmin(C_j) (strongest evidence for B)
    valid["driver"] = np.where(valid["choice"] == "A", valid["driver_A"], valid["driver_B"])
    
    top1_driver = (valid["premise_attr"] == valid["driver"]).mean()
    top1_weights = (valid["premise_attr"] == top_attr).mean()

    counts = valid["premise_attr"].value_counts()
    premise_series = pd.Series({attr: counts.get(attr, 0) for attr in ATTRIBUTES})
    weight_series = pd.Series({attr: weight_map.get(attr, 0.0) for attr in ATTRIBUTES})
    if premise_series.sum() == 0 or weight_series.sum() == 0:
        rank_corr = np.nan
    else:
        rank_corr = premise_series.rank().corr(weight_series.rank(), method="spearman")

    return {
        "ECRB_top1_driver": float(top1_driver),
        "ECRB_top1_weights": float(top1_weights),
        "rank_corr": float(rank_corr) if not np.isnan(rank_corr) else np.nan,
    }


def probe_deltas_and_pivots(df: pd.DataFrame) -> dict[str, Any]:
    baseline = df[df["manipulation"] == "short_reason"]
    if baseline.empty:
        baseline = df[df["manipulation"] == "split_reason"]
    if baseline.empty:
        return {}
    base_design = build_design_matrix(baseline)
    base_model = fit_glm_clustered(base_design)
    base_stats = compute_ames_and_weights(base_model)
    base_beta = base_stats["beta"]

    results: dict[str, Any] = {
        "baseline": {
            "beta": base_beta,
            "weights": base_stats["weights"],
        }
    }

    # Estimate probe effects conditioned on target attribute
    # This allows clean difference-in-differences: "redacting E drops β_E more than others"
    for manip in ["redact", "neutralize", "inject"]:
        manip_subset = df[df["manipulation"] == manip]
        if manip_subset.empty:
            continue
        
        # Get unique target attributes for this manipulation
        targets = manip_subset["attribute_target"].dropna().unique()
        
        if len(targets) == 0:
            # No target attribute recorded - fall back to pooled analysis
            design = build_design_matrix(manip_subset)
            model = fit_glm_clustered(design)
            stats = compute_ames_and_weights(model)
            beta = stats["beta"]
            delta = {attr: base_beta.get(attr, 0.0) - beta.get(attr, 0.0) for attr in ATTRIBUTES}
            results[manip] = {"beta": beta, "delta_beta": delta}
        else:
            # Fit separate model for each target attribute
            results[manip] = {"by_target": {}}
            for target in targets:
                target_subset = manip_subset[manip_subset["attribute_target"] == target]
                if target_subset.empty or len(target_subset) < 2:
                    continue
                try:
                    design = build_design_matrix(target_subset)
                    model = fit_glm_clustered(design)
                    stats = compute_ames_and_weights(model)
                    beta = stats["beta"]
                    delta = {attr: base_beta.get(attr, 0.0) - beta.get(attr, 0.0) for attr in ATTRIBUTES}
                    
                    # Compute targeted effect: how much did the TARGET attribute's β change
                    # relative to other attributes (difference-in-differences)
                    target_delta = delta.get(target, 0.0)
                    other_deltas = [delta.get(a, 0.0) for a in ATTRIBUTES if a != target]
                    avg_other_delta = np.mean(other_deltas) if other_deltas else 0.0
                    diff_in_diff = target_delta - avg_other_delta
                    
                    results[manip]["by_target"][target] = {
                        "beta": beta,
                        "delta_beta": delta,
                        "target_delta": target_delta,
                        "avg_other_delta": avg_other_delta,
                        "diff_in_diff": diff_in_diff,
                        "n_trials": len(target_subset),
                    }
                except (ValueError, np.linalg.LinAlgError):
                    # Skip if model fitting fails (e.g., singular matrix)
                    continue
            
            # Also compute pooled result for backwards compatibility
            try:
                design = build_design_matrix(manip_subset)
                model = fit_glm_clustered(design)
                stats = compute_ames_and_weights(model)
                beta = stats["beta"]
                delta = {attr: base_beta.get(attr, 0.0) - beta.get(attr, 0.0) for attr in ATTRIBUTES}
                results[manip]["pooled"] = {"beta": beta, "delta_beta": delta}
            except (ValueError, np.linalg.LinAlgError):
                pass
    
    return results


from .stageA import per_trial_contributions  # noqa: E402  # circular-safe
