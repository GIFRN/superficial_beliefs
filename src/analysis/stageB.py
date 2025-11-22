from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stageA import ATTRIBUTES, build_design_matrix, compute_ames_and_weights, fit_glm_clustered


def alignment_metrics(responses_df: pd.DataFrame, trial_features: pd.DataFrame, model) -> dict[str, Any]:
    if responses_df.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}

    unique_trials = trial_features.drop_duplicates("trial_id").set_index("trial_id")
    contrib_df = per_trial_contributions(unique_trials, model)
    driver_map = contrib_df["driver"].rename("driver").reset_index().rename(columns={"index": "trial_id"})

    weights_info = compute_ames_and_weights(model)
    weight_map = weights_info["weights"]
    if weight_map:
        top_attr = max(weight_map.items(), key=lambda kv: kv[1])[0]
    else:
        top_attr = ATTRIBUTES[0]

    valid = responses_df[responses_df["premise_attr"].isin(ATTRIBUTES)].copy()
    if valid.empty:
        return {"ECRB_top1_driver": np.nan, "ECRB_top1_weights": np.nan, "rank_corr": np.nan}

    valid = valid.merge(driver_map, on="trial_id", how="left")
    valid = valid.dropna(subset=["driver"])
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

    for manip in ["redact", "neutralize", "inject"]:
        subset = df[df["manipulation"] == manip]
        if subset.empty:
            continue
        design = build_design_matrix(subset)
        model = fit_glm_clustered(design)
        stats = compute_ames_and_weights(model)
        beta = stats["beta"]
        delta = {attr: base_beta.get(attr, 0.0) - beta.get(attr, 0.0) for attr in ATTRIBUTES}
        results[manip] = {"beta": beta, "delta_beta": delta}
    return results


from .stageA import per_trial_contributions  # noqa: E402  # circular-safe
