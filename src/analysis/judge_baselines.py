from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


ATTRIBUTES = ["E", "A", "S", "D"]
PAIRS = ["EA", "ES", "ED", "AS", "AD", "SD"]


def add_tau_predictions(responses_df: pd.DataFrame, trials_df: pd.DataFrame) -> pd.DataFrame:
    """Add tau-based predictions and drivers using prompt-visible deltas."""
    deltas = trials_df[["trial_id"] + [f"delta_{attr}" for attr in ATTRIBUTES]].copy()
    merged = responses_df.merge(deltas, on="trial_id", how="left")
    for attr in ATTRIBUTES:
        tau_col = f"tau_{attr}"
        delta_col = f"delta_{attr}"
        merged[tau_col] = pd.to_numeric(merged.get(tau_col), errors="coerce").fillna(0.0)
        merged[delta_col] = pd.to_numeric(merged.get(delta_col), errors="coerce").fillna(0.0)
        signed = np.where(merged[delta_col] > 0, merged[tau_col], 0.0)
        signed = np.where(merged[delta_col] < 0, -merged[tau_col], signed)
        merged[f"tau_signed_{attr}"] = signed

    merged["tau_score_A"] = merged[[f"tau_signed_{attr}" for attr in ATTRIBUTES]].sum(axis=1)
    merged["tau_pred_choice"] = np.where(
        merged["tau_score_A"] > 0, "A", np.where(merged["tau_score_A"] < 0, "B", "A")
    )
    merged["tau_driver"] = merged.apply(_argmax_abs_signed, axis=1)
    return merged


def add_pairwise_drivers(responses_df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise driver derived from pairwise winners."""
    df = responses_df.copy()
    df["pair_driver"] = df.apply(_pairwise_driver, axis=1)
    return df


def tau_stability(responses_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-attribute stability across responses for each trial."""
    results: dict[str, dict[str, float]] = {}
    for attr in ATTRIBUTES:
        col = f"tau_{attr}"
        if col not in responses_df:
            continue
        grouped = responses_df.groupby("trial_id")[col].std()
        grouped = grouped.dropna()
        results[attr] = {
            "mean_std": float(grouped.mean()) if not grouped.empty else float("nan"),
            "median_std": float(grouped.median()) if not grouped.empty else float("nan"),
        }
    return results


def behavioral_attribution(
    trials_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    *,
    baseline_manips: Iterable[str] | None = None,
    occlusion_manips: Iterable[str] | None = None,
    group_cols: Iterable[str] | None = None,
    bootstrap: int = 0,
    seed: int = 13,
) -> dict[str, Any]:
    """Estimate behavioral impact of occlusions on P(choose A).

    By default, this computes matched differences between baseline and occlusion
    conditions grouped by (`config_id`, `labelA`). If the dataset contains a
    `base_trial_id` column (e.g. from the occlusion suite generator), this
    function will instead group by `base_trial_id` to enable paired comparisons
    at the per-base-trial level (holding order/paraphrase/seed fixed).
    """
    baseline_manips = list(baseline_manips or ["short_reason"])
    occlusion_manips = list(occlusion_manips or ["occlude_equalize", "occlude_drop", "occlude_swap"])
    group_cols = list(group_cols) if group_cols is not None else []

    trial_cols = ["trial_id", "config_id", "labelA", "manipulation", "attribute_target"]
    if "base_trial_id" in trials_df.columns:
        trial_cols.append("base_trial_id")

    merged = responses_df[responses_df["choice_ok"]].merge(
        trials_df[trial_cols],
        on="trial_id",
        how="left",
        suffixes=("", "_trial"),
    )
    if "manipulation" not in merged and "manipulation_trial" in merged:
        merged["manipulation"] = merged["manipulation_trial"]
    if "config_id" not in merged and "config_id_trial" in merged:
        merged["config_id"] = merged["config_id_trial"]
    if "labelA" not in merged and "labelA_trial" in merged:
        merged["labelA"] = merged["labelA_trial"]
    if "attribute_target" not in merged and "attribute_target_trial" in merged:
        merged["attribute_target"] = merged["attribute_target_trial"]
    if "base_trial_id" not in merged and "base_trial_id_trial" in merged:
        merged["base_trial_id"] = merged["base_trial_id_trial"]
    if merged.empty:
        return {}
    if not group_cols:
        group_cols = ["base_trial_id"] if "base_trial_id" in merged.columns else ["config_id", "labelA"]
    merged["is_A"] = merged["choice"].eq("A")

    baseline = merged[merged["manipulation"].isin(baseline_manips)]
    if baseline.empty and "split_reason" not in baseline_manips:
        baseline = merged[merged["manipulation"] == "split_reason"]
    if baseline.empty:
        return {}

    base_rates = baseline.groupby(group_cols)["is_A"].mean().rename("base_pA")
    results: dict[str, Any] = {}
    rng = np.random.default_rng(seed)

    for manip in occlusion_manips:
        manip_df = merged[merged["manipulation"] == manip]
        if manip_df.empty:
            continue
        results[manip] = {"by_attribute": {}, "normalized": {}}
        for attr in ATTRIBUTES:
            subset = manip_df[manip_df["attribute_target"] == attr]
            if subset.empty:
                continue
            occl_rates = subset.groupby(group_cols)["is_A"].mean().rename("occl_pA")
            aligned = base_rates.to_frame().join(occl_rates, how="inner")
            if aligned.empty:
                continue
            deltas = aligned["occl_pA"] - aligned["base_pA"]
            delta_mean = float(deltas.mean())
            entry = {"delta_pA": delta_mean, "n": int(len(aligned))}
            if bootstrap > 0 and len(deltas) > 1:
                boots = rng.choice(deltas.to_numpy(), size=(bootstrap, len(deltas)), replace=True).mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                entry["ci95"] = [float(lo), float(hi)]
            results[manip]["by_attribute"][attr] = entry

        abs_vals = {
            attr: abs(info["delta_pA"])
            for attr, info in results[manip]["by_attribute"].items()
        }
        denom = sum(abs_vals.values())
        if denom:
            results[manip]["normalized"] = {
                attr: value / denom for attr, value in abs_vals.items()
            }

    return results


def _argmax_abs_signed(row: pd.Series) -> str:
    values = [abs(row.get(f"tau_signed_{attr}", 0.0)) for attr in ATTRIBUTES]
    max_val = max(values) if values else 0.0
    for attr, value in zip(ATTRIBUTES, values):
        if value == max_val:
            return attr
    return ATTRIBUTES[0]


def _pairwise_driver(row: pd.Series) -> str | None:
    wins = {attr: 0 for attr in ATTRIBUTES}
    any_pair = False
    for pair in PAIRS:
        winner = row.get(f"pair_{pair}")
        if winner in wins:
            wins[winner] += 1
            any_pair = True
    if not any_pair:
        return None
    max_val = max(wins.values()) if wins else 0
    for attr in ATTRIBUTES:
        if wins[attr] == max_val:
            return attr
    return ATTRIBUTES[0]
