from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .behavioral_models import (
    ATTRIBUTES,
    BehavioralModel,
    load_behavioral_model_summary,
    mean_binomial_nll,
)
from .features import aggregate_choices, prepare_stageA_data
from .judge_baselines import add_tau_predictions

PLACEBO_ATTRIBUTE = "D"


def _safe_match(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left == right).fillna(False).astype(bool)


def _safe_bool(values: pd.Series | np.ndarray | list[object]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(bool)
    return pd.Series(values).fillna(False).astype(bool)


def build_test_row_predictions(
    *,
    trials_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    model: BehavioralModel,
    condition_keys: Mapping[str, Any],
) -> pd.DataFrame:
    choice_agg = aggregate_choices(responses_df)
    stagea_df = prepare_stageA_data(trials_df, choice_agg)
    eta = model.linear_predictor(stagea_df, exclude_b1=False)
    probs = model.predict_proba(stagea_df, exclude_b1=False)
    choices = model.predict_choice(stagea_df, exclude_b1=False)
    contribs = model.attribute_contributions(stagea_df, exclude_b1=False)

    filtered = stagea_df.loc[eta.index].copy()
    row_predictions = pd.DataFrame(index=filtered.index)
    for key, value in condition_keys.items():
        row_predictions[key] = value
    row_predictions["behavioral_model"] = model.behavioral_model
    row_predictions["behavioral_model_name"] = model.model_name
    if "family_id" in filtered:
        row_predictions["family_id"] = filtered["family_id"].astype(str)
    elif "config_id" in filtered:
        row_predictions["family_id"] = filtered["config_id"].astype(str)
    else:
        row_predictions["family_id"] = ""
    row_predictions["trial_id"] = filtered["trial_id"].astype(str)
    row_predictions["y_A"] = filtered["successes"].astype(int)
    row_predictions["n_trials"] = filtered["trials"].astype(int)
    for attr in ATTRIBUTES:
        row_predictions[f"delta_{attr}"] = filtered[f"delta_{attr}"]
    row_predictions["eta"] = eta.to_numpy(dtype=float)
    row_predictions["p_choose_A"] = probs.to_numpy(dtype=float)
    row_predictions["predicted_choice"] = choices.to_numpy(dtype=object)
    for attr in ATTRIBUTES:
        row_predictions[f"contrib_{attr}"] = contribs[f"contrib_{attr}"].to_numpy(dtype=float)
    row_predictions["driver_A"] = contribs["driver_A"].to_numpy(dtype=object)
    row_predictions["driver_B"] = contribs["driver_B"].to_numpy(dtype=object)
    return row_predictions.reset_index(drop=True)


def build_per_draw_driver_table(
    *,
    trials_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    row_predictions: Mapping[str, pd.DataFrame],
    models: Mapping[str, BehavioralModel],
    condition_keys: Mapping[str, Any],
) -> pd.DataFrame:
    trial_meta = trials_df[["trial_id", "family_id"]].copy()
    trial_meta["trial_id"] = trial_meta["trial_id"].astype(str)
    trial_meta["family_id"] = trial_meta["family_id"].astype(str)

    draw_df = add_tau_predictions(responses_df, trials_df)
    draw_df["trial_id"] = draw_df["trial_id"].astype(str)
    draw_df = draw_df.merge(trial_meta, on="trial_id", how="left")

    output = pd.DataFrame(index=draw_df.index)
    for key, value in condition_keys.items():
        output[key] = value
    output["trial_id"] = draw_df["trial_id"]
    output["family_id"] = draw_df["family_id"]
    output["seed"] = draw_df.get("seed")
    output["choice_ok"] = draw_df.get("choice_ok", False)
    output["choice"] = draw_df.get("choice")
    output["actual_choice"] = output["choice"]
    output["premise_ok"] = draw_df.get("premise_ok", False)
    output["premise_attr"] = draw_df.get("premise_attr")
    output["tau_ok"] = draw_df.get("tau_ok", False)
    output["tau_pred_choice"] = draw_df.get("tau_pred_choice")
    output["tau_driver"] = draw_df.get("tau_driver")

    for label, model in models.items():
        prediction_cols = row_predictions[label][
            ["trial_id", "predicted_choice", "eta", "p_choose_A", "driver_A", "driver_B"]
        ].rename(
            columns={
                "predicted_choice": f"{label}_predicted_choice",
                "eta": f"{label}_eta",
                "p_choose_A": f"{label}_p_choose_A",
                "driver_A": f"{label}_driver_A",
                "driver_B": f"{label}_driver_B",
            }
        )
        enriched = draw_df.merge(prediction_cols, on="trial_id", how="left")
        driver_info = model.revealed_driver(enriched, enriched["choice"], exclude_b1=False)
        output[f"{label}_predicted_choice"] = enriched[f"{label}_predicted_choice"]
        output[f"{label}_eta"] = enriched[f"{label}_eta"]
        output[f"{label}_p_choose_A"] = enriched[f"{label}_p_choose_A"]
        output[f"{label}_driver_A"] = enriched[f"{label}_driver_A"]
        output[f"{label}_driver_B"] = enriched[f"{label}_driver_B"]
        output[f"{label}_predicted_side_driver"] = np.where(
            output[f"{label}_predicted_choice"] == "A",
            output[f"{label}_driver_A"],
            np.where(output[f"{label}_predicted_choice"] == "B", output[f"{label}_driver_B"], None),
        )
        output[f"{label}_revealed_driver"] = driver_info["revealed_driver"]
        output[f"{label}_driver_margin"] = driver_info["driver_margin"]
        for attr in ATTRIBUTES:
            output[f"{label}_influence_{attr}"] = driver_info[f"influence_{attr}"]
        output[f"{label}_driver_matches_stated_factor"] = np.where(
            output["choice_ok"] & output["premise_ok"],
            output[f"{label}_revealed_driver"] == output["premise_attr"],
            np.nan,
        )

    labels = list(models.keys())
    for label_a, label_b in combinations(labels, 2):
        output[f"{label_a}_{label_b}_driver_agreement"] = np.where(
            output[f"{label_a}_predicted_side_driver"].notna() & output[f"{label_b}_predicted_side_driver"].notna(),
            output[f"{label_a}_predicted_side_driver"] == output[f"{label_b}_predicted_side_driver"],
            np.nan,
        )
        output[f"{label_a}_{label_b}_actor_conditioned_driver_agreement"] = np.where(
            output["choice_ok"],
            output[f"{label_a}_revealed_driver"] == output[f"{label_b}_revealed_driver"],
            np.nan,
        )
    if "m0" in labels and "m1" in labels:
        output["cross_model_driver_agreement"] = output["m0_m1_driver_agreement"]
        output["cross_model_actor_conditioned_driver_agreement"] = output["m0_m1_actor_conditioned_driver_agreement"]
    return output


def build_condition_comparison(
    *,
    theme: str,
    row_predictions: Mapping[str, pd.DataFrame],
    per_draw_df: pd.DataFrame,
    condition_keys: Mapping[str, Any],
    model_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    m0_summary: Mapping[str, Any] | None = None,
    m1_summary: Mapping[str, Any] | None = None,
    m2_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {**condition_keys}
    labels = list(row_predictions.keys())
    summaries: dict[str, Mapping[str, Any]] = {}
    if model_summaries is not None:
        summaries.update(model_summaries)
    if m0_summary is not None:
        summaries.setdefault("m0", m0_summary)
    if m1_summary is not None:
        summaries.setdefault("m1", m1_summary)
    if m2_summary is not None:
        summaries.setdefault("m2", m2_summary)

    for label in labels:
        rows = row_predictions[label]
        comparison[f"{label}_heldout_nll"] = mean_binomial_nll(
            successes=rows["y_A"],
            weights=rows["n_trials"],
            probs=rows["p_choose_A"],
        )

    n_draws = len(per_draw_df)
    choice_ok = _safe_bool(per_draw_df["choice_ok"])
    premise_ok = _safe_bool(per_draw_df["premise_ok"])
    tau_ok = _safe_bool(per_draw_df["tau_ok"])

    for label in labels:
        predicted_choice = per_draw_df[f"{label}_predicted_choice"]
        predicted_side_driver = per_draw_df[f"{label}_predicted_side_driver"]
        actor_side_driver = per_draw_df[f"{label}_revealed_driver"]

        comparison[f"{label}_heldout_choice_accuracy"] = (
            float((choice_ok & _safe_match(predicted_choice, per_draw_df["choice"])).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_judge_choice_accuracy"] = (
            float((tau_ok & _safe_match(per_draw_df["tau_pred_choice"], per_draw_df["choice"])).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_driver_matches_stated_factor_rate"] = (
            float((premise_ok & _safe_match(predicted_side_driver, per_draw_df["premise_attr"])).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_judge_driver_matches_rate"] = (
            float((tau_ok & _safe_match(per_draw_df["tau_driver"], predicted_side_driver)).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_actor_conditioned_driver_matches_stated_factor_rate"] = (
            float((choice_ok & premise_ok & _safe_match(actor_side_driver, per_draw_df["premise_attr"])).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_placebo_driver_rate"] = (
            float((predicted_side_driver == PLACEBO_ATTRIBUTE).fillna(False).mean())
            if n_draws
            else None
        )
        comparison[f"{label}_actor_conditioned_placebo_driver_rate"] = (
            float((choice_ok & actor_side_driver.eq(PLACEBO_ATTRIBUTE).fillna(False)).mean())
            if n_draws
            else None
        )

    for label_a, label_b in combinations(labels, 2):
        pair_col = f"{label_a}_{label_b}_driver_agreement"
        comparison[pair_col] = float(per_draw_df[pair_col].mean()) if n_draws > 0 else None
        actor_pair_col = f"{label_a}_{label_b}_actor_conditioned_driver_agreement"
        comparison[actor_pair_col] = float(per_draw_df[actor_pair_col].mean()) if n_draws > 0 else None
    if "m0" in labels and "m1" in labels:
        comparison["cross_model_driver_agreement"] = comparison.get("m0_m1_driver_agreement")
        comparison["cross_model_actor_conditioned_driver_agreement"] = comparison.get(
            "m0_m1_actor_conditioned_driver_agreement"
        )

    comparison["n_test_rows"] = int(len(next(iter(row_predictions.values()))))
    comparison["n_draws"] = int(n_draws)
    comparison["n_choice_ok_draws"] = int(choice_ok.sum())
    comparison["n_premise_ok_draws"] = int(premise_ok.sum())
    comparison["n_tau_ok_draws"] = int(tau_ok.sum())
    for label in labels:
        summary = summaries.get(label, {})
        comparison[f"{label}_summary_model_name"] = summary.get("behavioral_model_name", label.upper())
        comparison[f"{label}_selected_ridge_lambda"] = float(summary.get("selected_ridge_lambda", 0.0) or 0.0)
        if summary.get("selected_shrinkage_lambda") is not None:
            comparison[f"{label}_selected_shrinkage_lambda"] = float(summary.get("selected_shrinkage_lambda") or 0.0)
        if summary.get("num_unique_observed_cells") is not None:
            comparison[f"{label}_num_unique_observed_cells"] = int(summary["num_unique_observed_cells"])
    return comparison


def load_models_for_comparison(
    *,
    summary_paths: Mapping[str, str],
) -> tuple[dict[str, BehavioralModel], dict[str, dict[str, Any]]]:
    models: dict[str, BehavioralModel] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for label, path in summary_paths.items():
        model, summary = load_behavioral_model_summary(path)
        models[label] = model
        summaries[label] = summary
    return models, summaries
