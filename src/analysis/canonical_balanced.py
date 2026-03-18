from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import aggregate_choices, load_responses, prepare_stageA_data
from .judge_baselines import add_tau_predictions
from .stageA import build_design_matrix, per_trial_contributions


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/v4_matchedsuite_drugs_20260208"
BALANCED_SPLIT_ROOT = ARTIFACT_ROOT / "splits/holdout80_canonical_balanced"
BALANCED_STAGE_ROOT = ARTIFACT_ROOT / "stage42_holdout80_canonical_balanced"

MODEL_MAP: dict[str, tuple[str, str]] = {
    "mini_min": ("GPT-5-mini", "minimal"),
    "mini_low": ("GPT-5-mini", "low"),
    "nano_min": ("GPT-5-nano", "minimal"),
    "nano_low": ("GPT-5-nano", "low"),
    "haiku45_min": ("Claude Haiku 4.5", "minimal"),
    "haiku45_low": ("Claude Haiku 4.5", "low"),
    "qwen35_14b_minimal": ("Qwen3.5-14B", "minimal"),
    "qwen35_14b_low": ("Qwen3.5-14B", "low"),
    "ministral3_14b_minimal": ("Ministral-3-14B", "minimal"),
    "ministral3_14b_low": ("Ministral-3-14B", "low"),
}

MAIN_MODEL_TAGS = [
    "mini_min",
    "mini_low",
    "nano_min",
    "nano_low",
    "haiku45_min",
    "haiku45_low",
    "qwen35_14b_minimal",
    "qwen35_14b_low",
    "ministral3_14b_minimal",
    "ministral3_14b_low",
]

NON_PLACEBO_THEMES = ["drugs", "policy", "software"]
PLACEBO_THEMES = ["placebo_packaging", "placebo_label_border"]


@dataclass
class PrefitStageAModel:
    params: pd.Series
    feature_info: dict[str, Any]


def load_stagea_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_prefit_stagea_model(path: str | Path) -> tuple[PrefitStageAModel, dict[str, Any]]:
    summary = load_stagea_summary(path)
    params = pd.Series(summary.get("model_params", {}), dtype=float)
    if params.empty:
        raise ValueError(f"Stage A summary missing model_params: {path}")
    model = PrefitStageAModel(
        params=params,
        feature_info=summary.get("feature_info", {}),
    )
    return model, summary


def resolve_run_dir(prefix: str | Path) -> Path | None:
    prefix_path = Path(prefix)
    if prefix_path.is_dir():
        return prefix_path
    matches = sorted(prefix_path.parent.glob(f"{prefix_path.name}__*"))
    if matches:
        def sort_key(path: Path) -> tuple[int, str]:
            batch_rank = 1 if path.name.endswith("_batch") else 0
            return (batch_rank, path.name)
        matches.sort(key=sort_key)
        return matches[0]
    return None


def balanced_dataset_dir(
    theme: str,
    split: str,
    *,
    artifact_root: str | Path = ARTIFACT_ROOT,
    stage_root: str | Path = BALANCED_STAGE_ROOT,
) -> Path:
    artifact_root = Path(artifact_root)
    stage_root = Path(stage_root)
    if theme == "drugs":
        return artifact_root / f"splits/holdout80_canonical_balanced/{split}"
    return stage_root / f"splits/{theme}/{split}"


def balanced_train_stagea_summary(
    theme: str,
    model_tag: str,
    *,
    stage_root: str | Path = BALANCED_STAGE_ROOT,
) -> Path:
    stage_root = Path(stage_root)
    theme_key = "holdout80" if theme == "drugs" else theme
    return stage_root / f"results/stage_A_{theme_key}_balanced_train_{model_tag}/stageA_summary.json"


def balanced_test_run_prefix(
    theme: str,
    model_tag: str,
    *,
    stage_root: str | Path = BALANCED_STAGE_ROOT,
) -> Path:
    stage_root = Path(stage_root)
    if theme == "drugs":
        return stage_root / f"runs/holdout80/test/{model_tag}_tau"
    return stage_root / f"runs/holdout/{theme}/test/{model_tag}_tau"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _safe_rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def build_balanced_eval_frame_from_model(
    *,
    trials_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    model: Any,
    include_interactions: bool = False,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choice_agg = aggregate_choices(responses_df)
    stagea_df = prepare_stageA_data(trials_df, choice_agg)
    design = build_design_matrix(
        stagea_df,
        include_interactions=include_interactions,
        exclude_b1=True,
    )
    feature_columns = feature_columns or list(getattr(model, "params").index) or list(design.X.columns)
    X = design.X.reindex(columns=feature_columns, fill_value=0.0)
    linear_score = X.to_numpy(dtype=float) @ model.params.reindex(feature_columns, fill_value=0.0).to_numpy(dtype=float)

    trial_preds = stagea_df[["trial_id"]].copy()
    trial_preds["linear_model_score_A"] = linear_score
    trial_preds["linear_model_prob_A"] = _sigmoid(linear_score)
    trial_preds["linear_model_pred_choice"] = np.where(trial_preds["linear_model_score_A"] > 0, "A", "B")

    # Keep trial ids from the prepared Stage A frame rather than relying on the
    # DataFrame index, which may be integer-typed after parquet loads/resets.
    contributions = per_trial_contributions(stagea_df, model).reset_index(drop=True)
    contribution_preds = stagea_df[["trial_id"]].copy()
    contribution_preds["trial_id"] = contribution_preds["trial_id"].astype(str)
    contribution_preds = pd.concat(
        [contribution_preds.reset_index(drop=True), contributions[["driver_A", "driver_B"]].reset_index(drop=True)],
        axis=1,
    )
    trial_preds["trial_id"] = trial_preds["trial_id"].astype(str)
    trial_preds = trial_preds.merge(
        contribution_preds,
        on="trial_id",
        how="left",
    )

    eval_df = add_tau_predictions(responses_df, trials_df)
    eval_df["trial_id"] = eval_df["trial_id"].astype(str)
    eval_df = eval_df.merge(trial_preds, on="trial_id", how="left")
    eval_df["linear_model_factor"] = np.where(
        eval_df["choice"] == "A",
        eval_df["driver_A"],
        np.where(eval_df["choice"] == "B", eval_df["driver_B"], None),
    )
    return eval_df, stagea_df


def build_balanced_eval_frame(
    *,
    dataset_dir: str | Path,
    responses_path: str | Path,
    stagea_summary_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    responses_df = load_responses(responses_path)
    model, stagea_summary = load_prefit_stagea_model(stagea_summary_path)
    eval_df, _ = build_balanced_eval_frame_from_model(
        trials_df=trials_df,
        responses_df=responses_df,
        model=model,
        include_interactions=bool(stagea_summary.get("include_interactions", False)),
        feature_columns=stagea_summary.get("feature_columns"),
    )
    return eval_df, trials_df, stagea_summary


def compute_main_metrics(eval_df: pd.DataFrame) -> dict[str, float | int | None]:
    linear_choice = eval_df[eval_df["choice_ok"]].copy()
    judge_choice = eval_df[eval_df["choice_ok"] & eval_df["tau_ok"]].copy()
    linear_factor = eval_df[eval_df["premise_ok"] & eval_df["choice_ok"]].copy()
    judge_factor = eval_df[eval_df["premise_ok"] & eval_df["tau_ok"]].copy()

    return {
        "linear_model_predicts_actor_choice": _safe_rate(
            linear_choice["linear_model_pred_choice"] == linear_choice["choice"]
        ),
        "judge_predicts_actor_choice": _safe_rate(
            judge_choice["tau_pred_choice"] == judge_choice["choice"]
        ),
        "linear_model_factor_matches_stated_factor": _safe_rate(
            linear_factor["linear_model_factor"] == linear_factor["premise_attr"]
        ),
        "judge_factor_matches_stated_factor": _safe_rate(
            judge_factor["tau_driver"] == judge_factor["premise_attr"]
        ),
        "n_choice_ok": int(len(linear_choice)),
        "n_judge_choice_ok": int(len(judge_choice)),
        "n_premise_ok": int(len(linear_factor)),
        "n_judge_premise_ok": int(len(judge_factor)),
    }


def compute_placebo_metrics(eval_df: pd.DataFrame, *, placebo_attr: str = "D") -> dict[str, float | int | None]:
    premise_rows = eval_df[eval_df["premise_ok"]].copy()
    choice_rows = eval_df[eval_df["choice_ok"]].copy()
    judge_rows = eval_df[eval_df["tau_ok"]].copy()
    return {
        "actor_states_placebo_as_key_factor": _safe_rate(premise_rows["premise_attr"] == placebo_attr),
        "linear_model_factor_is_placebo": _safe_rate(choice_rows["linear_model_factor"] == placebo_attr),
        "judge_factor_is_placebo": _safe_rate(judge_rows["tau_driver"] == placebo_attr),
        "n_premise_ok": int(len(premise_rows)),
        "n_choice_ok": int(len(choice_rows)),
        "n_tau_ok": int(len(judge_rows)),
    }
