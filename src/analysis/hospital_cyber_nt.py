from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.data.themes import ThemeConfig, load_theme_from_yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/hospital_cyber_nt"
THEME_NAME = "hospital_cyber_response"
THEME_PATH = ROOT / "data/themes/hospital_cyber_response.yml"
DATASET_CONFIG_PATH = ROOT / "data/configs/hospital_cyber_response.yml"
ATTRIBUTES = ["T", "D", "M", "C", "A", "P"]
FULL_VARIANTS = (
    "p_at_oa_vs_q_at_oa",
    "q_at_oa_vs_p_at_oa",
    "p_at_ob_vs_q_at_ob",
    "q_at_ob_vs_p_at_ob",
)
DEFAULT_DATASET_SEED = 13
DEFAULT_TRAIN_TARGET = 400
DEFAULT_TEST_TARGET = 100
DEFAULT_REPLICATES = 3
DEFAULT_OPENAI_CONCURRENCY = 8
DEFAULT_QWEN_CONCURRENCY = 1
DEFAULT_RESUME_MODE = "any"


@dataclass(frozen=True)
class HospitalModelSpec:
    tag: str
    family: str
    effort: str
    provider: str
    config_path: Path
    color: str


MODEL_SPECS = [
    HospitalModelSpec(
        tag="mini_min",
        family="GPT-5-mini",
        effort="minimal",
        provider="openai",
        config_path=ROOT / "data/models/mini_min.yml",
        color="#3b6fb6",
    ),
    HospitalModelSpec(
        tag="qwen_min_8030",
        family="Qwen3-14B",
        effort="minimal",
        provider="qwen",
        config_path=ROOT / "data/models/qwen_min_8030.yml",
        color="#c26d2d",
    ),
]
MODEL_BY_TAG = {spec.tag: spec for spec in MODEL_SPECS}


@dataclass
class HospitalBehavioralModel:
    behavioral_model: str
    attributes: list[str]
    params: pd.Series
    feature_columns: list[str]
    fit_mode: str

    @property
    def model_name(self) -> str:
        return self.behavioral_model.upper()

    def linear_predictor(self, df: pd.DataFrame) -> pd.Series:
        X = build_feature_matrix(df, self.attributes, self.behavioral_model)
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        params = self.params.reindex(self.feature_columns, fill_value=0.0).astype(float)
        eta = X.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
        return pd.Series(eta, index=df.index, name="eta")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        eta = self.linear_predictor(df)
        return pd.Series(_sigmoid(eta.to_numpy(dtype=float)), index=df.index, name="p_choose_A")

    def predict_choice(self, df: pd.DataFrame) -> pd.Series:
        eta = self.linear_predictor(df)
        return pd.Series(np.where(eta.to_numpy(dtype=float) > 0.0, "A", "B"), index=df.index, name="predicted_choice")

    def revealed_driver(self, df: pd.DataFrame, chosen_side: pd.Series | Iterable[str] | str) -> pd.DataFrame:
        chosen = coerce_choice_series(chosen_side, df.index)
        eta = self.linear_predictor(df)
        influences = pd.DataFrame(index=df.index)
        for attr in self.attributes:
            neutralized = df.copy()
            delta_col = f"delta_{attr}"
            if delta_col in neutralized.columns:
                neutralized[delta_col] = 0
            eta_neutralized = self.linear_predictor(neutralized)
            influence = np.where(
                chosen.eq("A"),
                eta.to_numpy(dtype=float) - eta_neutralized.to_numpy(dtype=float),
                np.where(
                    chosen.eq("B"),
                    eta_neutralized.to_numpy(dtype=float) - eta.to_numpy(dtype=float),
                    np.nan,
                ),
            )
            influences[f"influence_{attr}"] = influence

        matrix = influences[[f"influence_{attr}" for attr in self.attributes]].to_numpy(dtype=float)
        influences["revealed_driver"] = [canonical_argmax(self.attributes, row) for row in matrix]
        influences["driver_margin"] = [driver_margin(row) for row in matrix]
        return influences


def load_theme() -> ThemeConfig:
    return load_theme_from_yaml(THEME_PATH)


def output_root(path: str | Path | None = None) -> Path:
    return Path(path).resolve() if path else OUTPUT_ROOT


def datasets_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "datasets" / THEME_NAME


def dataset_dir(split: str, base: str | Path | None = None) -> Path:
    return datasets_root(base) / split


def runs_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "runs" / THEME_NAME


def run_prefix(split: str, model_tag: str, kind: str, *, base: str | Path | None = None) -> Path:
    return runs_root(base) / split / f"{model_tag}_{kind}"


def results_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "results"


def reports_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "reports"


def logs_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "logs"


def resolve_run_dir(prefix: str | Path) -> Path | None:
    prefix_path = Path(prefix)
    if prefix_path.is_dir():
        return prefix_path
    matches = sorted(prefix_path.parent.glob(f"{prefix_path.name}__*"))
    return matches[0] if matches else None


def flatten_actor_responses(responses_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(responses_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            for response in payload.get("responses", []):
                steps = {step.get("name"): step for step in response.get("steps", [])}
                choice_step = steps.get("choice")
                premise_step = steps.get("premise")
                choice_parsed = choice_step.get("parsed", {}) if choice_step else {}
                premise_parsed = premise_step.get("parsed", {}) if premise_step else {}
                if premise_step:
                    premise_ok = bool(premise_parsed.get("ok"))
                    premise_attr = premise_parsed.get("attr")
                    premise_text = premise_parsed.get("text")
                    premise_raw = premise_step.get("content", "")
                else:
                    premise_ok = bool(choice_parsed.get("premise_ok", False))
                    premise_attr = choice_parsed.get("attr")
                    premise_text = choice_parsed.get("text", "")
                    premise_raw = choice_step.get("content", "") if choice_step else ""
                rows.append(
                    {
                        "trial_id": str(payload.get("trial_id")),
                        "config_id": str(payload.get("config_id")),
                        "seed": response.get("seed"),
                        "choice_ok": bool(choice_parsed.get("choice_ok", choice_parsed.get("ok", False))),
                        "choice": choice_parsed.get("choice"),
                        "premise_ok": premise_ok,
                        "premise_attr": premise_attr,
                        "premise_text": premise_text,
                        "choice_raw": choice_step.get("content", "") if choice_step else "",
                        "premise_raw": premise_raw,
                    }
                )
    return pd.DataFrame(rows)


def flatten_judge_responses(responses_path: str | Path, attributes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(responses_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            for response in payload.get("responses", []):
                steps_list = response.get("steps", [])
                steps = {step.get("name"): step for step in steps_list}
                choice_step = steps.get("choice")
                premise_step = steps.get("premise")
                choice_parsed = choice_step.get("parsed", {}) if choice_step else {}
                premise_parsed = premise_step.get("parsed", {}) if premise_step else {}

                row: dict[str, Any] = {
                    "trial_id": str(payload.get("trial_id")),
                    "config_id": str(payload.get("config_id")),
                    "seed": response.get("seed"),
                    "choice_ok": bool(choice_parsed.get("choice_ok", choice_parsed.get("ok", False))),
                    "choice": choice_parsed.get("choice"),
                    "premise_ok": bool(premise_parsed.get("ok", choice_parsed.get("premise_ok", False))),
                    "premise_attr": premise_parsed.get("attr", choice_parsed.get("attr")),
                    "premise_text": premise_parsed.get("text", choice_parsed.get("text", "")),
                    "tau_ok": False,
                    "tau_missing": [],
                    "tau_raw": "",
                }
                for attr in attributes:
                    row[f"tau_{attr}"] = np.nan

                for step in steps_list:
                    parsed = step.get("parsed", {})
                    tau = parsed.get("tau")
                    if not isinstance(tau, dict):
                        continue
                    row["tau_ok"] = row["tau_ok"] or bool(parsed.get("ok", False))
                    row["tau_raw"] = step.get("content", "")
                    row["tau_missing"] = list(parsed.get("missing", []))
                    for attr in attributes:
                        if attr in tau:
                            row[f"tau_{attr}"] = float(tau[attr])
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_choice_counts(actor_draws: pd.DataFrame) -> pd.DataFrame:
    valid = actor_draws.loc[actor_draws["choice_ok"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=["trial_id", "successes", "trials"])
    valid["is_A"] = valid["choice"].eq("A")
    return (
        valid.groupby("trial_id", as_index=False)
        .agg(successes=("is_A", "sum"), trials=("is_A", "count"))
        .assign(successes=lambda df: df["successes"].astype(int), trials=lambda df: df["trials"].astype(int))
    )


def prepare_behavioral_frame(trials_df: pd.DataFrame, actor_draws: pd.DataFrame) -> pd.DataFrame:
    choice_counts = aggregate_choice_counts(actor_draws)
    merged = trials_df.merge(choice_counts, on="trial_id", how="left")
    merged["successes"] = merged["successes"].fillna(0).astype(int)
    merged["trials"] = merged["trials"].fillna(0).astype(int)
    return merged


def build_feature_matrix(df: pd.DataFrame, attributes: Sequence[str], behavioral_model: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0
    if behavioral_model == "m1":
        for attr in attributes:
            delta = pd.to_numeric(df.get(f"delta_{attr}", 0), errors="coerce").fillna(0).astype(int)
            X[f"z_{attr}_1"] = delta.eq(1).astype(float) - delta.eq(-1).astype(float)
            X[f"z_{attr}_2"] = delta.eq(2).astype(float) - delta.eq(-2).astype(float)
        return X
    for attr in attributes:
        X[f"diff_{attr}"] = pd.to_numeric(df.get(f"delta_{attr}", 0), errors="coerce").fillna(0.0).astype(float)
    return X


def fit_behavioral_model(
    df: pd.DataFrame,
    *,
    attributes: Sequence[str],
    behavioral_model: str,
) -> tuple[HospitalBehavioralModel, pd.DataFrame, dict[str, Any]]:
    fit_df = df.loc[df["trials"] > 0].copy()
    if fit_df.empty:
        raise ValueError("No actor training rows with observed choices")
    X = build_feature_matrix(fit_df, attributes, behavioral_model)
    y = fit_df["successes"].astype(float) / fit_df["trials"].astype(float)
    weights = fit_df["trials"].astype(float)
    glm = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    fit_mode = "unpenalized"
    try:
        result = glm.fit()
        params = pd.Series(result.params, index=X.columns, dtype=float)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        result = glm.fit_regularized(alpha=1e-6, L1_wt=0.0)
        params = pd.Series(np.asarray(result.params, dtype=float), index=X.columns, dtype=float)
        fit_mode = "ridge_fallback"
    model = HospitalBehavioralModel(
        behavioral_model=behavioral_model,
        attributes=list(attributes),
        params=params,
        feature_columns=list(X.columns),
        fit_mode=fit_mode,
    )
    train_probs = model.predict_proba(fit_df)
    summary = {
        "behavioral_model": behavioral_model,
        "behavioral_model_name": model.model_name,
        "fit_mode": fit_mode,
        "train_nll": mean_binomial_nll(fit_df["successes"], fit_df["trials"], train_probs),
        "model_params": {k: float(v) for k, v in params.items()},
        "feature_columns": list(X.columns),
        "attributes": list(attributes),
    }
    return model, fit_df, summary


def mean_binomial_nll(successes: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, probs: pd.Series | np.ndarray) -> float:
    success_arr = np.asarray(successes, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    prob_arr = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0 - 1e-9)
    failures = weight_arr - success_arr
    row_nll = -(success_arr * np.log(prob_arr) + failures * np.log(1.0 - prob_arr))
    return float(np.mean(row_nll))


def coerce_choice_series(chosen_side: pd.Series | Iterable[str] | str, index: pd.Index) -> pd.Series:
    if isinstance(chosen_side, pd.Series):
        return chosen_side.reindex(index)
    if isinstance(chosen_side, str):
        return pd.Series(chosen_side, index=index, dtype=object)
    values = list(chosen_side)
    if len(values) != len(index):
        raise ValueError("Choice series length mismatch")
    return pd.Series(values, index=index, dtype=object)


def canonical_argmax(attributes: Sequence[str], values: np.ndarray) -> str:
    return list(attributes)[int(np.nanargmax(np.asarray(values, dtype=float)))]


def driver_margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    if len(ordered) < 2:
        return float("nan")
    return float(ordered[0] - ordered[1])


def canonical_profile(choice_series: pd.Series, slot_a_series: pd.Series, slot_b_series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(choice_series.eq("A"), slot_a_series, np.where(choice_series.eq("B"), slot_b_series, None)),
        index=choice_series.index,
        dtype="object",
    )


def safe_match(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left == right).fillna(False).astype(bool)


def pairwise_agreement(values: pd.Series) -> float | None:
    valid = values.dropna()
    n = int(len(valid))
    if n < 2:
        return None
    counts = valid.value_counts(dropna=True)
    agree_pairs = int(sum(count * (count - 1) for count in counts.to_list()))
    return float(agree_pairs / (n * (n - 1)))


def mode_value(values: pd.Series) -> Any:
    valid = values.dropna()
    if valid.empty:
        return None
    counts = valid.value_counts(dropna=True)
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None
    return counts.index[0]


def build_actor_eval_frame(*, trials_df: pd.DataFrame, actor_draws: pd.DataFrame, model: HospitalBehavioralModel) -> pd.DataFrame:
    eval_df = actor_draws.merge(trials_df, on="trial_id", how="left")
    eval_df["p_choose_A"] = model.predict_proba(eval_df).to_numpy(dtype=float)
    eval_df["predicted_choice"] = model.predict_choice(eval_df).to_numpy(dtype=object)
    eval_df["actor_choice_profile"] = canonical_profile(eval_df["choice"], eval_df["slot_A_profile"], eval_df["slot_B_profile"])
    eval_df["latent_choice_profile"] = canonical_profile(eval_df["predicted_choice"], eval_df["slot_A_profile"], eval_df["slot_B_profile"])
    revealed = model.revealed_driver(eval_df, eval_df["choice"])
    eval_df["revealed_driver"] = revealed["revealed_driver"]
    eval_df["driver_margin"] = revealed["driver_margin"]
    eval_df["actor_choice_correct"] = safe_match(eval_df["actor_choice_profile"], eval_df["latent_choice_profile"])
    eval_df["self_report_driver_correct"] = safe_match(eval_df["premise_attr"], eval_df["revealed_driver"])
    return eval_df


def add_tau_predictions(df: pd.DataFrame, attributes: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=float)
    for attr in attributes:
        tau_col = f"tau_{attr}"
        delta_col = f"delta_{attr}"
        out[tau_col] = pd.to_numeric(out.get(tau_col), errors="coerce").fillna(0.0)
        out[delta_col] = pd.to_numeric(out.get(delta_col), errors="coerce").fillna(0.0)
        signed = np.where(out[delta_col] > 0, out[tau_col], 0.0)
        signed = np.where(out[delta_col] < 0, -out[tau_col], signed)
        out[f"tau_signed_{attr}"] = signed
        score = score + out[f"tau_signed_{attr}"].to_numpy(dtype=float)
    out["tau_score_A"] = score
    out["tau_pred_choice"] = np.where(score > 0, "A", np.where(score < 0, "B", "A"))
    tau_matrix = out[[f"tau_signed_{attr}" for attr in attributes]].to_numpy(dtype=float)
    out["tau_driver"] = [list(attributes)[int(np.argmax(np.abs(row)))] for row in tau_matrix]
    return out


def build_judge_eval_frame(*, trials_df: pd.DataFrame, judge_draws: pd.DataFrame, model: HospitalBehavioralModel) -> pd.DataFrame:
    eval_df = judge_draws.merge(trials_df, on="trial_id", how="left")
    eval_df = add_tau_predictions(eval_df, model.attributes)
    eval_df["p_choose_A"] = model.predict_proba(eval_df).to_numpy(dtype=float)
    eval_df["predicted_choice"] = model.predict_choice(eval_df).to_numpy(dtype=object)
    eval_df["actor_choice_profile"] = canonical_profile(eval_df["choice"], eval_df["slot_A_profile"], eval_df["slot_B_profile"])
    eval_df["judge_choice_profile"] = canonical_profile(eval_df["tau_pred_choice"], eval_df["slot_A_profile"], eval_df["slot_B_profile"])
    eval_df["latent_choice_profile"] = canonical_profile(eval_df["predicted_choice"], eval_df["slot_A_profile"], eval_df["slot_B_profile"])
    revealed = model.revealed_driver(eval_df, eval_df["choice"])
    eval_df["revealed_driver"] = revealed["revealed_driver"]
    eval_df["driver_margin"] = revealed["driver_margin"]
    eval_df["judge_choice_matches_actor"] = safe_match(eval_df["judge_choice_profile"], eval_df["actor_choice_profile"])
    eval_df["judge_choice_correct"] = safe_match(eval_df["judge_choice_profile"], eval_df["latent_choice_profile"])
    eval_df["judge_driver_correct"] = safe_match(eval_df["tau_driver"], eval_df["revealed_driver"])
    return eval_df


def compute_summary_metrics(*, test_trial_df: pd.DataFrame, actor_eval_df: pd.DataFrame, judge_eval_df: pd.DataFrame, model: HospitalBehavioralModel) -> dict[str, Any]:
    test_probs = model.predict_proba(test_trial_df)
    return {
        "direct_choice_agreement": float(actor_eval_df.loc[actor_eval_df["choice_ok"], "actor_choice_correct"].mean()),
        "judge_choice_agreement_with_latent_choice": float(
            judge_eval_df.loc[judge_eval_df["choice_ok"] & judge_eval_df["tau_ok"], "judge_choice_correct"].mean()
        ),
        "judge_choice_matches_actor": float(
            judge_eval_df.loc[judge_eval_df["choice_ok"] & judge_eval_df["tau_ok"], "judge_choice_matches_actor"].mean()
        ),
        "direct_attribute_agreement": float(
            actor_eval_df.loc[actor_eval_df["choice_ok"] & actor_eval_df["premise_ok"], "self_report_driver_correct"].mean()
        ),
        "judge_attribute_agreement": float(
            judge_eval_df.loc[judge_eval_df["choice_ok"] & judge_eval_df["tau_ok"], "judge_driver_correct"].mean()
        ),
        "heldout_direct_choice_nll": mean_binomial_nll(test_trial_df["successes"], test_trial_df["trials"], test_probs),
        "n_actor_draws": int(len(actor_eval_df)),
        "n_judge_draws": int(len(judge_eval_df)),
        "n_test_trials": int(len(test_trial_df)),
    }


def compute_family_metrics(*, actor_eval_df: pd.DataFrame, judge_eval_df: pd.DataFrame, expected_draws: int = len(FULL_VARIANTS) * DEFAULT_REPLICATES) -> dict[str, Any]:
    def summarize(grouped_df: pd.DataFrame, value_col: str, correct_col: str) -> tuple[float | None, float | None]:
        pairwise_values: list[float] = []
        majority_values: list[float] = []
        for _, group in grouped_df.groupby("family_id", sort=False):
            agreement = pairwise_agreement(group[value_col])
            if agreement is not None:
                pairwise_values.append(agreement)
            valid_correct = group[correct_col].dropna()
            if len(group) == expected_draws and len(valid_correct) == expected_draws:
                majority_values.append(float(valid_correct.mean() > 0.5))
        pairwise_mean = float(np.mean(pairwise_values)) if pairwise_values else None
        majority_mean = float(np.mean(majority_values)) if majority_values else None
        return pairwise_mean, majority_mean

    actor_choice_pairwise, actor_choice_majority = summarize(
        actor_eval_df.loc[actor_eval_df["choice_ok"]].copy(), "actor_choice_profile", "actor_choice_correct"
    )
    actor_driver_pairwise, actor_driver_majority = summarize(
        actor_eval_df.loc[actor_eval_df["premise_ok"]].copy(), "premise_attr", "self_report_driver_correct"
    )
    judge_choice_pairwise, judge_choice_majority = summarize(
        judge_eval_df.loc[judge_eval_df["tau_ok"]].copy(), "judge_choice_profile", "judge_choice_correct"
    )
    judge_driver_pairwise, judge_driver_majority = summarize(
        judge_eval_df.loc[judge_eval_df["tau_ok"]].copy(), "tau_driver", "judge_driver_correct"
    )
    return {
        "actor_choice_family_pairwise_agreement": actor_choice_pairwise,
        "self_report_driver_family_pairwise_agreement": actor_driver_pairwise,
        "score_judge_choice_family_pairwise_agreement": judge_choice_pairwise,
        "score_judge_driver_family_pairwise_agreement": judge_driver_pairwise,
        "actor_choice_family_majority_correct_rate": actor_choice_majority,
        "self_report_driver_family_majority_correct_rate": actor_driver_majority,
        "score_judge_choice_family_majority_correct_rate": judge_choice_majority,
        "score_judge_driver_family_majority_correct_rate": judge_driver_majority,
    }


def render_family_summary_figure(summary_rows: list[dict[str, Any]], out_prefix: str | Path) -> None:
    pairwise_metrics = [
        ("actor_choice_family_pairwise_agreement", "Direct report (choice)"),
        ("score_judge_choice_family_pairwise_agreement", "Score-based judge (choice)"),
        ("self_report_driver_family_pairwise_agreement", "Direct report (attribute)"),
        ("score_judge_driver_family_pairwise_agreement", "Score-based judge (attribute)"),
    ]
    majority_metrics = [
        ("actor_choice_family_majority_correct_rate", "Direct report (choice)"),
        ("score_judge_choice_family_majority_correct_rate", "Score-based judge (choice)"),
        ("self_report_driver_family_majority_correct_rate", "Direct report (attribute)"),
        ("score_judge_driver_family_majority_correct_rate", "Score-based judge (attribute)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharey=True)
    offsets = np.linspace(-0.1, 0.1, num=max(len(summary_rows), 1))
    for ax, metrics, xlabel, title in [
        (axes[0], pairwise_metrics, "Pairwise agreement", "Within-family reproducibility"),
        (axes[1], majority_metrics, "Majority-correct rate", "Revealed-target recovery"),
    ]:
        y_positions = np.arange(len(metrics))
        ax.axhline(1.5, color="#bbbbbb", linewidth=0.9, alpha=0.7)
        for idx, row in enumerate(summary_rows):
            spec = MODEL_BY_TAG[row["model_tag"]]
            values = [row.get(metric) for metric, _ in metrics]
            ax.scatter(values, y_positions + offsets[idx], s=38, color=spec.color, alpha=0.9, label=spec.tag if ax is axes[0] else None)
        ax.set_xlim(0.0, 1.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in metrics])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def compute_occlusion_summary(*, occlusion_draws: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, Any]] = []
    for (base_trial_id, manipulation, attribute_target), group in occlusion_draws.groupby(
        ["base_trial_id", "manipulation", "attribute_target"], dropna=False, sort=False
    ):
        choice_profile = canonical_profile(group["choice"], group["slot_A_profile"], group["slot_B_profile"])
        grouped_rows.append(
            {
                "base_trial_id": str(base_trial_id),
                "manipulation": manipulation,
                "attribute_target": attribute_target,
                "majority_choice_profile": mode_value(choice_profile),
                "majority_premise_attr": mode_value(group.loc[group["premise_ok"], "premise_attr"]),
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    baseline = grouped.loc[grouped["manipulation"] == "short_reason"].rename(
        columns={
            "majority_choice_profile": "baseline_choice_profile",
            "majority_premise_attr": "baseline_premise_attr",
        }
    )
    occluded = grouped.loc[grouped["manipulation"] == "occlude_equalize"].copy()
    merged = occluded.merge(
        baseline[["base_trial_id", "baseline_choice_profile", "baseline_premise_attr"]],
        on="base_trial_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for attr, group in merged.groupby("attribute_target", sort=False):
        choice_mask = group["majority_choice_profile"].notna() & group["baseline_choice_profile"].notna()
        premise_mask = group["majority_premise_attr"].notna() & group["baseline_premise_attr"].notna()
        rows.append(
            {
                "attribute": attr,
                "choice_flip_rate": float((group.loc[choice_mask, "majority_choice_profile"] != group.loc[choice_mask, "baseline_choice_profile"]).mean()) if choice_mask.any() else np.nan,
                "choice_flip_n": int(choice_mask.sum()),
                "stated_factor_flip_rate": float((group.loc[premise_mask, "majority_premise_attr"] != group.loc[premise_mask, "baseline_premise_attr"]).mean()) if premise_mask.any() else np.nan,
                "stated_factor_flip_n": int(premise_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def render_occlusion_figure(summary_df: pd.DataFrame, out_prefix: str | Path, *, title: str) -> None:
    attrs = summary_df["attribute"].tolist()
    x = np.arange(len(attrs))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    axes[0].bar(x, summary_df["choice_flip_rate"], color="#3b6fb6")
    axes[1].bar(x, summary_df["stated_factor_flip_rate"], color="#c26d2d")
    axes[0].set_title("Choice flips")
    axes[1].set_title("Stated-factor flips")
    axes[0].set_ylabel("Flip rate")
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(attrs)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_dataframe_artifacts(df: pd.DataFrame, out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_prefix.with_suffix(".parquet"), index=False)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result
