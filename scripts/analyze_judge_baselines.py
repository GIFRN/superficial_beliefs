#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.features import aggregate_choices, load_responses, prepare_stageA_data
from src.analysis.argllm_semantics import add_argllm_semantics_predictions
from src.analysis.judge_baselines import (
    ATTRIBUTES,
    add_pairwise_drivers,
    add_tau_predictions,
    behavioral_attribution,
    tau_stability,
)
from src.analysis.stageA import build_design_matrix, compute_ames_and_weights, fit_glm_clustered
from src.utils.io import ensure_dir, write_json


def _agreement_rate(df: pd.DataFrame, pred_col: str, actual_col: str) -> float:
    valid = df[df[actual_col].notna()].copy()
    if valid.empty:
        return float("nan")
    return float((valid[pred_col] == valid[actual_col]).mean())


def _driver_alignment(df: pd.DataFrame, driver_col: str, premise_col: str) -> float:
    valid = df[df[premise_col].isin(ATTRIBUTES)].copy()
    if valid.empty:
        return float("nan")
    return float((valid[driver_col] == valid[premise_col]).mean())


def _weight_correlation_from_effects(df: pd.DataFrame, prefix: str, weights: dict[str, float]) -> float:
    effect_series = pd.Series(
        {attr: df[f"{prefix}_effect_{attr}"].abs().mean() for attr in ATTRIBUTES}
    )
    weight_series = pd.Series(weights)
    if effect_series.sum() and weight_series.sum():
        return float(effect_series.rank().corr(weight_series.rank(), method="spearman"))
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judge baselines and behavioral attributions")
    parser.add_argument("--dataset", default="data/generated/v1_short", help="Path to generated dataset directory")
    parser.add_argument("--responses", default="data/runs/v1_short_openai_gpt5mini_high/responses.jsonl", help="Path to responses JSONL file")
    parser.add_argument("--out", default=None, help="Directory to write outputs (auto-generated if not specified)")
    parser.add_argument("--interactions", action="store_true", help="Include pairwise interactions for Stage A")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples for behavioral attribution CIs")
    parser.add_argument("--argllm-conservativeness", type=float, default=1.0, help="Conservativeness for dfq/qe semantics")
    parser.add_argument("--claim-base-score", type=float, default=0.5, help="Base score for the root claim 'Choose A over B'")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    trials_df = pd.read_parquet(dataset_dir / "dataset_trials.parquet")
    responses_df = load_responses(args.responses)

    responses_path = Path(args.responses)
    manifest_path = responses_path.parent / "MANIFEST.json"
    model_name = "unknown"
    reasoning_effort = None
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
            model_name = manifest.get("backend", "unknown")
            reasoning_effort = manifest.get("reasoning_effort", None)

    if args.out is None:
        suffix = "_interactions" if args.interactions else ""
        if reasoning_effort:
            out_dirname = f"judge_baselines_{model_name}_{reasoning_effort}{suffix}"
        else:
            out_dirname = f"judge_baselines_{model_name}{suffix}"
        args.out = f"results/{out_dirname}"

    print(f"Model: {model_name}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Output directory: {args.out}")

    choice_agg = aggregate_choices(responses_df)
    stageA_data = prepare_stageA_data(trials_df, choice_agg)
    design = build_design_matrix(stageA_data, include_interactions=args.interactions, exclude_b1=True)
    model = fit_glm_clustered(design)
    weights_info = compute_ames_and_weights(model)

    tau_df = add_tau_predictions(responses_df, trials_df)
    tau_df = add_pairwise_drivers(tau_df)
    tau_df = add_argllm_semantics_predictions(
        tau_df,
        semantics="dfq",
        conservativeness=args.argllm_conservativeness,
        claim_base_score=args.claim_base_score,
    )
    tau_df = add_argllm_semantics_predictions(
        tau_df,
        semantics="qe",
        conservativeness=args.argllm_conservativeness,
        claim_base_score=args.claim_base_score,
    )

    stability = tau_stability(tau_df)
    tau_ok_rate = float(tau_df["tau_ok"].mean()) if "tau_ok" in tau_df else float("nan")
    pair_ok_rate = float(tau_df["pairwise_ok"].mean()) if "pairwise_ok" in tau_df else float("nan")
    has_tau_signal = bool(np.isfinite(tau_ok_rate) and tau_ok_rate > 0)
    has_pairwise_signal = bool(np.isfinite(pair_ok_rate) and pair_ok_rate > 0)

    if has_tau_signal:
        tau_agreement = _agreement_rate(
            tau_df[tau_df["choice_ok"]], "tau_pred_choice", "choice"
        )
        tau_driver_align = _driver_alignment(
            tau_df[tau_df["premise_ok"]], "tau_driver", "premise_attr"
        )
        dfq_agreement = _agreement_rate(
            tau_df[tau_df["choice_ok"] & tau_df["dfq_ok"]], "dfq_pred_choice", "choice"
        )
        dfq_driver_align = _driver_alignment(
            tau_df[tau_df["premise_ok"] & tau_df["dfq_ok"]], "dfq_driver", "premise_attr"
        )
        qe_agreement = _agreement_rate(
            tau_df[tau_df["choice_ok"] & tau_df["qe_ok"]], "qe_pred_choice", "choice"
        )
        qe_driver_align = _driver_alignment(
            tau_df[tau_df["premise_ok"] & tau_df["qe_ok"]], "qe_driver", "premise_attr"
        )
    else:
        tau_agreement = float("nan")
        tau_driver_align = float("nan")
        dfq_agreement = float("nan")
        dfq_driver_align = float("nan")
        qe_agreement = float("nan")
        qe_driver_align = float("nan")
        stability = {}

    if has_pairwise_signal:
        pair_driver_align = _driver_alignment(
            tau_df[tau_df["premise_ok"]], "pair_driver", "premise_attr"
        )
    else:
        pair_driver_align = float("nan")

    if has_tau_signal:
        avg_abs_tau = pd.Series(
            {attr: tau_df[f"tau_signed_{attr}"].abs().mean() for attr in ATTRIBUTES}
        )
        weight_series = pd.Series(weights_info["weights"])
        if avg_abs_tau.sum() and weight_series.sum():
            tau_weight_corr = avg_abs_tau.rank().corr(weight_series.rank(), method="spearman")
        else:
            tau_weight_corr = float("nan")
        dfq_weight_corr = _weight_correlation_from_effects(tau_df, "dfq", weights_info["weights"])
        qe_weight_corr = _weight_correlation_from_effects(tau_df, "qe", weights_info["weights"])
    else:
        tau_weight_corr = float("nan")
        dfq_weight_corr = float("nan")
        qe_weight_corr = float("nan")

    behavioral = behavioral_attribution(
        trials_df,
        responses_df,
        bootstrap=args.bootstrap,
    )

    summary = {
        "model": model_name,
        "include_interactions": args.interactions,
        "agreement": {
            "tau_vs_choice": tau_agreement,
            "dfq_vs_choice": dfq_agreement,
            "qe_vs_choice": qe_agreement,
        },
        "alignment": {
            "tau_driver_vs_premise": tau_driver_align,
            "dfq_driver_vs_premise": dfq_driver_align,
            "qe_driver_vs_premise": qe_driver_align,
            "pair_driver_vs_premise": pair_driver_align,
        },
        "tau_ok_rate": tau_ok_rate,
        "pairwise_ok_rate": pair_ok_rate,
        "tau_stability": stability,
        "stageA_weights": weights_info["weights"],
        "tau_weight_correlation": float(tau_weight_corr) if not np.isnan(tau_weight_corr) else float("nan"),
        "dfq_weight_correlation": float(dfq_weight_corr) if not np.isnan(dfq_weight_corr) else float("nan"),
        "qe_weight_correlation": float(qe_weight_corr) if not np.isnan(qe_weight_corr) else float("nan"),
        "argllm_semantics": {
            "claim_base_score": args.claim_base_score,
            "conservativeness": args.argllm_conservativeness,
        },
        "behavioral_attribution": behavioral,
    }
    if reasoning_effort:
        summary["reasoning_effort"] = reasoning_effort

    out_dir = ensure_dir(args.out)
    write_json(summary, out_dir / "judge_baselines_summary.json")

    report_lines = [
        f"# Judge Baselines Summary ({model_name})",
        "",
        f"- Tau/choice agreement: {tau_agreement:.3f}" if not np.isnan(tau_agreement) else "- Tau/choice agreement: n/a",
        f"- DFQ/choice agreement: {dfq_agreement:.3f}" if not np.isnan(dfq_agreement) else "- DFQ/choice agreement: n/a",
        f"- QE/choice agreement: {qe_agreement:.3f}" if not np.isnan(qe_agreement) else "- QE/choice agreement: n/a",
        f"- Tau driver vs premise alignment: {tau_driver_align:.3f}" if not np.isnan(tau_driver_align) else "- Tau driver vs premise alignment: n/a",
        f"- DFQ driver vs premise alignment: {dfq_driver_align:.3f}" if not np.isnan(dfq_driver_align) else "- DFQ driver vs premise alignment: n/a",
        f"- QE driver vs premise alignment: {qe_driver_align:.3f}" if not np.isnan(qe_driver_align) else "- QE driver vs premise alignment: n/a",
        f"- Pairwise driver vs premise alignment: {pair_driver_align:.3f}" if not np.isnan(pair_driver_align) else "- Pairwise driver vs premise alignment: n/a",
        f"- Tau OK rate: {tau_ok_rate:.3f}" if not np.isnan(tau_ok_rate) else "- Tau OK rate: n/a",
        f"- Pairwise OK rate: {pair_ok_rate:.3f}" if not np.isnan(pair_ok_rate) else "- Pairwise OK rate: n/a",
        f"- Tau/weights rank correlation: {tau_weight_corr:.3f}" if not np.isnan(tau_weight_corr) else "- Tau/weights rank correlation: n/a",
        f"- DFQ/weights rank correlation: {dfq_weight_corr:.3f}" if not np.isnan(dfq_weight_corr) else "- DFQ/weights rank correlation: n/a",
        f"- QE/weights rank correlation: {qe_weight_corr:.3f}" if not np.isnan(qe_weight_corr) else "- QE/weights rank correlation: n/a",
        "",
        "## Stage A Weights",
        json.dumps(weights_info["weights"], indent=2),
        "",
        "## Behavioral Attribution",
        json.dumps(behavioral, indent=2),
        "",
    ]
    (out_dir / "judge_baselines_report.md").write_text("\n".join(report_lines))

    print("\nJudge baseline analysis complete!")
    print(f"   Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
