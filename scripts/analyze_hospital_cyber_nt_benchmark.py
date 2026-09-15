#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hospital_cyber_nt import (
    ATTRIBUTES,
    MODEL_BY_TAG,
    MODEL_SPECS,
    build_actor_eval_frame,
    build_judge_eval_frame,
    compute_family_metrics,
    compute_occlusion_summary,
    compute_summary_metrics,
    dataset_dir,
    fit_behavioral_model,
    flatten_actor_responses,
    flatten_judge_responses,
    output_root,
    prepare_behavioral_frame,
    render_family_summary_figure,
    reports_root,
    resolve_run_dir,
    results_root,
    run_prefix,
    write_dataframe_artifacts,
)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    columns = list(df.columns)
    rows = [columns, ["---"] * len(columns)]
    for record in df.to_dict("records"):
        rows.append([str(record.get(column, "")) for column in columns])
    text = "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def load_trials(base_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = dataset_dir(split, base=base_root)
    trials_df = pd.read_parquet(split_dir / "dataset_trials.parquet")
    configs_df = pd.read_parquet(split_dir / "dataset_configs.parquet")
    return trials_df, configs_df


def render_combined_occlusion_figure(model_frames: dict[str, pd.DataFrame], out_prefix: Path) -> None:
    model_tags = list(model_frames.keys())
    fig, axes = plt.subplots(len(model_tags), 2, figsize=(11.0, max(4.0, 3.6 * len(model_tags))), squeeze=False)
    for row_idx, model_tag in enumerate(model_tags):
        spec = MODEL_BY_TAG[model_tag]
        summary_df = model_frames[model_tag]
        x = range(len(summary_df))
        axes[row_idx, 0].bar(x, summary_df["choice_flip_rate"], color=spec.color)
        axes[row_idx, 1].bar(x, summary_df["stated_factor_flip_rate"], color=spec.color)
        axes[row_idx, 0].set_ylabel(f"{model_tag}\nflip rate")
        axes[row_idx, 0].set_title("Choice flips")
        axes[row_idx, 1].set_title("Stated-factor flips")
        for col_idx in range(2):
            axes[row_idx, col_idx].set_ylim(0.0, 1.0)
            axes[row_idx, col_idx].set_xticks(list(x))
            axes[row_idx, col_idx].set_xticklabels(summary_df["attribute"].tolist())
            axes[row_idx, col_idx].grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the hospital cyber NT benchmark")
    parser.add_argument("--output-root", default=None, help="Override benchmark output root")
    parser.add_argument("--models", nargs="+", default=[spec.tag for spec in MODEL_SPECS])
    parser.add_argument("--include-m1", action="store_true")
    args = parser.parse_args()

    base_root = output_root(args.output_root)
    train_trials, _ = load_trials(base_root, "train")
    test_trials, _ = load_trials(base_root, "test")
    occlusion_trials, _ = load_trials(base_root, "occlusion_test")

    summary_rows: list[dict[str, object]] = []
    m1_rows: list[dict[str, object]] = []
    occlusion_frames: dict[str, pd.DataFrame] = {}

    for model_tag in args.models:
        if model_tag not in MODEL_BY_TAG:
            raise SystemExit(f"Unknown model tag: {model_tag}")
        spec = MODEL_BY_TAG[model_tag]

        train_actor_dir = resolve_run_dir(run_prefix("train", model_tag, "actor", base=base_root))
        test_actor_dir = resolve_run_dir(run_prefix("test", model_tag, "actor", base=base_root))
        test_judge_dir = resolve_run_dir(run_prefix("test", model_tag, "judge", base=base_root))
        if train_actor_dir is None or test_actor_dir is None or test_judge_dir is None:
            raise SystemExit(f"Missing run directory for model {model_tag}")

        actor_train_draws = flatten_actor_responses(train_actor_dir / "responses.jsonl")
        actor_test_draws = flatten_actor_responses(test_actor_dir / "responses.jsonl")
        judge_test_draws = flatten_judge_responses(test_judge_dir / "responses.jsonl", ATTRIBUTES)

        parsed_root = results_root(base_root) / "parsed"
        write_dataframe_artifacts(actor_train_draws, parsed_root / f"{model_tag}_actor_train")
        write_dataframe_artifacts(actor_test_draws, parsed_root / f"{model_tag}_actor_test")
        write_dataframe_artifacts(judge_test_draws, parsed_root / f"{model_tag}_judge_test")

        train_model_df = prepare_behavioral_frame(train_trials, actor_train_draws)
        test_model_df = prepare_behavioral_frame(test_trials, actor_test_draws)

        m0_model, _, m0_fit_summary = fit_behavioral_model(
            train_model_df,
            attributes=ATTRIBUTES,
            behavioral_model="m0",
        )
        actor_eval_m0 = build_actor_eval_frame(trials_df=test_trials, actor_draws=actor_test_draws, model=m0_model)
        judge_eval_m0 = build_judge_eval_frame(trials_df=test_trials, judge_draws=judge_test_draws, model=m0_model)

        eval_root = results_root(base_root) / "eval"
        write_dataframe_artifacts(actor_eval_m0, eval_root / f"{model_tag}_actor_test_m0")
        write_dataframe_artifacts(judge_eval_m0, eval_root / f"{model_tag}_judge_test_m0")
        write_json(results_root(base_root) / "fits" / f"{model_tag}_m0_summary.json", m0_fit_summary)

        summary = compute_summary_metrics(
            test_trial_df=test_model_df,
            actor_eval_df=actor_eval_m0,
            judge_eval_df=judge_eval_m0,
            model=m0_model,
        )
        summary.update(compute_family_metrics(actor_eval_df=actor_eval_m0, judge_eval_df=judge_eval_m0))
        summary.update(
            {
                "model_tag": model_tag,
                "family": spec.family,
                "effort": spec.effort,
                "provider": spec.provider,
                "behavioral_model": "M0",
            }
        )
        summary_rows.append(summary)

        if args.include_m1:
            m1_model, _, m1_fit_summary = fit_behavioral_model(
                train_model_df,
                attributes=ATTRIBUTES,
                behavioral_model="m1",
            )
            actor_eval_m1 = build_actor_eval_frame(trials_df=test_trials, actor_draws=actor_test_draws, model=m1_model)
            judge_eval_m1 = build_judge_eval_frame(trials_df=test_trials, judge_draws=judge_test_draws, model=m1_model)
            write_dataframe_artifacts(actor_eval_m1, eval_root / f"{model_tag}_actor_test_m1")
            write_dataframe_artifacts(judge_eval_m1, eval_root / f"{model_tag}_judge_test_m1")
            write_json(results_root(base_root) / "fits" / f"{model_tag}_m1_summary.json", m1_fit_summary)

            m1_summary = compute_summary_metrics(
                test_trial_df=test_model_df,
                actor_eval_df=actor_eval_m1,
                judge_eval_df=judge_eval_m1,
                model=m1_model,
            )
            m1_rows.append(
                {
                    "model_tag": model_tag,
                    "m0_direct_choice_agreement": summary["direct_choice_agreement"],
                    "m1_direct_choice_agreement": m1_summary["direct_choice_agreement"],
                    "delta_direct_choice_agreement": m1_summary["direct_choice_agreement"] - summary["direct_choice_agreement"],
                    "m0_judge_choice_agreement_with_latent_choice": summary["judge_choice_agreement_with_latent_choice"],
                    "m1_judge_choice_agreement_with_latent_choice": m1_summary["judge_choice_agreement_with_latent_choice"],
                    "delta_judge_choice_agreement_with_latent_choice": (
                        m1_summary["judge_choice_agreement_with_latent_choice"]
                        - summary["judge_choice_agreement_with_latent_choice"]
                    ),
                    "m0_direct_attribute_agreement": summary["direct_attribute_agreement"],
                    "m1_direct_attribute_agreement": m1_summary["direct_attribute_agreement"],
                    "delta_direct_attribute_agreement": m1_summary["direct_attribute_agreement"] - summary["direct_attribute_agreement"],
                    "m0_judge_attribute_agreement": summary["judge_attribute_agreement"],
                    "m1_judge_attribute_agreement": m1_summary["judge_attribute_agreement"],
                    "delta_judge_attribute_agreement": m1_summary["judge_attribute_agreement"] - summary["judge_attribute_agreement"],
                    "m0_heldout_direct_choice_nll": summary["heldout_direct_choice_nll"],
                    "m1_heldout_direct_choice_nll": m1_summary["heldout_direct_choice_nll"],
                    "delta_heldout_direct_choice_nll": m1_summary["heldout_direct_choice_nll"] - summary["heldout_direct_choice_nll"],
                }
            )

        occlusion_actor_dir = resolve_run_dir(run_prefix("occlusion_test", model_tag, "actor", base=base_root))
        if occlusion_actor_dir is not None:
            occlusion_draws = flatten_actor_responses(occlusion_actor_dir / "responses.jsonl")
            occlusion_eval = occlusion_draws.merge(occlusion_trials, on="trial_id", how="left")
            occlusion_summary = compute_occlusion_summary(occlusion_draws=occlusion_eval)
            occlusion_summary["model_tag"] = model_tag
            occlusion_frames[model_tag] = occlusion_summary
            write_dataframe_artifacts(
                occlusion_summary,
                reports_root(base_root) / f"{model_tag}_equalise_occlusion_summary",
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("model_tag").reset_index(drop=True)
    report_prefix = reports_root(base_root) / "hospital_cyber_nt_summary"
    write_dataframe_artifacts(summary_df, report_prefix)
    write_json(report_prefix.with_suffix(".json"), summary_df.to_dict("records"))
    write_markdown_table(summary_df, report_prefix.with_suffix(".md"))
    render_family_summary_figure(summary_df.to_dict("records"), reports_root(base_root) / "hospital_cyber_nt_family_summary")

    if m1_rows:
        m1_df = pd.DataFrame(m1_rows).sort_values("model_tag").reset_index(drop=True)
        m1_prefix = reports_root(base_root) / "hospital_cyber_nt_m1_comparison"
        write_dataframe_artifacts(m1_df, m1_prefix)
        write_json(m1_prefix.with_suffix(".json"), m1_df.to_dict("records"))
        write_markdown_table(m1_df, m1_prefix.with_suffix(".md"))

    if occlusion_frames:
        combined_occlusion = pd.concat(occlusion_frames.values(), ignore_index=True)
        occlusion_prefix = reports_root(base_root) / "hospital_cyber_nt_equalise_occlusion"
        write_dataframe_artifacts(combined_occlusion, occlusion_prefix)
        write_json(occlusion_prefix.with_suffix(".json"), combined_occlusion.to_dict("records"))
        write_markdown_table(combined_occlusion, occlusion_prefix.with_suffix(".md"))
        render_combined_occlusion_figure(occlusion_frames, reports_root(base_root) / "hospital_cyber_nt_equalise_occlusion")


if __name__ == "__main__":
    main()
