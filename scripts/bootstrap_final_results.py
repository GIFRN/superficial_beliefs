#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_balanced import (
    build_balanced_eval_frame_from_model,
    compute_main_metrics,
    compute_placebo_metrics,
    load_prefit_stagea_model,
)
from src.analysis.features import aggregate_choices, load_responses, prepare_stageA_data
from src.analysis.final_benchmark import (
    MAIN_THEMES,
    MODEL_SPECS,
    PLACEBO_THEMES,
    THEME_CONFIGS,
    dataset_dir,
    output_root,
    reports_root,
    resolve_run_dir,
    run_prefix,
    stagea_dir,
)
from src.analysis.stageA import build_design_matrix, fit_glm_clustered


MAIN_METRICS = [
    "linear_model_predicts_actor_choice",
    "judge_predicts_actor_choice",
    "linear_model_factor_matches_stated_factor",
    "judge_factor_matches_stated_factor",
]

MAIN_COUNT_FIELDS = [
    "n_choice_ok",
    "n_judge_choice_ok",
    "n_premise_ok",
    "n_judge_premise_ok",
]

PLACEBO_METRICS = [
    "actor_states_placebo_as_key_factor",
    "linear_model_factor_is_placebo",
    "judge_factor_is_placebo",
]

PLACEBO_COUNT_FIELDS = [
    "n_premise_ok",
    "n_choice_ok",
    "n_tau_ok",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _group_frames(df: pd.DataFrame, group_col: str = "config_id") -> tuple[list[str], dict[str, pd.DataFrame]]:
    groups: dict[str, pd.DataFrame] = {}
    for key, group in df.groupby(df[group_col].astype(str), sort=False):
        groups[str(key)] = group.copy()
    return list(groups.keys()), groups


def _sample_grouped_frame(group_ids: list[str], group_map: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
    return pd.concat([group_map[str(group_id)] for group_id in sampled_ids], ignore_index=True)


def _fit_bootstrap_model(train_stagea_df: pd.DataFrame, *, include_interactions: bool, rng: np.random.Generator):
    train_ids, train_groups = _group_frames(train_stagea_df, "config_id")
    sampled_train = _sample_grouped_frame(train_ids, train_groups, rng)
    design = build_design_matrix(sampled_train, include_interactions=include_interactions, exclude_b1=True)
    model = fit_glm_clustered(design)
    return model, list(design.X.columns)


def _bootstrap_summary(values: list[float | None]) -> tuple[float | None, float | None, float | None]:
    cleaned = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not cleaned:
        return None, None, None
    arr = np.asarray(cleaned, dtype=float)
    se = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return se, float(lo), float(hi)


def _augment_row(row: dict[str, Any], metric_names: list[str], samples: dict[str, list[float | None]]) -> dict[str, Any]:
    out = dict(row)
    for metric in metric_names:
        se, lo, hi = _bootstrap_summary(samples[metric])
        out[f"{metric}_se"] = se
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
    return out


def _placebo_variant_label(theme: str) -> str:
    payload = yaml.safe_load(THEME_CONFIGS[theme].read_text())
    return str(payload["attributes"]["D"]["label"])


def _write_rows(
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
    payload: dict[str, Any],
    out_prefix: Path,
    title: str,
    summary_columns: list[str],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# {title}",
        "",
        f"- Expected rows: {payload['expected_rows']}",
        f"- Actual rows: {payload['actual_rows']}",
        f"- Bootstrap replicates: {payload['bootstrap']}",
        "",
        "## Summary",
        _markdown_table(rows, summary_columns),
    ]
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines))
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.json')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap final same-order result tables.")
    parser.add_argument("--out-root", default=str(output_root()))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--main-out-prefix",
        default=None,
        help="Optional output prefix for the main bootstrap table. Defaults to <out-root>/reports/FINAL_MAIN_RESULTS_BOOTSTRAP",
    )
    parser.add_argument(
        "--placebo-out-prefix",
        default=None,
        help="Optional output prefix for the placebo bootstrap table. Defaults to <out-root>/reports/FINAL_PLACEBO_RESULTS_BOOTSTRAP",
    )
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    main_out_prefix = Path(args.main_out_prefix).resolve() if args.main_out_prefix else reports_root(out_root) / "FINAL_MAIN_RESULTS_BOOTSTRAP"
    placebo_out_prefix = Path(args.placebo_out_prefix).resolve() if args.placebo_out_prefix else reports_root(out_root) / "FINAL_PLACEBO_RESULTS_BOOTSTRAP"

    main_rows: list[dict[str, Any]] = []
    for theme in MAIN_THEMES:
        train_dataset = dataset_dir(theme, "train", base=out_root)
        test_dataset = dataset_dir(theme, "test", base=out_root)
        train_trials_df = pd.read_parquet(train_dataset / "dataset_trials.parquet")
        test_trials_df = pd.read_parquet(test_dataset / "dataset_trials.parquet")
        for idx, spec in enumerate(MODEL_SPECS):
            train_run = resolve_run_dir(run_prefix(theme, "train", spec.tag, "actor", base=out_root))
            test_run = resolve_run_dir(run_prefix(theme, "test", spec.tag, "tau", base=out_root))
            stagea_summary_path = stagea_dir(theme, spec.tag, base=out_root) / "stageA_summary.json"
            if train_run is None or test_run is None or not stagea_summary_path.exists():
                raise SystemExit(f"Missing inputs for {theme}/{spec.tag}")

            train_responses_df = load_responses(train_run / "responses.jsonl")
            test_responses_df = load_responses(test_run / "responses.jsonl")
            choice_agg = aggregate_choices(train_responses_df)
            train_stagea_df = prepare_stageA_data(train_trials_df, choice_agg)
            model, stagea_summary = load_prefit_stagea_model(stagea_summary_path)
            include_interactions = bool(stagea_summary.get("include_interactions", False))
            feature_columns = stagea_summary.get("feature_columns")

            point_eval_df, _ = build_balanced_eval_frame_from_model(
                trials_df=test_trials_df,
                responses_df=test_responses_df,
                model=model,
                include_interactions=include_interactions,
                feature_columns=feature_columns,
            )
            point_metrics = compute_main_metrics(point_eval_df)
            row = {
                "theme": theme,
                "family": spec.family,
                "effort": spec.effort,
                "model_tag": spec.tag,
                **point_metrics,
            }
            metric_samples = {metric: [] for metric in MAIN_METRICS}
            rng = np.random.default_rng(args.seed + idx * 1000 + len(main_rows))
            for _ in range(args.bootstrap):
                boot_model, boot_features = _fit_bootstrap_model(
                    train_stagea_df,
                    include_interactions=include_interactions,
                    rng=rng,
                )
                boot_eval_df, _ = build_balanced_eval_frame_from_model(
                    trials_df=test_trials_df,
                    responses_df=test_responses_df,
                    model=boot_model,
                    include_interactions=include_interactions,
                    feature_columns=boot_features,
                )
                test_ids, test_groups = _group_frames(boot_eval_df, "config_id")
                sampled_eval_df = _sample_grouped_frame(test_ids, test_groups, rng)
                metrics = compute_main_metrics(sampled_eval_df)
                for metric in MAIN_METRICS:
                    metric_samples[metric].append(metrics.get(metric))
            main_rows.append(_augment_row(row, MAIN_METRICS, metric_samples))

    main_rows = sorted(main_rows, key=lambda row: (row["theme"], row["family"], row["effort"]))
    _write_rows(
        rows=main_rows,
        columns=[
            "theme",
            "family",
            "effort",
            *MAIN_METRICS,
            *MAIN_COUNT_FIELDS,
            *[f"{metric}_se" for metric in MAIN_METRICS],
            *[f"{metric}_ci_lo" for metric in MAIN_METRICS],
            *[f"{metric}_ci_hi" for metric in MAIN_METRICS],
            "model_tag",
        ],
        payload={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "out_root": _rel(out_root),
            "bootstrap": args.bootstrap,
            "expected_rows": len(MAIN_THEMES) * len(MODEL_SPECS),
            "actual_rows": len(main_rows),
            "rows": main_rows,
        },
        out_prefix=main_out_prefix,
        title="Final Main Results Bootstrap",
        summary_columns=[
            "theme",
            "family",
            "effort",
            "linear_model_predicts_actor_choice",
            "linear_model_predicts_actor_choice_ci_lo",
            "linear_model_predicts_actor_choice_ci_hi",
            "judge_predicts_actor_choice",
            "judge_predicts_actor_choice_ci_lo",
            "judge_predicts_actor_choice_ci_hi",
        ],
    )

    placebo_rows: list[dict[str, Any]] = []
    for theme in PLACEBO_THEMES:
        train_dataset = dataset_dir(theme, "train", base=out_root)
        test_dataset = dataset_dir(theme, "test", base=out_root)
        train_trials_df = pd.read_parquet(train_dataset / "dataset_trials.parquet")
        test_trials_df = pd.read_parquet(test_dataset / "dataset_trials.parquet")
        placebo_variant = _placebo_variant_label(theme)
        for idx, spec in enumerate(MODEL_SPECS):
            train_run = resolve_run_dir(run_prefix(theme, "train", spec.tag, "actor", base=out_root))
            test_run = resolve_run_dir(run_prefix(theme, "test", spec.tag, "tau", base=out_root))
            stagea_summary_path = stagea_dir(theme, spec.tag, base=out_root) / "stageA_summary.json"
            if train_run is None or test_run is None or not stagea_summary_path.exists():
                raise SystemExit(f"Missing inputs for {theme}/{spec.tag}")

            train_responses_df = load_responses(train_run / "responses.jsonl")
            test_responses_df = load_responses(test_run / "responses.jsonl")
            choice_agg = aggregate_choices(train_responses_df)
            train_stagea_df = prepare_stageA_data(train_trials_df, choice_agg)
            model, stagea_summary = load_prefit_stagea_model(stagea_summary_path)
            include_interactions = bool(stagea_summary.get("include_interactions", False))
            feature_columns = stagea_summary.get("feature_columns")

            point_eval_df, _ = build_balanced_eval_frame_from_model(
                trials_df=test_trials_df,
                responses_df=test_responses_df,
                model=model,
                include_interactions=include_interactions,
                feature_columns=feature_columns,
            )
            point_metrics = compute_placebo_metrics(point_eval_df, placebo_attr="D")
            row = {
                "theme": theme,
                "placebo_variant": placebo_variant,
                "family": spec.family,
                "effort": spec.effort,
                "model_tag": spec.tag,
                **point_metrics,
            }
            metric_samples = {metric: [] for metric in PLACEBO_METRICS}
            rng = np.random.default_rng(args.seed + 100000 + idx * 1000 + len(placebo_rows))
            for _ in range(args.bootstrap):
                boot_model, boot_features = _fit_bootstrap_model(
                    train_stagea_df,
                    include_interactions=include_interactions,
                    rng=rng,
                )
                boot_eval_df, _ = build_balanced_eval_frame_from_model(
                    trials_df=test_trials_df,
                    responses_df=test_responses_df,
                    model=boot_model,
                    include_interactions=include_interactions,
                    feature_columns=boot_features,
                )
                _, test_groups = _group_frames(boot_eval_df, "config_id")
                sampled_eval_df = _sample_grouped_frame(list(test_groups.keys()), test_groups, rng)
                metrics = compute_placebo_metrics(sampled_eval_df, placebo_attr="D")
                for metric in PLACEBO_METRICS:
                    metric_samples[metric].append(metrics.get(metric))
            placebo_rows.append(_augment_row(row, PLACEBO_METRICS, metric_samples))

    placebo_rows = sorted(placebo_rows, key=lambda row: (row["theme"], row["family"], row["effort"]))
    _write_rows(
        rows=placebo_rows,
        columns=[
            "theme",
            "placebo_variant",
            "family",
            "effort",
            *PLACEBO_METRICS,
            *PLACEBO_COUNT_FIELDS,
            *[f"{metric}_se" for metric in PLACEBO_METRICS],
            *[f"{metric}_ci_lo" for metric in PLACEBO_METRICS],
            *[f"{metric}_ci_hi" for metric in PLACEBO_METRICS],
            "model_tag",
        ],
        payload={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "out_root": _rel(out_root),
            "bootstrap": args.bootstrap,
            "expected_rows": len(PLACEBO_THEMES) * len(MODEL_SPECS),
            "actual_rows": len(placebo_rows),
            "rows": placebo_rows,
        },
        out_prefix=placebo_out_prefix,
        title="Final Placebo Results Bootstrap",
        summary_columns=[
            "placebo_variant",
            "family",
            "effort",
            "actor_states_placebo_as_key_factor",
            "actor_states_placebo_as_key_factor_ci_lo",
            "actor_states_placebo_as_key_factor_ci_hi",
            "judge_factor_is_placebo",
            "judge_factor_is_placebo_ci_lo",
            "judge_factor_is_placebo_ci_hi",
        ],
    )


if __name__ == "__main__":
    main()
