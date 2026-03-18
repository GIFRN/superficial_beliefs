#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_balanced import build_balanced_eval_frame
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

ATTR_CODES = ["E", "A", "S", "D"]


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


def _safe_rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def _weighted_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _theme_meta(theme: str) -> dict[str, str]:
    payload = yaml.safe_load(THEME_CONFIGS[theme].read_text())
    return {
        "theme": theme,
        "objective": payload["objective"],
        **{f"{code}_label": payload["attributes"][code]["label"] for code in ATTR_CODES},
    }


def _label(meta: dict[str, str], attr_code: str | None) -> str | None:
    if attr_code is None:
        return None
    return meta.get(f"{attr_code}_label", attr_code)


def _weight_order(weights: dict[str, float]) -> str:
    ordered = sorted(ATTR_CODES, key=lambda code: weights.get(code, float("-inf")), reverse=True)
    return ">".join(ordered)


def _top_pair(counter: Counter[tuple[str, str]]) -> tuple[tuple[str, str] | None, int]:
    if not counter:
        return None, 0
    pair, count = counter.most_common(1)[0]
    return pair, count


def _top_attr(counter: Counter[str]) -> tuple[str | None, int]:
    if not counter:
        return None, 0
    attr, count = counter.most_common(1)[0]
    return attr, count


def _visible_deltas(row: pd.Series, meta: dict[str, str]) -> str:
    parts = [f"{_label(meta, code)} `{int(row[f'delta_{code}']):+d}`" for code in ATTR_CODES]
    return "; ".join(parts)


def _annotate_consistency(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["premise_delta"] = pd.Series([math.nan] * len(work), index=work.index, dtype=float)
    for attr in ATTR_CODES:
        mask = work["premise_attr"] == attr
        if mask.any():
            work.loc[mask, "premise_delta"] = pd.to_numeric(work.loc[mask, f"delta_{attr}"], errors="coerce")
    work["premise_tied"] = work["premise_delta"] == 0
    work["premise_non_tied"] = work["premise_delta"].notna() & ~work["premise_tied"]
    work["premise_supports_choice"] = (
        ((work["choice"] == "A") & (work["premise_delta"] > 0))
        | ((work["choice"] == "B") & (work["premise_delta"] < 0))
    )
    work["premise_contradicts_choice"] = (
        ((work["choice"] == "A") & (work["premise_delta"] < 0))
        | ((work["choice"] == "B") & (work["premise_delta"] > 0))
    )
    return work


def _summarize_rows(rows: pd.DataFrame) -> dict[str, Any]:
    n_valid = int(len(rows))
    n_non_tied = int(rows["premise_non_tied"].sum())
    n_support = int(rows["premise_supports_choice"].sum())
    n_contradict = int(rows["premise_contradicts_choice"].sum())
    n_tied = int(rows["premise_tied"].sum())
    return {
        "n_choice_premise_ok": n_valid,
        "n_non_tied": n_non_tied,
        "n_support": n_support,
        "n_contradict": n_contradict,
        "n_tied": n_tied,
        "support_rate_all": _weighted_rate(n_support, n_valid),
        "support_rate_non_tied": _weighted_rate(n_support, n_non_tied),
        "contradiction_rate_non_tied": _weighted_rate(n_contradict, n_non_tied),
        "tied_rate": _weighted_rate(n_tied, n_valid),
    }


def _attr_summary(rows: pd.DataFrame, meta: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for attr in ATTR_CODES:
        attr_rows = rows[rows["premise_attr"] == attr].copy()
        out.append({"attr_code": attr, "attr_label": _label(meta, attr), **_summarize_rows(attr_rows)})
    return out


def _worst_attr(attr_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in attr_rows if row["n_non_tied"] > 0]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["contradiction_rate_non_tied"] if row["contradiction_rate_non_tied"] is not None else -1.0,
            row["n_non_tied"],
        ),
    )


def _load_eval_frame(theme: str, spec_tag: str, *, base: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = resolve_run_dir(run_prefix(theme, "test", spec_tag, "tau", base=base))
    if run_dir is None:
        raise SystemExit(f"Missing run dir for {theme}/{spec_tag}")
    eval_df, _, stagea_summary = build_balanced_eval_frame(
        dataset_dir=dataset_dir(theme, "test", base=base),
        responses_path=run_dir / "responses.jsonl",
        stagea_summary_path=stagea_dir(theme, spec_tag, base=base) / "stageA_summary.json",
    )
    return eval_df, stagea_summary


def _summary_file_counts() -> dict[str, int]:
    counts = {}
    for split in ["train", "test"]:
        with (ROOT / "data" / split / "base_samples.csv").open() as handle:
            counts[f"{split}_base"] = sum(1 for _ in handle) - 1
        with (ROOT / "data" / split / "full_samples.csv").open() as handle:
            counts[f"{split}_full"] = sum(1 for _ in handle) - 1
    return counts


def _write_results_summary(
    *,
    out_path: Path,
    main_rows: list[dict[str, Any]],
    main_boot_rows: list[dict[str, Any]],
    placebo_rows: list[dict[str, Any]],
    placebo_boot_rows: list[dict[str, Any]],
) -> None:
    boot_lookup = {
        (row["theme"], row["family"], row["effort"]): row
        for row in main_boot_rows
    }
    placebo_boot_lookup = {
        (row["theme"], row["family"], row["effort"]): row
        for row in placebo_boot_rows
    }
    counts = _summary_file_counts()

    def best_row(theme: str, metric: str) -> dict[str, Any]:
        rows = [row for row in main_rows if row["theme"] == theme]
        return max(rows, key=lambda row: row[metric])

    def ci_text(row: dict[str, Any], metric: str) -> str:
        boot = boot_lookup[(row["theme"], row["family"], row["effort"])]
        return f"{row[metric]:.3f} with 95% CI [{boot[f'{metric}_ci_lo']:.3f}, {boot[f'{metric}_ci_hi']:.3f}]"

    best_linear = {theme: best_row(theme, "linear_model_predicts_actor_choice") for theme in MAIN_THEMES}
    best_judge = {theme: best_row(theme, "judge_predicts_actor_choice") for theme in MAIN_THEMES}
    best_linear_factor = {theme: best_row(theme, "linear_model_factor_matches_stated_factor") for theme in MAIN_THEMES}
    best_judge_factor = {theme: best_row(theme, "judge_factor_matches_stated_factor") for theme in MAIN_THEMES}

    lowest_placebo = min(placebo_rows, key=lambda row: row["judge_factor_is_placebo"])
    highest_placebo = max(placebo_rows, key=lambda row: row["judge_factor_is_placebo"])

    lines = [
        "# Final Same-Order Results Summary",
        "",
        "This document is the standalone paper-facing summary for the completed final same-order benchmark.",
        "",
        "## Benchmark Definition",
        "",
        "The final benchmark uses a same-order four-variant construction built from a base train/test split. For each base family with profiles `P` and `Q` and source orders `oA` and `oB`, the benchmark evaluates four prompts:",
        "",
        "- `P@oA vs Q@oA`",
        "- `Q@oA vs P@oA`",
        "- `P@oB vs Q@oB`",
        "- `Q@oB vs P@oB`",
        "",
        "So every evaluated prompt is same-order within-row, and each source order is seen in both slot orientations.",
        "",
        f"- base families: `{counts['train_base']}` train and `{counts['test_base']}` test",
        f"- evaluated prompts per theme: `{counts['train_full']}` train and `{counts['test_full']}` test",
        f"- themes: `{len(MAIN_THEMES)}` substantive + `{len(PLACEBO_THEMES)}` placebo",
        f"- model settings: `{len(MODEL_SPECS)}` (`GPT-5-mini`, `GPT-5-nano`, `Qwen3.5-14B`, `Ministral-3-14B`, each at `minimal/low`)",
        "- actor replicates per prompt: `S=3`",
        "- bootstrap intervals use `500` resamples",
        "",
        "## Main Findings",
        "",
        "The fitted choice model still predicts held-out model choices better than the judge recovers them from textual factor attributions. Factor recovery remains noisier than behavioral choice prediction.",
        "",
        "The strongest linear-model choice rows are:",
        "",
    ]
    for theme in MAIN_THEMES:
        row = best_linear[theme]
        lines.append(f"- `{theme}`: `{row['family']} {row['effort']}` at {ci_text(row, 'linear_model_predicts_actor_choice')}")

    lines.extend(["", "The strongest judge-choice rows are:", ""])
    for theme in MAIN_THEMES:
        row = best_judge[theme]
        lines.append(f"- `{theme}`: `{row['family']} {row['effort']}` at {row['judge_predicts_actor_choice']:.3f}")

    lines.extend(["", "The strongest factor-alignment rows are:", ""])
    for metric, label, mapping in [
        ("linear_model_factor_matches_stated_factor", "linear-model factor match", best_linear_factor),
        ("judge_factor_matches_stated_factor", "judge factor match", best_judge_factor),
    ]:
        lines.append(f"- {label}:")
        for theme in MAIN_THEMES:
            row = mapping[theme]
            lines.append(f"  - `{theme}`: `{row['family']} {row['effort']}` at {row[metric]:.3f}")

    lines.extend(
        [
            "",
            "## Placebo Findings",
            "",
            "The placebo results remain bounded and interpretable rather than dominating the main story.",
            "",
            f"- lowest judge placebo row: `{lowest_placebo['theme']}` / `{lowest_placebo['family']} {lowest_placebo['effort']}` at `{lowest_placebo['judge_factor_is_placebo']:.3f}`",
            f"- highest judge placebo row: `{highest_placebo['theme']}` / `{highest_placebo['family']} {highest_placebo['effort']}` at `{highest_placebo['judge_factor_is_placebo']:.3f}`",
            "",
            "## Files",
            "",
            "- `outputs/final_same_order/reports/FINAL_MAIN_RESULTS.csv`",
            "- `outputs/final_same_order/reports/FINAL_PLACEBO_RESULTS.csv`",
            "- `outputs/final_same_order/reports/FINAL_MAIN_RESULTS_BOOTSTRAP.csv`",
            "- `outputs/final_same_order/reports/FINAL_PLACEBO_RESULTS_BOOTSTRAP.csv`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def _build_attribute_reports(*, base: Path, reports_dir: Path) -> tuple[Path, Path]:
    theme_meta_rows = [_theme_meta(theme) for theme in MAIN_THEMES]
    theme_meta = {row["theme"]: row for row in theme_meta_rows}

    conditional_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    stagea_rows: list[dict[str, Any]] = []
    aggregated_theme_rows: list[dict[str, Any]] = []
    exemplar_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    per_theme_stagea_top: dict[str, Counter[str]] = defaultdict(Counter)
    per_theme_actor_linear: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_actor_judge: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_linear_judge: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_linear_correct_judge_wrong: dict[str, Counter[str]] = defaultdict(Counter)
    per_theme_judge_correct_linear_wrong: dict[str, Counter[str]] = defaultdict(Counter)

    premise_row_summaries: list[dict[str, Any]] = []
    premise_theme_aggregates: list[dict[str, Any]] = []
    substantive_frames: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []

    for theme in MAIN_THEMES + PLACEBO_THEMES:
        theme_frames: list[pd.DataFrame] = []
        meta = _theme_meta(theme)
        for spec in MODEL_SPECS:
            eval_df, stagea_summary = _load_eval_frame(theme, spec.tag, base=base)
            if theme in MAIN_THEMES:
                weights = {code: float(stagea_summary["weights"][code]) for code in ATTR_CODES}
                top_attr = max(ATTR_CODES, key=lambda code: weights[code])
                per_theme_stagea_top[theme][top_attr] += 1
                stagea_rows.append(
                    {
                        "theme": theme,
                        "objective": meta["objective"],
                        "family": spec.family,
                        "effort": spec.effort,
                        "top_attr_code": top_attr,
                        "top_attr_label": _label(meta, top_attr),
                        "weight_order": _weight_order(weights),
                        **{f"weight_{code}": weights[code] for code in ATTR_CODES},
                    }
                )

                actor_rows = eval_df[eval_df["premise_ok"] & eval_df["choice_ok"]].copy()
                judge_rows = eval_df[eval_df["premise_ok"] & eval_df["tau_ok"]].copy()
                shared_rows = eval_df[eval_df["premise_ok"] & eval_df["choice_ok"] & eval_df["tau_ok"]].copy()

                for attr in ATTR_CODES:
                    actor_attr_rows = actor_rows[actor_rows["premise_attr"] == attr].copy()
                    judge_attr_rows = judge_rows[judge_rows["premise_attr"] == attr].copy()
                    conditional_rows.append(
                        {
                            "theme": theme,
                            "objective": meta["objective"],
                            "family": spec.family,
                            "effort": spec.effort,
                            "attr_code": attr,
                            "attr_label": _label(meta, attr),
                            "actor_share": _safe_rate(actor_rows["premise_attr"] == attr),
                            "linear_share": _safe_rate(actor_rows["linear_model_factor"] == attr),
                            "judge_share": _safe_rate(judge_rows["tau_driver"] == attr),
                            "n_actor_attr": int(len(actor_attr_rows)),
                            "linear_matches_actor_rate": _safe_rate(actor_attr_rows["linear_model_factor"] == attr),
                            "judge_matches_actor_rate": _safe_rate(judge_attr_rows["tau_driver"] == attr),
                            "linear_choice_accuracy_given_actor_attr": _safe_rate(
                                actor_attr_rows["linear_model_pred_choice"] == actor_attr_rows["choice"]
                            ),
                            "judge_choice_accuracy_given_actor_attr": _safe_rate(
                                judge_attr_rows["tau_pred_choice"] == judge_attr_rows["choice"]
                            ),
                        }
                    )

                actor_linear_mismatch = actor_rows[
                    actor_rows["premise_attr"].notna()
                    & actor_rows["linear_model_factor"].notna()
                    & (actor_rows["premise_attr"] != actor_rows["linear_model_factor"])
                ].copy()
                actor_judge_mismatch = judge_rows[
                    judge_rows["premise_attr"].notna()
                    & judge_rows["tau_driver"].notna()
                    & (judge_rows["premise_attr"] != judge_rows["tau_driver"])
                ].copy()
                linear_judge_mismatch = shared_rows[
                    shared_rows["linear_model_factor"].notna()
                    & shared_rows["tau_driver"].notna()
                    & (shared_rows["linear_model_factor"] != shared_rows["tau_driver"])
                ].copy()

                for label, df, left_col, right_col, counter in [
                    ("actor_vs_linear", actor_linear_mismatch, "premise_attr", "linear_model_factor", per_theme_actor_linear[theme]),
                    ("actor_vs_judge", actor_judge_mismatch, "premise_attr", "tau_driver", per_theme_actor_judge[theme]),
                    ("linear_vs_judge", linear_judge_mismatch, "linear_model_factor", "tau_driver", per_theme_linear_judge[theme]),
                ]:
                    pair_counter: Counter[tuple[str, str]] = Counter(zip(df[left_col], df[right_col]))
                    total = int(len(df))
                    for (left_attr, right_attr), count in pair_counter.items():
                        counter[(left_attr, right_attr)] += count
                        mismatch_rows.append(
                            {
                                "theme": theme,
                                "objective": meta["objective"],
                                "family": spec.family,
                                "effort": spec.effort,
                                "comparison": label,
                                "from_attr_code": left_attr,
                                "from_attr_label": _label(meta, left_attr),
                                "to_attr_code": right_attr,
                                "to_attr_label": _label(meta, right_attr),
                                "count": count,
                                "share_of_mismatches": (count / total) if total else None,
                                "n_mismatches": total,
                            }
                        )

                linear_right_judge_wrong = shared_rows[
                    (shared_rows["linear_model_pred_choice"] == shared_rows["choice"])
                    & (shared_rows["tau_pred_choice"] != shared_rows["choice"])
                ].copy()
                judge_right_linear_wrong = shared_rows[
                    (shared_rows["tau_pred_choice"] == shared_rows["choice"])
                    & (shared_rows["linear_model_pred_choice"] != shared_rows["choice"])
                ].copy()
                per_theme_linear_correct_judge_wrong[theme].update(linear_right_judge_wrong["premise_attr"].dropna().tolist())
                per_theme_judge_correct_linear_wrong[theme].update(judge_right_linear_wrong["premise_attr"].dropna().tolist())

                for mode, frame in [("linear_correct_judge_wrong", linear_right_judge_wrong), ("judge_correct_linear_wrong", judge_right_linear_wrong)]:
                    for _, row in frame.iterrows():
                        exemplar_candidates[theme].append(
                            {
                                "mode": mode,
                                "theme": theme,
                                "family": spec.family,
                                "effort": spec.effort,
                                "trial_id": row["trial_id"],
                                "actor_attr": row["premise_attr"],
                                "linear_attr": row["linear_model_factor"],
                                "judge_attr": row["tau_driver"],
                                "actor_choice": row["choice"],
                                "linear_choice": row["linear_model_pred_choice"],
                                "judge_choice": row["tau_pred_choice"],
                                "visible_deltas": _visible_deltas(row, meta),
                                "delta_actor_abs": abs(int(row[f"delta_{row['premise_attr']}"])) if row["premise_attr"] in ATTR_CODES else 0,
                                "delta_linear_abs": abs(int(row[f"delta_{row['linear_model_factor']}"])) if row["linear_model_factor"] in ATTR_CODES else 0,
                                "delta_judge_abs": abs(int(row[f"delta_{row['tau_driver']}"])) if row["tau_driver"] in ATTR_CODES else 0,
                            }
                        )

            valid = eval_df[eval_df["choice_ok"] & eval_df["premise_ok"]].copy()
            valid = _annotate_consistency(valid)
            theme_frames.append(valid)
            premise_row_summaries.append(
                {
                    "theme": theme,
                    "objective": meta["objective"],
                    "family": spec.family,
                    "effort": spec.effort,
                    "model_tag": spec.tag,
                    **_summarize_rows(valid),
                }
            )

        theme_all = pd.concat(theme_frames, ignore_index=True)
        all_frames.append(theme_all)
        if theme in MAIN_THEMES:
            substantive_frames.append(theme_all)
        attr_rows = _attr_summary(theme_all, meta)
        worst = _worst_attr(attr_rows)
        premise_theme_aggregates.append(
            {
                "theme": theme,
                "objective": meta["objective"],
                **_summarize_rows(theme_all),
                "worst_attr_code": worst["attr_code"] if worst else None,
                "worst_attr_label": worst["attr_label"] if worst else None,
                "worst_attr_contradiction_rate_non_tied": (
                    worst["contradiction_rate_non_tied"] if worst else None
                ),
                "worst_attr_n_non_tied": worst["n_non_tied"] if worst else None,
                "attribute_rows": attr_rows,
            }
        )

    for theme in MAIN_THEMES:
        meta = theme_meta[theme]
        top_attr_code, top_attr_count = _top_attr(per_theme_stagea_top[theme])
        actor_linear_pair, actor_linear_count = _top_pair(per_theme_actor_linear[theme])
        actor_judge_pair, actor_judge_count = _top_pair(per_theme_actor_judge[theme])
        linear_judge_pair, linear_judge_count = _top_pair(per_theme_linear_judge[theme])
        ljw_attr, ljw_count = _top_attr(per_theme_linear_correct_judge_wrong[theme])
        jlw_attr, jlw_count = _top_attr(per_theme_judge_correct_linear_wrong[theme])
        theme_cond = pd.DataFrame([row for row in conditional_rows if row["theme"] == theme])
        worst_linear_match = theme_cond.loc[theme_cond["linear_matches_actor_rate"].idxmin()].to_dict()
        worst_judge_match = theme_cond.loc[theme_cond["judge_matches_actor_rate"].idxmin()].to_dict()
        aggregated_theme_rows.append(
            {
                "theme": theme,
                "objective": meta["objective"],
                "E_label": meta["E_label"],
                "A_label": meta["A_label"],
                "S_label": meta["S_label"],
                "D_label": meta["D_label"],
                "most_common_top_attr_code": top_attr_code,
                "most_common_top_attr_label": _label(meta, top_attr_code),
                "most_common_top_attr_count": top_attr_count,
                "top_actor_vs_linear_pair": (
                    f"{_label(meta, actor_linear_pair[0])} -> {_label(meta, actor_linear_pair[1])}"
                    if actor_linear_pair else None
                ),
                "top_actor_vs_linear_pair_code": ">".join(actor_linear_pair) if actor_linear_pair else None,
                "top_actor_vs_linear_pair_count": actor_linear_count,
                "top_actor_vs_judge_pair": (
                    f"{_label(meta, actor_judge_pair[0])} -> {_label(meta, actor_judge_pair[1])}"
                    if actor_judge_pair else None
                ),
                "top_actor_vs_judge_pair_code": ">".join(actor_judge_pair) if actor_judge_pair else None,
                "top_actor_vs_judge_pair_count": actor_judge_count,
                "top_linear_vs_judge_pair": (
                    f"{_label(meta, linear_judge_pair[0])} -> {_label(meta, linear_judge_pair[1])}"
                    if linear_judge_pair else None
                ),
                "top_linear_vs_judge_pair_count": linear_judge_count,
                "linear_correct_judge_wrong_top_actor_attr": _label(meta, ljw_attr),
                "linear_correct_judge_wrong_top_actor_attr_count": ljw_count,
                "judge_correct_linear_wrong_top_actor_attr": _label(meta, jlw_attr),
                "judge_correct_linear_wrong_top_actor_attr_count": jlw_count,
                "worst_linear_match_attr": worst_linear_match["attr_label"],
                "worst_linear_match_rate": worst_linear_match["linear_matches_actor_rate"],
                "worst_judge_match_attr": worst_judge_match["attr_label"],
                "worst_judge_match_rate": worst_judge_match["judge_matches_actor_rate"],
            }
        )

    prefix = reports_dir / "FINAL_ATTRIBUTE_LEVEL_RESULTS"
    conditional_path = prefix.with_name(prefix.name + "_CONDITIONALS").with_suffix(".csv")
    mismatch_path = prefix.with_name(prefix.name + "_MISMATCHES").with_suffix(".csv")
    with conditional_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "theme",
                "objective",
                "family",
                "effort",
                "attr_code",
                "attr_label",
                "actor_share",
                "linear_share",
                "judge_share",
                "n_actor_attr",
                "linear_matches_actor_rate",
                "judge_matches_actor_rate",
                "linear_choice_accuracy_given_actor_attr",
                "judge_choice_accuracy_given_actor_attr",
            ],
        )
        writer.writeheader()
        for row in conditional_rows:
            writer.writerow(row)

    with mismatch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "theme",
                "objective",
                "family",
                "effort",
                "comparison",
                "from_attr_code",
                "from_attr_label",
                "to_attr_code",
                "to_attr_label",
                "count",
                "share_of_mismatches",
                "n_mismatches",
            ],
        )
        writer.writeheader()
        for row in mismatch_rows:
            writer.writerow(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "theme_meta": theme_meta_rows,
        "stagea_rows": stagea_rows,
        "aggregated_theme_rows": aggregated_theme_rows,
        "conditional_rows": conditional_rows,
        "mismatch_rows": mismatch_rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    prefix.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Attribute-Level Results Summary",
                "",
                "This report surfaces attribute-specific structure that is not visible in the headline score tables alone.",
                "",
                "## Theme Metadata",
                _markdown_table(theme_meta_rows, ["theme", "objective", "E_label", "A_label", "S_label", "D_label"]),
                "",
                "## Theme-Level Hotspots",
                _markdown_table(
                    aggregated_theme_rows,
                    [
                        "theme",
                        "most_common_top_attr_label",
                        "top_actor_vs_linear_pair",
                        "top_actor_vs_judge_pair",
                        "linear_correct_judge_wrong_top_actor_attr",
                        "judge_correct_linear_wrong_top_actor_attr",
                        "worst_linear_match_attr",
                        "worst_linear_match_rate",
                        "worst_judge_match_attr",
                        "worst_judge_match_rate",
                    ],
                ),
                "",
                "## Stage A Weight Orders",
                _markdown_table(
                    stagea_rows,
                    ["theme", "family", "effort", "top_attr_label", "weight_order", "weight_E", "weight_A", "weight_S", "weight_D"],
                ),
                "",
                "## Files",
                f"- `{conditional_path.relative_to(ROOT)}`",
                f"- `{mismatch_path.relative_to(ROOT)}`",
                f"- `{(reports_dir / 'FINAL_ATTRIBUTE_LEVEL_EXEMPLARS.md').relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )

    aggregate_lookup = {row["theme"]: row for row in aggregated_theme_rows}

    exemplar_rows: list[dict[str, Any]] = []
    for theme in MAIN_THEMES:
        meta = theme_meta[theme]
        wanted_linear = tuple((aggregate_lookup[theme]["top_actor_vs_linear_pair_code"] or "").split(">")) if aggregate_lookup[theme]["top_actor_vs_linear_pair_code"] else ()
        wanted_judge = tuple((aggregate_lookup[theme]["top_actor_vs_judge_pair_code"] or "").split(">")) if aggregate_lookup[theme]["top_actor_vs_judge_pair_code"] else ()
        theme_cases = exemplar_candidates[theme]

        def choose(mode: str, target_pair: tuple[str, str]) -> dict[str, Any] | None:
            rows = [row for row in theme_cases if row["mode"] == mode]
            if not rows:
                return None
            if target_pair:
                if mode == "linear_correct_judge_wrong":
                    pair_rows = [
                        row for row in rows
                        if (row["actor_attr"], row["linear_attr"]) == target_pair and row["actor_attr"] != row["linear_attr"]
                    ]
                else:
                    pair_rows = [
                        row for row in rows
                        if (row["actor_attr"], row["judge_attr"]) == target_pair and row["actor_attr"] != row["judge_attr"]
                    ]
                if pair_rows:
                    rows = pair_rows
            if mode == "linear_correct_judge_wrong":
                differing = [row for row in rows if row["actor_attr"] != row["linear_attr"]]
            else:
                differing = [row for row in rows if row["actor_attr"] != row["judge_attr"]]
            if differing:
                rows = differing
            rows = sorted(
                rows,
                key=lambda row: (
                    max(row["delta_actor_abs"], row["delta_linear_abs"], row["delta_judge_abs"]),
                    row["family"],
                    row["effort"],
                    row["trial_id"],
                ),
                reverse=True,
            )
            return rows[0]

        for mode, target_pair in [
            ("linear_correct_judge_wrong", wanted_linear),
            ("judge_correct_linear_wrong", wanted_judge),
        ]:
            row = choose(mode, target_pair)
            if row is None:
                continue
            if mode == "linear_correct_judge_wrong":
                why = (
                    f"Linear model is correct while judge is wrong. This illustrates the common "
                    f"`{_label(meta, row['actor_attr'])} -> {_label(meta, row['linear_attr'])}` actor-vs-linear substitution."
                )
            else:
                why = (
                    f"Judge is correct while linear model is wrong. This highlights how the judge shifts toward "
                    f"`{_label(meta, row['judge_attr'])}` while the actor states `{_label(meta, row['actor_attr'])}`."
                )
            exemplar_rows.append(
                {
                    "theme": theme,
                    "family": row["family"],
                    "effort": row["effort"],
                    "trial_id": row["trial_id"],
                    "visible_deltas": row["visible_deltas"],
                    "actor": f"choice `{row['actor_choice']}`; factor `{_label(meta, row['actor_attr'])}`",
                    "linear_model": f"choice `{row['linear_choice']}`; factor `{_label(meta, row['linear_attr'])}`",
                    "judge": f"choice `{row['judge_choice']}`; factor `{_label(meta, row['judge_attr'])}`",
                    "why_this_case_matters": why,
                }
            )

    (reports_dir / "FINAL_ATTRIBUTE_LEVEL_EXEMPLARS.md").write_text(
        "\n".join(
            [
                "# Attribute-Level Exemplars",
                "",
                "This table gives concrete disagreement cases from the final same-order benchmark.",
                "",
                "## Case Table",
                _markdown_table(
                    exemplar_rows,
                    ["theme", "family", "effort", "trial_id", "visible_deltas", "actor", "linear_model", "judge", "why_this_case_matters"],
                ),
            ]
        )
        + "\n"
    )

    all_rows = pd.concat(all_frames, ignore_index=True)
    substantive_rows = pd.concat(substantive_frames, ignore_index=True)
    overall = _summarize_rows(all_rows)
    substantive = _summarize_rows(substantive_rows)
    premise_csv = reports_dir / "FINAL_PREMISE_CHOICE_CONSISTENCY.csv"
    with premise_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "theme",
                "objective",
                "family",
                "effort",
                "model_tag",
                "n_choice_premise_ok",
                "n_non_tied",
                "n_support",
                "n_contradict",
                "n_tied",
                "support_rate_all",
                "support_rate_non_tied",
                "contradiction_rate_non_tied",
                "tied_rate",
            ],
        )
        writer.writeheader()
        for row in premise_row_summaries:
            writer.writerow(row)

    worst_theme_attrs = []
    for row in premise_theme_aggregates:
        if row["theme"] not in MAIN_THEMES:
            continue
        worst_theme_attrs.append(
            {
                "theme": row["theme"],
                "worst_attr_label": row["worst_attr_label"],
                "worst_attr_contradiction_rate_non_tied": row["worst_attr_contradiction_rate_non_tied"],
                "worst_attr_n_non_tied": row["worst_attr_n_non_tied"],
            }
        )
    worst_cases = sorted(
        [row for row in premise_row_summaries if row["theme"] in MAIN_THEMES and row["n_non_tied"] > 0],
        key=lambda row: (row["contradiction_rate_non_tied"], row["n_non_tied"]),
        reverse=True,
    )[:6]

    (reports_dir / "FINAL_PREMISE_CHOICE_CONSISTENCY.md").write_text(
        "\n".join(
            [
                "# Premise-Choice Consistency",
                "",
                "This report checks whether the actor's stated premise attribute actually favors the chosen option on the final same-order benchmark test split.",
                "",
                f"- all themes: support on `{overall['support_rate_non_tied']:.3f}` of non-tied rows; contradiction on `{overall['contradiction_rate_non_tied']:.3f}`; tie rate `{overall['tied_rate']:.3f}`",
                f"- substantive themes only: support on `{substantive['support_rate_non_tied']:.3f}` of non-tied rows; contradiction on `{substantive['contradiction_rate_non_tied']:.3f}`; tie rate `{substantive['tied_rate']:.3f}`",
                "",
                "## Theme Aggregates",
                _markdown_table(
                    [row for row in premise_theme_aggregates if row["theme"] in MAIN_THEMES],
                    ["theme", "support_rate_non_tied", "contradiction_rate_non_tied", "tied_rate", "worst_attr_label", "worst_attr_contradiction_rate_non_tied"],
                ),
                "",
                "## Worst Attributes",
                _markdown_table(
                    worst_theme_attrs,
                    ["theme", "worst_attr_label", "worst_attr_contradiction_rate_non_tied", "worst_attr_n_non_tied"],
                ),
                "",
                "## Weakest Theme/Model Cases",
                _markdown_table(
                    worst_cases,
                    ["theme", "family", "effort", "contradiction_rate_non_tied", "tied_rate", "n_non_tied"],
                ),
                "",
                "## File",
                f"- `{premise_csv.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )

    return conditional_path, mismatch_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the artifact-era analysis bundle against the final same-order benchmark.")
    parser.add_argument("--out-root", default=str(output_root()))
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    rep_dir = reports_root(out_root)
    rep_dir.mkdir(parents=True, exist_ok=True)

    main_payload = json.loads((rep_dir / "FINAL_MAIN_RESULTS.json").read_text())
    main_boot_payload = json.loads((rep_dir / "FINAL_MAIN_RESULTS_BOOTSTRAP.json").read_text())
    placebo_payload = json.loads((rep_dir / "FINAL_PLACEBO_RESULTS.json").read_text())
    placebo_boot_payload = json.loads((rep_dir / "FINAL_PLACEBO_RESULTS_BOOTSTRAP.json").read_text())

    _write_results_summary(
        out_path=rep_dir / "FINAL_RESULTS_SUMMARY.md",
        main_rows=main_payload["rows"],
        main_boot_rows=main_boot_payload["rows"],
        placebo_rows=placebo_payload["rows"],
        placebo_boot_rows=placebo_boot_payload["rows"],
    )

    conditional_path, mismatch_path = _build_attribute_reports(base=out_root, reports_dir=rep_dir)

    _run(
        [
            sys.executable,
            "scripts/plot_attribute_match_rates.py",
            "--input",
            str(conditional_path),
            "--out-prefix",
            str(rep_dir / "FINAL_ATTRIBUTE_MATCH_RATES"),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/plot_attribute_match_rates.py",
            "--input",
            str(conditional_path),
            "--out-prefix",
            str(rep_dir / "FINAL_ATTRIBUTE_MATCH_RATES_NOTITLE"),
            "--no-header",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/plot_attribute_confusion_heatmaps.py",
            "--mismatches",
            str(mismatch_path),
            "--conditionals",
            str(conditional_path),
            "--out-prefix",
            str(rep_dir / "FINAL_ATTRIBUTE_CONFUSION_HEATMAPS_MAIN"),
            "--disable-expected-checks",
        ]
    )

    print(f"wrote {rep_dir / 'FINAL_RESULTS_SUMMARY.md'}")
    print(f"wrote {rep_dir / 'FINAL_ATTRIBUTE_LEVEL_RESULTS.md'}")
    print(f"wrote {rep_dir / 'FINAL_ATTRIBUTE_LEVEL_EXEMPLARS.md'}")
    print(f"wrote {rep_dir / 'FINAL_PREMISE_CHOICE_CONSISTENCY.md'}")


if __name__ == "__main__":
    main()
