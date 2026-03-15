#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

from src.analysis.canonical_balanced import (  # noqa: E402
    ARTIFACT_ROOT,
    MODEL_MAP,
    MAIN_MODEL_TAGS,
    NON_PLACEBO_THEMES,
    balanced_dataset_dir,
    balanced_test_run_prefix,
    balanced_train_stagea_summary,
    build_balanced_eval_frame,
    resolve_run_dir,
)


THEME_CONFIGS: dict[str, Path] = {
    "drugs": ROOT / "configs/themes/drugs.yml",
    "policy": ROOT / "configs/themes/policy_intervention.yml",
    "software": ROOT / "configs/themes/software_library.yml",
    "placebo_packaging": ROOT / "configs/themes/drugs_placebo_packaging.yml",
    "placebo_label_border": ROOT / "configs/themes/drugs_placebo_label_border.yml",
}

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


def _theme_meta(theme: str) -> dict[str, str]:
    payload = yaml.safe_load(THEME_CONFIGS[theme].read_text())
    attrs = payload["attributes"]
    return {
        "theme": theme,
        "objective": payload["objective"],
        "E_label": attrs["E"]["label"],
        "A_label": attrs["A"]["label"],
        "S_label": attrs["S"]["label"],
        "D_label": attrs["D"]["label"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute-level story report for the canonical short-reason benchmark.")
    parser.add_argument(
        "--artifact-root",
        default=str(ARTIFACT_ROOT),
        help="Artifact root containing the stage43 outputs.",
    )
    parser.add_argument(
        "--stage-root",
        default=str(ARTIFACT_ROOT / "stage43_holdout80_canonical_balanced_short_reason"),
        help="Stage root containing the stage43 balanced short-reason runs/results.",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(ARTIFACT_ROOT / "reports/ATTRIBUTE_LEVEL_RESULTS_20260315"),
        help="Output prefix for the CSV/JSON/MD summary files.",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    stage_root = Path(args.stage_root).resolve()
    out_prefix = Path(args.out_prefix).resolve()

    theme_meta_rows = [_theme_meta(theme) for theme in NON_PLACEBO_THEMES]
    theme_meta = {row["theme"]: row for row in theme_meta_rows}

    conditional_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    stagea_rows: list[dict[str, Any]] = []
    aggregated_theme_rows: list[dict[str, Any]] = []

    per_theme_stagea_top: dict[str, Counter[str]] = defaultdict(Counter)
    per_theme_actor_linear: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_actor_judge: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_linear_judge: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_theme_linear_correct_judge_wrong: dict[str, Counter[str]] = defaultdict(Counter)
    per_theme_judge_correct_linear_wrong: dict[str, Counter[str]] = defaultdict(Counter)

    for theme in NON_PLACEBO_THEMES:
        meta = theme_meta[theme]
        for model_tag in MAIN_MODEL_TAGS:
            family, effort = MODEL_MAP[model_tag]
            dataset_dir = balanced_dataset_dir(theme, "test", artifact_root=artifact_root, stage_root=stage_root)
            stagea_path = balanced_train_stagea_summary(theme, model_tag, stage_root=stage_root)
            run_dir = resolve_run_dir(balanced_test_run_prefix(theme, model_tag, stage_root=stage_root))
            if run_dir is None:
                raise SystemExit(f"Missing run dir for {theme}/{model_tag}")
            responses_path = run_dir / "responses.jsonl"
            eval_df, _, stagea_summary = build_balanced_eval_frame(
                dataset_dir=dataset_dir,
                responses_path=responses_path,
                stagea_summary_path=stagea_path,
            )

            weights = {code: float(stagea_summary["weights"][code]) for code in ATTR_CODES}
            top_attr = max(ATTR_CODES, key=lambda code: weights[code])
            per_theme_stagea_top[theme][top_attr] += 1
            stagea_rows.append(
                {
                    "theme": theme,
                    "objective": meta["objective"],
                    "family": family,
                    "effort": effort,
                    "top_attr_code": top_attr,
                    "top_attr_label": _label(meta, top_attr),
                    "weight_order": _weight_order(weights),
                    **{f"weight_{code}": weights[code] for code in ATTR_CODES},
                }
            )

            actor_rows = eval_df[eval_df["premise_ok"] & eval_df["choice_ok"]].copy()
            judge_rows = eval_df[eval_df["premise_ok"] & eval_df["tau_ok"]].copy()
            shared_rows = eval_df[eval_df["premise_ok"] & eval_df["choice_ok"] & eval_df["tau_ok"]].copy()
            linear_choice_rows = eval_df[eval_df["choice_ok"]].copy()
            judge_choice_rows = eval_df[eval_df["choice_ok"] & eval_df["tau_ok"]].copy()

            for attr in ATTR_CODES:
                actor_attr_rows = actor_rows[actor_rows["premise_attr"] == attr].copy()
                cond_row = {
                    "theme": theme,
                    "objective": meta["objective"],
                    "family": family,
                    "effort": effort,
                    "attr_code": attr,
                    "attr_label": _label(meta, attr),
                    "actor_share": _safe_rate(actor_rows["premise_attr"] == attr),
                    "linear_share": _safe_rate(actor_rows["linear_model_factor"] == attr),
                    "judge_share": _safe_rate(judge_rows["tau_driver"] == attr),
                    "n_actor_attr": int(len(actor_attr_rows)),
                    "linear_matches_actor_rate": _safe_rate(actor_attr_rows["linear_model_factor"] == attr),
                    "judge_matches_actor_rate": _safe_rate(
                        judge_rows[judge_rows["premise_attr"] == attr]["tau_driver"] == attr
                    ),
                    "linear_choice_accuracy_given_actor_attr": _safe_rate(
                        actor_attr_rows["linear_model_pred_choice"] == actor_attr_rows["choice"]
                    ),
                    "judge_choice_accuracy_given_actor_attr": _safe_rate(
                        judge_rows[judge_rows["premise_attr"] == attr]["tau_pred_choice"]
                        == judge_rows[judge_rows["premise_attr"] == attr]["choice"]
                    ),
                }
                conditional_rows.append(cond_row)

            actor_linear_mismatch = actor_rows[
                actor_rows["premise_attr"].notna() & actor_rows["linear_model_factor"].notna()
                & (actor_rows["premise_attr"] != actor_rows["linear_model_factor"])
            ]
            actor_judge_mismatch = judge_rows[
                judge_rows["premise_attr"].notna() & judge_rows["tau_driver"].notna()
                & (judge_rows["premise_attr"] != judge_rows["tau_driver"])
            ]
            linear_judge_mismatch = shared_rows[
                shared_rows["linear_model_factor"].notna() & shared_rows["tau_driver"].notna()
                & (shared_rows["linear_model_factor"] != shared_rows["tau_driver"])
            ]

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
                            "family": family,
                            "effort": effort,
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
            ]
            judge_right_linear_wrong = shared_rows[
                (shared_rows["tau_pred_choice"] == shared_rows["choice"])
                & (shared_rows["linear_model_pred_choice"] != shared_rows["choice"])
            ]
            per_theme_linear_correct_judge_wrong[theme].update(linear_right_judge_wrong["premise_attr"].dropna().tolist())
            per_theme_judge_correct_linear_wrong[theme].update(judge_right_linear_wrong["premise_attr"].dropna().tolist())

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
                "top_actor_vs_linear_pair_count": actor_linear_count,
                "top_actor_vs_judge_pair": (
                    f"{_label(meta, actor_judge_pair[0])} -> {_label(meta, actor_judge_pair[1])}"
                    if actor_judge_pair else None
                ),
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

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    conditional_columns = [
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
    ]
    with out_prefix.with_name(out_prefix.name + "_CONDITIONALS").with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=conditional_columns)
        writer.writeheader()
        for row in conditional_rows:
            writer.writerow(row)

    mismatch_columns = [
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
    ]
    with out_prefix.with_name(out_prefix.name + "_MISMATCHES").with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mismatch_columns)
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
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Attribute-Level Results Summary",
        "",
        "This report surfaces attribute-specific structure that is not visible in the headline score tables alone.",
        "",
        "## Theme Metadata",
        _markdown_table(
            theme_meta_rows,
            ["theme", "objective", "E_label", "A_label", "S_label", "D_label"],
        ),
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
        "## Notes",
        "- `top_actor_vs_linear_pair` is the most common attribute mismatch between the actor's stated factor and the linear-model trial factor.",
        "- `top_actor_vs_judge_pair` is the most common mismatch between the actor's stated factor and the judge's trial factor.",
        "- `linear_correct_judge_wrong_top_actor_attr` identifies which actor-stated attribute most often appears when the linear model predicts the actor's choice correctly but the judge does not.",
        "- `worst_*_match_attr` is computed from conditional rates given the actor's stated attribute.",
        "",
        "## Files",
        f"- `{out_prefix.with_name(out_prefix.name + '_CONDITIONALS').with_suffix('.csv').relative_to(ROOT)}`",
        f"- `{out_prefix.with_name(out_prefix.name + '_MISMATCHES').with_suffix('.csv').relative_to(ROOT)}`",
    ]
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines))

    print(f"wrote {out_prefix.with_suffix('.md')}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_CONDITIONALS').with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_MISMATCHES').with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
