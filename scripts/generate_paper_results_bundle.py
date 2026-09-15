#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import transforms as mtransforms

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    }
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_balanced import build_balanced_eval_frame
from src.analysis.final_benchmark import (
    MAIN_THEMES,
    MODEL_SPECS,
    PLACEBO_THEMES,
    THEME_CONFIGS,
    output_root,
    reports_root,
    resolve_run_dir,
    run_prefix,
    stagea_dir,
)
from src.analysis.judge_baselines import add_pairwise_drivers


SUBSTANTIVE_THEMES = tuple(MAIN_THEMES)
PLACEBO_THEMES_ORDER = tuple(PLACEBO_THEMES)
PLACEBO_ATTR = "D"
ATTR_CODES = ("E", "A", "S", "D")
BOOTSTRAP_DEFAULT = 500
SEED_DEFAULT = 17

THEME_DISPLAY = {
    "drugs": "Drugs",
    "policy": "Policy",
    "software": "Software",
    "placebo_packaging": "Packaging Symmetry",
    "placebo_label_border": "Label Border Thickness",
}

THEME_COLORS = {
    "drugs": "#3b6fb6",
    "policy": "#c26d2d",
    "software": "#4d8b5b",
}

FIGURE_METHOD_COLORS = {
    "drop_choice": "#4a74a8",
    "equalize_choice": "#86a9d5",
    "drop_premise": "#b4553d",
    "equalize_premise": "#d7957d",
}

TABLE1_SUBSTANTIVE_METRICS = (
    "heldout_choice_prediction",
    "self_report_revealed_driver",
    "score_judge_revealed_driver",
    "pairwise_top_driver_revealed_driver",
    "pairwise_mirror_consistency",
)
TABLE1_SUBSTANTIVE_METRICS_NO_PAIRWISE = TABLE1_SUBSTANTIVE_METRICS[:3]

FIGURE1_CHOICE_METRICS = (
    "heldout_choice_prediction",
    "score_judge_choice_agreement_with_latent_choice",
)

FIGURE1_DRIVER_METRICS = (
    "self_report_revealed_driver",
    "score_judge_revealed_driver",
)

TABLE1_PLACEBO_METRICS = (
    "actor_names_placebo",
    "revealed_driver_is_placebo",
    "score_judge_top_driver_is_placebo",
)

SUBSTANTIVE_GRID_METRICS = (
    "heldout_choice_prediction",
    "score_judge_predicts_actor_choice",
    "self_report_revealed_driver",
    "score_judge_revealed_driver",
    "pairwise_top_driver_revealed_driver",
    "pairwise_mirror_consistency",
)
SUBSTANTIVE_GRID_METRICS_NO_PAIRWISE = SUBSTANTIVE_GRID_METRICS[:4]

PLACEBO_GRID_METRICS = (
    "actor_names_placebo",
    "revealed_driver_is_placebo",
    "score_judge_top_driver_is_placebo",
    "pairwise_top_driver_is_placebo",
)
PLACEBO_GRID_METRICS_NO_PAIRWISE = PLACEBO_GRID_METRICS[:3]

PAIRWISE_DIAG_METRICS = (
    "pairwise_top_driver_revealed_driver",
    "pairwise_mirror_consistency",
    "pairwise_cycle_rate",
    "pairwise_tie_rate",
    "pairwise_step_parse_ok_rate",
)

FAMILY_EXPECTED_DRAWS = 12

FAMILY_CONSISTENCY_METRICS = (
    "actor_choice_family_complete_rate",
    "actor_choice_family_consistency_rate",
    "actor_choice_family_modal_share",
    "self_report_driver_family_complete_rate",
    "self_report_driver_family_consistency_rate",
    "self_report_driver_family_modal_share",
    "score_judge_choice_family_complete_rate",
    "score_judge_choice_family_consistency_rate",
    "score_judge_choice_family_modal_share",
    "score_judge_driver_family_complete_rate",
    "score_judge_driver_family_consistency_rate",
    "score_judge_driver_family_modal_share",
)

FAMILY_CONSISTENCY_DISPLAY_METRICS = (
    "actor_choice_family_consistency_rate",
    "self_report_driver_family_consistency_rate",
    "score_judge_choice_family_consistency_rate",
    "score_judge_driver_family_consistency_rate",
)

FAMILY_PAIRWISE_AGREEMENT_METRICS = (
    "actor_choice_family_pairwise_agreement",
    "self_report_driver_family_pairwise_agreement",
    "score_judge_choice_family_pairwise_agreement",
    "score_judge_driver_family_pairwise_agreement",
)

FAMILY_MAJORITY_CORRECTNESS_METRICS = (
    "actor_choice_family_majority_correct_rate",
    "self_report_driver_family_majority_correct_rate",
    "score_judge_choice_family_majority_correct_rate",
    "score_judge_driver_family_majority_correct_rate",
)

FAMILY_ALL_METRICS = (
    FAMILY_CONSISTENCY_METRICS
    + FAMILY_PAIRWISE_AGREEMENT_METRICS
    + FAMILY_MAJORITY_CORRECTNESS_METRICS
)

MODEL_TAG_ORDER = tuple(spec.tag for spec in MODEL_SPECS)

FIGURE1_METHOD_STYLES = {
    "heldout_choice_prediction": {
        "label": "Actor choice vs latent choice",
        "color": "#2f5d8a",
        "marker": "o",
    },
    "score_judge_choice_agreement_with_latent_choice": {
        "label": "Score judge choice vs latent choice",
        "color": "#c26d2d",
        "marker": "s",
    },
    "self_report_revealed_driver": {
        "label": "Actor self-report vs revealed driver",
        "color": "#2f5d8a",
        "marker": "o",
    },
    "score_judge_revealed_driver": {
        "label": "Score judge top driver vs revealed driver",
        "color": "#c26d2d",
        "marker": "s",
    },
    "pairwise_top_driver_revealed_driver": {
        "label": "Pairwise top driver -> revealed driver",
        "color": "#4d8b5b",
        "marker": "^",
    },
}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _safe_mean(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _percentile_ci(values: list[float | None]) -> tuple[float | None, float | None]:
    cleaned = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not cleaned:
        return None, None
    lo, hi = np.percentile(np.asarray(cleaned, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(point: float | None, lo: float | None, hi: float | None) -> str:
    if point is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{point:.3f}"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2))


def _theme_meta(theme: str) -> dict[str, Any]:
    payload = yaml.safe_load(THEME_CONFIGS[theme].read_text())
    return {
        "theme": theme,
        "display_name": THEME_DISPLAY[theme],
        "objective": payload["objective"],
        "attr_labels": {code: payload["attributes"][code]["label"] for code in ATTR_CODES},
        "placebo_variant": payload["attributes"][PLACEBO_ATTR]["label"],
    }


def _build_run_eval(theme: str, model_tag: str, kind: str, *, out_root: Path) -> tuple[pd.DataFrame, Path]:
    run_dir = resolve_run_dir(run_prefix(theme, "test", model_tag, kind, base=out_root))
    if run_dir is None:
        raise SystemExit(f"Missing run dir for {theme}/{model_tag}/{kind}")
    responses_path = run_dir / "responses.jsonl"
    eval_df, _, _ = build_balanced_eval_frame(
        dataset_dir=out_root / "datasets" / theme / "test",
        responses_path=responses_path,
        stagea_summary_path=stagea_dir(theme, model_tag, base=out_root) / "stageA_summary.json",
    )
    return eval_df, responses_path


def _bool_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(bool)


def _match_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left == right).fillna(False).astype(bool)


def _tau_group_frame(eval_df: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "config_id": eval_df["config_id"].astype(str),
            "n_rows": 1,
            "choice_correct": (
                _bool_series(eval_df["choice_ok"])
                & _match_series(eval_df["linear_model_pred_choice"], eval_df["choice"])
            ).astype(int),
            "self_report_revealed_driver": (
                _bool_series(eval_df["premise_ok"])
                & _match_series(eval_df["premise_attr"], eval_df["linear_model_factor"])
            ).astype(int),
            "score_judge_revealed_driver": (
                _bool_series(eval_df["tau_ok"])
                & _match_series(eval_df["tau_driver"], eval_df["linear_model_factor"])
            ).astype(int),
            "score_judge_predicts_actor_choice": (
                _bool_series(eval_df["tau_ok"])
                & _match_series(eval_df["tau_pred_choice"], eval_df["choice"])
            ).astype(int),
            "score_judge_choice_agreement_with_latent_choice": (
                _bool_series(eval_df["tau_ok"])
                & _match_series(eval_df["tau_pred_choice"], eval_df["linear_model_pred_choice"])
            ).astype(int),
            "actor_names_placebo": (
                _bool_series(eval_df["premise_ok"])
                & eval_df["premise_attr"].eq(PLACEBO_ATTR)
            ).astype(int),
            "revealed_driver_is_placebo": eval_df["linear_model_factor"].eq(PLACEBO_ATTR).fillna(False).astype(int),
            "score_judge_top_driver_is_placebo": (
                _bool_series(eval_df["tau_ok"])
                & eval_df["tau_driver"].eq(PLACEBO_ATTR)
            ).astype(int),
        }
    )
    return work.groupby("config_id", as_index=False).sum()


def _pair_group_frame(eval_df: pd.DataFrame) -> pd.DataFrame:
    mirror_complete = _bool_series(eval_df["pairwise_mirror_complete"])
    mirror_consistent = _bool_series(eval_df["pairwise_mirror_consistent"])
    pairwise_ok = _bool_series(eval_df["pairwise_ok"])
    has_cycle = _bool_series(eval_df["pairwise_has_cycle"])
    work = pd.DataFrame(
        {
            "config_id": eval_df["config_id"].astype(str),
            "n_rows": 1,
            "pairwise_top_driver_revealed_driver": _match_series(
                eval_df["pair_driver"], eval_df["linear_model_factor"]
            ).astype(int),
            "pairwise_top_driver_is_placebo": eval_df["pair_driver"].eq(PLACEBO_ATTR).fillna(False).astype(int),
            "mirror_complete_n": mirror_complete.astype(int),
            "mirror_consistent_n": (mirror_complete & mirror_consistent).astype(int),
            "pairwise_ok_n": pairwise_ok.astype(int),
            "cycle_n": (pairwise_ok & has_cycle).astype(int),
            "pairwise_tie_pairs": pd.to_numeric(eval_df["pairwise_tie_pairs"], errors="coerce").fillna(0.0),
            "pairwise_parsed_pairs": pd.to_numeric(eval_df["pairwise_parsed_pairs"], errors="coerce").fillna(0.0),
        }
    )
    return work.groupby("config_id", as_index=False).sum()


def _tau_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    return {
        "heldout_choice_prediction": _ratio(totals["choice_correct"], totals["n_rows"]),
        "score_judge_predicts_actor_choice": _ratio(
            totals["score_judge_predicts_actor_choice"], totals["n_rows"]
        ),
        "score_judge_choice_agreement_with_latent_choice": _ratio(
            totals["score_judge_choice_agreement_with_latent_choice"], totals["n_rows"]
        ),
        "self_report_revealed_driver": _ratio(totals["self_report_revealed_driver"], totals["n_rows"]),
        "score_judge_revealed_driver": _ratio(totals["score_judge_revealed_driver"], totals["n_rows"]),
        "actor_names_placebo": _ratio(totals["actor_names_placebo"], totals["n_rows"]),
        "revealed_driver_is_placebo": _ratio(totals["revealed_driver_is_placebo"], totals["n_rows"]),
        "score_judge_top_driver_is_placebo": _ratio(
            totals["score_judge_top_driver_is_placebo"], totals["n_rows"]
        ),
    }


def _pair_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    return {
        "pairwise_top_driver_revealed_driver": _ratio(
            totals["pairwise_top_driver_revealed_driver"], totals["n_rows"]
        ),
        "pairwise_top_driver_is_placebo": _ratio(totals["pairwise_top_driver_is_placebo"], totals["n_rows"]),
        "pairwise_mirror_consistency": _ratio(totals["mirror_consistent_n"], totals["mirror_complete_n"]),
        "pairwise_cycle_rate": _ratio(totals["cycle_n"], totals["pairwise_ok_n"]),
        "pairwise_tie_rate": _ratio(totals["pairwise_tie_pairs"], totals["pairwise_parsed_pairs"]),
    }


def _bootstrap_group_totals(
    group_df: pd.DataFrame,
    *,
    metric_builder: Callable[[dict[str, float]], dict[str, float | None]],
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, float | None], dict[str, list[float | None]], dict[str, float]]:
    value_cols = [column for column in group_df.columns if column != "config_id"]
    arrays = {
        column: group_df[column].to_numpy(dtype=float)
        for column in value_cols
    }
    totals = {column: float(values.sum()) for column, values in arrays.items()}
    point = metric_builder(totals)
    samples = {metric: [] for metric in point}

    n_groups = len(group_df)
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        picks = rng.integers(0, n_groups, size=n_groups)
        sampled_totals = {column: float(values[picks].sum()) for column, values in arrays.items()}
        metrics = metric_builder(sampled_totals)
        for metric, value in metrics.items():
            samples[metric].append(value)

    return point, samples, totals


def _step_parse_stats(responses_path: Path) -> dict[str, float | int | None]:
    total_steps = 0
    ok_steps = 0
    total_responses = 0
    with responses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            for response in payload.get("responses", []):
                total_responses += 1
                for step in response.get("steps", []):
                    if not step.get("name", "").startswith("judge_pair_"):
                        continue
                    total_steps += 1
                    ok_steps += int(bool(step.get("parsed", {}).get("ok", False)))
    return {
        "pairwise_step_parse_ok_rate": _ratio(ok_steps, total_steps),
        "pairwise_step_parse_ok_steps": ok_steps,
        "pairwise_step_parse_total_steps": total_steps,
        "pairwise_step_parse_total_responses": total_responses,
    }


def _attach_ci_fields(
    row: dict[str, Any],
    samples: dict[str, list[float | None]],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    out = dict(row)
    for metric in metrics:
        lo, hi = _percentile_ci(samples.get(metric, []))
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
    return out


def _mean_bootstrap_samples(sample_lists: list[list[float | None]]) -> list[float | None]:
    if not sample_lists:
        return []
    n = min(len(values) for values in sample_lists)
    out: list[float | None] = []
    for idx in range(n):
        values = [samples[idx] for samples in sample_lists]
        out.append(_safe_mean(values))
    return out


def _mode_share(values: pd.Series) -> float | None:
    valid = values.dropna()
    if valid.empty:
        return None
    counts = valid.value_counts(dropna=True)
    if counts.empty:
        return None
    return float(counts.iloc[0] / len(valid))


def _pairwise_agreement(values: pd.Series) -> float | None:
    valid = values.dropna()
    n = int(len(valid))
    if n < 2:
        return None
    counts = valid.value_counts(dropna=True)
    agree_pairs = int(sum(count * (count - 1) for count in counts.to_list()))
    total_pairs = n * (n - 1)
    return float(agree_pairs / total_pairs)


def _canonical_profile(choice_series: pd.Series, slot_a_series: pd.Series, slot_b_series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            choice_series.eq("A"),
            slot_a_series,
            np.where(choice_series.eq("B"), slot_b_series, None),
        ),
        index=choice_series.index,
        dtype="object",
    )


def _aggregate_rows(
    rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, list[float | None]]],
    metrics: tuple[str, ...],
    *,
    label: str,
    selector: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [row for row in rows if selector(row)]
    if not selected:
        raise ValueError(f"No rows selected for aggregate {label}")
    out = {
        "row_label": label,
        "n_cells": len(selected),
    }
    for metric in metrics:
        out[metric] = _safe_mean([row.get(metric) for row in selected])
        agg_samples = _mean_bootstrap_samples([sample_lookup[row["cell_key"]][metric] for row in selected])
        lo, hi = _percentile_ci(agg_samples)
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
    return out


def _substantive_summary_rows(
    rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, list[float | None]]],
    metrics: tuple[str, ...] = TABLE1_SUBSTANTIVE_METRICS,
) -> list[dict[str, Any]]:
    out = []
    for theme in SUBSTANTIVE_THEMES:
        out.append(
            _aggregate_rows(
                rows,
                sample_lookup,
                metrics,
                label=THEME_DISPLAY[theme],
                selector=lambda row, theme=theme: row["theme"] == theme,
            )
        )
    out.append(
        _aggregate_rows(
            rows,
            sample_lookup,
            metrics,
            label="Pooled substantive",
            selector=lambda row: row["theme"] in SUBSTANTIVE_THEMES,
        )
    )
    return out


def _family_consistency_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    return {
        "actor_choice_family_complete_rate": _ratio(totals["actor_choice_complete_n"], totals["n_families"]),
        "actor_choice_family_consistency_rate": _ratio(
            totals["actor_choice_consistent_n"], totals["actor_choice_complete_n"]
        ),
        "actor_choice_family_modal_share": _ratio(
            totals["actor_choice_modal_share_sum"], totals["actor_choice_modal_share_n"]
        ),
        "self_report_driver_family_complete_rate": _ratio(
            totals["self_report_driver_complete_n"], totals["n_families"]
        ),
        "self_report_driver_family_consistency_rate": _ratio(
            totals["self_report_driver_consistent_n"], totals["self_report_driver_complete_n"]
        ),
        "self_report_driver_family_modal_share": _ratio(
            totals["self_report_driver_modal_share_sum"], totals["self_report_driver_modal_share_n"]
        ),
        "score_judge_choice_family_complete_rate": _ratio(
            totals["score_judge_choice_complete_n"], totals["n_families"]
        ),
        "score_judge_choice_family_consistency_rate": _ratio(
            totals["score_judge_choice_consistent_n"], totals["score_judge_choice_complete_n"]
        ),
        "score_judge_choice_family_modal_share": _ratio(
            totals["score_judge_choice_modal_share_sum"], totals["score_judge_choice_modal_share_n"]
        ),
        "score_judge_driver_family_complete_rate": _ratio(
            totals["score_judge_driver_complete_n"], totals["n_families"]
        ),
        "score_judge_driver_family_consistency_rate": _ratio(
            totals["score_judge_driver_consistent_n"], totals["score_judge_driver_complete_n"]
        ),
        "score_judge_driver_family_modal_share": _ratio(
            totals["score_judge_driver_modal_share_sum"], totals["score_judge_driver_modal_share_n"]
        ),
    }


def _family_pairwise_agreement_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    return {
        "actor_choice_family_pairwise_agreement": _ratio(
            totals["actor_choice_pairwise_agreement_sum"], totals["actor_choice_pairwise_agreement_n"]
        ),
        "self_report_driver_family_pairwise_agreement": _ratio(
            totals["self_report_driver_pairwise_agreement_sum"], totals["self_report_driver_pairwise_agreement_n"]
        ),
        "score_judge_choice_family_pairwise_agreement": _ratio(
            totals["score_judge_choice_pairwise_agreement_sum"], totals["score_judge_choice_pairwise_agreement_n"]
        ),
        "score_judge_driver_family_pairwise_agreement": _ratio(
            totals["score_judge_driver_pairwise_agreement_sum"], totals["score_judge_driver_pairwise_agreement_n"]
        ),
    }


def _family_majority_correctness_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    return {
        "actor_choice_family_majority_correct_rate": _ratio(
            totals["actor_choice_majority_correct_n"], totals["actor_choice_complete_n"]
        ),
        "self_report_driver_family_majority_correct_rate": _ratio(
            totals["self_report_driver_majority_correct_n"], totals["self_report_driver_complete_n"]
        ),
        "score_judge_choice_family_majority_correct_rate": _ratio(
            totals["score_judge_choice_majority_correct_n"], totals["score_judge_choice_complete_n"]
        ),
        "score_judge_driver_family_majority_correct_rate": _ratio(
            totals["score_judge_driver_majority_correct_n"], totals["score_judge_driver_complete_n"]
        ),
    }


def _family_all_metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    metrics = {}
    metrics.update(_family_consistency_metrics_from_totals(totals))
    metrics.update(_family_pairwise_agreement_metrics_from_totals(totals))
    metrics.update(_family_majority_correctness_metrics_from_totals(totals))
    return metrics


def _family_consistency_group_frame(
    eval_df: pd.DataFrame,
    *,
    expected_draws: int = FAMILY_EXPECTED_DRAWS,
) -> pd.DataFrame:
    work = eval_df.copy()
    work["family_id"] = work["family_id"].astype(str)
    work["actor_choice_profile"] = _canonical_profile(work["choice"], work["slot_A_profile"], work["slot_B_profile"])
    work["latent_choice_profile"] = _canonical_profile(
        work["linear_model_pred_choice"], work["slot_A_profile"], work["slot_B_profile"]
    )
    work["score_judge_choice_profile"] = _canonical_profile(
        work["tau_pred_choice"], work["slot_A_profile"], work["slot_B_profile"]
    )

    records: list[dict[str, Any]] = []
    for family_id, group in work.groupby("family_id", sort=False):
        total_rows = int(len(group))

        actor_choice_valid = _bool_series(group["choice_ok"]) & group["actor_choice_profile"].notna()
        self_report_driver_valid = _bool_series(group["premise_ok"])
        score_judge_choice_valid = _bool_series(group["tau_ok"]) & group["score_judge_choice_profile"].notna()
        score_judge_driver_valid = _bool_series(group["tau_ok"])

        actor_choice_values = group.loc[actor_choice_valid, "actor_choice_profile"]
        self_report_driver_values = group.loc[self_report_driver_valid, "premise_attr"]
        score_judge_choice_values = group.loc[score_judge_choice_valid, "score_judge_choice_profile"]
        score_judge_driver_values = group.loc[score_judge_driver_valid, "tau_driver"]

        actor_choice_correct = _match_series(
            group.loc[actor_choice_valid, "actor_choice_profile"],
            group.loc[actor_choice_valid, "latent_choice_profile"],
        )
        self_report_driver_correct = _match_series(
            group.loc[self_report_driver_valid, "premise_attr"],
            group.loc[self_report_driver_valid, "linear_model_factor"],
        )
        score_judge_choice_correct = _match_series(
            group.loc[score_judge_choice_valid, "score_judge_choice_profile"],
            group.loc[score_judge_choice_valid, "latent_choice_profile"],
        )
        score_judge_driver_correct = _match_series(
            group.loc[score_judge_driver_valid, "tau_driver"],
            group.loc[score_judge_driver_valid, "linear_model_factor"],
        )

        def summarize(values: pd.Series) -> tuple[int, int, float, int, float, int]:
            valid = values.dropna()
            n_valid = int(len(valid))
            complete = int(total_rows == expected_draws and n_valid == expected_draws)
            consistent = int(bool(complete) and valid.nunique(dropna=True) == 1)
            modal_share = _mode_share(valid)
            pairwise = _pairwise_agreement(valid)
            return (
                complete,
                consistent,
                float(modal_share or 0.0),
                int(modal_share is not None),
                float(pairwise or 0.0),
                int(pairwise is not None),
            )

        def summarize_majority(correct_values: pd.Series) -> int:
            n_valid = int(len(correct_values))
            complete = int(total_rows == expected_draws and n_valid == expected_draws)
            return int(bool(complete) and float(correct_values.mean()) > 0.5)

        (
            actor_choice_complete,
            actor_choice_consistent,
            actor_choice_modal_sum,
            actor_choice_modal_n,
            actor_choice_pairwise_sum,
            actor_choice_pairwise_n,
        ) = summarize(
            actor_choice_values
        )
        (
            self_report_driver_complete,
            self_report_driver_consistent,
            self_report_driver_modal_sum,
            self_report_driver_modal_n,
            self_report_driver_pairwise_sum,
            self_report_driver_pairwise_n,
        ) = summarize(
            self_report_driver_values
        )
        (
            score_judge_choice_complete,
            score_judge_choice_consistent,
            score_judge_choice_modal_sum,
            score_judge_choice_modal_n,
            score_judge_choice_pairwise_sum,
            score_judge_choice_pairwise_n,
        ) = summarize(
            score_judge_choice_values
        )
        (
            score_judge_driver_complete,
            score_judge_driver_consistent,
            score_judge_driver_modal_sum,
            score_judge_driver_modal_n,
            score_judge_driver_pairwise_sum,
            score_judge_driver_pairwise_n,
        ) = summarize(
            score_judge_driver_values
        )

        actor_choice_majority_correct = summarize_majority(actor_choice_correct)
        self_report_driver_majority_correct = summarize_majority(self_report_driver_correct)
        score_judge_choice_majority_correct = summarize_majority(score_judge_choice_correct)
        score_judge_driver_majority_correct = summarize_majority(score_judge_driver_correct)

        records.append(
            {
                "config_id": family_id,
                "n_families": 1,
                "actor_choice_complete_n": actor_choice_complete,
                "actor_choice_consistent_n": actor_choice_consistent,
                "actor_choice_modal_share_sum": actor_choice_modal_sum,
                "actor_choice_modal_share_n": actor_choice_modal_n,
                "actor_choice_pairwise_agreement_sum": actor_choice_pairwise_sum,
                "actor_choice_pairwise_agreement_n": actor_choice_pairwise_n,
                "actor_choice_majority_correct_n": actor_choice_majority_correct,
                "self_report_driver_complete_n": self_report_driver_complete,
                "self_report_driver_consistent_n": self_report_driver_consistent,
                "self_report_driver_modal_share_sum": self_report_driver_modal_sum,
                "self_report_driver_modal_share_n": self_report_driver_modal_n,
                "self_report_driver_pairwise_agreement_sum": self_report_driver_pairwise_sum,
                "self_report_driver_pairwise_agreement_n": self_report_driver_pairwise_n,
                "self_report_driver_majority_correct_n": self_report_driver_majority_correct,
                "score_judge_choice_complete_n": score_judge_choice_complete,
                "score_judge_choice_consistent_n": score_judge_choice_consistent,
                "score_judge_choice_modal_share_sum": score_judge_choice_modal_sum,
                "score_judge_choice_modal_share_n": score_judge_choice_modal_n,
                "score_judge_choice_pairwise_agreement_sum": score_judge_choice_pairwise_sum,
                "score_judge_choice_pairwise_agreement_n": score_judge_choice_pairwise_n,
                "score_judge_choice_majority_correct_n": score_judge_choice_majority_correct,
                "score_judge_driver_complete_n": score_judge_driver_complete,
                "score_judge_driver_consistent_n": score_judge_driver_consistent,
                "score_judge_driver_modal_share_sum": score_judge_driver_modal_sum,
                "score_judge_driver_modal_share_n": score_judge_driver_modal_n,
                "score_judge_driver_pairwise_agreement_sum": score_judge_driver_pairwise_sum,
                "score_judge_driver_pairwise_agreement_n": score_judge_driver_pairwise_n,
                "score_judge_driver_majority_correct_n": score_judge_driver_majority_correct,
            }
        )

    return pd.DataFrame(records)


def _family_metric_summary_rows(
    rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, list[float | None]]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    out = []
    for theme in SUBSTANTIVE_THEMES:
        out.append(
            _aggregate_rows(
                rows,
                sample_lookup,
                metrics,
                label=THEME_DISPLAY[theme],
                selector=lambda row, theme=theme: row["theme"] == theme,
            )
        )
    out.append(
        _aggregate_rows(
            rows,
            sample_lookup,
            metrics,
            label="Pooled substantive",
            selector=lambda row: row["theme"] in SUBSTANTIVE_THEMES,
        )
    )
    return out


def _figure1_summary_rows(
    rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, list[float | None]]],
) -> list[dict[str, Any]]:
    metrics = FIGURE1_CHOICE_METRICS + FIGURE1_DRIVER_METRICS
    out = []
    for theme in SUBSTANTIVE_THEMES:
        out.append(
            _aggregate_rows(
                rows,
                sample_lookup,
                metrics,
                label=THEME_DISPLAY[theme],
                selector=lambda row, theme=theme: row["theme"] == theme,
            )
        )
    out.append(
        _aggregate_rows(
            rows,
            sample_lookup,
            metrics,
            label="Pooled substantive",
            selector=lambda row: row["theme"] in SUBSTANTIVE_THEMES,
        )
    )
    return out


def _placebo_summary_rows(
    rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, list[float | None]]],
    metrics: tuple[str, ...] = TABLE1_PLACEBO_METRICS,
) -> list[dict[str, Any]]:
    out = []
    for theme in PLACEBO_THEMES_ORDER:
        out.append(
            _aggregate_rows(
                rows,
                sample_lookup,
                metrics,
                label=THEME_DISPLAY[theme],
                selector=lambda row, theme=theme: row["theme"] == theme,
            )
        )
    return out


def _pairwise_theme_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for theme in SUBSTANTIVE_THEMES:
        selected = [row for row in rows if row["theme"] == theme]
        out.append(
            {
                "theme": theme,
                "display_name": THEME_DISPLAY[theme],
                "pairwise_cycle_rate": _safe_mean([row.get("pairwise_cycle_rate") for row in selected]),
                "pairwise_tie_rate": _safe_mean([row.get("pairwise_tie_rate") for row in selected]),
                "pairwise_step_parse_ok_rate": _safe_mean(
                    [row.get("pairwise_step_parse_ok_rate") for row in selected]
                ),
            }
        )
    return out


def _summary_display_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rendered = {"row_label": row["row_label"]}
        for metric in metrics:
            rendered[metric] = _fmt_ci(row.get(metric), row.get(f"{metric}_ci_lo"), row.get(f"{metric}_ci_hi"))
        out.append(rendered)
    return out


def _grid_display_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rendered = {
            "theme": THEME_DISPLAY[row["theme"]],
            "family": row["family"],
            "effort": row["effort"],
        }
        for metric in metrics:
            rendered[metric] = _fmt_ci(row.get(metric), row.get(f"{metric}_ci_lo"), row.get(f"{metric}_ci_hi"))
        out.append(rendered)
    return out


def _wrap_label(label: str) -> str:
    return label.replace(" ", "\n") if " " in label else label


def _write_table1_files(
    reports_dir: Path,
    substantive_rows: list[dict[str, Any]],
    placebo_rows: list[dict[str, Any]],
    *,
    substantive_metrics: tuple[str, ...] = TABLE1_SUBSTANTIVE_METRICS,
    placebo_metrics: tuple[str, ...] = TABLE1_PLACEBO_METRICS,
    include_pairwise: bool = True,
) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in substantive_rows:
        csv_rows.append({"panel": "substantive", **row})
    for row in placebo_rows:
        csv_rows.append({"panel": "placebo", **row})

    csv_columns = [
        "panel",
        "row_label",
        "n_cells",
        *[
            column
            for metric in substantive_metrics + placebo_metrics
            for column in (metric, f"{metric}_ci_lo", f"{metric}_ci_hi")
        ],
    ]
    csv_path = reports_dir / "PAPER_TABLE1_SUMMARY.csv"
    _write_csv(csv_path, csv_rows, csv_columns)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "substantive_panel_rows": substantive_rows,
        "placebo_panel_rows": placebo_rows,
        "notes": [
            "Held-out actor-choice accuracy is the share of held-out test prompts where the latent behavioural model predicts the actor's A/B choice correctly.",
            "Explicit columns report recovery of the behaviourally revealed driver.",
            *(
                ["Pairwise mirror consistency is computed separately from pairwise step parse success."]
                if include_pairwise
                else []
            ),
        ],
    }
    _write_json(reports_dir / "PAPER_TABLE1_SUMMARY.json", payload)

    substantive_display = _summary_display_rows(substantive_rows, substantive_metrics)
    placebo_display = _summary_display_rows(placebo_rows, placebo_metrics)
    md_lines = [
        "# Paper Table 1 Summary",
        "",
        "## Panel A. Substantive Themes",
        "",
        _markdown_table(
            substantive_display,
            ["row_label", *substantive_metrics],
        ),
        "",
        "## Panel B. Placebo Falsification",
        "",
        _markdown_table(
            placebo_display,
            ["row_label", *placebo_metrics],
        ),
        "",
        "## Notes",
        "",
        "- `Held-out choice prediction` is held-out actor-choice accuracy for the latent behavioural model.",
        "- The explicit columns report recovery of the behaviourally revealed driver.",
        *(
            ["- `Pairwise mirror consistency` is reported separately from pairwise step parse success."]
            if include_pairwise
            else []
        ),
    ]
    (reports_dir / "PAPER_TABLE1_SUMMARY.md").write_text("\n".join(md_lines) + "\n")


def _write_appendix_grid(
    *,
    reports_dir: Path,
    stem: str,
    payload_rows: list[dict[str, Any]],
    display_rows: list[dict[str, Any]],
    csv_columns: list[str],
    md_columns: list[str],
    title: str,
    notes: list[str],
) -> None:
    _write_csv(reports_dir / f"{stem}.csv", payload_rows, csv_columns)
    _write_json(
        reports_dir / f"{stem}.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": payload_rows,
            "notes": notes,
        },
    )
    md_lines = [f"# {title}", "", *[f"- {note}" for note in notes], "", _markdown_table(display_rows, md_columns)]
    (reports_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n")


def _write_figure(fig: plt.Figure, out_prefix: Path) -> None:
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axis(ax: plt.Axes, *, ylabel: str, ylim: tuple[float, float] = (0.0, 1.0)) -> None:
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def _panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.set_title(title, loc="left", pad=10, fontweight="bold")


def _model_setting_label(row: dict[str, Any]) -> str:
    effort_short = "min" if row["effort"] == "minimal" else row["effort"]
    return f"{row['family']} / {effort_short}"


def _figure1_row_entries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_order = {tag: idx for idx, tag in enumerate(MODEL_TAG_ORDER)}
    entries: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    y = 0

    for theme_idx, theme in enumerate(SUBSTANTIVE_THEMES):
        theme_rows = sorted(
            [row for row in rows if row["theme"] == theme],
            key=lambda row: model_order.get(row["model_tag"], 999),
        )
        cell_start = y
        for row in theme_rows:
            entries.append(
                {
                    "kind": "cell",
                    "theme": theme,
                    "y": y,
                    "label": _model_setting_label(row),
                    "row": row,
                }
            )
            y += 1

        theme_mean_y = y
        entries.append(
            {
                "kind": "theme_mean",
                "theme": theme,
                "y": theme_mean_y,
                "label": "Theme mean",
            }
        )
        y += 1

        blocks.append(
            {
                "theme": theme,
                "cell_start": cell_start,
                "cell_end": theme_mean_y - 1,
                "block_end": theme_mean_y,
                "center": (cell_start + theme_mean_y) / 2.0,
                "theme_mean_y": theme_mean_y,
            }
        )

        if theme_idx < len(SUBSTANTIVE_THEMES) - 1:
            entries.append({"kind": "gap", "y": y, "label": ""})
            y += 1

    entries.append(
        {
            "kind": "pooled",
            "theme": "pooled",
            "y": y,
            "label": "Pooled substantive",
        }
    )
    return entries, blocks


def _plot_figure1_row_panel(
    *,
    ax: plt.Axes,
    row_entries: list[dict[str, Any]],
    theme_blocks: list[dict[str, Any]],
    summary_lookup: dict[str, dict[str, Any]],
    metrics: tuple[str, ...],
    title: str,
    letter: str,
    show_row_labels: bool,
) -> None:
    offsets = np.linspace(-0.18, 0.18, len(metrics)) if len(metrics) > 1 else np.array([0.0])
    pooled_summary = summary_lookup["Pooled substantive"]

    for block_idx, block in enumerate(theme_blocks):
        facecolor = "#f7f7f7" if block_idx % 2 == 0 else "#ffffff"
        ax.axhspan(block["cell_start"] - 0.5, block["block_end"] + 0.5, facecolor=facecolor, zorder=0)
        ax.axhline(block["block_end"] + 0.5, color="#d0d0d0", linewidth=1.0, zorder=1)

    pooled_entry = next(entry for entry in row_entries if entry["kind"] == "pooled")
    ax.axhline(pooled_entry["y"] - 0.5, color="#a9a9a9", linewidth=1.2, zorder=1)
    for metric in metrics:
        pooled_mean = pooled_summary.get(metric)
        if pooled_mean is None:
            continue
        style = FIGURE1_METHOD_STYLES[metric]
        ax.axvline(
            pooled_mean,
            color=style["color"],
            linewidth=1.1,
            linestyle=(0, (1.4, 2.2)),
            alpha=0.6,
            zorder=1.5,
        )

    for entry in row_entries:
        kind = entry["kind"]
        if kind == "gap":
            continue

        y = entry["y"]
        if kind == "cell":
            row = entry["row"]
            values = [row.get(metric) for metric in metrics if row.get(metric) is not None]
            if len(values) >= 2:
                ax.plot([min(values), max(values)], [y, y], color="#c6c6c6", linewidth=1.0, zorder=2)
            for offset, metric in zip(offsets, metrics):
                value = row.get(metric)
                if value is None:
                    continue
                style = FIGURE1_METHOD_STYLES[metric]
                ax.scatter(
                    value,
                    y + offset,
                    s=44,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=4,
                )
            continue

        if kind == "theme_mean":
            summary = summary_lookup[THEME_DISPLAY[entry["theme"]]]
        else:
            summary = summary_lookup["Pooled substantive"]

        mean_values = [summary.get(metric) for metric in metrics if summary.get(metric) is not None]
        if len(mean_values) >= 2:
            ax.plot([min(mean_values), max(mean_values)], [y, y], color="#8d8d8d", linewidth=1.3, zorder=3)

        for offset, metric in zip(offsets, metrics):
            mean = summary.get(metric)
            lo = summary.get(f"{metric}_ci_lo")
            hi = summary.get(f"{metric}_ci_hi")
            if mean is None:
                continue
            style = FIGURE1_METHOD_STYLES[metric]
            ax.errorbar(
                mean,
                y + offset,
                xerr=None if lo is None or hi is None else [[mean - lo], [hi - mean]],
                fmt=style["marker"],
                color=style["color"],
                markersize=7.4 if kind == "theme_mean" else 8.2,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.7,
                elinewidth=1.6,
                capsize=3,
                linestyle="none",
                zorder=5,
            )

    tick_entries = [entry for entry in row_entries if entry["kind"] != "gap"]
    ax.set_yticks([entry["y"] for entry in tick_entries])
    ax.set_yticklabels([entry["label"] for entry in tick_entries] if show_row_labels else [])
    ax.tick_params(axis="y", length=0, labelleft=show_row_labels)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Rate")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    _panel_label(ax, letter, title)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=FIGURE1_METHOD_STYLES[metric]["marker"],
            linestyle="",
            markersize=7.5,
            color=FIGURE1_METHOD_STYLES[metric]["color"],
            label=FIGURE1_METHOD_STYLES[metric]["label"],
        )
        for metric in metrics
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        loc="upper right",
        fontsize=8.8,
    )

    if show_row_labels:
        text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for block in theme_blocks:
            ax.text(
                -0.56,
                block["center"],
                THEME_DISPLAY[block["theme"]],
                transform=text_transform,
                ha="left",
                va="center",
                fontsize=10.8,
                fontweight="bold",
            )


def _plot_figure1(
    *,
    reports_dir: Path,
    substantive_rows: list[dict[str, Any]],
    figure1_summary: list[dict[str, Any]],
) -> None:
    rng = np.random.default_rng(321)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.16, wspace=0.17)

    theme_order = list(SUBSTANTIVE_THEMES)
    summary_lookup = {row["row_label"]: row for row in figure1_summary}
    offsets = {"drugs": -0.20, "policy": 0.0, "software": 0.20}

    def plot_panel(
        ax: plt.Axes,
        *,
        method_specs: list[tuple[str, str]],
        title: str,
        letter: str,
    ) -> None:
        for method_idx, (metric, label) in enumerate(method_specs):
            theme_means = [
                summary_lookup[THEME_DISPLAY[theme]].get(metric)
                for theme in theme_order
                if summary_lookup[THEME_DISPLAY[theme]].get(metric) is not None
            ]
            across_theme_mean = _safe_mean(theme_means)
            if across_theme_mean is not None:
                ax.plot(
                    [method_idx - 0.30, method_idx + 0.30],
                    [across_theme_mean, across_theme_mean],
                    color="#777777",
                    linewidth=1.2,
                    linestyle=(0, (1.4, 2.2)),
                    zorder=1,
                )

            for theme in theme_order:
                theme_rows = [row for row in substantive_rows if row["theme"] == theme]
                xs = method_idx + offsets[theme] + rng.normal(0.0, 0.025, size=len(theme_rows))
                ys = [row[metric] for row in theme_rows]
                ax.scatter(xs, ys, s=38, alpha=0.30, color=THEME_COLORS[theme], edgecolor="none", zorder=2)

                summary = summary_lookup[THEME_DISPLAY[theme]]
                mean = summary.get(metric)
                lo = summary.get(f"{metric}_ci_lo")
                hi = summary.get(f"{metric}_ci_hi")
                if mean is None:
                    continue
                if lo is not None and hi is not None:
                    ax.errorbar(
                        method_idx + offsets[theme],
                        mean,
                        yerr=[[mean - lo], [hi - mean]],
                        fmt="o",
                        color=THEME_COLORS[theme],
                        ms=8,
                        capsize=4,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        [method_idx + offsets[theme]],
                        [mean],
                        s=70,
                        color=THEME_COLORS[theme],
                        zorder=3,
                    )

        ax.set_xticks(range(len(method_specs)))
        ax.set_xticklabels([label for _, label in method_specs])
        _style_axis(ax, ylabel="Rate")
        _panel_label(ax, letter, title)

    plot_panel(
        axes[0],
        method_specs=[
            ("heldout_choice_prediction", "Actor\nchoice"),
            ("score_judge_choice_agreement_with_latent_choice", "Score judge\nchoice"),
        ],
        title="Agreement with latent-model choice",
        letter="A",
    )
    plot_panel(
        axes[1],
        method_specs=[
            ("self_report_revealed_driver", "Actor\nself-report"),
            ("score_judge_revealed_driver", "Score judge\ntop driver"),
        ],
        title="Agreement with latent-model driver",
        letter="B",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=THEME_COLORS[theme], label=THEME_DISPLAY[theme])
        for theme in theme_order
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower left", ncol=1)

    _write_figure(fig, reports_dir / "PAPER_FIG1_DISSOCIATION")


def _plot_figure2(
    *,
    reports_dir: Path,
    substantive_rows: list[dict[str, Any]],
    substantive_summary: list[dict[str, Any]],
    pairwise_annotations: list[dict[str, Any]] | None,
    occlusion_row: dict[str, Any],
    drugs_meta: dict[str, Any],
    include_pairwise: bool,
) -> None:
    rng = np.random.default_rng(321)
    if include_pairwise:
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.2), constrained_layout=True)
        axes = [ax]

    theme_order = list(SUBSTANTIVE_THEMES)
    summary_lookup = {row["row_label"]: row for row in substantive_summary}

    if include_pairwise:
        ax = axes[0]
        method_specs = [
            ("pairwise_top_driver_revealed_driver", "Pairwise ->\nrevealed driver"),
            ("pairwise_mirror_consistency", "Mirror\nconsistency"),
        ]
        offsets = {"drugs": -0.20, "policy": 0.0, "software": 0.20}
        for method_idx, (metric, label) in enumerate(method_specs):
            for theme in theme_order:
                theme_rows = [row for row in substantive_rows if row["theme"] == theme]
                xs = method_idx + offsets[theme] + rng.normal(0.0, 0.025, size=len(theme_rows))
                ys = [row[metric] for row in theme_rows]
                ax.scatter(xs, ys, s=38, alpha=0.30, color=THEME_COLORS[theme], edgecolor="none")
                summary = summary_lookup[THEME_DISPLAY[theme]]
                mean = summary[metric]
                lo = summary[f"{metric}_ci_lo"]
                hi = summary[f"{metric}_ci_hi"]
                if lo is not None and hi is not None:
                    ax.errorbar(
                        method_idx + offsets[theme],
                        mean,
                        yerr=[[mean - lo], [hi - mean]],
                        fmt="o",
                        color=THEME_COLORS[theme],
                        ms=8,
                        capsize=4,
                    )
        ax.set_xticks(range(len(method_specs)))
        ax.set_xticklabels([label for _, label in method_specs])
        _style_axis(ax, ylabel="Rate")
        _panel_label(ax, "A", "Mirrored stepwise pairwise explicit-access probe")
        annotation_lines = ["Cycle / tie / step parse:"]
        for row in pairwise_annotations or []:
            annotation_lines.append(
                f"{row['display_name']}: cycle {_fmt(row['pairwise_cycle_rate'])}, "
                f"tie {_fmt(row['pairwise_tie_rate'])}, "
                f"step parse {_fmt(row['pairwise_step_parse_ok_rate'])}"
            )
        ax.text(
            1.53,
            0.97,
            "\n".join(annotation_lines),
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"},
        )

    ax = axes[1] if include_pairwise else axes[0]
    baseline_order = str(occlusion_row["baseline_weight_order"]).split(">")
    attr_labels = drugs_meta["attr_labels"]
    x = np.arange(len(baseline_order))
    width = 0.18
    series = [
        ("drop_choice", "Drop: choice flip", "drop_choice_flip"),
        ("equalize_choice", "Equalize: choice flip", "equalize_choice_flip"),
        ("drop_premise", "Drop: premise flip", "drop_premise_flip"),
        ("equalize_premise", "Equalize: premise flip", "equalize_premise_flip"),
    ]
    offsets_bars = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, (series_key, label, prefix) in zip(offsets_bars, series):
        values = [float(occlusion_row[f"{prefix}_{attr}"]) for attr in baseline_order]
        ax.bar(x + offset, values, width=width, color=FIGURE_METHOD_COLORS[series_key], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(attr_labels[attr]) for attr in baseline_order])
    _style_axis(ax, ylabel="Flip rate", ylim=(0.0, 0.75))
    _panel_label(
        ax,
        "B" if include_pairwise else "A",
        "Drugs occlusion corroboration\n"
        f"GPT-5-mini minimal; baseline order: {' > '.join(baseline_order)}",
    )
    ax.legend(frameon=False, fontsize=8.8, loc="upper right", ncol=2)

    _write_figure(
        fig,
        reports_dir / ("PAPER_FIG2_PAIRWISE_OCCLUSION" if include_pairwise else "PAPER_FIG2_OCCLUSION"),
    )


def _write_manuscript_captions(
    reports_dir: Path,
    substantive_summary: list[dict[str, Any]],
    figure1_summary: list[dict[str, Any]],
    placebo_rows: list[dict[str, Any]],
    pairwise_annotations: list[dict[str, Any]] | None,
    substantive_rows: list[dict[str, Any]],
    occlusion_row: dict[str, Any],
    *,
    include_pairwise: bool,
) -> None:
    summary_lookup = {row["row_label"]: row for row in substantive_summary}
    pooled = summary_lookup["Pooled substantive"]
    figure1_lookup = {row["row_label"]: row for row in figure1_summary}
    pooled_figure1 = figure1_lookup["Pooled substantive"]

    max_placebo_outlier = max(
        placebo_rows,
        key=lambda row: (
            row.get("revealed_driver_is_placebo") or 0.0,
            row.get("score_judge_top_driver_is_placebo") or 0.0,
        ),
    )
    parse_rates = [
        row["pairwise_step_parse_ok_rate"] for row in substantive_rows if row.get("pairwise_step_parse_ok_rate") is not None
    ]
    parse_mean = _safe_mean(parse_rates)
    parse_min = min(parse_rates) if parse_rates else None
    parse_max = max(parse_rates) if parse_rates else None

    baseline_order = str(occlusion_row["baseline_weight_order"]).split(">")
    strongest_attr = baseline_order[0]
    weakest_attr = baseline_order[-1]
    strongest_drop_choice = float(occlusion_row[f"drop_choice_flip_{strongest_attr}"])
    strongest_drop_premise = float(occlusion_row[f"drop_premise_flip_{strongest_attr}"])
    weakest_equalize_choice = float(occlusion_row[f"equalize_choice_flip_{weakest_attr}"])
    weakest_equalize_premise = float(occlusion_row[f"equalize_premise_flip_{weakest_attr}"])

    captions = {
        "table_1": (
            "Table 1. Benchmark summary. Panel A reports substantive themes averaged equally across the 8 model settings per theme. "
            "The first column is held-out actor-choice accuracy for the latent behavioural model; the remaining substantive columns report "
            "recovery of the behaviourally revealed driver. Pooled across substantive conditions, held-out choice prediction was "
            f"{_fmt_ci(pooled['heldout_choice_prediction'], pooled['heldout_choice_prediction_ci_lo'], pooled['heldout_choice_prediction_ci_hi'])}, "
            "compared with "
            f"{_fmt_ci(pooled['self_report_revealed_driver'], pooled['self_report_revealed_driver_ci_lo'], pooled['self_report_revealed_driver_ci_hi'])} "
            "for direct self-report and "
            f"{_fmt_ci(pooled['score_judge_revealed_driver'], pooled['score_judge_revealed_driver_ci_lo'], pooled['score_judge_revealed_driver_ci_hi'])} "
            "for the score judge top driver. "
            "Panel B shows placebo falsification rates aggregated within each placebo theme; rates remain low overall. The largest cell-level placebo outlier "
            f"was {THEME_DISPLAY[max_placebo_outlier['theme']]} / {max_placebo_outlier['family']} {max_placebo_outlier['effort']}, "
            f"where the revealed driver was placebo at {_fmt(max_placebo_outlier['revealed_driver_is_placebo'])} and the score judge top driver was placebo at "
            f"{_fmt(max_placebo_outlier['score_judge_top_driver_is_placebo'])}."
        ),
        "figure_1": (
            "Figure 1. Panel A compares agreement with the latent-model choice for actor choice and score-judge reconstructed choice. "
            "Panel B compares agreement with the latent-model driver for actor self-report and the score judge top driver. "
            "Each faint point is one substantive theme x family x effort condition; colours denote themes. Larger points and vertical error bars show theme means with grouped-bootstrap 95% percentile intervals. Dotted horizontal reference segments mark the average across the three substantive themes for each method; because each theme contributes eight conditions, these averages are numerically identical to the pooled substantive means. "
            f"Pooled substantive agreement with latent-model choice was {_fmt_ci(pooled_figure1['heldout_choice_prediction'], pooled_figure1['heldout_choice_prediction_ci_lo'], pooled_figure1['heldout_choice_prediction_ci_hi'])} for actor choice "
            f"and {_fmt_ci(pooled_figure1['score_judge_choice_agreement_with_latent_choice'], pooled_figure1['score_judge_choice_agreement_with_latent_choice_ci_lo'], pooled_figure1['score_judge_choice_agreement_with_latent_choice_ci_hi'])} for score-judge reconstructed choice. "
            f"Pooled latent-model-driver agreement was {_fmt_ci(pooled_figure1['self_report_revealed_driver'], pooled_figure1['self_report_revealed_driver_ci_lo'], pooled_figure1['self_report_revealed_driver_ci_hi'])} for actor self-report, "
            f"and {_fmt_ci(pooled_figure1['score_judge_revealed_driver'], pooled_figure1['score_judge_revealed_driver_ci_lo'], pooled_figure1['score_judge_revealed_driver_ci_hi'])} for the score judge top driver."
        ),
        "figure_2": (
            (
                "Figure 2. Panel A shows the mirrored stepwise pairwise explicit-access probe on the substantive themes. "
                f"Pairwise step parse success was near-ceiling across substantive conditions (mean {_fmt(parse_mean)}; range {_fmt(parse_min)}-{_fmt(parse_max)}), "
                "but mirror consistency was substantially lower, indicating instability under order mirroring rather than primarily formatting failure. "
                f"The pooled substantive pairwise mirror consistency estimate was {_fmt_ci(pooled['pairwise_mirror_consistency'], pooled['pairwise_mirror_consistency_ci_lo'], pooled['pairwise_mirror_consistency_ci_hi'])}. "
                "Panel B uses the drugs occlusion suite for GPT-5-mini minimal as single-theme, single-model corroboration rather than as the main generalization claim. "
            )
            if include_pairwise
            else "Figure 2. Drugs occlusion corroboration using GPT-5-mini minimal as single-theme, single-model support rather than as the main generalization claim. "
        )
        + f"The baseline revealed order was {' > '.join(baseline_order)}. The strongest revealed attribute ({strongest_attr}) produced the largest disruption "
        f"(drop choice flip {_fmt(strongest_drop_choice)}; drop premise flip {_fmt(strongest_drop_premise)}), while the weakest revealed attribute ({weakest_attr}) produced the smallest disruption "
        f"(equalize choice flip {_fmt(weakest_equalize_choice)}; equalize premise flip {_fmt(weakest_equalize_premise)}).",
    }

    lines = [
        "# Paper Manuscript Captions",
        "",
        "## Table 1",
        captions["table_1"],
        "",
        "## Figure 1",
        captions["figure_1"],
        "",
        "## Figure 2",
        captions["figure_2"],
        "",
        "## Asset Links",
        f"- Table summary: `{_rel(reports_dir / 'PAPER_TABLE1_SUMMARY.md')}`",
        f"- Figure 1: `{_rel(reports_dir / 'PAPER_FIG1_DISSOCIATION.svg')}`",
        f"- Figure 2: `{_rel(reports_dir / ('PAPER_FIG2_PAIRWISE_OCCLUSION.svg' if include_pairwise else 'PAPER_FIG2_OCCLUSION.svg'))}`",
        f"- Family consistency summary: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_SUMMARY.md')}`",
        f"- Family consistency grid: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_GRID.md')}`",
        f"- Family majority correctness summary: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_SUMMARY.md')}`",
        f"- Family majority correctness grid: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_GRID.md')}`",
        *(
            [
                f"- Family pairwise agreement summary: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_SUMMARY.md')}`",
                f"- Family pairwise agreement grid: `{_rel(reports_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_GRID.md')}`",
            ]
            if include_pairwise
            else []
        ),
    ]
    (reports_dir / "PAPER_MANUSCRIPT_CAPTIONS.md").write_text("\n".join(lines) + "\n")


def _build_substantive_artifacts(
    *,
    out_root: Path,
    bootstrap: int,
    seed: int,
    include_pairwise: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float | None]]]]:
    rows: list[dict[str, Any]] = []
    sample_lookup: dict[str, dict[str, list[float | None]]] = {}

    for theme_idx, theme in enumerate(SUBSTANTIVE_THEMES):
        theme_meta = _theme_meta(theme)
        for spec_idx, spec in enumerate(MODEL_SPECS):
            cell_key = f"{theme}:{spec.tag}"
            tau_eval_df, tau_responses_path = _build_run_eval(theme, spec.tag, "tau", out_root=out_root)

            tau_group = _tau_group_frame(tau_eval_df)
            tau_point, tau_samples, tau_totals = _bootstrap_group_totals(
                tau_group,
                metric_builder=_tau_metrics_from_totals,
                bootstrap=bootstrap,
                seed=seed + theme_idx * 1000 + spec_idx * 10 + 1,
            )

            if include_pairwise:
                pair_eval_df, pair_responses_path = _build_run_eval(theme, spec.tag, "pair", out_root=out_root)
                pair_eval_df = add_pairwise_drivers(pair_eval_df)
                pair_group = _pair_group_frame(pair_eval_df)
                pair_point, pair_samples, pair_totals = _bootstrap_group_totals(
                    pair_group,
                    metric_builder=_pair_metrics_from_totals,
                    bootstrap=bootstrap,
                    seed=seed + theme_idx * 1000 + spec_idx * 10 + 2,
                )
                parse_stats = _step_parse_stats(pair_responses_path)
                pair_relpath = _rel(pair_responses_path)
            else:
                pair_point = {}
                pair_samples = {}
                pair_totals = {}
                parse_stats = {}
                pair_relpath = None

            row = {
                "cell_key": cell_key,
                "theme": theme,
                "theme_display": theme_meta["display_name"],
                "family": spec.family,
                "effort": spec.effort,
                "model_tag": spec.tag,
                "n_responses": int(tau_totals["n_rows"]),
                "n_config_ids": int(len(tau_group)),
                **tau_point,
                **pair_point,
                **parse_stats,
                "tau_responses_relpath": _rel(tau_responses_path),
                "pair_responses_relpath": pair_relpath,
            }
            row = _attach_ci_fields(
                row,
                tau_samples,
                (
                    "heldout_choice_prediction",
                    "score_judge_predicts_actor_choice",
                    "self_report_revealed_driver",
                    "score_judge_revealed_driver",
                ),
            )
            if include_pairwise:
                row = _attach_ci_fields(row, pair_samples, TABLE1_SUBSTANTIVE_METRICS[3:])
            rows.append(row)

            combined_samples = {}
            combined_samples.update({metric: tau_samples[metric] for metric in tau_point})
            if include_pairwise:
                combined_samples.update({metric: pair_samples[metric] for metric in pair_point})
            sample_lookup[cell_key] = combined_samples

    rows.sort(key=lambda row: (row["theme"], row["family"], row["effort"]))
    return rows, sample_lookup


def _build_placebo_artifacts(
    *,
    out_root: Path,
    bootstrap: int,
    seed: int,
    include_pairwise: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float | None]]]]:
    rows: list[dict[str, Any]] = []
    sample_lookup: dict[str, dict[str, list[float | None]]] = {}

    for theme_idx, theme in enumerate(PLACEBO_THEMES_ORDER):
        theme_meta = _theme_meta(theme)
        for spec_idx, spec in enumerate(MODEL_SPECS):
            cell_key = f"{theme}:{spec.tag}"
            tau_eval_df, tau_responses_path = _build_run_eval(theme, spec.tag, "tau", out_root=out_root)

            tau_group = _tau_group_frame(tau_eval_df)
            tau_point, tau_samples, tau_totals = _bootstrap_group_totals(
                tau_group,
                metric_builder=_tau_metrics_from_totals,
                bootstrap=bootstrap,
                seed=seed + 5000 + theme_idx * 1000 + spec_idx * 10 + 1,
            )

            if include_pairwise:
                pair_eval_df, pair_responses_path = _build_run_eval(theme, spec.tag, "pair", out_root=out_root)
                pair_eval_df = add_pairwise_drivers(pair_eval_df)
                pair_group = _pair_group_frame(pair_eval_df)
                pair_point, pair_samples, _ = _bootstrap_group_totals(
                    pair_group,
                    metric_builder=_pair_metrics_from_totals,
                    bootstrap=bootstrap,
                    seed=seed + 5000 + theme_idx * 1000 + spec_idx * 10 + 2,
                )
                parse_stats = _step_parse_stats(pair_responses_path)
                pair_relpath = _rel(pair_responses_path)
            else:
                pair_point = {}
                pair_samples = {}
                parse_stats = {}
                pair_relpath = None

            row = {
                "cell_key": cell_key,
                "theme": theme,
                "theme_display": theme_meta["display_name"],
                "placebo_variant": theme_meta["placebo_variant"],
                "family": spec.family,
                "effort": spec.effort,
                "model_tag": spec.tag,
                "n_responses": int(tau_totals["n_rows"]),
                "n_config_ids": int(len(tau_group)),
                **tau_point,
                **pair_point,
                **parse_stats,
                "tau_responses_relpath": _rel(tau_responses_path),
                "pair_responses_relpath": pair_relpath,
            }
            row = _attach_ci_fields(row, tau_samples, TABLE1_PLACEBO_METRICS)
            if include_pairwise:
                row = _attach_ci_fields(row, pair_samples, ("pairwise_top_driver_is_placebo",))
            rows.append(row)

            combined_samples = {}
            combined_samples.update({metric: tau_samples[metric] for metric in tau_point})
            if include_pairwise:
                combined_samples.update({metric: pair_samples[metric] for metric in pair_point})
            sample_lookup[cell_key] = combined_samples

    rows.sort(key=lambda row: (row["theme"], row["family"], row["effort"]))
    return rows, sample_lookup


def _build_family_consistency_artifacts(
    *,
    out_root: Path,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float | None]]]]:
    rows: list[dict[str, Any]] = []
    sample_lookup: dict[str, dict[str, list[float | None]]] = {}

    for theme_idx, theme in enumerate(SUBSTANTIVE_THEMES):
        theme_meta = _theme_meta(theme)
        trials_path = out_root / "datasets" / theme / "test" / "dataset_trials.parquet"
        trial_meta = (
            pd.read_parquet(trials_path)[["trial_id", "family_id", "slot_A_profile", "slot_B_profile"]]
            .drop_duplicates(subset=["trial_id"])
            .copy()
        )
        trial_meta["trial_id"] = trial_meta["trial_id"].astype(str)
        for spec_idx, spec in enumerate(MODEL_SPECS):
            cell_key = f"{theme}:{spec.tag}"
            tau_eval_df, tau_responses_path = _build_run_eval(theme, spec.tag, "tau", out_root=out_root)
            tau_eval_df = tau_eval_df.drop(
                columns=[column for column in trial_meta.columns if column != "trial_id" and column in tau_eval_df.columns]
            )
            tau_eval_df["trial_id"] = tau_eval_df["trial_id"].astype(str)
            tau_eval_df = tau_eval_df.merge(trial_meta, on="trial_id", how="left", validate="many_to_one")
            if tau_eval_df["family_id"].isna().any():
                missing = int(tau_eval_df["family_id"].isna().sum())
                raise ValueError(f"Missing family metadata for {theme}/{spec.tag}: {missing} rows")

            family_group = _family_consistency_group_frame(tau_eval_df)
            family_point, family_samples, family_totals = _bootstrap_group_totals(
                family_group,
                metric_builder=_family_all_metrics_from_totals,
                bootstrap=bootstrap,
                seed=seed + 9000 + theme_idx * 1000 + spec_idx * 10 + 1,
            )

            row = {
                "cell_key": cell_key,
                "theme": theme,
                "theme_display": theme_meta["display_name"],
                "family": spec.family,
                "effort": spec.effort,
                "model_tag": spec.tag,
                "n_responses": int(len(tau_eval_df)),
                "n_families": int(family_totals["n_families"]),
                "expected_draws_per_family": FAMILY_EXPECTED_DRAWS,
                **family_point,
                "tau_responses_relpath": _rel(tau_responses_path),
            }
            row = _attach_ci_fields(row, family_samples, FAMILY_ALL_METRICS)
            rows.append(row)
            sample_lookup[cell_key] = {metric: family_samples[metric] for metric in family_point}

    rows.sort(key=lambda row: (row["theme"], row["family"], row["effort"]))
    return rows, sample_lookup


def _write_family_metric_files(
    reports_dir: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    stem: str,
    title: str,
    metrics: tuple[str, ...],
    summary_notes: list[str],
    grid_notes: list[str],
) -> None:
    summary_csv_columns = [
        "row_label",
        "n_cells",
        *[
            column
            for metric in metrics
            for column in (metric, f"{metric}_ci_lo", f"{metric}_ci_hi")
        ],
    ]
    _write_csv(reports_dir / f"{stem}_SUMMARY.csv", summary_rows, summary_csv_columns)
    _write_json(
        reports_dir / f"{stem}_SUMMARY.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": summary_rows,
            "notes": summary_notes,
        },
    )
    summary_display = _summary_display_rows(summary_rows, metrics)
    summary_md_lines = [f"# {title} Summary", "", *[f"- {note}" for note in summary_notes], "", _markdown_table(summary_display, ["row_label", *metrics])]
    (reports_dir / f"{stem}_SUMMARY.md").write_text("\n".join(summary_md_lines) + "\n")

    payload_rows = [
        {
            key: row.get(key)
            for key in [
                "theme",
                "theme_display",
                "family",
                "effort",
                "model_tag",
                "n_responses",
                "n_families",
                "expected_draws_per_family",
                *[
                    column
                    for metric in metrics
                    for column in (metric, f"{metric}_ci_lo", f"{metric}_ci_hi")
                ],
                "tau_responses_relpath",
            ]
        }
        for row in rows
    ]
    display_rows = _grid_display_rows(rows, metrics)

    _write_appendix_grid(
        reports_dir=reports_dir,
        stem=f"{stem}_GRID",
        payload_rows=payload_rows,
        display_rows=display_rows,
        csv_columns=list(payload_rows[0].keys()),
        md_columns=["theme", "family", "effort", *metrics],
        title=f"{title} Grid",
        notes=grid_notes,
    )


def _occlusion_row(out_root: Path) -> dict[str, Any]:
    csv_path = ROOT / "outputs/occlusion_suite_drugs_mini_min/reports/OCCLUSION_SUITE_RESULTS.csv"
    if out_root != (ROOT / "outputs/final_same_order").resolve():
        csv_path = ROOT / "outputs/occlusion_suite_drugs_mini_min/reports/OCCLUSION_SUITE_RESULTS.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly 1 occlusion row in {csv_path}, found {len(rows)}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper-facing results bundle from current final outputs.")
    parser.add_argument("--out-root", default=str(output_root()))
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--exclude-pairwise", action="store_true")
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    rep_dir = Path(args.reports_dir).resolve() if args.reports_dir else reports_root(out_root)
    rep_dir.mkdir(parents=True, exist_ok=True)
    include_pairwise = not args.exclude_pairwise
    substantive_metrics = TABLE1_SUBSTANTIVE_METRICS if include_pairwise else TABLE1_SUBSTANTIVE_METRICS_NO_PAIRWISE
    substantive_grid_metrics = SUBSTANTIVE_GRID_METRICS if include_pairwise else SUBSTANTIVE_GRID_METRICS_NO_PAIRWISE
    placebo_grid_metrics = PLACEBO_GRID_METRICS if include_pairwise else PLACEBO_GRID_METRICS_NO_PAIRWISE

    substantive_rows, substantive_samples = _build_substantive_artifacts(
        out_root=out_root,
        bootstrap=args.bootstrap,
        seed=args.seed,
        include_pairwise=include_pairwise,
    )
    placebo_rows, placebo_samples = _build_placebo_artifacts(
        out_root=out_root,
        bootstrap=args.bootstrap,
        seed=args.seed,
        include_pairwise=include_pairwise,
    )
    family_consistency_rows, family_consistency_samples = _build_family_consistency_artifacts(
        out_root=out_root,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    substantive_summary = _substantive_summary_rows(substantive_rows, substantive_samples, substantive_metrics)
    figure1_summary = _figure1_summary_rows(substantive_rows, substantive_samples)
    placebo_summary = _placebo_summary_rows(placebo_rows, placebo_samples)
    family_consistency_summary = _family_metric_summary_rows(
        family_consistency_rows, family_consistency_samples, FAMILY_CONSISTENCY_METRICS
    )
    family_pairwise_agreement_summary = (
        _family_metric_summary_rows(
            family_consistency_rows, family_consistency_samples, FAMILY_PAIRWISE_AGREEMENT_METRICS
        )
        if include_pairwise
        else None
    )
    family_majority_correctness_summary = _family_metric_summary_rows(
        family_consistency_rows, family_consistency_samples, FAMILY_MAJORITY_CORRECTNESS_METRICS
    )
    pairwise_theme_annotations = _pairwise_theme_annotation_rows(substantive_rows) if include_pairwise else None
    occlusion = _occlusion_row(out_root)

    _write_table1_files(
        rep_dir,
        substantive_summary,
        placebo_summary,
        substantive_metrics=substantive_metrics,
        placebo_metrics=TABLE1_PLACEBO_METRICS,
        include_pairwise=include_pairwise,
    )
    _write_family_metric_files(
        rep_dir,
        family_consistency_rows,
        family_consistency_summary,
        stem="PAPER_APPENDIX_FAMILY_CONSISTENCY",
        title="Paper Appendix Family Consistency",
        metrics=FAMILY_CONSISTENCY_METRICS,
        summary_notes=[
            "Consistency is computed at the base-family level over 12 draws: 4 prompt variants x 3 samples.",
            "Choice consistency is canonicalized to the underlying chosen profile rather than raw A/B labels.",
            "Consistency rates are conditional on all 12 draws being valid for that measure; completeness rates and modal-share diagnostics are reported in the CSV and JSON outputs.",
        ],
        grid_notes=[
            "Each row is one substantive theme x model-family x effort condition.",
            "Consistency is computed over the 12 responses for each base family: 4 prompt variants x 3 samples.",
            "Choice consistency is canonicalized to the underlying chosen profile so left-right reversals do not count as disagreement.",
            "The markdown table shows all-12 consistency rates; completeness and modal-share diagnostics are included in the CSV and JSON outputs.",
        ],
    )
    if include_pairwise and family_pairwise_agreement_summary is not None:
        _write_family_metric_files(
            rep_dir,
            family_consistency_rows,
            family_pairwise_agreement_summary,
            stem="PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT",
            title="Paper Appendix Family Pairwise Agreement",
            metrics=FAMILY_PAIRWISE_AGREEMENT_METRICS,
            summary_notes=[
                "Pairwise agreement is computed at the base-family level over 12 draws: 4 prompt variants x 3 samples.",
                "Choice agreement is canonicalized to the underlying chosen profile rather than raw A/B labels.",
                "Each value is the mean within-family agreement probability for two random draws, averaged across families.",
            ],
            grid_notes=[
                "Each row is one substantive theme x model-family x effort condition.",
                "Pairwise agreement is computed over the 12 responses for each base family: 4 prompt variants x 3 samples.",
                "Choice agreement is canonicalized to the underlying chosen profile so left-right reversals do not count as disagreement.",
                "The markdown table shows mean within-family pairwise agreement; completeness diagnostics remain available in the family consistency CSV and JSON outputs.",
            ],
        )
    _write_family_metric_files(
        rep_dir,
        family_consistency_rows,
        family_majority_correctness_summary,
        stem="PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS",
        title="Paper Appendix Family Majority Correctness",
        metrics=FAMILY_MAJORITY_CORRECTNESS_METRICS,
        summary_notes=[
            "Majority correctness is computed at the base-family level over 12 draws: 4 prompt variants x 3 samples.",
            "A family counts as majority-correct when more than half of its valid draws are correct relative to the row-level latent or revealed target.",
            "For choice metrics, correctness uses canonicalized underlying profiles rather than raw A/B labels.",
        ],
        grid_notes=[
            "Each row is one substantive theme x model-family x effort condition.",
            "Majority correctness is computed over the 12 responses for each base family: 4 prompt variants x 3 samples.",
            "For driver metrics, correctness is evaluated draw-by-draw against that draw's revealed driver rather than a single fixed family-level target.",
            "The markdown table shows the share of complete families whose response stream is majority-correct.",
        ],
    )

    substantive_payload_rows = [
        {
            key: row.get(key)
            for key in [
                "theme",
                "theme_display",
                "family",
                "effort",
                "model_tag",
                "n_responses",
                "n_config_ids",
                *[
                    column
                    for metric in substantive_grid_metrics
                    for column in (metric, f"{metric}_ci_lo", f"{metric}_ci_hi")
                ],
                "tau_responses_relpath",
                *(["pair_responses_relpath"] if include_pairwise else []),
            ]
        }
        for row in substantive_rows
    ]
    _write_appendix_grid(
        reports_dir=rep_dir,
        stem="PAPER_APPENDIX_SUBSTANTIVE_GRID",
        payload_rows=substantive_payload_rows,
        display_rows=_grid_display_rows(substantive_rows, substantive_grid_metrics),
        csv_columns=list(substantive_payload_rows[0].keys()),
        md_columns=["theme", "family", "effort", *substantive_grid_metrics],
        title="Paper Appendix Substantive Grid",
        notes=[
            "Each row is one substantive theme x model-family x effort condition.",
            "Point estimates are paired with grouped-bootstrap 95% percentile intervals.",
        ],
    )

    placebo_payload_rows = [
        {
            key: row.get(key)
            for key in [
                "theme",
                "theme_display",
                "placebo_variant",
                "family",
                "effort",
                "model_tag",
                "n_responses",
                "n_config_ids",
                *[
                    column
                    for metric in placebo_grid_metrics
                    for column in (metric, f"{metric}_ci_lo", f"{metric}_ci_hi")
                ],
                "tau_responses_relpath",
                *(["pair_responses_relpath"] if include_pairwise else []),
            ]
        }
        for row in placebo_rows
    ]
    _write_appendix_grid(
        reports_dir=rep_dir,
        stem="PAPER_APPENDIX_PLACEBO_GRID",
        payload_rows=placebo_payload_rows,
        display_rows=_grid_display_rows(placebo_rows, placebo_grid_metrics),
        csv_columns=list(placebo_payload_rows[0].keys()),
        md_columns=["theme", "family", "effort", *placebo_grid_metrics],
        title="Paper Appendix Placebo Grid",
        notes=[
            "Each row is one placebo theme x model-family x effort condition.",
            *(
                ["Pairwise placebo rates are appendix-only by default."]
                if include_pairwise
                else []
            ),
        ],
    )

    if include_pairwise:
        pairwise_payload_rows = [
            {
                key: row.get(key)
                for key in [
                    "theme",
                    "theme_display",
                    "family",
                    "effort",
                    "model_tag",
                    "n_responses",
                    "n_config_ids",
                    *PAIRWISE_DIAG_METRICS,
                    "pair_responses_relpath",
                ]
            }
            for row in substantive_rows + placebo_rows
        ]
        pairwise_display_rows = []
        for row in substantive_rows + placebo_rows:
            pairwise_display_rows.append(
                {
                    "theme": THEME_DISPLAY[row["theme"]],
                    "family": row["family"],
                    "effort": row["effort"],
                    **{metric: _fmt(row.get(metric)) for metric in PAIRWISE_DIAG_METRICS},
                }
            )
        _write_appendix_grid(
            reports_dir=rep_dir,
            stem="PAPER_APPENDIX_PAIRWISE_DIAGNOSTICS",
            payload_rows=pairwise_payload_rows,
            display_rows=pairwise_display_rows,
            csv_columns=list(pairwise_payload_rows[0].keys()),
            md_columns=["theme", "family", "effort", *PAIRWISE_DIAG_METRICS],
            title="Paper Appendix Pairwise Diagnostics",
            notes=[
                "Pairwise mirror consistency is conditional on mirror-complete rows.",
                "Pairwise cycle rate is conditional on pairwise-ok rows.",
                "Pairwise step parse success is reported separately from mirror consistency.",
            ],
        )

    _plot_figure1(
        reports_dir=rep_dir,
        substantive_rows=substantive_rows,
        figure1_summary=figure1_summary,
    )

    _plot_figure2(
        reports_dir=rep_dir,
        substantive_rows=substantive_rows,
        substantive_summary=substantive_summary,
        pairwise_annotations=pairwise_theme_annotations,
        occlusion_row=occlusion,
        drugs_meta=_theme_meta("drugs"),
        include_pairwise=include_pairwise,
    )
    _write_manuscript_captions(
        rep_dir,
        substantive_summary,
        figure1_summary,
        placebo_rows,
        pairwise_theme_annotations,
        substantive_rows,
        occlusion,
        include_pairwise=include_pairwise,
    )

    print(f"wrote {rep_dir / 'PAPER_TABLE1_SUMMARY.csv'}")
    print(f"wrote {rep_dir / 'PAPER_TABLE1_SUMMARY.json'}")
    print(f"wrote {rep_dir / 'PAPER_TABLE1_SUMMARY.md'}")
    print(f"wrote {rep_dir / 'PAPER_FIG1_DISSOCIATION.svg'}")
    print(f"wrote {rep_dir / 'PAPER_FIG1_DISSOCIATION.pdf'}")
    print(
        f"wrote {rep_dir / ('PAPER_FIG2_PAIRWISE_OCCLUSION.svg' if include_pairwise else 'PAPER_FIG2_OCCLUSION.svg')}"
    )
    print(
        f"wrote {rep_dir / ('PAPER_FIG2_PAIRWISE_OCCLUSION.pdf' if include_pairwise else 'PAPER_FIG2_OCCLUSION.pdf')}"
    )
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_SUBSTANTIVE_GRID.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_SUBSTANTIVE_GRID.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_SUBSTANTIVE_GRID.md'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_PLACEBO_GRID.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_PLACEBO_GRID.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_PLACEBO_GRID.md'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_SUMMARY.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_SUMMARY.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_SUMMARY.md'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_GRID.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_GRID.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_CONSISTENCY_GRID.md'}")
    if include_pairwise:
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_PAIRWISE_DIAGNOSTICS.csv'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_PAIRWISE_DIAGNOSTICS.json'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_PAIRWISE_DIAGNOSTICS.md'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_SUMMARY.csv'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_SUMMARY.json'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_SUMMARY.md'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_GRID.csv'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_GRID.json'}")
        print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_PAIRWISE_AGREEMENT_GRID.md'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_SUMMARY.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_SUMMARY.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_SUMMARY.md'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_GRID.csv'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_GRID.json'}")
    print(f"wrote {rep_dir / 'PAPER_APPENDIX_FAMILY_MAJORITY_CORRECTNESS_GRID.md'}")
    print(f"wrote {rep_dir / 'PAPER_MANUSCRIPT_CAPTIONS.md'}")


if __name__ == "__main__":
    main()
