#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import methodc_extra_diagnostics as mxd

from src.analysis.diagnostics import evaluate_model
from src.analysis.features import aggregate_choices, prepare_stageA_data
from src.analysis.stageA import ATTRIBUTES, build_design_matrix, compute_ames_and_weights, fit_glm_clustered
from src.utils.io import ensure_dir, write_json

MANIPS = ["occlude_drop", "occlude_equalize"]
BOOT_METRICS = [
    "delta_favored_mean",
    "choice_flip_rate",
    "premise_flip_rate",
    "shift_away_from_target_rate",
    "p_choice_flip_given_premise_flip",
    "p_choice_flip_given_no_premise_flip",
    "shared_flip_fraction_of_choice_flips",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        value = float(value)
        if math.isnan(value):
            return None
        return value
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(point: float | None, ci: list[float] | None) -> str:
    if point is None:
        return "n/a"
    if not ci:
        return f"{point:.3f}"
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _rank_order(mapping: dict[str, float | None]) -> list[str]:
    usable = [(attr, value) for attr, value in mapping.items() if value is not None]
    return [attr for attr, _ in sorted(usable, key=lambda kv: kv[1], reverse=True)]


def _spearman(a: dict[str, float | None], b: dict[str, float | None]) -> float | None:
    a_series = pd.Series({k: v for k, v in a.items() if v is not None}, dtype=float)
    b_series = pd.Series({k: v for k, v in b.items() if v is not None}, dtype=float)
    keys = sorted(set(a_series.index).intersection(b_series.index))
    if len(keys) < 2:
        return None
    return float(a_series[keys].rank().corr(b_series[keys].rank(), method="spearman"))


def _bootstrap_ci(values: list[float | None]) -> list[float] | None:
    usable = np.asarray([value for value in values if value is not None and not math.isnan(value)], dtype=float)
    if usable.size == 0:
        return None
    lo, hi = np.percentile(usable, [2.5, 97.5])
    return [float(lo), float(hi)]


def fit_baseline_only_stagea(trials_df: pd.DataFrame, responses_df: pd.DataFrame) -> dict[str, Any]:
    baseline_trials = trials_df[trials_df["manipulation"] == "short_reason"].copy()
    baseline_responses = responses_df[responses_df["manipulation"] == "short_reason"].copy()
    choice_agg = aggregate_choices(baseline_responses)
    stagea_df = prepare_stageA_data(baseline_trials, choice_agg)
    design = build_design_matrix(stagea_df, exclude_b1=True)
    model = fit_glm_clustered(design)
    weights = compute_ames_and_weights(model)
    evaluation = evaluate_model(model, design)
    return {
        "n_trials": int(len(baseline_trials)),
        "n_configs": int(baseline_trials["config_id"].nunique()),
        "weights": {attr: float(weights["weights"][attr]) for attr in ATTRIBUTES},
        "beta": {attr: float(weights["beta"][attr]) for attr in ATTRIBUTES},
        "se": {attr: float(weights["se"][attr]) for attr in ATTRIBUTES},
        "evaluation": {k: float(v) for k, v in evaluation.items()},
    }


def _directional_and_raw(sub: pd.DataFrame, delta_col: str) -> tuple[pd.Series, pd.Series]:
    sign = np.sign(pd.to_numeric(sub[delta_col], errors="coerce").fillna(0.0))
    raw = pd.to_numeric(sub["pA"], errors="coerce") - pd.to_numeric(sub["base_pA"], errors="coerce")
    directional = np.where(sign > 0, raw, np.where(sign < 0, -raw, np.nan))
    mask = np.isfinite(directional) & np.isfinite(raw)
    return pd.Series(directional[mask], dtype=float), pd.Series(raw[mask], dtype=float)


def summarize_attr_pair(sub: pd.DataFrame, attr: str, delta_col: str) -> dict[str, Any]:
    directional, raw = _directional_and_raw(sub, delta_col)
    valid_choice = sub["choice_majority"].isin(["A", "B"]) & sub["base_choice_majority"].isin(["A", "B"])
    choice_flip = (sub.loc[valid_choice, "choice_majority"] != sub.loc[valid_choice, "base_choice_majority"]).astype(float)

    valid_premise = sub["premise_majority"].notna() & sub["base_premise_majority"].notna()
    premise_flip = (sub.loc[valid_premise, "premise_majority"] != sub.loc[valid_premise, "base_premise_majority"]).astype(float)
    shift_away = (
        (sub.loc[valid_premise, "base_premise_majority"] == attr)
        & (sub.loc[valid_premise, "premise_majority"] != attr)
    ).astype(float)

    valid_med = valid_choice & valid_premise
    med = sub.loc[valid_med].copy()
    if med.empty:
        p_flip_given_prem = None
        p_flip_given_no_prem = None
        shared = None
    else:
        med_choice_flip = med["choice_majority"] != med["base_choice_majority"]
        med_premise_flip = med["premise_majority"] != med["base_premise_majority"]
        both = med_choice_flip & med_premise_flip
        with_premise = med[med_premise_flip]
        without_premise = med[~med_premise_flip]
        p_flip_given_prem = float((with_premise["choice_majority"] != with_premise["base_choice_majority"]).mean()) if len(with_premise) else None
        p_flip_given_no_prem = float((without_premise["choice_majority"] != without_premise["base_choice_majority"]).mean()) if len(without_premise) else None
        shared = float(both.sum() / med_choice_flip.sum()) if med_choice_flip.sum() > 0 else None

    return {
        "n": int(sub["pair_key"].nunique()),
        "delta_favored_mean": float(directional.mean()) if len(directional) else None,
        "delta_pA_mean": float(raw.mean()) if len(raw) else None,
        "choice_flip_rate": float(choice_flip.mean()) if len(choice_flip) else None,
        "premise_flip_rate": float(premise_flip.mean()) if len(premise_flip) else None,
        "shift_away_from_target_rate": float(shift_away.mean()) if len(shift_away) else None,
        "p_choice_flip_given_premise_flip": p_flip_given_prem,
        "p_choice_flip_given_no_premise_flip": p_flip_given_no_prem,
        "shared_flip_fraction_of_choice_flips": shared,
    }


def _resample_pairs(sub: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pair_ids = list(sub["pair_key"].astype(str).unique())
    picks = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    frames: list[pd.DataFrame] = []
    for idx, pair_id in enumerate(picks):
        part = sub[sub["pair_key"].astype(str) == pair_id].copy()
        part["pair_key"] = part["pair_key"].astype(str) + f"__bs{idx}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def bootstrap_attr_pair(sub: pd.DataFrame, attr: str, delta_col: str, *, bootstrap: int, seed: int) -> dict[str, Any]:
    point = summarize_attr_pair(sub, attr, delta_col)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float | None]] = defaultdict(list)
    for _ in range(bootstrap):
        boot = _resample_pairs(sub, rng)
        stats = summarize_attr_pair(boot, attr, delta_col)
        for metric in BOOT_METRICS:
            draws[metric].append(_safe_float(stats.get(metric)))
    out = dict(point)
    for metric in BOOT_METRICS:
        out[f"{metric}_ci95"] = _bootstrap_ci(draws[metric])
    return out


def summarize_magnitude(sub: pd.DataFrame, delta_col: str) -> dict[str, Any]:
    directional, _ = _directional_and_raw(sub, delta_col)
    work = sub.copy()
    sign = np.sign(pd.to_numeric(work[delta_col], errors="coerce").fillna(0.0))
    raw = pd.to_numeric(work["pA"], errors="coerce") - pd.to_numeric(work["base_pA"], errors="coerce")
    work["delta_favored"] = np.where(sign > 0, raw, np.where(sign < 0, -raw, np.nan))
    work["mag"] = np.abs(pd.to_numeric(work[delta_col], errors="coerce").fillna(0.0))
    valid_choice = work["choice_majority"].isin(["A", "B"]) & work["base_choice_majority"].isin(["A", "B"])
    rows: dict[str, Any] = {}
    for mag in sorted(x for x in work["mag"].dropna().unique() if x > 0):
        grp = work[work["mag"] == mag].copy()
        if grp.empty:
            continue
        choice_grp = grp[valid_choice.loc[grp.index]]
        label = str(int(mag)) if float(mag).is_integer() else str(mag)
        rows[label] = {
            "n": int(grp["pair_key"].nunique()),
            "delta_favored_mean": float(pd.to_numeric(grp["delta_favored"], errors="coerce").mean()),
            "choice_flip_rate": float((choice_grp["choice_majority"] != choice_grp["base_choice_majority"]).mean()) if len(choice_grp) else None,
        }
    return rows


def compare_manips(
    trial_df: pd.DataFrame,
    attr: str,
    delta_col: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    drop = mxd._pair_baseline_vs_manip(trial_df, "occlude_drop")
    drop = drop[drop["attribute_target"] == attr].copy()
    equalize = mxd._pair_baseline_vs_manip(trial_df, "occlude_equalize")
    equalize = equalize[equalize["attribute_target"] == attr].copy()
    if drop.empty or equalize.empty:
        return {}

    left = drop[["pair_key", "pA", "base_pA", "choice_majority", "base_choice_majority", "premise_majority", "base_premise_majority", delta_col]].rename(
        columns={
            "pA": "drop_pA",
            "base_pA": "drop_base_pA",
            "choice_majority": "drop_choice_majority",
            "base_choice_majority": "drop_base_choice_majority",
            "premise_majority": "drop_premise_majority",
            "base_premise_majority": "drop_base_premise_majority",
            delta_col: "drop_delta",
        }
    )
    right = equalize[["pair_key", "pA", "base_pA", "choice_majority", "base_choice_majority", "premise_majority", "base_premise_majority", delta_col]].rename(
        columns={
            "pA": "eq_pA",
            "base_pA": "eq_base_pA",
            "choice_majority": "eq_choice_majority",
            "base_choice_majority": "eq_base_choice_majority",
            "premise_majority": "eq_premise_majority",
            "base_premise_majority": "eq_base_premise_majority",
            delta_col: "eq_delta",
        }
    )
    merged = left.merge(right, on="pair_key", how="inner")
    if merged.empty:
        return {}

    def _diff_stats(df: pd.DataFrame) -> dict[str, Any]:
        sign_drop = np.sign(pd.to_numeric(df["drop_delta"], errors="coerce").fillna(0.0))
        sign_eq = np.sign(pd.to_numeric(df["eq_delta"], errors="coerce").fillna(0.0))
        raw_drop = pd.to_numeric(df["drop_pA"], errors="coerce") - pd.to_numeric(df["drop_base_pA"], errors="coerce")
        raw_eq = pd.to_numeric(df["eq_pA"], errors="coerce") - pd.to_numeric(df["eq_base_pA"], errors="coerce")
        dir_drop = np.where(sign_drop > 0, raw_drop, np.where(sign_drop < 0, -raw_drop, np.nan))
        dir_eq = np.where(sign_eq > 0, raw_eq, np.where(sign_eq < 0, -raw_eq, np.nan))
        choice_drop = (df["drop_choice_majority"] != df["drop_base_choice_majority"]).astype(float)
        choice_eq = (df["eq_choice_majority"] != df["eq_base_choice_majority"]).astype(float)
        premise_valid_drop = df["drop_premise_majority"].notna() & df["drop_base_premise_majority"].notna()
        premise_valid_eq = df["eq_premise_majority"].notna() & df["eq_base_premise_majority"].notna()
        premise_drop = (df.loc[premise_valid_drop, "drop_premise_majority"] != df.loc[premise_valid_drop, "drop_base_premise_majority"]).astype(float)
        premise_eq = (df.loc[premise_valid_eq, "eq_premise_majority"] != df.loc[premise_valid_eq, "eq_base_premise_majority"]).astype(float)
        return {
            "n": int(len(df)),
            "delta_equalize_minus_drop_directional": float(np.nanmean(dir_eq - dir_drop)),
            "delta_equalize_minus_drop_choice_flip_rate": float(choice_eq.mean() - choice_drop.mean()),
            "delta_equalize_minus_drop_premise_flip_rate": float(premise_eq.mean() - premise_drop.mean()),
        }

    point = _diff_stats(merged)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float | None]] = defaultdict(list)
    pair_ids = list(merged["pair_key"].astype(str).unique())
    for _ in range(bootstrap):
        picks = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        parts = []
        for idx, pair_id in enumerate(picks):
            part = merged[merged["pair_key"].astype(str) == pair_id].copy()
            part["pair_key"] = part["pair_key"].astype(str) + f"__bs{idx}"
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        stats = _diff_stats(boot)
        for metric in [
            "delta_equalize_minus_drop_directional",
            "delta_equalize_minus_drop_choice_flip_rate",
            "delta_equalize_minus_drop_premise_flip_rate",
        ]:
            draws[metric].append(_safe_float(stats[metric]))
    for metric, values in draws.items():
        point[f"{metric}_ci95"] = _bootstrap_ci(values)
    return point


def premise_transition_summary(trial_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for manip in MANIPS:
        paired = mxd._pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            valid = (
                sub["base_premise_majority"].eq(attr)
                & sub["premise_majority"].notna()
                & sub["premise_majority"].ne(attr)
            )
            shifted = sub[valid]
            if shifted.empty:
                continue
            counts = Counter(shifted["premise_majority"].astype(str))
            total = sum(counts.values())
            top_dest, top_n = counts.most_common(1)[0]
            by_attr[attr] = {
                "n_shifted_from_target": int(total),
                "top_destination": top_dest,
                "top_destination_rate_among_shifted": float(top_n / total),
                "destination_distribution": {k: float(v / total) for k, v in sorted(counts.items())},
            }
        if by_attr:
            out[manip] = {"by_attribute": by_attr}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-facing analysis for a completed Method C run")
    parser.add_argument("--dataset", required=True, help="Method C dataset directory")
    parser.add_argument("--responses", required=True, help="Responses jsonl path or run directory")
    parser.add_argument("--full-stagea", default=None, help="Optional full Method C Stage A summary json for comparison")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap resamples by base_trial_id/pair_key")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    trials_df = mxd._load_dataset_trials(Path(args.dataset))
    responses_df, responses_path = mxd._load_run_responses(Path(args.responses))
    joined = mxd._prepare_joined(trials_df, responses_df, include_tau=True, include_pairwise=True)
    trial_df = mxd._trial_level_table(joined, stagea_map=None)
    delta_cols = mxd._pick_delta_columns(trials_df)

    baseline_stagea = fit_baseline_only_stagea(trials_df, responses_df)
    full_stagea = None
    if args.full_stagea:
        full_stagea = json.loads(Path(args.full_stagea).read_text())

    directional_tables: dict[str, Any] = {}
    magnitude_tables: dict[str, Any] = {}
    manip_comparisons: dict[str, Any] = {}
    for manip in MANIPS:
        paired = mxd._pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        by_attr_mag: dict[str, Any] = {}
        for idx, attr in enumerate(ATTRIBUTES, start=1):
            sub = paired[paired["attribute_target"] == attr].copy()
            if sub.empty:
                continue
            delta_col = delta_cols.get(attr)
            if not delta_col:
                continue
            by_attr[attr] = bootstrap_attr_pair(
                sub,
                attr,
                delta_col,
                bootstrap=args.bootstrap,
                seed=args.seed + idx * (101 if manip == "occlude_drop" else 211),
            )
            by_attr_mag[attr] = summarize_magnitude(sub, delta_col)
        directional_tables[manip] = {"by_attribute": by_attr}
        magnitude_tables[manip] = {"by_attribute": by_attr_mag}

    for idx, attr in enumerate(ATTRIBUTES, start=1):
        delta_col = delta_cols.get(attr)
        if not delta_col:
            continue
        manip_comparisons[attr] = compare_manips(
            trial_df,
            attr,
            delta_col,
            bootstrap=args.bootstrap,
            seed=args.seed + idx * 997,
        )

    intervention_rankings = {}
    for manip, payload in directional_tables.items():
        effect_map = {
            attr: _safe_float(info.get("delta_favored_mean"))
            for attr, info in payload["by_attribute"].items()
        }
        effect_abs = {attr: (abs(val) if val is not None else None) for attr, val in effect_map.items()}
        normalized = None
        usable = {attr: value for attr, value in effect_abs.items() if value is not None}
        if usable:
            denom = sum(usable.values())
            normalized = {attr: value / denom for attr, value in usable.items()}
        intervention_rankings[manip] = {
            "effect_abs_normalized": normalized,
            "rank_order": _rank_order(normalized or {}),
            "spearman_vs_baseline_stageA_weights": _spearman(baseline_stagea["weights"], normalized or {}),
        }

    baseline_weight_order = _rank_order(baseline_stagea["weights"])
    full_weight_order = _rank_order(full_stagea.get("weights", {})) if full_stagea else None
    transitions = premise_transition_summary(trial_df)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.dataset)),
        "responses": str(responses_path),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "baseline_only_stageA": baseline_stagea,
        "full_stageA_summary": full_stagea,
        "baseline_weight_order": baseline_weight_order,
        "full_weight_order": full_weight_order,
        "directional_summaries": directional_tables,
        "magnitude_summaries": magnitude_tables,
        "drop_vs_equalize_differences": manip_comparisons,
        "intervention_rankings": intervention_rankings,
        "premise_transition_summary": transitions,
    }

    out_dir = ensure_dir(args.out_dir)
    write_json(payload, out_dir / "methodc_current_run_analysis.json")

    baseline_weight_row = {"weight_order": " > ".join(baseline_weight_order)}
    for attr in ATTRIBUTES:
        baseline_weight_row[f"weight_{attr}"] = baseline_stagea["weights"][attr]
        baseline_weight_row[f"beta_{attr}"] = baseline_stagea["beta"][attr]

    ranking_rows = []
    for manip, info in intervention_rankings.items():
        row = {
            "manipulation": manip,
            "baseline_weight_order": " > ".join(baseline_weight_order),
            "intervention_order": " > ".join(info["rank_order"]),
            "spearman_vs_baseline": info["spearman_vs_baseline_stageA_weights"],
        }
        normalized = info["effect_abs_normalized"] or {}
        for attr in ATTRIBUTES:
            row[f"effect_{attr}"] = normalized.get(attr)
        ranking_rows.append(row)

    directional_rows = []
    for manip, payload_row in directional_tables.items():
        for attr, info in payload_row["by_attribute"].items():
            directional_rows.append(
                {
                    "manipulation": manip,
                    "attribute": attr,
                    "n": info["n"],
                    "delta_favored_mean": _fmt_ci(info["delta_favored_mean"], info["delta_favored_mean_ci95"]),
                    "choice_flip_rate": _fmt_ci(info["choice_flip_rate"], info["choice_flip_rate_ci95"]),
                    "premise_flip_rate": _fmt_ci(info["premise_flip_rate"], info["premise_flip_rate_ci95"]),
                    "shift_away_from_target_rate": _fmt_ci(
                        info["shift_away_from_target_rate"],
                        info["shift_away_from_target_rate_ci95"],
                    ),
                }
            )

    comparison_rows = []
    for attr, info in manip_comparisons.items():
        if not info:
            continue
        comparison_rows.append(
            {
                "attribute": attr,
                "delta_equalize_minus_drop_directional": _fmt_ci(
                    info["delta_equalize_minus_drop_directional"],
                    info.get("delta_equalize_minus_drop_directional_ci95"),
                ),
                "delta_equalize_minus_drop_choice_flip_rate": _fmt_ci(
                    info["delta_equalize_minus_drop_choice_flip_rate"],
                    info.get("delta_equalize_minus_drop_choice_flip_rate_ci95"),
                ),
                "delta_equalize_minus_drop_premise_flip_rate": _fmt_ci(
                    info["delta_equalize_minus_drop_premise_flip_rate"],
                    info.get("delta_equalize_minus_drop_premise_flip_rate_ci95"),
                ),
            }
        )

    transition_rows = []
    for manip, info in transitions.items():
        for attr, payload_row in info["by_attribute"].items():
            transition_rows.append(
                {
                    "manipulation": manip,
                    "attribute": attr,
                    "n_shifted_from_target": payload_row["n_shifted_from_target"],
                    "top_destination": payload_row["top_destination"],
                    "top_destination_rate_among_shifted": payload_row["top_destination_rate_among_shifted"],
                }
            )

    md_lines = [
        "# Method C Current-Run Analysis",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Responses: `{responses_path}`",
        f"- Bootstrap resamples: `{args.bootstrap}` by `base_trial_id` / matched pair",
        "",
        "## Baseline-Only Stage A",
        _markdown_table(
            [baseline_weight_row],
            ["weight_order", "weight_E", "weight_A", "weight_S", "weight_D", "beta_E", "beta_A", "beta_S", "beta_D"],
        ),
        "",
        "## Baseline Vs Intervention Ranking",
        _markdown_table(
            ranking_rows,
            [
                "manipulation",
                "baseline_weight_order",
                "intervention_order",
                "spearman_vs_baseline",
                "effect_E",
                "effect_A",
                "effect_S",
                "effect_D",
            ],
        ),
        "",
        "## Directional Intervention Summary",
        _markdown_table(
            directional_rows,
            [
                "manipulation",
                "attribute",
                "n",
                "delta_favored_mean",
                "choice_flip_rate",
                "premise_flip_rate",
                "shift_away_from_target_rate",
            ],
        ),
        "",
        "## Drop Vs Equalize Differences",
        _markdown_table(
            comparison_rows,
            [
                "attribute",
                "delta_equalize_minus_drop_directional",
                "delta_equalize_minus_drop_choice_flip_rate",
                "delta_equalize_minus_drop_premise_flip_rate",
            ],
        ),
        "",
        "## Premise Transition Destinations",
        _markdown_table(
            transition_rows,
            [
                "manipulation",
                "attribute",
                "n_shifted_from_target",
                "top_destination",
                "top_destination_rate_among_shifted",
            ],
        ),
        "",
        "## Notes",
        "- `delta_favored_mean` is orientation-corrected: negative values mean the intervention reduced support for the option favored by the targeted attribute.",
        "- The baseline-only Stage A fit uses only `short_reason` rows as the reference preference model.",
        "- The JSON output contains the full mediation-style and magnitude-response summaries.",
    ]
    (out_dir / "methodc_current_run_analysis.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_dir / 'methodc_current_run_analysis.json'}")
    print(f"Wrote {out_dir / 'methodc_current_run_analysis.md'}")


if __name__ == "__main__":
    main()
