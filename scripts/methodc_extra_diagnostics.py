#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.features import load_responses
from src.analysis.judge_baselines import ATTRIBUTES, add_pairwise_drivers, add_tau_predictions
from src.utils.io import ensure_dir, write_json


@dataclass
class MetricStatus:
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"available": self.available}
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _parse_compare(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --compare-run format: {spec} (expected label=path)")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise SystemExit(f"Invalid --compare-run format: {spec} (empty label)")
        out.append((label, path))
    return out


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.float64, np.float32)):
        val = float(value)
        if np.isnan(val):
            return None
        return val
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Series):
        return _as_jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _as_jsonable(value.to_dict(orient="records"))
    return value


def _safe_rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.mean())


def _safe_mean_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _mode_or_none(series: pd.Series) -> str | None:
    vals = series.dropna().astype(str)
    if vals.empty:
        return None
    counts = vals.value_counts()
    top_n = int(counts.max())
    top = sorted(counts[counts == top_n].index.tolist())
    return top[0] if top else None


def _safe_first(series: pd.Series) -> Any:
    vals = series.dropna()
    if vals.empty:
        return None
    return vals.iloc[0]


def _load_dataset_trials(dataset_dir: Path) -> pd.DataFrame:
    trials_path = dataset_dir / "dataset_trials.parquet"
    if not trials_path.exists():
        raise SystemExit(f"Missing dataset file: {trials_path}")
    trials = pd.read_parquet(trials_path).copy()
    trials["trial_id"] = trials["trial_id"].astype(str)
    return trials


def _load_run_responses(path_or_dir: Path) -> tuple[pd.DataFrame, Path]:
    if path_or_dir.is_dir():
        responses_path = path_or_dir / "responses.jsonl"
    else:
        responses_path = path_or_dir
    if not responses_path.exists():
        raise SystemExit(f"Missing responses file: {responses_path}")
    responses = load_responses(responses_path).copy()
    responses["trial_id"] = responses["trial_id"].astype(str)
    return responses, responses_path


def _pick_delta_columns(trials_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for attr in ATTRIBUTES:
        base_col = f"delta_base_{attr}"
        shown_col = f"delta_{attr}"
        if base_col in trials_df.columns:
            out[attr] = base_col
        elif shown_col in trials_df.columns:
            out[attr] = shown_col
    return out


def _prepare_joined(
    trials_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    *,
    include_tau: bool = True,
    include_pairwise: bool = True,
) -> pd.DataFrame:
    merged = responses_df.copy()
    if include_tau:
        merged = add_tau_predictions(merged, trials_df)
    else:
        delta_cols = [f"delta_{a}" for a in ATTRIBUTES if f"delta_{a}" in trials_df.columns]
        merged = merged.merge(trials_df[["trial_id"] + delta_cols], on="trial_id", how="left")
    if include_pairwise:
        merged = add_pairwise_drivers(merged)

    trial_cols = [
        "trial_id",
        "manipulation",
        "attribute_target",
        "base_trial_id",
        "config_id",
        "labelA",
    ] + [c for c in trials_df.columns if c.startswith("delta_")]
    trial_cols = [c for c in trial_cols if c in trials_df.columns]
    trial_meta = trials_df[trial_cols].drop_duplicates("trial_id")
    merged = merged.merge(trial_meta, on="trial_id", how="left", suffixes=("", "_trial"))

    for col in ["manipulation", "attribute_target", "base_trial_id", "config_id", "labelA"]:
        trial_col = f"{col}_trial"
        if trial_col in merged.columns and col not in merged.columns:
            merged[col] = merged[trial_col]

    merged["is_A"] = merged["choice"].eq("A")
    return merged


def _build_pair_key(df: pd.DataFrame) -> pd.Series:
    key = pd.Series(index=df.index, dtype=object)

    if "base_trial_id" in df.columns:
        base_str = df["base_trial_id"].astype(str)
        ok_base = df["base_trial_id"].notna() & base_str.ne("")
        key.loc[ok_base] = base_str.loc[ok_base]

    if "config_id" in df.columns and "labelA" in df.columns:
        cfg = df["config_id"]
        lab = df["labelA"]
        ok_cfg = cfg.notna() & cfg.astype(str).ne("")
        ok_lab = lab.notna() & lab.astype(str).ne("")
        ok = ok_cfg & ok_lab
        fallback = cfg.astype(str) + "::" + lab.astype(str)
        key.loc[key.isna() & ok] = fallback.loc[key.isna() & ok]

    key = key.fillna(df["trial_id"].astype(str))
    return key.astype(str)


def _trial_level_table(rows_df: pd.DataFrame, stagea_map: pd.DataFrame | None = None) -> pd.DataFrame:
    work = rows_df.copy()
    if stagea_map is not None:
        work = work.merge(stagea_map, on="trial_id", how="left")
        work["stageA_driver_choice_cond"] = np.where(
            work["choice"] == "A",
            work.get("driver_A"),
            np.where(work["choice"] == "B", work.get("driver_B"), None),
        )
        valid = work["premise_ok"] & work["choice_ok"] & work["stageA_driver_choice_cond"].notna()
        work["align_stageA_driver_row"] = np.where(
            valid,
            work["premise_attr"] == work["stageA_driver_choice_cond"],
            np.nan,
        )
    else:
        work["align_stageA_driver_row"] = np.nan

    if "tau_driver" in work.columns:
        valid = work["premise_ok"] & work["tau_ok"] & work["tau_driver"].notna()
        work["align_tau_driver_row"] = np.where(valid, work["premise_attr"] == work["tau_driver"], np.nan)
    else:
        work["align_tau_driver_row"] = np.nan

    if "pair_driver" in work.columns:
        valid = work["premise_ok"] & work["pairwise_ok"] & work["pair_driver"].notna()
        work["align_pair_driver_row"] = np.where(valid, work["premise_attr"] == work["pair_driver"], np.nan)
    else:
        work["align_pair_driver_row"] = np.nan

    meta_cols = [
        c
        for c in [
            "manipulation",
            "attribute_target",
            "base_trial_id",
            "config_id",
            "labelA",
            "block",
            "variant",
        ]
        if c in work.columns
    ]
    meta_cols += [c for c in work.columns if c.startswith("delta_")]
    meta_cols = sorted(set(meta_cols))

    rows: list[dict[str, Any]] = []
    for trial_id, grp in work.groupby("trial_id", sort=False):
        rec: dict[str, Any] = {"trial_id": str(trial_id)}
        for col in meta_cols:
            rec[col] = _safe_first(grp[col])

        choice_grp = grp[grp["choice_ok"]].copy()
        rec["n_choice"] = int(len(choice_grp))
        if len(choice_grp):
            p_a = float(choice_grp["choice"].eq("A").mean())
            rec["pA"] = p_a
            rec["choice_majority"] = "A" if p_a >= 0.5 else "B"
        else:
            rec["pA"] = float("nan")
            rec["choice_majority"] = None

        premise_grp = grp[grp["premise_ok"] & grp["premise_attr"].notna()].copy()
        rec["n_premise"] = int(len(premise_grp))
        rec["premise_majority"] = _mode_or_none(premise_grp["premise_attr"])

        if "tau_driver" in grp.columns:
            tau_grp = grp[grp["tau_ok"] & grp["tau_driver"].notna()].copy()
            rec["tau_driver_majority"] = _mode_or_none(tau_grp["tau_driver"])
        else:
            rec["tau_driver_majority"] = None

        if "pair_driver" in grp.columns:
            pair_grp = grp[grp["pairwise_ok"] & grp["pair_driver"].notna()].copy()
            rec["pair_driver_majority"] = _mode_or_none(pair_grp["pair_driver"])
        else:
            rec["pair_driver_majority"] = None

        rec["align_stageA_driver_rate"] = _safe_mean_numeric(grp["align_stageA_driver_row"])
        rec["align_tau_driver_rate"] = _safe_mean_numeric(grp["align_tau_driver_row"])
        rec["align_pair_driver_rate"] = _safe_mean_numeric(grp["align_pair_driver_row"])
        rows.append(rec)

    trial_df = pd.DataFrame(rows)
    trial_df["pair_key"] = _build_pair_key(trial_df)
    return trial_df


def _pair_baseline_vs_manip(trial_df: pd.DataFrame, manip: str) -> pd.DataFrame:
    baseline = trial_df[trial_df["manipulation"] == "short_reason"].copy()
    if baseline.empty:
        baseline = trial_df[trial_df["manipulation"] == "split_reason"].copy()
    baseline = baseline[baseline["pA"].notna()].copy()

    occl = trial_df[(trial_df["manipulation"] == manip) & (trial_df["pA"].notna())].copy()
    if baseline.empty or occl.empty:
        return pd.DataFrame()

    baseline = baseline.sort_values("trial_id").drop_duplicates("pair_key", keep="first")
    keep_base = [
        "pair_key",
        "trial_id",
        "pA",
        "choice_majority",
        "premise_majority",
        "tau_driver_majority",
        "pair_driver_majority",
        "align_stageA_driver_rate",
        "align_tau_driver_rate",
        "align_pair_driver_rate",
    ]
    keep_base = [c for c in keep_base if c in baseline.columns]
    base = baseline[keep_base].rename(
        columns={
            "trial_id": "base_trial_id_row",
            "pA": "base_pA",
            "choice_majority": "base_choice_majority",
            "premise_majority": "base_premise_majority",
            "tau_driver_majority": "base_tau_driver_majority",
            "pair_driver_majority": "base_pair_driver_majority",
            "align_stageA_driver_rate": "base_align_stageA_driver_rate",
            "align_tau_driver_rate": "base_align_tau_driver_rate",
            "align_pair_driver_rate": "base_align_pair_driver_rate",
        }
    )

    paired = occl.merge(base, on="pair_key", how="inner")
    return paired


def _directional_effects(trial_df: pd.DataFrame, delta_cols: dict[str, str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue

        out: dict[str, Any] = {"by_attribute": {}, "normalized_abs": {}}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            if sub.empty:
                continue
            delta_col = delta_cols.get(attr)
            if not delta_col or delta_col not in sub.columns:
                continue

            sign = np.sign(pd.to_numeric(sub[delta_col], errors="coerce").fillna(0.0))
            raw = pd.to_numeric(sub["pA"], errors="coerce") - pd.to_numeric(sub["base_pA"], errors="coerce")
            directional = np.where(sign > 0, raw, np.where(sign < 0, -raw, np.nan))

            mask = np.isfinite(directional) & np.isfinite(raw)
            directional = pd.Series(directional[mask], dtype=float)
            raw_valid = pd.Series(raw[mask], dtype=float)
            signs = pd.Series(sign[mask], dtype=float)
            if directional.empty:
                continue

            entry = {
                "delta_pA_mean": float(raw_valid.mean()),
                "delta_favored_mean": float(directional.mean()),
                "flip_rate_any": float((raw_valid != 0).mean()),
                "n": int(len(directional)),
                "n_sign_pos": int((signs > 0).sum()),
                "n_sign_neg": int((signs < 0).sum()),
                "n_sign_zero": int((signs == 0).sum()),
            }
            out["by_attribute"][attr] = entry

        abs_vals = {
            attr: abs(info["delta_favored_mean"])
            for attr, info in out["by_attribute"].items()
            if np.isfinite(info["delta_favored_mean"])
        }
        denom = sum(abs_vals.values())
        if denom > 0:
            out["normalized_abs"] = {attr: val / denom for attr, val in abs_vals.items()}
        if out["by_attribute"]:
            results[manip] = out
    return results


def _magnitude_response(trial_df: pd.DataFrame, delta_cols: dict[str, str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            if sub.empty:
                continue
            delta_col = delta_cols.get(attr)
            if not delta_col or delta_col not in sub.columns:
                continue
            d = pd.to_numeric(sub[delta_col], errors="coerce").fillna(0.0)
            sign = np.sign(d)
            mag = np.abs(d)
            raw = pd.to_numeric(sub["pA"], errors="coerce") - pd.to_numeric(sub["base_pA"], errors="coerce")
            directional = np.where(sign > 0, raw, np.where(sign < 0, -raw, np.nan))
            sub = sub.assign(mag=mag, delta_favored=directional)
            sub = sub[np.isfinite(sub["delta_favored"])]
            if sub.empty:
                continue
            mag_out: dict[str, Any] = {}
            for m in sorted(sub["mag"].unique()):
                grp = sub[sub["mag"] == m]
                if grp.empty:
                    continue
                key = str(int(m)) if float(m).is_integer() else str(m)
                mag_out[key] = {
                    "delta_favored_mean": float(grp["delta_favored"].mean()),
                    "flip_rate_any": float((grp["delta_favored"] != 0).mean()),
                    "n": int(len(grp)),
                }
            if mag_out:
                by_attr[attr] = mag_out
        if by_attr:
            results[manip] = {"by_attribute": by_attr}
    return results


def _choice_flip(trial_df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            if sub.empty:
                continue
            valid = sub["choice_majority"].isin(["A", "B"]) & sub["base_choice_majority"].isin(["A", "B"])
            sub = sub[valid]
            if sub.empty:
                continue
            choice_flip = sub["choice_majority"] != sub["base_choice_majority"]
            to_a = (sub["base_choice_majority"] == "B") & (sub["choice_majority"] == "A")
            to_b = (sub["base_choice_majority"] == "A") & (sub["choice_majority"] == "B")
            by_attr[attr] = {
                "choice_flip_rate": _safe_rate(choice_flip),
                "flip_to_A_rate": _safe_rate(to_a),
                "flip_to_B_rate": _safe_rate(to_b),
                "n": int(len(sub)),
            }
        if by_attr:
            results[manip] = {"by_attribute": by_attr}
    return results


def _premise_shift(trial_df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            valid = sub["premise_majority"].notna() & sub["base_premise_majority"].notna()
            sub = sub[valid]
            if sub.empty:
                continue
            premise_flip = sub["premise_majority"] != sub["base_premise_majority"]
            to_target = (sub["premise_majority"] == attr) & (sub["base_premise_majority"] != attr)
            away_target = (sub["premise_majority"] != attr) & (sub["base_premise_majority"] == attr)
            by_attr[attr] = {
                "premise_flip_rate": _safe_rate(premise_flip),
                "shift_to_target_rate": _safe_rate(to_target),
                "shift_away_from_target_rate": _safe_rate(away_target),
                "n": int(len(sub)),
            }
        if by_attr:
            results[manip] = {"by_attribute": by_attr}
    return results


def _mediation_proxy(trial_df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            valid = (
                sub["choice_majority"].isin(["A", "B"])
                & sub["base_choice_majority"].isin(["A", "B"])
                & sub["premise_majority"].notna()
                & sub["base_premise_majority"].notna()
            )
            sub = sub[valid]
            if sub.empty:
                continue
            choice_flip = sub["choice_majority"] != sub["base_choice_majority"]
            premise_flip = sub["premise_majority"] != sub["base_premise_majority"]
            both = choice_flip & premise_flip
            with_premise = sub[premise_flip]
            without_premise = sub[~premise_flip]
            by_attr[attr] = {
                "choice_flip_rate": _safe_rate(choice_flip),
                "premise_flip_rate": _safe_rate(premise_flip),
                "both_flip_rate": _safe_rate(both),
                "p_choice_flip_given_premise_flip": _safe_rate(
                    with_premise["choice_majority"] != with_premise["base_choice_majority"]
                )
                if len(with_premise)
                else float("nan"),
                "p_choice_flip_given_no_premise_flip": _safe_rate(
                    without_premise["choice_majority"] != without_premise["base_choice_majority"]
                )
                if len(without_premise)
                else float("nan"),
                "shared_flip_fraction_of_choice_flips": float(both.sum() / choice_flip.sum())
                if choice_flip.sum() > 0
                else float("nan"),
                "n": int(len(sub)),
            }
        if by_attr:
            results[manip] = {"by_attribute": by_attr}
    return results


def _resolve_stagea_driver_map(stagea_dir: Path | None) -> tuple[pd.DataFrame | None, MetricStatus]:
    if stagea_dir is None:
        return None, MetricStatus(False, "No --stageA directory provided")
    contrib_path = stagea_dir / "stageA_contributions.parquet"
    design_path = stagea_dir / "stageA_design.parquet"
    if not contrib_path.exists():
        return None, MetricStatus(False, f"Missing contributions file: {contrib_path}")
    if not design_path.exists():
        return None, MetricStatus(False, f"Missing design file: {design_path}")

    contrib = pd.read_parquet(contrib_path).copy()
    design = pd.read_parquet(design_path).copy().reset_index(drop=True)
    if "trial_id" not in design.columns:
        return None, MetricStatus(False, "stageA_design.parquet has no trial_id column")

    if "trial_id" in contrib.columns and pd.api.types.is_numeric_dtype(contrib["trial_id"]):
        idx = pd.to_numeric(contrib["trial_id"], errors="coerce")
    else:
        idx = pd.to_numeric(contrib.index, errors="coerce")
    contrib = contrib.assign(_idx=idx).dropna(subset=["_idx"])
    contrib["_idx"] = contrib["_idx"].astype(int)
    contrib = contrib[(contrib["_idx"] >= 0) & (contrib["_idx"] < len(design))]
    contrib["trial_id_real"] = design.iloc[contrib["_idx"]]["trial_id"].astype(str).to_numpy()

    keep = ["trial_id_real"]
    for col in ["driver_A", "driver_B", "driver"]:
        if col in contrib.columns:
            keep.append(col)
    if len(keep) <= 1:
        return None, MetricStatus(False, "No driver columns in stageA_contributions.parquet")

    out = contrib[keep].drop_duplicates("trial_id_real").rename(columns={"trial_id_real": "trial_id"})
    return out, MetricStatus(True)


def _alignment_deltas(trial_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for manip in sorted(trial_df["manipulation"].dropna().unique()):
        if manip not in {"occlude_drop", "occlude_equalize", "occlude_swap"}:
            continue
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if paired.empty:
            continue
        by_attr: dict[str, Any] = {}
        for attr in ATTRIBUTES:
            sub = paired[paired["attribute_target"] == attr].copy()
            if sub.empty:
                continue
            entry = {"n": int(len(sub))}
            for prefix, col, base_col in [
                ("stageA", "align_stageA_driver_rate", "base_align_stageA_driver_rate"),
                ("tau", "align_tau_driver_rate", "base_align_tau_driver_rate"),
                ("pair", "align_pair_driver_rate", "base_align_pair_driver_rate"),
            ]:
                if col not in sub.columns or base_col not in sub.columns:
                    continue
                x = pd.to_numeric(sub[col], errors="coerce")
                y = pd.to_numeric(sub[base_col], errors="coerce")
                valid = x.notna() & y.notna()
                if not valid.any():
                    continue
                entry[f"{prefix}_align_occl"] = float(x[valid].mean())
                entry[f"{prefix}_align_base"] = float(y[valid].mean())
                entry[f"{prefix}_align_delta"] = float(x[valid].mean() - y[valid].mean())
            by_attr[attr] = entry
        if by_attr:
            out[manip] = {"by_attribute": by_attr}
    return out


def _cross_model_agreement(
    trials_df: pd.DataFrame,
    run_specs: list[tuple[str, Path]],
    delta_cols: dict[str, str],
) -> tuple[dict[str, Any], MetricStatus]:
    if len(run_specs) < 2:
        return {}, MetricStatus(False, "Need at least two --compare-run inputs to compute agreement")

    vectors: dict[str, dict[tuple[str, str], float]] = {}
    for label, path in run_specs:
        responses_df, _ = _load_run_responses(path)
        joined = _prepare_joined(trials_df, responses_df, include_tau=False, include_pairwise=False)
        trial_df = _trial_level_table(joined, stagea_map=None)
        directionals = _directional_effects(trial_df, delta_cols)
        vec: dict[tuple[str, str], float] = {}
        for manip, payload in directionals.items():
            for attr, info in payload.get("by_attribute", {}).items():
                val = info.get("delta_favored_mean")
                if val is None or not np.isfinite(val):
                    continue
                vec[(manip, attr)] = float(val)
        if vec:
            vectors[label] = vec

    labels = sorted(vectors.keys())
    if len(labels) < 2:
        return {}, MetricStatus(False, "Insufficient comparable directional vectors across compare runs")

    pairwise: list[dict[str, Any]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            keys = sorted(set(vectors[a]).intersection(vectors[b]))
            if len(keys) < 3:
                pairwise.append(
                    {
                        "run_a": a,
                        "run_b": b,
                        "n_common_dims": len(keys),
                        "pearson": None,
                        "spearman": None,
                        "note": "Need >=3 common dims for stable correlation",
                    }
                )
                continue
            x = np.array([vectors[a][k] for k in keys], dtype=float)
            y = np.array([vectors[b][k] for k in keys], dtype=float)
            pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
            x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
            y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
            spearman = float(np.corrcoef(x_rank, y_rank)[0, 1]) if len(x_rank) > 1 else float("nan")
            pairwise.append(
                {
                    "run_a": a,
                    "run_b": b,
                    "n_common_dims": len(keys),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )

    payload = {
        "pairwise": pairwise,
        "vectors": {k: {f"{m}:{a}": v for (m, a), v in vec.items()} for k, vec in vectors.items()},
    }
    return payload, MetricStatus(True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute extra Method-C diagnostics from existing run outputs."
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory containing dataset_trials.parquet")
    parser.add_argument("--responses", required=True, help="Responses jsonl path or run directory containing responses.jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--stageA", default=None, help="Optional Stage A output directory (for Stage A driver alignment deltas)")
    parser.add_argument(
        "--compare-run",
        action="append",
        default=[],
        help="Optional cross-run agreement input: label=responses_path_or_run_dir. Repeat as needed.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    responses_df, responses_path = _load_run_responses(Path(args.responses))
    trials_df = _load_dataset_trials(dataset_dir)
    joined = _prepare_joined(trials_df, responses_df, include_tau=True, include_pairwise=True)

    stagea_map, stagea_status = _resolve_stagea_driver_map(Path(args.stageA) if args.stageA else None)
    trial_df = _trial_level_table(joined, stagea_map=stagea_map)
    delta_cols = _pick_delta_columns(trials_df)

    availability: dict[str, Any] = {}

    directional = _directional_effects(trial_df, delta_cols)
    availability["directional_effects"] = MetricStatus(
        bool(directional),
        None if directional else "No matched baseline/occlusion pairs at trial-family level",
    ).to_dict()

    magnitude = _magnitude_response(trial_df, delta_cols)
    availability["magnitude_response"] = MetricStatus(
        bool(magnitude),
        None if magnitude else "No usable non-zero target deltas after pairing",
    ).to_dict()

    flips = _choice_flip(trial_df)
    availability["choice_flip_rates"] = MetricStatus(
        bool(flips),
        None if flips else "No matched baseline/occlusion majority-choice pairs",
    ).to_dict()

    premise_shift = _premise_shift(trial_df)
    availability["premise_shift_rates"] = MetricStatus(
        bool(premise_shift),
        None if premise_shift else "Premise parsing unavailable or no paired premise-majority rows",
    ).to_dict()

    mediation = _mediation_proxy(trial_df)
    availability["mediation_proxy"] = MetricStatus(
        bool(mediation),
        None if mediation else "Need paired rows with both valid majority choices and premise labels",
    ).to_dict()

    alignment_deltas = _alignment_deltas(trial_df)
    if stagea_status.available:
        availability["intervention_alignment_deltas"] = MetricStatus(
            bool(alignment_deltas),
            None if alignment_deltas else "Stage A map found, but no paired rows with valid alignment rates",
        ).to_dict()
    else:
        availability["intervention_alignment_deltas"] = stagea_status.to_dict()

    compare_specs = _parse_compare(args.compare_run)
    cross_agreement, cross_status = _cross_model_agreement(trials_df, compare_specs, delta_cols)
    availability["cross_model_causal_agreement"] = cross_status.to_dict()

    unavailable_extra = {
        "formal_causal_mediation_effects": {
            "available": False,
            "reason": "Not identifiable from current saved artifacts without extra causal assumptions and an explicit mediation model",
        },
        "path_specific_counterfactual_effects": {
            "available": False,
            "reason": "Would require intervention-specific counterfactual labels not present in current outputs",
        },
    }

    pairing_counts: dict[str, Any] = {}
    for manip in ["occlude_drop", "occlude_equalize", "occlude_swap"]:
        paired = _pair_baseline_vs_manip(trial_df, manip)
        if not paired.empty:
            pairing_counts[manip] = int(len(paired))

    summary = {
        "dataset": str(dataset_dir),
        "responses": str(responses_path),
        "n_rows_responses": int(len(responses_df)),
        "n_trials_dataset": int(trials_df["trial_id"].nunique()),
        "n_trials_in_responses": int(joined["trial_id"].nunique()),
        "trial_level_rows": int(len(trial_df)),
        "pairing_counts": pairing_counts,
        "delta_columns_used": delta_cols,
        "availability": availability,
        "directional_effects": directional,
        "magnitude_response": magnitude,
        "choice_flip_rates": flips,
        "premise_shift_rates": premise_shift,
        "mediation_proxy": mediation,
        "intervention_alignment_deltas": alignment_deltas,
        "cross_model_causal_agreement": cross_agreement,
        "not_currently_computable": unavailable_extra,
    }

    out_dir = ensure_dir(args.out)
    write_json(_as_jsonable(summary), out_dir / "methodc_extra_diagnostics.json")

    md_lines = [
        "# Method-C Extra Diagnostics",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Responses: `{responses_path}`",
        f"- Parsed response rows: {len(responses_df)}",
        f"- Distinct response trials: {joined['trial_id'].nunique()}",
        f"- Trial-level rows: {len(trial_df)}",
        "",
        "## Pairing",
    ]
    if pairing_counts:
        for manip, n in pairing_counts.items():
            md_lines.append(f"- {manip}: {n} paired trials")
    else:
        md_lines.append("- no baseline/occlusion pairs found")

    md_lines.extend(
        [
            "",
            "## Availability",
        ]
    )
    for key, status in availability.items():
        flag = "yes" if status.get("available") else "no"
        reason = status.get("reason")
        md_lines.append(f"- {key}: {flag}" + (f" ({reason})" if reason else ""))

    md_lines.extend(
        [
            "",
            "## Not Currently Computable",
        ]
    )
    for key, payload in unavailable_extra.items():
        md_lines.append(f"- {key}: {payload['reason']}")

    (out_dir / "methodc_extra_diagnostics.md").write_text("\n".join(md_lines) + "\n")
    print(f"OK wrote {out_dir / 'methodc_extra_diagnostics.json'}")
    print(f"OK wrote {out_dir / 'methodc_extra_diagnostics.md'}")


if __name__ == "__main__":
    main()
