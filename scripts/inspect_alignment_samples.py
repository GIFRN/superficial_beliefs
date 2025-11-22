#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
from typing import Any, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from src.analysis.features import load_responses

ATTRIBUTES = ["E", "A", "S", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Stage A vs Stage B alignment samples."
    )
    parser.add_argument(
        "--contributions",
        default="results/stage_A_openai_gpt5mini/stageA_contributions.parquet",
        help="Path to Stage A per-trial contribution parquet.",
    )
    parser.add_argument(
        "--design",
        default="results/stage_A_openai_gpt5mini/stageA_design.parquet",
        help="Path to Stage A design parquet (used to recover true trial_ids).",
    )
    parser.add_argument(
        "--configs",
        default="data/generated/v1_short/dataset_configs.parquet",
        help="Path to dataset configuration parquet (used to attach drug configurations).",
    )
    parser.add_argument(
        "--responses",
        default="data/runs/v1_short_openai_gpt5mini/responses.jsonl",
        help="Path to responses JSONL file.",
    )
    parser.add_argument(
        "--aligned-n",
        type=int,
        default=5,
        help="Number of aligned samples to show.",
    )
    parser.add_argument(
        "--misaligned-n",
        type=int,
        default=5,
        help="Number of misaligned samples to show.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling rows.",
    )
    return parser.parse_args()


def load_contributions(
    path: str | Path,
    design_path: str | Path | None,
    configs_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrib = pd.read_parquet(path).reset_index(drop=True)
    if "trial_id" in contrib.columns:
        contrib = contrib.rename(columns={"trial_id": "row_idx"})

    used_design_ids = False
    design_info = pd.DataFrame()
    design_subset = pd.DataFrame()

    if design_path is not None:
        try:
            design = pd.read_parquet(design_path).reset_index(drop=True)
        except FileNotFoundError:
            design = None
        if design is not None and "trial_id" in design.columns and len(design) == len(contrib):
            contrib["trial_id"] = design["trial_id"].astype(str)
            used_design_ids = True
            design_subset = design.assign(trial_id=design["trial_id"].astype(str))
        else:
            contrib["trial_id"] = contrib.index.astype(str)
    else:
        contrib["trial_id"] = contrib.index.astype(str)

    if not design_subset.empty:
        detail_cols = [
            col
            for col in [
                "trial_id",
                "config_id",
                "labelA",
                "delta_E",
                "delta_A",
                "delta_S",
                "delta_D",
            ]
            if col in design_subset.columns
        ]
        design_info = design_subset[detail_cols].drop_duplicates("trial_id")

        if configs_path:
            try:
                configs = pd.read_parquet(configs_path)
            except FileNotFoundError:
                configs = None
            if configs is not None and "config_id" in configs.columns:
                configs = configs.copy()
                configs["config_id"] = configs["config_id"].astype(str)
                design_info = design_info.merge(
                    configs[["config_id", "levels_left", "levels_right"]],
                    on="config_id",
                    how="left",
                )

    cols = ["trial_id", "driver"] + [col for col in contrib.columns if col.startswith("C_")]
    result = contrib[cols].drop_duplicates("trial_id")
    if not used_design_ids:
        print(
            "Warning: Stage A design mapping unavailable or mismatched; "
            "using positional indices for trial identifiers."
        )
    return result, design_info


def load_valid_responses(path: str | Path) -> pd.DataFrame:
    responses = load_responses(path)
    if "trial_id" in responses.columns:
        responses["trial_id"] = responses["trial_id"].astype(str)
    valid = responses[
        responses["premise_attr"].isin(ATTRIBUTES) & responses["premise_ok"]
    ].copy()
    return valid


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.head(0)
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).sort_values(["trial_id", "seed"])


def format_contrib_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col.startswith("C_")]


def render_levels(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
    if isinstance(parsed, dict):
        return ", ".join(f"{k}:{parsed[k]}" for k in sorted(parsed))
    return str(parsed)


def describe_drug_a(label: Any) -> str:
    if isinstance(label, str):
        label = label.strip().upper()
        if label == "A":
            return "Drug A uses Left arm (levels_left)"
        if label == "B":
            return "Drug A uses Right arm (levels_right)"
    return "Drug A arm unknown"


def format_deltas(row: pd.Series) -> str:
    pieces = []
    for attr in ATTRIBUTES:
        col = f"delta_{attr}"
        if col in row and pd.notna(row[col]):
            value = row[col]
            pieces.append(f"{attr}:{value:+g}")
    return ", ".join(pieces)


def main() -> None:
    args = parse_args()
    contributions, design_info = load_contributions(args.contributions, args.design, args.configs)
    responses = load_valid_responses(args.responses)
    merged = responses.merge(contributions, on="trial_id", how="left")
    if not design_info.empty:
        merged = merged.merge(design_info, on="trial_id", how="left")
    if "config_id_x" in merged.columns or "config_id_y" in merged.columns:
        merged["config_id"] = merged.get("config_id_x", pd.Series(dtype=object)).fillna(
            merged.get("config_id_y", pd.Series(dtype=object))
        )
        merged = merged.drop(columns=[col for col in ["config_id_x", "config_id_y"] if col in merged.columns])
    merged["aligned"] = merged["premise_attr"] == merged["driver"]
    print(
        f"Loaded {len(contributions)} contribution rows "
        f"(missing driver: {contributions['driver'].isna().sum()})"
    )
    print(f"Loaded {len(responses)} valid responses")
    print(
        f"Merged responses: {len(merged)} rows; "
        f"missing driver assignments: {merged['driver'].isna().sum()}"
    )

    contrib_cols = format_contrib_columns(contributions.columns)
    config_cols = [col for col in ["config_id", "levels_left", "levels_right"] if col in merged.columns]

    aligned = merged[merged["aligned"]]
    misaligned = merged[~merged["aligned"]]

    aligned_sample = sample_rows(aligned, args.aligned_n, args.seed)
    misaligned_sample = sample_rows(misaligned, args.misaligned_n, args.seed + 1)

    def print_block(title: str, df: pd.DataFrame, total: int) -> None:
        print(f"\n== {title} (showing {len(df)} of {total} total) ==")
        if df.empty:
            print("No rows available.")
            return
        for _, row in df.iterrows():
            print(
                f"- trial_id={row['trial_id']} config_id={row.get('config_id', 'NA')} "
                f"block={row.get('block', 'NA')} manipulation={row.get('manipulation', 'NA')} "
                f"variant={row.get('variant', 'NA')}"
            )
            driver = row.get("driver", "NA")
            premise_attr = row.get("premise_attr", "NA")
            print(f"  StageA_driver={driver} | StageB_premise={premise_attr}")
            contrib_details = ", ".join(
                f"{col}:{row.get(col):.3f}" for col in contrib_cols if pd.notna(row.get(col))
            )
            if contrib_details:
                print(f"  Contributions -> {contrib_details}")
            labelA = row.get("labelA")
            orientation = describe_drug_a(labelA)
            if labelA is not None and not (isinstance(labelA, float) and pd.isna(labelA)):
                print(f"  LabelA={labelA} ({orientation})")
            else:
                print(f"  LabelA=<unknown> ({orientation})")
            if config_cols:
                config_bits: list[str] = []
                if "config_id" in config_cols:
                    config_bits.append(f"id={row.get('config_id', 'NA')}")
                left = render_levels(row.get("levels_left")) if "levels_left" in config_cols else ""
                right = render_levels(row.get("levels_right")) if "levels_right" in config_cols else ""
                if left or right:
                    config_bits.append(f"L[{left or '?'}] vs R[{right or '?'}]")
                if config_bits:
                    print("  Drug config: " + " | ".join(config_bits))
            delta_summary = format_deltas(row)
            if delta_summary:
                print(f"  Deltas (A minus B): {delta_summary}")
            premise_text = row.get("premise_text") or ""
            if premise_text:
                snippet = " ".join(premise_text.split())
                print(f"  Premise: {snippet}")
            else:
                print("  Premise: <empty>")
            print(f"  Response seed={row.get('seed', 'NA')} choice={row.get('choice', 'NA')}")

    print_block("Aligned samples", aligned_sample, len(aligned))
    print_block("Misaligned samples", misaligned_sample, len(misaligned))


if __name__ == "__main__":
    main()
