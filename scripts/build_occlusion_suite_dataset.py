#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.occlusions import apply_occlusion_to_deltas
from src.data.orders import positions_for_order
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_json


def _variant_rows(
    row: pd.Series,
    *,
    attributes: list[str],
    orders: list[list[str]],
) -> list[dict[str, object]]:
    base_trial_id = str(row["trial_id"])
    base_deltas = {attr: int(row[f"delta_base_{attr}"]) for attr in attributes}
    base_order_a = list(orders[int(row["order_id_A"])])
    base_order_b = list(orders[int(row["order_id_B"])])

    variants: list[tuple[str, str | None]] = [("short_reason", None)]
    variants.extend([("occlude_equalize", attr) for attr in attributes])
    variants.extend([("occlude_drop", attr) for attr in attributes])

    out: list[dict[str, object]] = []
    for manipulation, attribute_target in variants:
        visible_deltas = apply_occlusion_to_deltas(base_deltas, manipulation, attribute_target)
        order_a = list(base_order_a)
        order_b = list(base_order_b)
        if manipulation == "occlude_drop" and attribute_target is not None:
            order_a = [attr for attr in order_a if attr != attribute_target]
            order_b = [attr for attr in order_b if attr != attribute_target]

        pos_a = positions_for_order(order_a)
        pos_b = positions_for_order(order_b)
        if manipulation == "occlude_drop" and attribute_target is not None:
            pos_a[attribute_target] = 0
            pos_b[attribute_target] = 0

        variant_suffix = "baseline" if manipulation == "short_reason" else f"{manipulation}_{attribute_target}"
        variant_name = str(row.get("variant_name", ""))
        trial_id = f"{base_trial_id}__{variant_suffix}"

        payload = row.to_dict()
        payload["trial_id"] = trial_id
        payload["base_trial_id"] = base_trial_id
        payload["base_variant_name"] = variant_name
        payload["occlusion_variant_name"] = variant_suffix
        payload["manipulation"] = manipulation
        payload["attribute_target"] = attribute_target
        payload["inject_offset"] = 0
        payload["source_full_trial_id"] = base_trial_id
        payload["source_full_family_id"] = str(row.get("family_id", ""))

        for attr in attributes:
            payload[f"delta_{attr}"] = int(visible_deltas[attr])
            payload[f"posA_{attr}"] = int(pos_a.get(attr, 0))
            payload[f"posB_{attr}"] = int(pos_b.get(attr, 0))
            payload[f"delta_pos_{attr}"] = int(pos_a.get(attr, 0) - pos_b.get(attr, 0))

        out.append(payload)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the final same-order occlusion suite from data/test/full_trials.parquet."
    )
    parser.add_argument("--config", default="data/configs/dataset.yml")
    parser.add_argument("--source-dir", default="data/test")
    parser.add_argument("--out-dir", default="data/occlusion_suite/test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    attributes = list(cfg.profiles.attributes)
    orders = cfg.orders_permutations

    source_dir = Path(args.source_dir).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    configs_df = pd.read_parquet(source_dir / "dataset_configs.parquet").copy()
    trials_df = pd.read_parquet(source_dir / "full_trials.parquet").copy()
    trials_df = trials_df[(trials_df["block"] == "B3") & (trials_df["manipulation"] == "short_reason")].copy()
    trials_df = trials_df.sort_values(["family_id", "variant_name"]).reset_index(drop=True)

    if not trials_df["same_order_prompt"].astype(bool).all():
        raise SystemExit("Expected all source full-test rows to already be same-order prompts.")

    variant_rows: list[dict[str, object]] = []
    for _, row in trials_df.iterrows():
        variant_rows.extend(_variant_rows(row, attributes=attributes, orders=orders))
    suite_df = pd.DataFrame(variant_rows)

    manip_order = {"short_reason": 0, "occlude_equalize": 1, "occlude_drop": 2}
    attr_order = {attr: idx for idx, attr in enumerate(attributes)}
    suite_df["_manip_order"] = suite_df["manipulation"].map(manip_order).fillna(99).astype(int)
    suite_df["_attr_order"] = suite_df["attribute_target"].map(attr_order).fillna(-1).astype(int)
    suite_df = (
        suite_df.sort_values(["base_trial_id", "_manip_order", "_attr_order", "trial_id"])
        .drop(columns=["_manip_order", "_attr_order"])
        .reset_index(drop=True)
    )

    configs_path = out_dir / "dataset_configs.parquet"
    trials_path = out_dir / "dataset_trials.parquet"
    configs_df.to_parquet(configs_path, index=False)
    suite_df.to_parquet(trials_path, index=False)

    sample_cols = [
        "trial_id",
        "base_trial_id",
        "family_id",
        "variant_name",
        "base_variant_name",
        "occlusion_variant_name",
        "labelA",
        "manipulation",
        "attribute_target",
        "order_id_A",
        "order_id_B",
        "source_order_id_A",
        "source_order_id_B",
        "same_order_prompt",
        "slot_A_profile",
        "slot_B_profile",
        "delta_E",
        "delta_A",
        "delta_S",
        "delta_D",
        "delta_base_E",
        "delta_base_A",
        "delta_base_S",
        "delta_base_D",
    ]
    suite_df[sample_cols].to_csv(out_dir / "samples.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.relative_to(ROOT)),
        "source_trials_filename": "full_trials.parquet",
        "source_configs_filename": "dataset_configs.parquet",
        "n_configs": int(configs_df["config_id"].nunique()),
        "n_base_trials": int(trials_df["trial_id"].nunique()),
        "n_trials": int(len(suite_df)),
        "blocks": {k: int(v) for k, v in suite_df["block"].value_counts().to_dict().items()},
        "counts_by_manipulation": {k: int(v) for k, v in suite_df["manipulation"].value_counts().to_dict().items()},
        "counts_by_attribute_target": {
            str(k): int(v)
            for k, v in suite_df["attribute_target"].fillna("None").value_counts().to_dict().items()
        },
        "suite": {
            "type": "occlusion_suite",
            "baseline_manipulation": "short_reason",
            "occlusion_manipulations": ["occlude_equalize", "occlude_drop"],
            "paired_key": "base_trial_id",
            "attributes": attributes,
            "variants_per_base_trial": 1 + 2 * len(attributes),
            "note": "Baseline and occlusions share order_id_A/B, paraphrase_id, and seed within each base_trial_id.",
        },
    }
    write_json(manifest, out_dir / "MANIFEST.json")

    readme_lines = [
        "# Occlusion Suite Test Dataset",
        "",
        "This dataset is built directly from `data/test/full_trials.parquet`.",
        "",
        "For each same-order source trial, it includes:",
        "- 1 baseline `short_reason` row",
        "- 4 `occlude_equalize` rows, one per attribute",
        "- 4 `occlude_drop` rows, one per attribute",
        "",
        "All members of a matched family share `order_id_A`, `order_id_B`, `paraphrase_id`, and `seed`.",
        f"",
        f"- Base trials: `{manifest['n_base_trials']}`",
        f"- Total rows: `{manifest['n_trials']}`",
        "",
        "Use `base_trial_id` as the paired-family key.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"wrote {trials_path}")
    print(f"base trials: {manifest['n_base_trials']}")
    print(f"total rows: {manifest['n_trials']}")
    print(f"counts by manipulation: {manifest['counts_by_manipulation']}")


if __name__ == "__main__":
    main()
