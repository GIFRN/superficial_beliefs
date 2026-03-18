#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make import make_dataset_from_yaml
from src.data.orders import positions_for_order
from src.data.schema import LEVEL_SCORES
from src.utils.config import load_config


ATTRS = ["E", "A", "S", "D"]
DELTA_BASE_COLS = [f"delta_base_{attr}" for attr in ATTRS]

MODEL_COPY_MAP = {
    "mini_min.yml": "configs/models_gpt5mini_minimal.yml",
    "mini_low.yml": "configs/models_gpt5mini_low.yml",
    "nano_min.yml": "configs/models_gpt5nano_minimal.yml",
    "nano_low.yml": "configs/models_gpt5nano_low.yml",
    "haiku45_min.yml": "configs/models_claude_haiku45_minimal.yml",
    "haiku45_low.yml": "configs/models_claude_haiku45_low.yml",
    "qwen_min.yml": "configs/models_qwen35_14b_vllm_minimal.yml",
    "qwen_low.yml": "configs/models_qwen35_14b_vllm_low.yml",
    "ministral_min.yml": "configs/models_ministral3_14b_reasoning_vllm_minimal.yml",
    "ministral_low.yml": "configs/models_ministral3_14b_reasoning_vllm_low.yml",
}

THEME_COPY_MAP = {
    "drugs.yml": "configs/themes/drugs.yml",
    "policy.yml": "configs/themes/policy_intervention.yml",
    "software.yml": "configs/themes/software_library.yml",
    "placebo_packaging.yml": "configs/themes/drugs_placebo_packaging.yml",
    "placebo_label_border.yml": "configs/themes/drugs_placebo_label_border.yml",
}

BASE_VARIANT = "p_at_oa_vs_q_at_oa"
FULL_VARIANTS = [
    "p_at_oa_vs_q_at_oa",
    "q_at_oa_vs_p_at_oa",
    "p_at_ob_vs_q_at_ob",
    "q_at_ob_vs_p_at_ob",
]


def tradeoff_mask(df: pd.DataFrame) -> pd.Series:
    positive = (df[DELTA_BASE_COLS] > 0).any(axis=1)
    negative = (df[DELTA_BASE_COLS] < 0).any(axis=1)
    return positive & negative


def order_text(orders: list[list[str]], order_id: int) -> str:
    return ">".join(orders[int(order_id)])


def canonical_deltas(levels_left_json: str, levels_right_json: str) -> dict[str, int]:
    left = json.loads(levels_left_json)
    right = json.loads(levels_right_json)
    return {attr: int(LEVEL_SCORES[left[attr]] - LEVEL_SCORES[right[attr]]) for attr in ATTRS}


def base_signature(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                df["levels_left"].astype(str),
                df["levels_right"].astype(str),
                df["order_id_A"].astype(int),
                df["order_id_B"].astype(int),
                df["paraphrase_id"].astype(int),
                df["seed"].astype(int),
            )
        ),
        index=df.index,
    )


def load_stage43_split(split_dir: Path, source_split: str) -> pd.DataFrame:
    trials = pd.read_parquet(split_dir / "dataset_trials.parquet")
    configs = pd.read_parquet(split_dir / "dataset_configs.parquet")[["config_id", "levels_left", "levels_right"]]
    trials = trials[(trials["is_mirrored"] == False) & (trials["block"] == "B3")].copy()
    trials = trials[trials["manipulation"] == "short_reason"].copy()
    merged = trials.merge(configs, on="config_id", how="left")
    merged["source_origin"] = "stage43"
    merged["source_split"] = source_split
    merged["source_trial_id"] = merged["trial_id"].astype(str)
    merged["source_config_id"] = merged["config_id"].astype(str)
    merged["same_order"] = merged["order_id_A"].astype(int) == merged["order_id_B"].astype(int)
    merged["base_sig"] = base_signature(merged)
    if merged["base_sig"].duplicated().any():
        raise ValueError(f"stage43 {source_split} contains duplicate base families")
    return merged


def build_probe_pool(config_path: Path, seed: int) -> pd.DataFrame:
    payload = make_dataset_from_yaml(config_path, seed=seed)
    trials = payload["trials"].copy()
    configs = payload["configs"][["config_id", "levels_left", "levels_right"]].copy()
    trials = trials[(trials["block"] == "B3") & (trials["manipulation"] == "short_reason")].copy()
    trials = trials[tradeoff_mask(trials)].copy()
    merged = trials.merge(configs, on="config_id", how="left")
    merged["source_origin"] = "regen"
    merged["source_split"] = "pool"
    merged["source_trial_id"] = merged["trial_id"].astype(str)
    merged["source_config_id"] = merged["config_id"].astype(str)
    merged["same_order"] = merged["order_id_A"].astype(int) == merged["order_id_B"].astype(int)
    merged["base_sig"] = base_signature(merged)
    if merged["base_sig"].duplicated().any():
        raise ValueError("regenerated pool contains duplicate base families")
    return merged


def quota_same_order(current_df: pd.DataFrame, target_total: int) -> int:
    current_same = int(current_df["same_order"].sum())
    target_same_total = round(float(current_df["same_order"].mean()) * target_total)
    need = target_same_total - current_same
    return max(0, min(target_total - len(current_df), need))


def choose_rows(
    candidates: pd.DataFrame,
    *,
    current_df: pd.DataFrame,
    n_same: int,
    n_diff: int,
    used_base_sigs: set[tuple],
    used_probe_configs: set[str],
) -> pd.DataFrame:
    pair_counts = current_df[["order_id_A", "order_id_B"]].value_counts().to_dict()

    def pick(pool: pd.DataFrame, needed: int) -> list[int]:
        ranked = pool.copy()
        ranked["pair_count"] = [
            pair_counts.get((int(a), int(b)), 0)
            for a, b in zip(ranked["order_id_A"], ranked["order_id_B"], strict=False)
        ]
        ranked = ranked.sort_values(
            ["pair_count", "order_id_A", "order_id_B", "paraphrase_id", "source_config_id", "source_trial_id"]
        )
        chosen: list[int] = []
        for idx, row in ranked.iterrows():
            if row["base_sig"] in used_base_sigs:
                continue
            if row["source_config_id"] in used_probe_configs:
                continue
            chosen.append(idx)
            used_base_sigs.add(row["base_sig"])
            used_probe_configs.add(str(row["source_config_id"]))
            if len(chosen) == needed:
                break
        if len(chosen) != needed:
            raise ValueError(f"unable to select {needed} rows from pool of size {len(pool)}")
        return chosen

    same_pool = candidates[candidates["same_order"]].copy()
    diff_pool = candidates[~candidates["same_order"]].copy()
    chosen_idx = pick(same_pool, n_same) + pick(diff_pool, n_diff)
    return candidates.loc[chosen_idx].copy()


def assign_family_and_config_ids(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    config_lookup: dict[tuple[str, str], str] = {}
    config_rows: list[dict[str, str]] = []
    config_ids: list[str] = []

    for _, row in df.iterrows():
        key = (str(row["levels_left"]), str(row["levels_right"]))
        if key not in config_lookup:
            config_id = f"{split_name}_cfg_{len(config_lookup) + 1:04d}"
            config_lookup[key] = config_id
            config_rows.append(
                {
                    "config_id": config_id,
                    "block": "B3",
                    "levels_left": key[0],
                    "levels_right": key[1],
                }
            )
        config_ids.append(config_lookup[key])

    df["config_id"] = config_ids
    df["family_id"] = [f"{split_name}_{idx:04d}" for idx in range(1, len(df) + 1)]
    return df, pd.DataFrame(config_rows)


def build_trial_payload(
    row: pd.Series,
    *,
    trial_id: str,
    label_a: str,
    order_id_a: int,
    order_id_b: int,
    variant_name: str,
    orders: list[list[str]],
) -> dict[str, object]:
    deltas = canonical_deltas(str(row["levels_left"]), str(row["levels_right"]))
    if label_a == "B":
        deltas = {attr: -value for attr, value in deltas.items()}

    pos_a = positions_for_order(orders[int(order_id_a)])
    pos_b = positions_for_order(orders[int(order_id_b)])

    payload: dict[str, object] = {
        "trial_id": trial_id,
        "config_id": str(row["config_id"]),
        "family_id": str(row["family_id"]),
        "variant_name": variant_name,
        "block": "B3",
        "labelA": label_a,
        "manipulation": "short_reason",
        "attribute_target": None,
        "inject_offset": 0,
        "seed": int(row["seed"]),
        "order_id_A": int(order_id_a),
        "order_id_B": int(order_id_b),
        "paraphrase_id": 0,
        "source_origin": str(row["source_origin"]),
        "source_split": str(row["source_split"]),
        "source_trial_id": str(row["source_trial_id"]),
        "source_config_id": str(row["source_config_id"]),
        "source_labelA": str(row["labelA"]),
        "source_order_id_A": int(row["order_id_A"]),
        "source_order_id_B": int(row["order_id_B"]),
        "source_paraphrase_id": int(row["paraphrase_id"]),
        "same_order_prompt": int(order_id_a) == int(order_id_b),
        "slot_A_profile": "P" if label_a == "A" else "Q",
        "slot_B_profile": "Q" if label_a == "A" else "P",
    }
    for attr in ATTRS:
        payload[f"delta_{attr}"] = int(deltas[attr])
        payload[f"delta_base_{attr}"] = int(deltas[attr])
        payload[f"source_delta_{attr}"] = int(row[f"delta_base_{attr}"])
        payload[f"posA_{attr}"] = int(pos_a[attr])
        payload[f"posB_{attr}"] = int(pos_b[attr])
        payload[f"delta_pos_{attr}"] = int(pos_a[attr] - pos_b[attr])
    return payload


def build_split_trials(df: pd.DataFrame, orders: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        family_id = str(row["family_id"])
        oa = int(row["order_id_A"])
        ob = int(row["order_id_B"])

        base_rows.append(
            build_trial_payload(
                row,
                trial_id=family_id,
                label_a="A",
                order_id_a=oa,
                order_id_b=oa,
                variant_name=BASE_VARIANT,
                orders=orders,
            )
        )

        full_rows.extend(
            [
                build_trial_payload(
                    row,
                    trial_id=f"{family_id}__{FULL_VARIANTS[0]}",
                    label_a="A",
                    order_id_a=oa,
                    order_id_b=oa,
                    variant_name=FULL_VARIANTS[0],
                    orders=orders,
                ),
                build_trial_payload(
                    row,
                    trial_id=f"{family_id}__{FULL_VARIANTS[1]}",
                    label_a="B",
                    order_id_a=oa,
                    order_id_b=oa,
                    variant_name=FULL_VARIANTS[1],
                    orders=orders,
                ),
                build_trial_payload(
                    row,
                    trial_id=f"{family_id}__{FULL_VARIANTS[2]}",
                    label_a="A",
                    order_id_a=ob,
                    order_id_b=ob,
                    variant_name=FULL_VARIANTS[2],
                    orders=orders,
                ),
                build_trial_payload(
                    row,
                    trial_id=f"{family_id}__{FULL_VARIANTS[3]}",
                    label_a="B",
                    order_id_a=ob,
                    order_id_b=ob,
                    variant_name=FULL_VARIANTS[3],
                    orders=orders,
                ),
            ]
        )

    return pd.DataFrame(base_rows), pd.DataFrame(full_rows)


def write_inspection_csv(
    out_path: Path,
    trials: pd.DataFrame,
    configs: pd.DataFrame,
    *,
    orders: list[list[str]],
) -> None:
    inspection = trials.merge(configs, on="config_id", how="left").copy()
    levels_left = inspection["levels_left"].map(json.loads)
    levels_right = inspection["levels_right"].map(json.loads)
    for attr in ATTRS:
        inspection[f"P_{attr}"] = [levels[attr] for levels in levels_left]
        inspection[f"Q_{attr}"] = [levels[attr] for levels in levels_right]
    inspection["order_A"] = inspection["order_id_A"].map(lambda value: order_text(orders, value))
    inspection["order_B"] = inspection["order_id_B"].map(lambda value: order_text(orders, value))
    inspection["source_order_A"] = inspection["source_order_id_A"].map(lambda value: order_text(orders, value))
    inspection["source_order_B"] = inspection["source_order_id_B"].map(lambda value: order_text(orders, value))
    inspection["variant_order_source"] = inspection["variant_name"].map(
        lambda value: "oA" if "_oa_" in str(value) else "oB"
    )
    for attr in ATTRS:
        inspection[f"variant_delta_{attr}"] = inspection[f"delta_base_{attr}"]

    inspect_cols = [
        "trial_id",
        "family_id",
        "variant_name",
        "variant_order_source",
        "slot_A_profile",
        "slot_B_profile",
        "source_origin",
        "source_split",
        "source_trial_id",
        "source_config_id",
        "source_labelA",
        "P_E",
        "P_A",
        "P_S",
        "P_D",
        "Q_E",
        "Q_A",
        "Q_S",
        "Q_D",
        "order_id_A",
        "order_A",
        "order_id_B",
        "order_B",
        "source_order_id_A",
        "source_order_A",
        "source_order_id_B",
        "source_order_B",
        "source_paraphrase_id",
        "paraphrase_id",
        "seed",
        "variant_delta_E",
        "variant_delta_A",
        "variant_delta_S",
        "variant_delta_D",
        "source_delta_E",
        "source_delta_A",
        "source_delta_S",
        "source_delta_D",
        "delta_base_E",
        "delta_base_A",
        "delta_base_S",
        "delta_base_D",
        "labelA",
    ]
    inspection[inspect_cols].to_csv(out_path, index=False)


def write_split(
    out_dir: Path,
    split_name: str,
    base_trials: pd.DataFrame,
    full_trials: pd.DataFrame,
    configs: pd.DataFrame,
    *,
    current_kept: int,
    added: int,
    source_same_order_rows: int,
    source_different_order_rows: int,
    orders: list[list[str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "samples.csv",
        "base_samples.csv",
        "full_samples.csv",
        "dataset_trials.parquet",
        "dataset_configs.parquet",
        "full_trials.parquet",
        "full_configs.parquet",
        "MANIFEST.json",
    ]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    base_trials.to_parquet(out_dir / "dataset_trials.parquet", index=False)
    configs.to_parquet(out_dir / "dataset_configs.parquet", index=False)
    full_trials.to_parquet(out_dir / "full_trials.parquet", index=False)
    configs.to_parquet(out_dir / "full_configs.parquet", index=False)

    write_inspection_csv(out_dir / "base_samples.csv", base_trials, configs, orders=orders)
    write_inspection_csv(out_dir / "full_samples.csv", full_trials, configs, orders=orders)

    manifest = {
        "split": split_name,
        "n_base_trials": int(len(base_trials)),
        "n_full_trials": int(len(full_trials)),
        "n_configs": int(len(configs)),
        "block": "B3",
        "manipulation": "short_reason",
        "current_stage43_rows_kept": int(current_kept),
        "rows_added_from_regenerated_pool": int(added),
        "source_same_order_rows": int(source_same_order_rows),
        "source_different_order_rows": int(source_different_order_rows),
        "base_variant": BASE_VARIANT,
        "full_variants": FULL_VARIANTS,
        "source_origin_counts": base_trials["source_origin"].value_counts().to_dict(),
        "notes": "dataset_trials.parquet is the canonical same-order base dataset; full_trials.parquet expands each family into all four same-order variants; paraphrase_id is fixed to 0 because profile rendering is now canonical.",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def copy_layout_files(out_root: Path, dataset_config: Path) -> None:
    themes_dir = out_root / "themes"
    models_dir = out_root / "models"
    configs_dir = out_root / "configs"
    themes_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, src_name in THEME_COPY_MAP.items():
        shutil.copy2(src_name, themes_dir / dest_name)
    for dest_name, src_name in MODEL_COPY_MAP.items():
        shutil.copy2(src_name, models_dir / dest_name)
    shutil.copy2(dataset_config, configs_dir / "dataset.yml")


def write_root_readme(out_root: Path) -> None:
    readme = """# Final Base Dataset

This directory contains the train/test source families and the derived same-order datasets for the final experiments.

- `train/dataset_trials.parquet` and `test/dataset_trials.parquet` are the canonical same-order base datasets.
- Each base row is the `p_at_oa_vs_q_at_oa` variant for one family.
- `train/full_trials.parquet` and `test/full_trials.parquet` contain the full four-variant expansion:
  - `p_at_oa_vs_q_at_oa`
  - `q_at_oa_vs_p_at_oa`
  - `p_at_ob_vs_q_at_ob`
  - `q_at_ob_vs_p_at_ob`
- `paraphrase_id` is fixed to `0`; profile rendering no longer varies by paraphrase template.
- `base_samples.csv` and `full_samples.csv` are the inspection-friendly CSV versions.
- `themes/` contains copied theme configs with simplified names.
- `models/` contains copied model configs with simplified names.
- `configs/dataset.yml` is the copied dataset-generation config.
"""
    (out_root / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final base train/test dataset layout under data/.")
    parser.add_argument(
        "--stage43-root",
        default="artifacts/v4_matchedsuite_drugs_20260208/splits/holdout80_canonical_balanced_short_reason",
        help="Stage43 canonical short_reason split root",
    )
    parser.add_argument("--config", default="configs/default.yml", help="Dataset config used to regenerate the top-up pool")
    parser.add_argument("--out-root", default="data", help="Output root directory")
    parser.add_argument("--train-target", type=int, default=400, help="Target number of base train families")
    parser.add_argument("--test-target", type=int, default=100, help="Target number of base test families")
    parser.add_argument("--seed", type=int, default=13, help="Seed for regenerating the top-up pool")
    args = parser.parse_args()

    stage43_root = Path(args.stage43_root)
    out_root = Path(args.out_root)
    orders = load_config(args.config).orders_permutations

    current_train = load_stage43_split(stage43_root / "train", "train")
    current_test = load_stage43_split(stage43_root / "test", "test")

    if len(current_train) > args.train_target or len(current_test) > args.test_target:
        raise ValueError("targets must be at least as large as the current stage43 base split")

    probe_pool = build_probe_pool(Path(args.config), args.seed)
    used_base_sigs = set(current_train["base_sig"]) | set(current_test["base_sig"])
    probe_pool = probe_pool[~probe_pool["base_sig"].isin(used_base_sigs)].copy()

    used_probe_configs: set[str] = set()
    add_test = args.test_target - len(current_test)
    add_train = args.train_target - len(current_train)

    test_same = quota_same_order(current_test, args.test_target)
    test_extra = choose_rows(
        probe_pool,
        current_df=current_test,
        n_same=test_same,
        n_diff=add_test - test_same,
        used_base_sigs=used_base_sigs,
        used_probe_configs=used_probe_configs,
    )

    train_same = quota_same_order(current_train, args.train_target)
    train_extra = choose_rows(
        probe_pool,
        current_df=current_train,
        n_same=train_same,
        n_diff=add_train - train_same,
        used_base_sigs=used_base_sigs,
        used_probe_configs=used_probe_configs,
    )

    final_train = pd.concat([current_train, train_extra], ignore_index=True)
    final_test = pd.concat([current_test, test_extra], ignore_index=True)
    train_source_same = int(final_train["same_order"].sum())
    test_source_same = int(final_test["same_order"].sum())

    final_train, final_train_configs = assign_family_and_config_ids(final_train, "train")
    final_test, final_test_configs = assign_family_and_config_ids(final_test, "test")

    base_train_trials, full_train_trials = build_split_trials(final_train, orders)
    base_test_trials, full_test_trials = build_split_trials(final_test, orders)

    write_split(
        out_root / "train",
        "train",
        base_train_trials,
        full_train_trials,
        final_train_configs,
        current_kept=len(current_train),
        added=len(train_extra),
        source_same_order_rows=train_source_same,
        source_different_order_rows=len(final_train) - train_source_same,
        orders=orders,
    )
    write_split(
        out_root / "test",
        "test",
        base_test_trials,
        full_test_trials,
        final_test_configs,
        current_kept=len(current_test),
        added=len(test_extra),
        source_same_order_rows=test_source_same,
        source_different_order_rows=len(final_test) - test_source_same,
        orders=orders,
    )
    copy_layout_files(out_root, Path(args.config))
    write_root_readme(out_root)

    root_manifest = {
        "train_target": args.train_target,
        "test_target": args.test_target,
        "train_base_trials": int(len(base_train_trials)),
        "test_base_trials": int(len(base_test_trials)),
        "train_full_trials": int(len(full_train_trials)),
        "test_full_trials": int(len(full_test_trials)),
        "train_kept_from_stage43": int(len(current_train)),
        "test_kept_from_stage43": int(len(current_test)),
        "train_added_from_regenerated_pool": int(len(train_extra)),
        "test_added_from_regenerated_pool": int(len(test_extra)),
        "regenerated_pool_available_rows": int(len(probe_pool)),
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(root_manifest, indent=2))

    print(f"wrote {out_root / 'train'} (base={len(base_train_trials)} full={len(full_train_trials)})")
    print(f"wrote {out_root / 'test'} (base={len(base_test_trials)} full={len(full_test_trials)})")
    print(f"added extras: train={len(train_extra)} test={len(test_extra)}")


if __name__ == "__main__":
    main()
