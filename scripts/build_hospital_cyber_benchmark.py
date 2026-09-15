#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hospital_cyber_nt import (
    ATTRIBUTES,
    DEFAULT_DATASET_SEED,
    DEFAULT_TEST_TARGET,
    DEFAULT_TRAIN_TARGET,
    FULL_VARIANTS,
    THEME_NAME,
    THEME_PATH,
    dataset_dir,
    load_theme,
    output_root,
)
from src.data.orders import positions_for_order
from src.data.schema import LEVEL_SCORES
from src.utils.config import load_config


LEVELS = ["Low", "Medium", "High"]
SCORE_TO_LEVEL = {value: key for key, value in LEVEL_SCORES.items()}
VARIANT_SPECS = (
    ("p_at_oa_vs_q_at_oa", "A", "P", "Q", "oa"),
    ("q_at_oa_vs_p_at_oa", "B", "Q", "P", "oa"),
    ("p_at_ob_vs_q_at_ob", "A", "P", "Q", "ob"),
    ("q_at_ob_vs_p_at_ob", "B", "Q", "P", "ob"),
)


def levels_json(levels: dict[str, str]) -> str:
    return json.dumps({attr: levels[attr] for attr in ATTRIBUTES})


def enumerate_candidate_families() -> list[dict[str, object]]:
    profiles = list(product([-1, 0, 1], repeat=len(ATTRIBUTES)))
    candidates: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(profiles)), 2):
        left = profiles[left_idx]
        right = profiles[right_idx]
        deltas = [left[pos] - right[pos] for pos in range(len(ATTRIBUTES))]
        if not (any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas)):
            continue
        p_levels = {attr: SCORE_TO_LEVEL[left[pos]] for pos, attr in enumerate(ATTRIBUTES)}
        q_levels = {attr: SCORE_TO_LEVEL[right[pos]] for pos, attr in enumerate(ATTRIBUTES)}
        candidates.append(
            {
                "p_levels": p_levels,
                "q_levels": q_levels,
                "source_delta": {attr: int(deltas[pos]) for pos, attr in enumerate(ATTRIBUTES)},
            }
        )
    return candidates


def assign_order_pairs(n_families: int, n_orders: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    pool = [(oa, ob) for oa in range(n_orders) for ob in range(n_orders) if oa != ob]
    reps = (n_families + len(pool) - 1) // len(pool)
    paired = (pool * reps)[:n_families]
    rng.shuffle(paired)
    return paired


def sample_families(*, train_target: int, test_target: int, seed: int, orders: list[list[str]]) -> list[dict[str, object]]:
    total = train_target + test_target
    candidates = enumerate_candidate_families()
    if total > len(candidates):
        raise ValueError(f"Requested {total} families but only {len(candidates)} tradeoff families are available")
    rng = np.random.default_rng(seed)
    chosen_indices = rng.permutation(len(candidates))[:total]
    chosen = [dict(candidates[idx]) for idx in chosen_indices]
    order_pairs = assign_order_pairs(total, len(orders), rng)
    family_seeds = rng.integers(0, 2**31 - 1, size=total, dtype=np.int64)
    for idx, family in enumerate(chosen):
        oa, ob = order_pairs[idx]
        family["oa"] = int(oa)
        family["ob"] = int(ob)
        family["family_seed"] = int(family_seeds[idx])
    return chosen


def build_split_frames(split_name: str, families: list[dict[str, object]], orders: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for idx, family in enumerate(families, start=1):
        family_id = f"{split_name}_{idx:04d}"
        config_id = f"{split_name}_cfg_{idx:04d}"
        p_levels = family["p_levels"]
        q_levels = family["q_levels"]
        source_delta = family["source_delta"]
        oa = int(family["oa"])
        ob = int(family["ob"])
        family_seed = int(family["family_seed"])

        config_rows.append(
            {
                "config_id": config_id,
                "block": "B3",
                "levels_left": levels_json(p_levels),
                "levels_right": levels_json(q_levels),
            }
        )

        source_row: dict[str, object] = {
            "family_id": family_id,
            "config_id": config_id,
            "split": split_name,
            "block": "B3",
            "seed": family_seed,
            "order_id_A": oa,
            "order_id_B": ob,
            "order_A_text": ">".join(orders[oa]),
            "order_B_text": ">".join(orders[ob]),
            "levels_left": levels_json(p_levels),
            "levels_right": levels_json(q_levels),
            "paraphrase_id": 0,
        }
        for attr in ATTRIBUTES:
            source_row[f"source_delta_{attr}"] = int(source_delta[attr])
        source_rows.append(source_row)

        for variant_name, label_a, slot_a_profile, slot_b_profile, order_key in VARIANT_SPECS:
            order_id = oa if order_key == "oa" else ob
            positions = positions_for_order(orders[order_id])
            trial_row: dict[str, object] = {
                "trial_id": f"{family_id}__{variant_name}",
                "config_id": config_id,
                "family_id": family_id,
                "variant_name": variant_name,
                "block": "B3",
                "labelA": label_a,
                "manipulation": "short_reason",
                "attribute_target": None,
                "inject_offset": 0,
                "seed": family_seed,
                "order_id_A": order_id,
                "order_id_B": order_id,
                "paraphrase_id": 0,
                "source_origin": "hospital_cyber_response_synthetic",
                "source_split": split_name,
                "source_trial_id": family_id,
                "source_config_id": config_id,
                "source_labelA": "A",
                "source_order_id_A": oa,
                "source_order_id_B": ob,
                "source_paraphrase_id": 0,
                "same_order_prompt": True,
                "slot_A_profile": slot_a_profile,
                "slot_B_profile": slot_b_profile,
            }
            for attr in ATTRIBUTES:
                displayed_delta = int(source_delta[attr]) if label_a == "A" else int(-source_delta[attr])
                trial_row[f"delta_{attr}"] = displayed_delta
                trial_row[f"delta_base_{attr}"] = displayed_delta
                trial_row[f"source_delta_{attr}"] = int(source_delta[attr])
                trial_row[f"posA_{attr}"] = int(positions[attr])
                trial_row[f"posB_{attr}"] = int(positions[attr])
                trial_row[f"delta_pos_{attr}"] = 0
            trial_rows.append(trial_row)

    return pd.DataFrame(config_rows), pd.DataFrame(source_rows), pd.DataFrame(trial_rows)


def build_occlusion_frame(test_trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in test_trials.to_dict("records"):
        base_trial_id = str(row["trial_id"])
        baseline = dict(row)
        baseline["base_trial_id"] = base_trial_id
        rows.append(baseline)
        for attr in ATTRIBUTES:
            occluded = dict(row)
            occluded["trial_id"] = f"{base_trial_id}__occlude_equalize_{attr}"
            occluded["variant_name"] = f"{row['variant_name']}__occlude_equalize_{attr}"
            occluded["manipulation"] = "occlude_equalize"
            occluded["attribute_target"] = attr
            occluded["base_trial_id"] = base_trial_id
            occluded[f"delta_{attr}"] = 0
            occluded[f"delta_base_{attr}"] = 0
            rows.append(occluded)
    return pd.DataFrame(rows)


def write_manifest(
    out_dir: Path,
    *,
    split_name: str,
    n_trials: int,
    n_configs: int,
    n_source_families: int,
    seed: int,
    order_library: list[list[str]],
    extra: dict[str, object] | None = None,
) -> None:
    manifest = {
        "theme_name": THEME_NAME,
        "theme": load_theme().to_dict(),
        "split": split_name,
        "seed": seed,
        "attributes": ATTRIBUTES,
        "levels": LEVELS,
        "order_library": order_library,
        "n_trials": int(n_trials),
        "n_configs": int(n_configs),
        "n_source_families": int(n_source_families),
        "source_configs_filename": "dataset_configs.parquet",
        "source_trials_filename": "dataset_trials.parquet",
        "source_families_filename": "source_families.parquet",
        "full_variants": list(FULL_VARIANTS),
        "generated_by": "scripts/build_hospital_cyber_benchmark.py",
        "theme_path": str(THEME_PATH),
    }
    if extra:
        manifest.update(extra)
    with (out_dir / "MANIFEST.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def write_split(
    out_dir: Path,
    configs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    *,
    split_name: str,
    seed: int,
    orders: list[list[str]],
    extra: dict[str, object] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_df.to_parquet(out_dir / "dataset_configs.parquet", index=False)
    source_df.to_parquet(out_dir / "source_families.parquet", index=False)
    trials_df.to_parquet(out_dir / "dataset_trials.parquet", index=False)
    write_manifest(
        out_dir,
        split_name=split_name,
        n_trials=len(trials_df),
        n_configs=len(configs_df),
        n_source_families=len(source_df),
        seed=seed,
        order_library=orders,
        extra=extra,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hospital cyber NT benchmark datasets")
    parser.add_argument("--output-root", default=None, help="Override benchmark output root")
    parser.add_argument("--train-target", type=int, default=DEFAULT_TRAIN_TARGET)
    parser.add_argument("--test-target", type=int, default=DEFAULT_TEST_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--force", action="store_true", help="Remove any existing hospital benchmark datasets first")
    args = parser.parse_args()

    cfg = load_config(ROOT / "data/configs/hospital_cyber_response.yml")
    orders = [list(order) for order in cfg.orders_permutations]
    base_root = output_root(args.output_root)
    dataset_root = base_root / "datasets" / THEME_NAME
    if args.force and dataset_root.exists():
        shutil.rmtree(dataset_root)

    selected = sample_families(
        train_target=args.train_target,
        test_target=args.test_target,
        seed=args.seed,
        orders=orders,
    )
    train_families = selected[: args.train_target]
    test_families = selected[args.train_target :]

    train_configs, train_source, train_trials = build_split_frames("train", train_families, orders)
    test_configs, test_source, test_trials = build_split_frames("test", test_families, orders)
    occlusion_trials = build_occlusion_frame(test_trials)

    write_split(
        dataset_dir("train", base=base_root),
        train_configs,
        train_source,
        train_trials,
        split_name="train",
        seed=args.seed,
        orders=orders,
        extra={"target_total": int(args.train_target * len(FULL_VARIANTS))},
    )
    write_split(
        dataset_dir("test", base=base_root),
        test_configs,
        test_source,
        test_trials,
        split_name="test",
        seed=args.seed,
        orders=orders,
        extra={"target_total": int(args.test_target * len(FULL_VARIANTS))},
    )
    write_split(
        dataset_dir("occlusion_test", base=base_root),
        test_configs,
        test_source,
        occlusion_trials,
        split_name="occlusion_test",
        seed=args.seed,
        orders=orders,
        extra={
            "base_split": "test",
            "n_baseline_trials": int(len(test_trials)),
            "n_occlusion_trials": int(len(occlusion_trials) - len(test_trials)),
        },
    )

    print(f"Built {THEME_NAME} datasets under {dataset_root}")
    print(f"Train families: {len(train_source)}  Train trials: {len(train_trials)}")
    print(f"Test families: {len(test_source)}  Test trials: {len(test_trials)}")
    print(f"Occlusion test trials: {len(occlusion_trials)}")


if __name__ == "__main__":
    main()
