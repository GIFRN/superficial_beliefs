from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import json
import numpy as np
import pandas as pd

from .build_B1 import build_B1
from .build_B2 import build_B2
from .build_B3_alternative import build_B3
from .occlusions import apply_occlusion_to_deltas
from .orders import assign_orders, positions_for_order
from .paraphrases import PARAPHRASES, choose_paraphrase_ids

from .schema import BaseConfiguration, Profile, compute_deltas
from ..utils.config import Config, load_config
from ..utils.io import ensure_dir, write_json
from ..utils.rng import make_generator

if TYPE_CHECKING:
    from .themes import ThemeConfig



def _build_dominance_configs(cfg: Config, rng: np.random.Generator) -> list[BaseConfiguration]:
    n_dom = cfg.dominance_items.n
    if n_dom <= 0:
        return []
    attributes = cfg.profiles.attributes
    levels = cfg.profiles.levels
    configs: list[BaseConfiguration] = []
    for _ in range(n_dom):
        dominant: dict[str, str] = {}
        dominant_idx: dict[str, int] = {}
        for attr in attributes:
            idx = int(rng.integers(1, len(levels)))
            dominant[attr] = levels[idx]
            dominant_idx[attr] = idx
        dominated: dict[str, str] = {}
        for attr in attributes:
            low_idx = int(rng.integers(0, dominant_idx[attr]))
            dominated[attr] = levels[low_idx]
        configs.append(
            BaseConfiguration(
                block="DOM",
                profile_left=Profile(dominant),
                profile_right=Profile(dominated),
            )
        )
    return configs


def _collect_base_configs(cfg: Config, rng: np.random.Generator) -> list[BaseConfiguration]:
    configs: list[BaseConfiguration] = []
    if cfg.blocks.B1.enable:
        configs.extend(build_B1(cfg))
    if cfg.blocks.B2.enable:
        configs.extend(build_B2(cfg))
    if cfg.blocks.B3.enable:
        configs.extend(build_B3(cfg, rng))
    configs.extend(_build_dominance_configs(cfg, rng))
    return configs


def _assign_config_ids(configs: list[BaseConfiguration]) -> dict[str, BaseConfiguration]:
    id_map: dict[str, BaseConfiguration] = {}
    counters: dict[str, int] = {}
    for config in configs:
        counters.setdefault(config.block, 0)
        counters[config.block] += 1
        config_id = f"{config.block}-{counters[config.block]:04d}"
        id_map[config_id] = config
    return id_map


def _profile_by_orientation(config: BaseConfiguration, label_of_left: str) -> tuple[Profile, Profile]:
    if label_of_left == "A":
        return config.profile_left, config.profile_right
    return config.profile_right, config.profile_left


def _manipulation_payload(cfg: Config, rng: np.random.Generator) -> tuple[str, str | None, int]:
    shares = cfg.manipulations.share.normalized
    manipulations = list(shares.keys())
    probabilities = np.array(list(shares.values()))
    choice = rng.choice(manipulations, p=probabilities)
    attribute: str | None = None
    inject_offset = 0
    if choice in {"redact", "neutralize", "occlude_drop", "occlude_equalize", "occlude_swap"}:
        attribute = rng.choice(cfg.probe_targets)
    elif choice == "inject":
        attribute = str(cfg.manipulations.injection.get("attribute", "A"))
        inject_offset = int(cfg.manipulations.injection.get("offset", 1))
    return choice, attribute, inject_offset


def make_dataset(
    cfg: Config, 
    *, 
    seed: int | None = None, 
    output_dir: str | Path | None = None,
    theme_config: "ThemeConfig | None" = None
) -> dict[str, Any]:
    rng = make_generator(cfg.seed_global if seed is None else seed)
    attributes = cfg.profiles.attributes
    base_configs = _collect_base_configs(cfg, rng)
    id_map = _assign_config_ids(base_configs)
    config_rows: list[dict[str, Any]] = []
    for config_id, config in id_map.items():
        config_rows.append(
            {
                "config_id": config_id,
                "block": config.block,
                "levels_left": json.dumps(config.profile_left.levels),
                "levels_right": json.dumps(config.profile_right.levels),
            }
        )
    configs_df = pd.DataFrame(config_rows).sort_values("config_id").reset_index(drop=True)

    replicate_plan = {
        "B1": cfg.blocks.B1.R,
        "B2": cfg.blocks.B2.R,
        "B3": 1,
        "DOM": 1,
    }
    orientation_plan = {
        "DOM": ["A"],
    }
    trial_rows: list[dict[str, Any]] = []
    for config_id, config in id_map.items():
        orientation_labels = orientation_plan.get(config.block, ["A", "B"])
        repeats = replicate_plan.get(config.block, 1)
        for orientation in orientation_labels:
            for _ in range(repeats):
                profile_a, profile_b = _profile_by_orientation(config, orientation)
                base_deltas = compute_deltas(profile_a, profile_b, attributes)
                if config.block == "DOM":
                    manipulation, attribute_target, inject_offset = "short_reason", None, 0
                else:
                    manipulation, attribute_target, inject_offset = _manipulation_payload(cfg, rng)
                visible_deltas = apply_occlusion_to_deltas(base_deltas, manipulation, attribute_target)
                trial_rows.append(
                    {
                        "config_id": config_id,
                        "block": config.block,
                        "labelA": orientation,
                        "manipulation": manipulation,
                        "attribute_target": attribute_target,
                        "inject_offset": inject_offset,
                        "delta_E": int(visible_deltas.get("E", 0)),
                        "delta_A": int(visible_deltas.get("A", 0)),
                        "delta_S": int(visible_deltas.get("S", 0)),
                        "delta_D": int(visible_deltas.get("D", 0)),
                        "delta_base_E": int(base_deltas.get("E", 0)),
                        "delta_base_A": int(base_deltas.get("A", 0)),
                        "delta_base_S": int(base_deltas.get("S", 0)),
                        "delta_base_D": int(base_deltas.get("D", 0)),
                        "seed": int(rng.integers(0, 2**32 - 1)),
                    }
                )
    trials_df = pd.DataFrame(trial_rows)
    n_trials = len(trials_df)
    orders = cfg.orders_permutations
    order_ids_a, order_ids_b = assign_orders(n_trials, orders, rng)
    trials_df["order_id_A"] = order_ids_a
    trials_df["order_id_B"] = order_ids_b

    n_templates = min(cfg.paraphrases.get("n", len(PARAPHRASES)), len(PARAPHRASES))
    paraphrase_ids = choose_paraphrase_ids(n_trials, n_templates, rng)
    trials_df["paraphrase_id"] = paraphrase_ids

    trials_df["trial_id"] = [f"T-{idx:05d}" for idx in range(1, n_trials + 1)]

    # Ensure attribute target missing entries stored as None
    trials_df["attribute_target"] = trials_df["attribute_target"].where(trials_df["attribute_target"].notna(), None)

    # Compute order position features for diagnostics
    order_positions_a = []
    order_positions_b = []
    for order_id_a, order_id_b, manip, target in zip(
        trials_df["order_id_A"],
        trials_df["order_id_B"],
        trials_df["manipulation"],
        trials_df["attribute_target"],
    ):
        order_a = list(orders[order_id_a])
        order_b = list(orders[order_id_b])
        if manip == "occlude_drop" and target:
            order_a = [attr for attr in order_a if attr != target]
            order_b = [attr for attr in order_b if attr != target]
        pos_a = positions_for_order(order_a)
        pos_b = positions_for_order(order_b)
        if manip == "occlude_drop" and target:
            pos_a[target] = 0
            pos_b[target] = 0
        order_positions_a.append(pos_a)
        order_positions_b.append(pos_b)
    for attr in attributes:
        trials_df[f"posA_{attr}"] = [positions.get(attr, 0) for positions in order_positions_a]
        trials_df[f"posB_{attr}"] = [positions.get(attr, 0) for positions in order_positions_b]
        trials_df[f"delta_pos_{attr}"] = trials_df[f"posA_{attr}"] - trials_df[f"posB_{attr}"]

    trials_df = trials_df.sort_values("trial_id").reset_index(drop=True)

    payload = {
        "configs": configs_df,
        "trials": trials_df,
    }

    if output_dir is not None:
        output_path = ensure_dir(output_dir)
        configs_path = output_path / "dataset_configs.parquet"
        trials_path = output_path / "dataset_trials.parquet"
        configs_df.to_parquet(configs_path, index=False)
        trials_df.to_parquet(trials_path, index=False)
        
        # Build manifest with optional theme metadata
        target_total = getattr(cfg, "target_total", None)
        if target_total is None:
            target_total = len(trials_df)
        manifest = {
            "n_configs": len(configs_df),
            "n_trials": len(trials_df),
            "target_total": target_total,
            "actual_total": len(trials_df),
            "blocks": trials_df["block"].value_counts().to_dict(),
        }
        
        if theme_config:
            manifest["theme"] = theme_config.to_dict()
        
        write_json(manifest, output_path / "MANIFEST.json")
        payload["configs_path"] = str(configs_path)
        payload["trials_path"] = str(trials_path)
    return payload


def make_occlusion_suite_dataset(
    cfg: Config,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    theme_config: "ThemeConfig | None" = None,
    include_swap: bool = False,
    include_dom: bool = False,
) -> dict[str, Any]:
    """Generate a matched occlusion suite dataset.

    For each base trial (config_id, orientation, repeat), emit:
    - baseline short_reason
    - occlude_equalize for each attribute
    - occlude_drop for each attribute
    - (optional) occlude_swap for each attribute

    The baseline and all occlusions share the same order/paraphrase/seed to enable
    paired (within-base) comparisons.
    """
    rng = make_generator(cfg.seed_global if seed is None else seed)
    attributes = cfg.profiles.attributes
    base_configs = _collect_base_configs(cfg, rng)
    id_map = _assign_config_ids(base_configs)

    config_rows: list[dict[str, Any]] = []
    for config_id, config in id_map.items():
        config_rows.append(
            {
                "config_id": config_id,
                "block": config.block,
                "levels_left": json.dumps(config.profile_left.levels),
                "levels_right": json.dumps(config.profile_right.levels),
            }
        )
    configs_df = pd.DataFrame(config_rows).sort_values("config_id").reset_index(drop=True)

    replicate_plan = {
        "B1": cfg.blocks.B1.R,
        "B2": cfg.blocks.B2.R,
        "B3": 1,
        "DOM": 1,
    }
    orientation_plan = {
        "DOM": ["A"],
    }

    base_trials: list[dict[str, Any]] = []
    base_counter = 0
    for config_id, config in id_map.items():
        orientation_labels = orientation_plan.get(config.block, ["A", "B"])
        repeats = replicate_plan.get(config.block, 1)
        for orientation in orientation_labels:
            for _ in range(repeats):
                base_counter += 1
                profile_a, profile_b = _profile_by_orientation(config, orientation)
                base_deltas = compute_deltas(profile_a, profile_b, attributes)
                base_trials.append(
                    {
                        "base_trial_id": f"BT-{base_counter:05d}",
                        "config_id": config_id,
                        "block": config.block,
                        "labelA": orientation,
                        "base_deltas": base_deltas,
                        "seed": int(rng.integers(0, 2**32 - 1)),
                    }
                )

    n_base = len(base_trials)
    orders = cfg.orders_permutations
    order_ids_a, order_ids_b = assign_orders(n_base, orders, rng)

    n_templates = min(cfg.paraphrases.get("n", len(PARAPHRASES)), len(PARAPHRASES))
    paraphrase_ids = choose_paraphrase_ids(n_base, n_templates, rng)

    trial_rows: list[dict[str, Any]] = []
    for idx, base in enumerate(base_trials):
        block = base["block"]
        base_deltas = base["base_deltas"]
        base_seed = int(base["seed"])
        order_id_a = int(order_ids_a[idx])
        order_id_b = int(order_ids_b[idx])
        paraphrase_id = int(paraphrase_ids[idx])

        variants: list[tuple[str, str | None]] = [("short_reason", None)]
        if include_dom or block != "DOM":
            variants.extend([("occlude_equalize", attr) for attr in attributes])
            variants.extend([("occlude_drop", attr) for attr in attributes])
            if include_swap:
                variants.extend([("occlude_swap", attr) for attr in attributes])

        for manipulation, attribute_target in variants:
            visible_deltas = apply_occlusion_to_deltas(base_deltas, manipulation, attribute_target)
            trial_rows.append(
                {
                    "base_trial_id": base["base_trial_id"],
                    "config_id": base["config_id"],
                    "block": block,
                    "labelA": base["labelA"],
                    "manipulation": manipulation,
                    "attribute_target": attribute_target,
                    "inject_offset": 0,
                    "delta_E": int(visible_deltas.get("E", 0)),
                    "delta_A": int(visible_deltas.get("A", 0)),
                    "delta_S": int(visible_deltas.get("S", 0)),
                    "delta_D": int(visible_deltas.get("D", 0)),
                    "delta_base_E": int(base_deltas.get("E", 0)),
                    "delta_base_A": int(base_deltas.get("A", 0)),
                    "delta_base_S": int(base_deltas.get("S", 0)),
                    "delta_base_D": int(base_deltas.get("D", 0)),
                    "order_id_A": order_id_a,
                    "order_id_B": order_id_b,
                    "paraphrase_id": paraphrase_id,
                    "seed": base_seed,
                }
            )

    trials_df = pd.DataFrame(trial_rows)
    trials_df["trial_id"] = [f"T-{idx:05d}" for idx in range(1, len(trials_df) + 1)]

    trials_df["attribute_target"] = trials_df["attribute_target"].where(trials_df["attribute_target"].notna(), None)

    order_positions_a = []
    order_positions_b = []
    for order_id_a, order_id_b, manip, target in zip(
        trials_df["order_id_A"],
        trials_df["order_id_B"],
        trials_df["manipulation"],
        trials_df["attribute_target"],
    ):
        order_a = list(orders[int(order_id_a)])
        order_b = list(orders[int(order_id_b)])
        if manip == "occlude_drop" and target:
            order_a = [attr for attr in order_a if attr != target]
            order_b = [attr for attr in order_b if attr != target]
        pos_a = positions_for_order(order_a)
        pos_b = positions_for_order(order_b)
        if manip == "occlude_drop" and target:
            pos_a[target] = 0
            pos_b[target] = 0
        order_positions_a.append(pos_a)
        order_positions_b.append(pos_b)
    for attr in attributes:
        trials_df[f"posA_{attr}"] = [positions.get(attr, 0) for positions in order_positions_a]
        trials_df[f"posB_{attr}"] = [positions.get(attr, 0) for positions in order_positions_b]
        trials_df[f"delta_pos_{attr}"] = trials_df[f"posA_{attr}"] - trials_df[f"posB_{attr}"]

    trials_df = trials_df.sort_values("trial_id").reset_index(drop=True)

    payload = {"configs": configs_df, "trials": trials_df}
    if output_dir is not None:
        output_path = ensure_dir(output_dir)
        configs_path = output_path / "dataset_configs.parquet"
        trials_path = output_path / "dataset_trials.parquet"
        configs_df.to_parquet(configs_path, index=False)
        trials_df.to_parquet(trials_path, index=False)

        target_total = getattr(cfg, "target_total", None)
        if target_total is None:
            target_total = len(trials_df)
        occlusion_manips = ["occlude_equalize", "occlude_drop"] + (["occlude_swap"] if include_swap else [])
        variants_non_dom = int(1 + len(occlusion_manips) * len(attributes))
        variants_dom = variants_non_dom if include_dom else 1
        manifest: dict[str, Any] = {
            "n_configs": len(configs_df),
            "n_trials": len(trials_df),
            "target_total": target_total,
            "actual_total": len(trials_df),
            "blocks": trials_df["block"].value_counts().to_dict(),
            "suite": {
                "type": "occlusion_suite",
                "base_trials": int(n_base),
                "paired_key": "base_trial_id",
                "attributes": list(attributes),
                "baseline_manipulation": "short_reason",
                "occlusion_manipulations": occlusion_manips,
                "include_swap": bool(include_swap),
                "include_dom": bool(include_dom),
                "variants_per_base_non_dom": variants_non_dom,
                "variants_per_base_dom": variants_dom,
                "note": "Baseline and occlusions share order_id_A/B, paraphrase_id, and seed within each base_trial_id.",
            },
        }
        if theme_config:
            manifest["theme"] = theme_config.to_dict()
        write_json(manifest, output_path / "MANIFEST.json")
        payload["configs_path"] = str(configs_path)
        payload["trials_path"] = str(trials_path)
    return payload


def make_dataset_from_yaml(
    config_path: str | Path, 
    *, 
    seed: int | None = None, 
    output_dir: str | Path | None = None,
    theme_config: "ThemeConfig | None" = None
) -> dict[str, Any]:
    cfg = load_config(config_path)
    return make_dataset(cfg, seed=seed, output_dir=output_dir, theme_config=theme_config)


def make_occlusion_suite_dataset_from_yaml(
    config_path: str | Path,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    theme_config: "ThemeConfig | None" = None,
    include_swap: bool = False,
    include_dom: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    return make_occlusion_suite_dataset(
        cfg,
        seed=seed,
        output_dir=output_dir,
        theme_config=theme_config,
        include_swap=include_swap,
        include_dom=include_dom,
    )
