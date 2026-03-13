from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import torch

from src.semantics import parse_semantics


ATTRIBUTES = ("E", "A", "S", "D")
CLAIM_IDX = 0
CLAIM_BASE_SCORE = 0.5
NODE_INDEX = {attr: idx + 1 for idx, attr in enumerate(ATTRIBUTES)}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=None)
def _get_semantics(name: str, conservativeness: float):
    return parse_semantics(name, conservativeness)


def _base_scores_tensor(row: pd.Series, claim_base_score: float) -> torch.Tensor:
    values = [claim_base_score]
    for attr in ATTRIBUTES:
        tau_value = _safe_float(row.get(f"tau_{attr}"), default=0.0)
        values.append(min(max(tau_value, 0.0), 1.0))
    return torch.tensor(values, dtype=torch.float32)


def _adjacency_tensor(row: pd.Series) -> torch.Tensor:
    adjacency = torch.zeros((1 + len(ATTRIBUTES), 1 + len(ATTRIBUTES)), dtype=torch.float32)
    for attr, node_idx in NODE_INDEX.items():
        delta_value = _safe_float(row.get(f"delta_{attr}"), default=0.0)
        if delta_value > 0:
            adjacency[node_idx, CLAIM_IDX] = 1.0
        elif delta_value < 0:
            adjacency[node_idx, CLAIM_IDX] = -1.0
    return adjacency


def _driver_from_effects(effects: dict[str, float]) -> str:
    best_attr = ATTRIBUTES[0]
    best_value = abs(effects.get(best_attr, 0.0))
    for attr in ATTRIBUTES[1:]:
        value = abs(effects.get(attr, 0.0))
        if value > best_value:
            best_attr = attr
            best_value = value
    return best_attr


def evaluate_tau_argument_framework_row(
    row: pd.Series,
    *,
    semantics: str,
    conservativeness: float = 1.0,
    claim_base_score: float = CLAIM_BASE_SCORE,
) -> dict[str, float | str]:
    gradual_semantics = _get_semantics(semantics, conservativeness)
    base_scores = _base_scores_tensor(row, claim_base_score)
    adjacency = _adjacency_tensor(row)

    final_strengths = gradual_semantics(adjacency, base_scores)
    claim_strength = float(final_strengths[CLAIM_IDX].item())

    effects: dict[str, float] = {}
    for attr, node_idx in NODE_INDEX.items():
        single_adjacency = torch.zeros_like(adjacency)
        single_adjacency[node_idx, CLAIM_IDX] = adjacency[node_idx, CLAIM_IDX]
        single_strengths = gradual_semantics(single_adjacency, base_scores)
        effects[attr] = float(single_strengths[CLAIM_IDX].item() - claim_base_score)

    driver = _driver_from_effects(effects)
    pred_choice = "A" if claim_strength >= claim_base_score else "B"

    return {
        "claim_strength": claim_strength,
        "pred_choice": pred_choice,
        "driver": driver,
        **{f"effect_{attr}": effect for attr, effect in effects.items()},
    }


def add_argllm_semantics_predictions(
    df: pd.DataFrame,
    *,
    semantics: str,
    conservativeness: float = 1.0,
    claim_base_score: float = CLAIM_BASE_SCORE,
    ok_col: str = "tau_ok",
) -> pd.DataFrame:
    out = df.copy()
    prefix = semantics.lower()

    records = [
        evaluate_tau_argument_framework_row(
            row,
            semantics=semantics,
            conservativeness=conservativeness,
            claim_base_score=claim_base_score,
        )
        for _, row in out.iterrows()
    ]
    metrics_df = pd.DataFrame.from_records(records, index=out.index)

    rename_map = {
        "claim_strength": f"{prefix}_claim_strength",
        "pred_choice": f"{prefix}_pred_choice",
        "driver": f"{prefix}_driver",
        **{f"effect_{attr}": f"{prefix}_effect_{attr}" for attr in ATTRIBUTES},
    }
    metrics_df = metrics_df.rename(columns=rename_map)
    out = pd.concat([out, metrics_df], axis=1)

    if ok_col in out.columns:
        out[f"{prefix}_ok"] = out[ok_col].fillna(False).astype(bool)
    else:
        out[f"{prefix}_ok"] = True

    return out
