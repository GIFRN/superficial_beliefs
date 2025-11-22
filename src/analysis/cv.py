from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .stageA import DesignMatrix, fit_glm_clustered


def grouped_kfold(groups: Iterable[str], n_splits: int, *, random_state: int | None = None):
    groups = np.array(list(groups))
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)
    folds = np.array_split(unique_groups, n_splits)
    for fold in folds:
        test_mask = np.isin(groups, fold)
        train_mask = ~test_mask
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def cross_validate_design(design: DesignMatrix, *, n_splits: int = 5, random_state: int | None = None) -> dict[str, float]:
    groups = design.groups.to_numpy()
    metrics = []
    for train_idx, test_idx in grouped_kfold(groups, n_splits, random_state=random_state):
        train_design = DesignMatrix(
            X=design.X.iloc[train_idx],
            y=design.y.iloc[train_idx],
            groups=design.groups.iloc[train_idx],
            weights=design.weights.iloc[train_idx],
            feature_info=design.feature_info,
        )
        model = fit_glm_clustered(train_design)
        test_X = design.X.iloc[test_idx]
        probs = model.predict(test_X)
        weights = design.weights.iloc[test_idx].to_numpy()
        outcomes = design.y.iloc[test_idx].to_numpy()
        metrics.append(_metric_bundle(probs, outcomes, weights))
    df = pd.DataFrame(metrics)
    return df.mean().to_dict()


def _metric_bundle(probs: np.ndarray, outcomes: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    eps = 1e-9
    probs = np.clip(probs, eps, 1 - eps)
    successes = outcomes * weights
    failures = weights - successes
    log_loss = -np.sum(successes * np.log(probs) + failures * np.log(1 - probs)) / np.sum(weights)
    brier = np.sum(weights * (outcomes - probs) ** 2) / np.sum(weights)
    predicted = (probs >= 0.5).astype(int)
    accuracy = np.sum(weights * (predicted == (outcomes >= 0.5))) / np.sum(weights)
    return {
        "log_loss": log_loss,
        "brier": brier,
        "accuracy": accuracy,
    }
