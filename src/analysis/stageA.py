from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import prepare_stageA_data

ATTRIBUTES = ["E", "A", "S", "D"]


@dataclass
class DesignMatrix:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    weights: pd.Series
    feature_info: dict[str, Any]

    def __iter__(self):  # allows tuple unpacking into X, y, groups
        yield self.X
        yield self.y
        yield self.groups


def build_design_matrix(
    df: pd.DataFrame,
    *,
    include_interactions: bool = False,
    include_order_terms: bool = True,
    exclude_b1: bool = True,
) -> DesignMatrix:
    filtered = df[df["trials"] > 0].copy()
    
    # Exclude B1 trials from weight estimation (B1 is for rationality check only)
    if exclude_b1:
        filtered = filtered[filtered["block"] != "B1"]
    
    if filtered.empty:
        raise ValueError("No trials with valid responses available for Stage A fit")

    X = pd.DataFrame(index=filtered.index)
    X["Intercept"] = 1.0
    feature_info: dict[str, Any] = {
        "main": {},
        "interactions": {},
        "order_bias": {},
        "order_interactions": {},
    }

    for attr in ATTRIBUTES:
        col = f"delta_{attr}"
        if col not in filtered:
            filtered[col] = 0
        feature_name = f"diff_{attr}"
        X[feature_name] = filtered[col]
        feature_info["main"][attr] = feature_name

    if include_interactions:
        for attr_i, attr_j in combinations(ATTRIBUTES, 2):
            name = f"inter_{attr_i}{attr_j}"
            X[name] = filtered[f"delta_{attr_i}"] * filtered[f"delta_{attr_j}"]
            feature_info["interactions"][tuple(sorted((attr_i, attr_j)))] = name

    if include_order_terms:
        for attr in ATTRIBUTES:
            delta_pos = filtered.get(f"delta_pos_{attr}", 0)
            order_name = f"order_{attr}"
            X[order_name] = delta_pos
            feature_info["order_bias"][attr] = order_name
            interact_name = f"order_x_{attr}"
            X[interact_name] = filtered[f"delta_{attr}"] * delta_pos
            feature_info["order_interactions"][attr] = interact_name

    weights = filtered["trials"].astype(float)
    successes = filtered["successes"].astype(float)
    y = successes / weights
    groups = filtered["config_id"].astype(str)

    return DesignMatrix(X=X, y=y, groups=groups, weights=weights, feature_info=feature_info)


def fit_glm_clustered(design: DesignMatrix | pd.DataFrame, y=None, groups=None, weights=None):
    if isinstance(design, DesignMatrix):
        X = design.X.copy()
        y_data = design.y
        groups_data = design.groups
        weights_data = design.weights
        feature_info = design.feature_info
    else:
        X = pd.DataFrame(design)
        if y is None or groups is None:
            raise ValueError("y and groups must be provided when passing raw matrices")
        y_data = pd.Series(y)
        groups_data = pd.Series(groups)
        weights_data = pd.Series(weights if weights is not None else np.ones_like(y_data))
        feature_info = {}

    model = sm.GLM(y_data, X, family=sm.families.Binomial(), freq_weights=weights_data)
    result = model.fit()
    try:
        robust = result.get_robustcov_results(cov_type="cluster", groups=groups_data)
    except AttributeError:
        # Older statsmodels versions lack get_robustcov_results; fall back to sandwich estimator.
        from statsmodels.stats.sandwich_covariance import cov_cluster

        cov = cov_cluster(result, np.asarray(groups_data))
        result.cov_params_default = cov
        result.cov_type = "cluster"
        result.cov_kwds = {"groups": groups_data}
        robust = result
    robust.feature_info = feature_info
    robust.weights = weights_data
    return robust


def compute_ames_and_weights(model) -> dict[str, Any]:
    feature_info = getattr(model, "feature_info", {})
    main = feature_info.get("main", {})
    betas: dict[str, float] = {}
    ses: dict[str, float] = {}
    ames: dict[str, float] = {}
    for attr, feature_name in main.items():
        betas[attr] = float(model.params.get(feature_name, np.nan))
        ses[attr] = float(model.bse.get(feature_name, np.nan))
        ames[attr] = betas[attr]
    positive = {attr: max(value, 0.0) for attr, value in ames.items()}
    denom = sum(positive.values())
    weights = {attr: (value / denom if denom else 0.0) for attr, value in positive.items()}
    return {
        "beta": betas,
        "se": ses,
        "AME": ames,
        "weights": weights,
    }


def per_trial_contributions(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_info = getattr(model, "feature_info", {})
    main = feature_info.get("main", {})
    interactions = feature_info.get("interactions", {})
    contributions = pd.DataFrame(index=df.index)

    for attr, feature_name in main.items():
        coeff = model.params.get(feature_name, 0.0)
        contributions[f"C_{attr}"] = coeff * df[f"delta_{attr}"]

    for (attr_i, attr_j), feature_name in interactions.items():
        coeff = model.params.get(feature_name, 0.0)
        term = 0.5 * coeff * df[f"delta_{attr_i}"] * df[f"delta_{attr_j}"]
        contributions[f"C_{attr_i}"] += term
        contributions[f"C_{attr_j}"] += term

    driver = contributions.apply(_argmax_attr, axis=1)
    contributions["driver"] = driver
    return contributions


def _argmax_attr(row: pd.Series) -> str:
    attrs = [col.split("_")[1] for col in row.index if col.startswith("C_")]
    values = [row[f"C_{attr}"] for attr in attrs]
    max_idx = int(np.argmax(values)) if values else 0
    return attrs[max_idx] if attrs else "E"


def fit_stageA_with_validation(
    trials_df: pd.DataFrame, 
    choice_agg: pd.DataFrame,
    include_interactions: bool = False
) -> dict[str, Any]:
    """Fit Stage A model with B1 validation"""
    from .diagnostics import validate_b1_rationality, validate_b1_probes
    
    # Validate B1 rationality first
    b1_validation = validate_b1_rationality(trials_df, choice_agg)
    
    # Validate B1 probes
    b1_probes = validate_b1_probes(trials_df, choice_agg)
    
    # Prepare data and fit model excluding B1
    stageA_data = prepare_stageA_data(trials_df, choice_agg)
    design = build_design_matrix(stageA_data, include_interactions=include_interactions, exclude_b1=True)
    model = fit_glm_clustered(design)
    
    # Compute AMEs and weights
    ames_weights = compute_ames_and_weights(model)
    
    # Compute per-trial contributions
    contributions = per_trial_contributions(stageA_data, model)
    
    return {
        "model": model,
        "design_matrix": design,
        "ames_weights": ames_weights,
        "contributions": contributions,
        "b1_validation": b1_validation,
        "b1_probes": b1_probes,
        "stageA_data": stageA_data
    }
