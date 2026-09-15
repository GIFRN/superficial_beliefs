from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .stageA import ATTRIBUTES

RIDGE_LAMBDAS = (0.0, 1e-6, 1e-4, 1e-2, 1e-1)
M2_SHRINKAGE_LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
NLL_CLIP = 1e-9
ETA_CLIP = 1e-6
CELL_VALUES = (-2, -1, 0, 1, 2)


@dataclass
class BehavioralDesignMatrix:
    X: pd.DataFrame
    y: pd.Series
    successes: pd.Series
    weights: pd.Series
    groups: pd.Series
    filtered_df: pd.DataFrame
    feature_info: dict[str, Any]
    behavioral_model: str


@dataclass
class BehavioralModel:
    behavioral_model: str
    params: pd.Series | None
    feature_columns: list[str]
    feature_info: dict[str, Any]
    include_interactions: bool = False
    selected_ridge_lambda: float = 0.0
    fit_mode: str = "unpenalized"
    convergence_status: str = "ok"
    bse: pd.Series | None = None
    lookup_table: pd.DataFrame | None = None
    selected_shrinkage_lambda: float | None = None
    m1_selected_ridge_lambda: float | None = None
    m1_convergence_status: str | None = None
    num_unique_observed_cells: int | None = None

    @property
    def model_name(self) -> str:
        return {
            "m0": "M0",
            "m1": "M1",
            "m2": "M2",
        }.get(self.behavioral_model, str(self.behavioral_model).upper())

    def _require_params(self) -> pd.Series:
        if self.params is None:
            raise ValueError(f"{self.model_name} does not expose coefficient parameters")
        return self.params

    def _m2_lookup_index(self) -> pd.DataFrame:
        if self.lookup_table is None:
            raise ValueError("M2 model is missing lookup_table")
        indexed = getattr(self, "_lookup_indexed", None)
        if indexed is None:
            table = self.lookup_table.copy()
            for column in _cell_columns():
                table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0).astype(int)
            indexed = table.set_index(_cell_columns()).sort_index()
            setattr(self, "_lookup_indexed", indexed)
        return indexed

    def _lookup_m2_values(
        self,
        df: pd.DataFrame,
        *,
        column: str,
    ) -> np.ndarray:
        lookup = self._m2_lookup_index()
        cells = _cell_index_from_frame(df)
        values = lookup[column].reindex(cells)
        if values.isna().any():
            missing = values[values.isna()]
            raise ValueError(f"M2 lookup missing {len(missing)} cells for column {column}")
        return values.to_numpy(dtype=float)

    def _prepare_scoring_matrix(
        self,
        df: pd.DataFrame,
        *,
        exclude_b1: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        filtered_df = _filtered_scoring_rows(df, exclude_b1=exclude_b1)
        if self.behavioral_model == "m2":
            return filtered_df, pd.DataFrame(index=filtered_df.index)
        if self.behavioral_model == "m1":
            X = _build_m1_feature_matrix(filtered_df)
            return filtered_df, X.reindex(columns=self.feature_columns, fill_value=0.0)

        X = _build_legacy_scoring_matrix(
            filtered_df,
            include_interactions=self.include_interactions,
            include_order_terms=True,
        )
        return filtered_df, X.reindex(columns=self.feature_columns, fill_value=0.0)

    def linear_predictor(
        self,
        df: pd.DataFrame,
        *,
        exclude_b1: bool = False,
    ) -> pd.Series:
        if self.behavioral_model == "m2":
            filtered_df = _filtered_scoring_rows(df, exclude_b1=exclude_b1)
            probs = self.predict_proba(filtered_df, exclude_b1=False)
            eta = _logit(np.clip(probs.to_numpy(dtype=float), ETA_CLIP, 1.0 - ETA_CLIP))
            return pd.Series(eta, index=filtered_df.index, name="eta")
        filtered_df, X = self._prepare_scoring_matrix(df, exclude_b1=exclude_b1)
        params = self._require_params().reindex(self.feature_columns, fill_value=0.0).astype(float)
        eta = X.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
        return pd.Series(eta, index=filtered_df.index, name="eta")

    def predict_proba(
        self,
        df: pd.DataFrame,
        *,
        exclude_b1: bool = False,
    ) -> pd.Series:
        if self.behavioral_model == "m2":
            filtered_df = _filtered_scoring_rows(df, exclude_b1=exclude_b1)
            probs = self._lookup_m2_values(filtered_df, column="p_M2")
            return pd.Series(probs, index=filtered_df.index, name="p_choose_A")
        eta = self.linear_predictor(df, exclude_b1=exclude_b1)
        return pd.Series(_sigmoid(eta.to_numpy(dtype=float)), index=eta.index, name="p_choose_A")

    def predict_choice(
        self,
        df: pd.DataFrame,
        *,
        exclude_b1: bool = False,
    ) -> pd.Series:
        eta = self.linear_predictor(df, exclude_b1=exclude_b1)
        return pd.Series(np.where(eta.to_numpy(dtype=float) > 0.0, "A", "B"), index=eta.index, name="predicted_choice")

    def attribute_contributions(
        self,
        df: pd.DataFrame,
        *,
        exclude_b1: bool = False,
    ) -> pd.DataFrame:
        filtered_df, X = self._prepare_scoring_matrix(df, exclude_b1=exclude_b1)
        contributions = pd.DataFrame(index=filtered_df.index)

        if self.behavioral_model == "m2":
            eta = self.linear_predictor(filtered_df, exclude_b1=False)
            for attr in ATTRIBUTES:
                neutralized = filtered_df.copy()
                neutralized[f"delta_{attr}"] = 0
                eta_neutralized = self.linear_predictor(neutralized, exclude_b1=False)
                delta = eta.to_numpy(dtype=float) - eta_neutralized.to_numpy(dtype=float)
                contributions[f"C_{attr}"] = delta
        elif self.behavioral_model == "m1":
            main = self.feature_info.get("main", {})
            for attr in ATTRIBUTES:
                names = main.get(attr, {})
                value = np.zeros(len(filtered_df), dtype=float)
                feature_1 = names.get("step1")
                feature_2 = names.get("step2")
                if feature_1 in X.columns:
                    value += float(self._require_params().get(feature_1, 0.0)) * X[feature_1].to_numpy(dtype=float)
                if feature_2 in X.columns:
                    value += float(self._require_params().get(feature_2, 0.0)) * X[feature_2].to_numpy(dtype=float)
                contributions[f"C_{attr}"] = value
        else:
            main = self.feature_info.get("main", {})
            for attr in ATTRIBUTES:
                feature_name = main.get(attr, f"diff_{attr}")
                coeff = float(self._require_params().get(feature_name, 0.0))
                contributions[f"C_{attr}"] = coeff * _delta_series(filtered_df, attr).to_numpy(dtype=float)

            interactions = self.feature_info.get("interactions", {})
            for key, feature_name in interactions.items():
                attr_i, attr_j = _parse_interaction_key(key)
                if attr_i is None or attr_j is None:
                    continue
                coeff = float(self._require_params().get(feature_name, 0.0))
                term = 0.5 * coeff * (
                    _delta_series(filtered_df, attr_i).to_numpy(dtype=float)
                    * _delta_series(filtered_df, attr_j).to_numpy(dtype=float)
                )
                contributions[f"C_{attr_i}"] += term
                contributions[f"C_{attr_j}"] += term

        for attr in ATTRIBUTES:
            contributions[f"contrib_{attr}"] = contributions[f"C_{attr}"]

        values = contributions[[f"C_{attr}" for attr in ATTRIBUTES]].to_numpy(dtype=float)
        contributions["driver_A"] = [_canonical_argmax(row) for row in values]
        contributions["driver_B"] = [_canonical_argmin(row) for row in values]
        contributions["driver"] = contributions["driver_A"]
        return contributions

    def revealed_driver(
        self,
        df: pd.DataFrame,
        chosen_side: pd.Series | Iterable[str] | str,
        *,
        exclude_b1: bool = False,
    ) -> pd.DataFrame:
        filtered_df, _ = self._prepare_scoring_matrix(df, exclude_b1=exclude_b1)
        chosen = _coerce_choice_series(chosen_side, filtered_df.index)
        eta = self.linear_predictor(filtered_df, exclude_b1=False)
        influences = pd.DataFrame(index=filtered_df.index)

        for attr in ATTRIBUTES:
            neutralized = filtered_df.copy()
            neutralized[f"delta_{attr}"] = 0
            eta_neutralized = self.linear_predictor(neutralized, exclude_b1=False)
            influence = np.where(
                chosen.eq("A"),
                eta.to_numpy(dtype=float) - eta_neutralized.to_numpy(dtype=float),
                np.where(
                    chosen.eq("B"),
                    eta_neutralized.to_numpy(dtype=float) - eta.to_numpy(dtype=float),
                    np.nan,
                ),
            )
            influences[f"influence_{attr}"] = influence

        influence_values = influences[[f"influence_{attr}" for attr in ATTRIBUTES]].to_numpy(dtype=float)
        drivers = []
        margins = []
        for row in influence_values:
            drivers.append(_canonical_argmax(row))
            margins.append(_driver_margin(row))

        influences["revealed_driver"] = drivers
        influences["driver_margin"] = margins
        return influences

    @classmethod
    def from_summary_dict(
        cls,
        summary: Mapping[str, Any],
        *,
        base_path: str | PathLike[str] | None = None,
    ) -> "BehavioralModel":
        behavioral_model = summary.get("behavioral_model")
        if not behavioral_model:
            feature_columns_guess = list(summary.get("feature_columns", []))
            behavioral_model = "m1" if any(str(col).startswith("z_") for col in feature_columns_guess) else "m0"
        if behavioral_model == "m2":
            lookup_path = summary.get("lookup_table_path") or summary.get("lookup_table_relpath")
            if not lookup_path:
                raise ValueError("M2 summary is missing lookup_table_path")
            resolved_lookup = pd.read_parquet(_resolve_artifact_path(lookup_path, base_path=base_path))
            return cls(
                behavioral_model="m2",
                params=None,
                feature_columns=list(summary.get("feature_columns", _cell_columns())),
                feature_info=dict(summary.get("feature_info", {"canonical_attributes": list(ATTRIBUTES)})),
                include_interactions=False,
                selected_ridge_lambda=0.0,
                fit_mode=str(summary.get("fit_mode", "cell_lookup")),
                convergence_status=str(summary.get("convergence_status", "cell_lookup_ok")),
                bse=None,
                lookup_table=resolved_lookup,
                selected_shrinkage_lambda=float(summary.get("selected_shrinkage_lambda", 0.0) or 0.0),
                m1_selected_ridge_lambda=float(summary.get("m1_selected_ridge_lambda", 0.0) or 0.0),
                m1_convergence_status=summary.get("m1_convergence_status"),
                num_unique_observed_cells=(
                    int(summary["num_unique_observed_cells"])
                    if summary.get("num_unique_observed_cells") is not None
                    else None
                ),
            )

        params = pd.Series(summary.get("model_params", {}), dtype=float)
        if params.empty:
            raise ValueError("Summary is missing model_params")
        feature_columns = list(summary.get("feature_columns", list(params.index)))
        bse_raw = summary.get("model_bse")
        bse = pd.Series(bse_raw, dtype=float) if isinstance(bse_raw, dict) else None
        return cls(
            behavioral_model=str(behavioral_model),
            params=params,
            feature_columns=feature_columns,
            feature_info=dict(summary.get("feature_info", {})),
            include_interactions=bool(summary.get("include_interactions", False)),
            selected_ridge_lambda=float(summary.get("selected_ridge_lambda", 0.0) or 0.0),
            fit_mode=str(summary.get("fit_mode", "unpenalized")),
            convergence_status=str(summary.get("convergence_status", "ok")),
            bse=bse,
        )


def build_behavioral_design_matrix(
    df: pd.DataFrame,
    *,
    behavioral_model: str,
    exclude_b1: bool = True,
    group_col: str = "family_id",
) -> BehavioralDesignMatrix:
    filtered = _filtered_stagea_rows(df, exclude_b1=exclude_b1)

    if behavioral_model in {"m1", "m2"}:
        X = _build_m1_feature_matrix(filtered)
        feature_info: dict[str, Any] = {
            "behavioral_model": behavioral_model,
            "main": {
                attr: {"step1": f"z_{attr}_1", "step2": f"z_{attr}_2"}
                for attr in ATTRIBUTES
            },
            "canonical_attributes": list(ATTRIBUTES),
            "cell_columns": _cell_columns(),
        }
    else:
        X = _build_m0_feature_matrix(filtered)
        feature_info = {
            "behavioral_model": behavioral_model,
            "main": {attr: f"diff_{attr}" for attr in ATTRIBUTES},
            "canonical_attributes": list(ATTRIBUTES),
        }

    weights = filtered["trials"].astype(float)
    successes = filtered["successes"].astype(float)
    y = successes / weights
    groups = _resolve_groups(filtered, group_col=group_col)
    return BehavioralDesignMatrix(
        X=X,
        y=y,
        successes=successes,
        weights=weights,
        groups=groups,
        filtered_df=filtered,
        feature_info=feature_info,
        behavioral_model=behavioral_model,
    )


def fit_behavioral_model(
    df: pd.DataFrame,
    *,
    behavioral_model: str,
    exclude_b1: bool = True,
    group_col: str = "family_id",
    ridge_lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    shrinkage_lambdas: tuple[float, ...] = M2_SHRINKAGE_LAMBDAS,
    n_splits: int = 5,
) -> tuple[BehavioralModel, BehavioralDesignMatrix, dict[str, Any]]:
    design = build_behavioral_design_matrix(
        df,
        behavioral_model=behavioral_model,
        exclude_b1=exclude_b1,
        group_col=group_col,
    )

    if behavioral_model == "m2":
        model, summary = _fit_m2_model(
            design,
            group_col=group_col,
            ridge_lambdas=ridge_lambdas,
            shrinkage_lambdas=shrinkage_lambdas,
            n_splits=n_splits,
        )
        return model, design, summary

    cv_scores: dict[str, float] = {}
    selected_lambda = 0.0
    fit_mode = "unpenalized"
    convergence_status = "unpenalized_ok"

    if behavioral_model == "m1":
        trigger_reason = None
        try:
            raw_result = _fit_unpenalized(design)
            if _has_numerical_problems(raw_result, design):
                trigger_reason = "non_finite_or_non_converged_unpenalized_fit"
        except (np.linalg.LinAlgError, PerfectSeparationError, ValueError) as exc:
            raw_result = None
            trigger_reason = f"unpenalized_fit_error:{type(exc).__name__}"

        if trigger_reason is not None:
            cv_scores = cross_validate_ridge_lambda(
                design,
                ridge_lambdas=ridge_lambdas,
                n_splits=n_splits,
            )
            selected_lambda = select_ridge_lambda(cv_scores)
            raw_result, selected_lambda = _fit_best_available_ridge(
                design,
                cv_scores=cv_scores,
                selected_lambda=selected_lambda,
            )
            fit_mode = "ridge"
            convergence_status = f"ridge_ok_after_{trigger_reason}"
        else:
            selected_lambda = 0.0
    else:
        raw_result = _fit_unpenalized(design)

    model = _wrap_fitted_model(
        behavioral_model=behavioral_model,
        raw_result=raw_result,
        design=design,
        selected_ridge_lambda=selected_lambda,
        fit_mode=fit_mode,
        convergence_status=convergence_status,
    )

    train_probs = model.predict_proba(design.filtered_df, exclude_b1=False)
    train_nll = mean_binomial_nll(
        successes=design.successes,
        weights=design.weights,
        probs=train_probs.to_numpy(dtype=float),
    )
    summary = {
        "behavioral_model": behavioral_model,
        "behavioral_model_name": model.model_name,
        "selected_ridge_lambda": float(selected_lambda),
        "fit_mode": fit_mode,
        "convergence_status": convergence_status,
        "train_nll": float(train_nll),
        "cv_mean_nll_by_lambda": cv_scores,
        "coefficient_table": coefficient_table(model),
    }
    return model, design, summary


def cross_validate_ridge_lambda(
    design: BehavioralDesignMatrix,
    *,
    ridge_lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    n_splits: int = 5,
) -> dict[str, float]:
    splits = list(iter_grouped_splits(design.groups, n_splits=n_splits))
    if not splits:
        return {str(value): float("inf") for value in ridge_lambdas}
    scores: dict[str, list[float]] = {str(value): [] for value in ridge_lambdas}

    for train_idx, val_idx in splits:
        train_design = _slice_design(design, train_idx)
        val_design = _slice_design(design, val_idx)
        for ridge_lambda in ridge_lambdas:
            try:
                raw_result = _fit_with_lambda(train_design, ridge_lambda)
                probs = raw_result.predict(val_design.X).to_numpy(dtype=float)
                fold_score = mean_binomial_nll(
                    successes=val_design.successes,
                    weights=val_design.weights,
                    probs=probs,
                )
            except (np.linalg.LinAlgError, PerfectSeparationError, ValueError):
                fold_score = float("inf")
            scores[str(ridge_lambda)].append(float(fold_score))

    return {key: float(np.mean(values)) if values else float("inf") for key, values in scores.items()}


def select_ridge_lambda(cv_scores: Mapping[str, float]) -> float:
    finite_scores = [(float(key), float(value)) for key, value in cv_scores.items() if np.isfinite(value)]
    if not finite_scores:
        return 1e-1
    best_score = min(value for _, value in finite_scores)
    eligible = [ridge_lambda for ridge_lambda, value in finite_scores if abs(value - best_score) <= 1e-8]
    return float(min(eligible))


def coefficient_table(model: BehavioralModel) -> list[dict[str, Any]]:
    if model.params is None:
        return []
    bse = model.bse if model.bse is not None else pd.Series(np.nan, index=model.feature_columns)
    rows = []
    for feature in model.feature_columns:
        standard_error = bse.get(feature, np.nan)
        rows.append(
            {
                "feature": feature,
                "coefficient": float(model.params.get(feature, 0.0)),
                "standard_error": None if pd.isna(standard_error) else float(standard_error),
            }
        )
    return rows


def mean_binomial_nll(
    *,
    successes: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    probs: pd.Series | np.ndarray,
) -> float:
    successes_arr = np.asarray(successes, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    probs_arr = np.clip(np.asarray(probs, dtype=float), NLL_CLIP, 1.0 - NLL_CLIP)
    failures_arr = weights_arr - successes_arr
    row_nll = -(successes_arr * np.log(probs_arr) + failures_arr * np.log(1.0 - probs_arr))
    return float(np.mean(row_nll))


def _fit_m2_model(
    design: BehavioralDesignMatrix,
    *,
    group_col: str,
    ridge_lambdas: tuple[float, ...],
    shrinkage_lambdas: tuple[float, ...],
    n_splits: int,
) -> tuple[BehavioralModel, dict[str, Any]]:
    cv_scores = cross_validate_m2_shrinkage(
        design,
        group_col=group_col,
        ridge_lambdas=ridge_lambdas,
        shrinkage_lambdas=shrinkage_lambdas,
        n_splits=n_splits,
    )
    selected_lambda = _select_smallest_best_value(cv_scores, default=float(shrinkage_lambdas[0]))
    m1_model, _, m1_summary = fit_behavioral_model(
        design.filtered_df,
        behavioral_model="m1",
        exclude_b1=False,
        group_col=group_col,
        ridge_lambdas=ridge_lambdas,
        n_splits=n_splits,
    )
    observed_counts = aggregate_m2_cell_counts(design.filtered_df)
    lookup_table = build_m2_lookup_table(
        observed_counts=observed_counts,
        prior_model=m1_model,
        shrinkage_lambda=selected_lambda,
    )
    feature_info = {
        **design.feature_info,
        "behavioral_model": "m2",
        "canonical_attributes": list(ATTRIBUTES),
        "cell_columns": _cell_columns(),
    }
    model = BehavioralModel(
        behavioral_model="m2",
        params=None,
        feature_columns=_cell_columns(),
        feature_info=feature_info,
        include_interactions=False,
        selected_ridge_lambda=0.0,
        fit_mode="cell_lookup",
        convergence_status="cell_lookup_ok",
        bse=None,
        lookup_table=lookup_table,
        selected_shrinkage_lambda=float(selected_lambda),
        m1_selected_ridge_lambda=float(m1_summary.get("selected_ridge_lambda", 0.0) or 0.0),
        m1_convergence_status=str(m1_summary.get("convergence_status", "ok")),
        num_unique_observed_cells=int((lookup_table["n_c"] > 0).sum()),
    )
    train_probs = model.predict_proba(design.filtered_df, exclude_b1=False)
    train_nll = mean_binomial_nll(
        successes=design.successes,
        weights=design.weights,
        probs=train_probs.to_numpy(dtype=float),
    )
    summary = {
        "behavioral_model": "m2",
        "behavioral_model_name": "M2",
        "selected_ridge_lambda": 0.0,
        "selected_shrinkage_lambda": float(selected_lambda),
        "fit_mode": "cell_lookup",
        "convergence_status": "cell_lookup_ok",
        "train_nll": float(train_nll),
        "cv_mean_nll_by_lambda": cv_scores,
        "coefficient_table": [],
        "num_unique_observed_cells": int((lookup_table["n_c"] > 0).sum()),
        "m1_selected_ridge_lambda": float(m1_summary.get("selected_ridge_lambda", 0.0) or 0.0),
        "m1_convergence_status": str(m1_summary.get("convergence_status", "ok")),
    }
    return model, summary


def cross_validate_m2_shrinkage(
    design: BehavioralDesignMatrix,
    *,
    group_col: str,
    ridge_lambdas: tuple[float, ...],
    shrinkage_lambdas: tuple[float, ...],
    n_splits: int,
) -> dict[str, float]:
    splits = list(iter_grouped_splits(design.groups, n_splits=n_splits))
    if not splits:
        return {str(value): float("inf") for value in shrinkage_lambdas}

    scores: dict[str, list[float]] = {str(value): [] for value in shrinkage_lambdas}
    for train_idx, val_idx in splits:
        train_design = _slice_design(design, train_idx)
        val_design = _slice_design(design, val_idx)
        m1_fold, _, _ = fit_behavioral_model(
            train_design.filtered_df,
            behavioral_model="m1",
            exclude_b1=False,
            group_col=group_col,
            ridge_lambdas=ridge_lambdas,
            n_splits=n_splits,
        )
        observed_counts = aggregate_m2_cell_counts(train_design.filtered_df)
        for shrinkage_lambda in shrinkage_lambdas:
            lookup_table = build_m2_lookup_table(
                observed_counts=observed_counts,
                prior_model=m1_fold,
                shrinkage_lambda=shrinkage_lambda,
            )
            probs = lookup_m2_probabilities(val_design.filtered_df, lookup_table)
            fold_score = mean_binomial_nll(
                successes=val_design.successes,
                weights=val_design.weights,
                probs=probs,
            )
            scores[str(shrinkage_lambda)].append(float(fold_score))

    return {key: float(np.mean(values)) if values else float("inf") for key, values in scores.items()}


def aggregate_m2_cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    working = _filtered_scoring_rows(df, exclude_b1=False)
    return (
        working.groupby(_cell_columns(), dropna=False, as_index=False)
        .agg(
            k_c=("successes", "sum"),
            n_c=("trials", "sum"),
        )
        .reset_index(drop=True)
    )


def build_m2_lookup_table(
    *,
    observed_counts: pd.DataFrame,
    prior_model: BehavioralModel,
    shrinkage_lambda: float,
) -> pd.DataFrame:
    full = all_behavioral_cells()
    prior_probs = prior_model.predict_proba(full, exclude_b1=False)
    full["p0"] = prior_probs.to_numpy(dtype=float)

    counts = observed_counts.copy()
    if counts.empty:
        counts = full[_cell_columns()].copy()
        counts["k_c"] = 0.0
        counts["n_c"] = 0.0
    merged = full.merge(counts, on=_cell_columns(), how="left")
    merged["k_c"] = pd.to_numeric(merged["k_c"], errors="coerce").fillna(0.0).astype(float)
    merged["n_c"] = pd.to_numeric(merged["n_c"], errors="coerce").fillna(0.0).astype(float)
    merged["p_M2"] = (merged["k_c"] + float(shrinkage_lambda) * merged["p0"]) / (merged["n_c"] + float(shrinkage_lambda))
    return merged


def lookup_m2_probabilities(df: pd.DataFrame, lookup_table: pd.DataFrame) -> np.ndarray:
    indexed = lookup_table.copy()
    for column in _cell_columns():
        indexed[column] = pd.to_numeric(indexed[column], errors="coerce").fillna(0).astype(int)
    indexed = indexed.set_index(_cell_columns()).sort_index()
    values = indexed["p_M2"].reindex(_cell_index_from_frame(df))
    if values.isna().any():
        raise ValueError(f"M2 lookup missing {int(values.isna().sum())} cells")
    return values.to_numpy(dtype=float)


def load_behavioral_model_summary(path: str | PathLike[str]) -> tuple[BehavioralModel, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    model = BehavioralModel.from_summary_dict(summary, base_path=path)
    return model, summary


def iter_grouped_splits(groups: pd.Series | Iterable[str], *, n_splits: int = 5):
    groups_array = np.asarray(list(groups), dtype=str)
    unique_groups = np.unique(groups_array)
    if len(unique_groups) < 2:
        return
    n_splits = min(n_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups_array), dtype=float)
    for train_idx, val_idx in splitter.split(dummy, groups=groups_array):
        yield train_idx, val_idx


def m0_weights_from_model(model: BehavioralModel) -> dict[str, dict[str, float]]:
    params = model._require_params()
    betas = {attr: float(params.get(f"diff_{attr}", 0.0)) for attr in ATTRIBUTES}
    ames = dict(betas)
    positive = {attr: max(value, 0.0) for attr, value in ames.items()}
    denom = sum(positive.values())
    weights = {attr: (value / denom if denom else 0.0) for attr, value in positive.items()}
    return {"beta": betas, "AME": ames, "weights": weights}


def _filtered_stagea_rows(df: pd.DataFrame, *, exclude_b1: bool) -> pd.DataFrame:
    filtered = df[df["trials"] > 0].copy()
    if exclude_b1 and "block" in filtered:
        filtered = filtered[filtered["block"] != "B1"]
    if filtered.empty:
        raise ValueError("No trials with valid responses available for behavioral model fit")
    return filtered


def _filtered_scoring_rows(df: pd.DataFrame, *, exclude_b1: bool) -> pd.DataFrame:
    filtered = df.copy()
    if "trials" in filtered:
        filtered = filtered[filtered["trials"] > 0].copy()
    if exclude_b1 and "block" in filtered:
        filtered = filtered[filtered["block"] != "B1"].copy()
    if filtered.empty:
        raise ValueError("No rows available for behavioral model scoring")
    return filtered


def _resolve_groups(df: pd.DataFrame, *, group_col: str) -> pd.Series:
    if group_col in df:
        return df[group_col].astype(str)
    if "family_id" in df:
        return df["family_id"].astype(str)
    if "config_id" in df:
        return df["config_id"].astype(str)
    raise ValueError("No grouped-CV key found; expected family_id or config_id")


def _delta_series(df: pd.DataFrame, attr: str) -> pd.Series:
    column = f"delta_{attr}"
    if column in df:
        return df[column]
    return pd.Series(0.0, index=df.index)


def _delta_pos_series(df: pd.DataFrame, attr: str) -> pd.Series:
    column = f"delta_pos_{attr}"
    if column in df:
        return df[column]
    return pd.Series(0.0, index=df.index)


def _build_m0_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0
    for attr in ATTRIBUTES:
        X[f"diff_{attr}"] = _delta_series(df, attr).astype(float)
    return X


def _build_m1_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0
    for attr in ATTRIBUTES:
        delta = _delta_series(df, attr)
        X[f"z_{attr}_1"] = (delta.eq(1).astype(float) - delta.eq(-1).astype(float))
        X[f"z_{attr}_2"] = (delta.eq(2).astype(float) - delta.eq(-2).astype(float))
    return X


def _build_legacy_scoring_matrix(
    df: pd.DataFrame,
    *,
    include_interactions: bool,
    include_order_terms: bool,
) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0
    for attr in ATTRIBUTES:
        X[f"diff_{attr}"] = _delta_series(df, attr).astype(float)
    if include_interactions:
        for i, attr_i in enumerate(ATTRIBUTES):
            for attr_j in ATTRIBUTES[i + 1 :]:
                X[f"inter_{attr_i}{attr_j}"] = _delta_series(df, attr_i).astype(float) * _delta_series(df, attr_j).astype(float)
    if include_order_terms:
        for attr in ATTRIBUTES:
            order = _delta_pos_series(df, attr).astype(float)
            X[f"order_{attr}"] = order
            X[f"order_x_{attr}"] = _delta_series(df, attr).astype(float) * order
    return X


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log(values / (1.0 - values))


def _coerce_choice_series(
    chosen_side: pd.Series | Iterable[str] | str,
    index: pd.Index,
) -> pd.Series:
    if isinstance(chosen_side, pd.Series):
        return chosen_side.reindex(index)
    if isinstance(chosen_side, str):
        return pd.Series(chosen_side, index=index, dtype=object)
    values = list(chosen_side)
    if len(values) != len(index):
        raise ValueError("chosen_side must be scalar or align with the filtered rows")
    return pd.Series(values, index=index, dtype=object)


def _canonical_argmax(values: np.ndarray) -> str:
    return ATTRIBUTES[int(np.argmax(np.asarray(values, dtype=float)))]


def _canonical_argmin(values: np.ndarray) -> str:
    return ATTRIBUTES[int(np.argmin(np.asarray(values, dtype=float)))]


def _driver_margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def _parse_interaction_key(key: Any) -> tuple[str | None, str | None]:
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return str(key[0]), str(key[1])
    if isinstance(key, str):
        raw = key.strip("()").replace("'", "").replace('"', "")
        parts = [part.strip() for part in raw.replace("|", ",").split(",") if part.strip()]
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, None


def _slice_design(design: BehavioralDesignMatrix, indices: np.ndarray) -> BehavioralDesignMatrix:
    return BehavioralDesignMatrix(
        X=design.X.iloc[indices].copy(),
        y=design.y.iloc[indices].copy(),
        successes=design.successes.iloc[indices].copy(),
        weights=design.weights.iloc[indices].copy(),
        groups=design.groups.iloc[indices].copy(),
        filtered_df=design.filtered_df.iloc[indices].copy(),
        feature_info=design.feature_info,
        behavioral_model=design.behavioral_model,
    )


def _cell_columns() -> list[str]:
    return [f"delta_{attr}" for attr in ATTRIBUTES]


def all_behavioral_cells() -> pd.DataFrame:
    rows = list(product(CELL_VALUES, repeat=len(ATTRIBUTES)))
    return pd.DataFrame(rows, columns=_cell_columns(), dtype=int)


def _cell_index_from_frame(df: pd.DataFrame) -> pd.MultiIndex:
    normalized = pd.DataFrame(index=df.index)
    for column in _cell_columns():
        normalized[column] = pd.to_numeric(df.get(column, 0), errors="coerce").fillna(0).astype(int)
    return pd.MultiIndex.from_frame(normalized[_cell_columns()])


def _select_smallest_best_value(cv_scores: Mapping[str, float], *, default: float) -> float:
    finite_scores = [(float(key), float(value)) for key, value in cv_scores.items() if np.isfinite(value)]
    if not finite_scores:
        return float(default)
    best_score = min(value for _, value in finite_scores)
    eligible = [candidate for candidate, value in finite_scores if abs(value - best_score) <= 1e-8]
    return float(min(eligible))


def _resolve_artifact_path(
    path: str | PathLike[str],
    *,
    base_path: str | PathLike[str] | None,
) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    if base_path is None:
        return str(candidate)
    return str(Path(base_path).resolve().parent / candidate)


def _fit_glm(design: BehavioralDesignMatrix):
    return sm.GLM(
        design.y,
        design.X,
        family=sm.families.Binomial(),
        freq_weights=design.weights,
    )


def _fit_unpenalized(design: BehavioralDesignMatrix):
    return _fit_glm(design).fit()


def _fit_with_lambda(design: BehavioralDesignMatrix, ridge_lambda: float):
    glm = _fit_glm(design)
    if ridge_lambda <= 0.0:
        return glm.fit()
    return glm.fit_regularized(alpha=ridge_lambda, L1_wt=0.0)


def _has_numerical_problems(raw_result, design: BehavioralDesignMatrix) -> bool:
    params = pd.Series(raw_result.params, index=design.X.columns, dtype=float)
    if not np.isfinite(params.to_numpy(dtype=float)).all():
        return True
    converged = getattr(raw_result, "converged", True)
    if converged is False:
        return True
    probs = np.asarray(raw_result.predict(design.X), dtype=float)
    eta = design.X.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
    return (not np.isfinite(probs).all()) or (not np.isfinite(eta).all())


def _fit_best_available_ridge(
    design: BehavioralDesignMatrix,
    *,
    cv_scores: Mapping[str, float],
    selected_lambda: float,
):
    ordered_candidates = sorted(
        ((float(key), float(value)) for key, value in cv_scores.items() if np.isfinite(value)),
        key=lambda item: (0 if item[0] == selected_lambda else 1, item[1], item[0]),
    )
    if not ordered_candidates:
        ordered_candidates = [(selected_lambda, float("inf"))]

    last_error = None
    for ridge_lambda, _ in ordered_candidates:
        try:
            raw_result = _fit_with_lambda(design, ridge_lambda)
            if _has_numerical_problems(raw_result, design):
                last_error = ValueError(f"non-finite ridge fit for lambda={ridge_lambda}")
                continue
            return raw_result, ridge_lambda
        except (np.linalg.LinAlgError, PerfectSeparationError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No valid ridge fit found")


def _wrap_fitted_model(
    *,
    behavioral_model: str,
    raw_result,
    design: BehavioralDesignMatrix,
    selected_ridge_lambda: float,
    fit_mode: str,
    convergence_status: str,
) -> BehavioralModel:
    params = pd.Series(raw_result.params, index=design.X.columns, dtype=float)
    raw_bse = getattr(raw_result, "bse", None)
    bse = pd.Series(raw_bse, index=design.X.columns, dtype=float) if raw_bse is not None else None
    return BehavioralModel(
        behavioral_model=behavioral_model,
        params=params,
        feature_columns=list(design.X.columns),
        feature_info=design.feature_info,
        selected_ridge_lambda=selected_ridge_lambda,
        fit_mode=fit_mode,
        convergence_status=convergence_status,
        bse=bse,
    )
