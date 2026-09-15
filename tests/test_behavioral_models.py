import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.behavioral_models import (
    BehavioralModel,
    all_behavioral_cells,
    build_m2_lookup_table,
    build_behavioral_design_matrix,
    fit_behavioral_model,
    iter_grouped_splits,
    load_behavioral_model_summary,
    select_ridge_lambda,
)
from src.analysis.canonical_balanced import build_balanced_eval_frame
from src.analysis.behavioral_robustness import (
    build_condition_comparison,
    build_per_draw_driver_table,
    build_test_row_predictions,
)
from src.analysis.features import load_responses
from src.analysis.final_benchmark import dataset_dir, output_root, resolve_run_dir, run_prefix, stagea_dir


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [f"t{i}" for i in range(5)],
            "family_id": [f"f{i}" for i in range(5)],
            "config_id": [f"c{i}" for i in range(5)],
            "block": ["B3"] * 5,
            "delta_E": [-2, -1, 0, 1, 2],
            "delta_A": [0, 0, 0, 0, 0],
            "delta_S": [0, 0, 0, 0, 0],
            "delta_D": [0, 0, 0, 0, 0],
            "successes": [0, 0, 1, 1, 1],
            "trials": [1, 1, 1, 1, 1],
        }
    )


def test_m1_feature_encoding():
    design = build_behavioral_design_matrix(_base_df(), behavioral_model="m1")
    assert design.X["z_E_1"].tolist() == [0.0, -1.0, 0.0, 1.0, 0.0]
    assert design.X["z_E_2"].tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_all_behavioral_cells_has_625_rows():
    cells = all_behavioral_cells()
    assert len(cells) == 625
    assert list(cells.columns) == ["delta_E", "delta_A", "delta_S", "delta_D"]


def test_grouped_cv_integrity_uses_disjoint_family_ids():
    df = pd.concat([_base_df(), _base_df().assign(trial_id=lambda x: x["trial_id"] + "_x", family_id=lambda x: x["family_id"] + "_x")], ignore_index=True)
    design = build_behavioral_design_matrix(df, behavioral_model="m1")
    splits = list(iter_grouped_splits(design.groups, n_splits=5))
    assert splits
    for train_idx, val_idx in splits:
        train_groups = set(design.groups.iloc[train_idx])
        val_groups = set(design.groups.iloc[val_idx])
        assert train_groups.isdisjoint(val_groups)


def test_choice_tie_convention_matches_existing_m0_behavior():
    df = _base_df().iloc[[2]].copy()
    design = build_behavioral_design_matrix(df, behavioral_model="m1")
    model = BehavioralModel(
        behavioral_model="m1",
        params=pd.Series(0.0, index=design.X.columns),
        feature_columns=list(design.X.columns),
        feature_info=design.feature_info,
    )
    assert model.predict_choice(df).iloc[0] == "B"


def test_neutralized_driver_tie_breaks_by_canonical_order():
    df = _base_df().iloc[[2]].copy()
    design = build_behavioral_design_matrix(df, behavioral_model="m1")
    model = BehavioralModel(
        behavioral_model="m1",
        params=pd.Series(0.0, index=design.X.columns),
        feature_columns=list(design.X.columns),
        feature_info=design.feature_info,
    )
    driver = model.revealed_driver(df, pd.Series(["A"], index=df.index))
    assert driver["revealed_driver"].iloc[0] == "E"
    assert driver["driver_margin"].iloc[0] == 0.0


def _simple_m1_prior() -> BehavioralModel:
    feature_columns = ["Intercept"]
    for attr in ("E", "A", "S", "D"):
        feature_columns.extend([f"z_{attr}_1", f"z_{attr}_2"])
    params = pd.Series(
        {
            "Intercept": 0.2,
            "z_E_1": 0.4,
            "z_E_2": 0.8,
            "z_A_1": -0.3,
            "z_A_2": -0.6,
            "z_S_1": 0.2,
            "z_S_2": 0.5,
            "z_D_1": -0.1,
            "z_D_2": -0.2,
        }
    ).reindex(feature_columns, fill_value=0.0)
    return BehavioralModel(
        behavioral_model="m1",
        params=params,
        feature_columns=feature_columns,
        feature_info={"canonical_attributes": ["E", "A", "S", "D"]},
    )


def test_m2_unseen_cells_fall_back_to_m1_exactly():
    prior_model = _simple_m1_prior()
    observed_counts = pd.DataFrame(
        {
            "delta_E": [2],
            "delta_A": [0],
            "delta_S": [0],
            "delta_D": [0],
            "k_c": [3.0],
            "n_c": [4.0],
        }
    )
    lookup = build_m2_lookup_table(
        observed_counts=observed_counts,
        prior_model=prior_model,
        shrinkage_lambda=2.0,
    )
    unseen_row = pd.DataFrame({"delta_E": [-2], "delta_A": [1], "delta_S": [0], "delta_D": [0]})
    p0 = float(prior_model.predict_proba(unseen_row, exclude_b1=False).iloc[0])
    p_m2 = float(
        lookup.set_index(["delta_E", "delta_A", "delta_S", "delta_D"])
        .loc[(-2, 1, 0, 0), "p_M2"]
    )
    assert np.isclose(p_m2, p0)


def test_m2_seen_cell_matches_shrinkage_formula():
    prior_model = _simple_m1_prior()
    observed_counts = pd.DataFrame(
        {
            "delta_E": [2],
            "delta_A": [0],
            "delta_S": [0],
            "delta_D": [0],
            "k_c": [3.0],
            "n_c": [4.0],
        }
    )
    shrinkage_lambda = 2.0
    lookup = build_m2_lookup_table(
        observed_counts=observed_counts,
        prior_model=prior_model,
        shrinkage_lambda=shrinkage_lambda,
    )
    p0 = float(prior_model.predict_proba(pd.DataFrame({"delta_E": [2], "delta_A": [0], "delta_S": [0], "delta_D": [0]}), exclude_b1=False).iloc[0])
    expected = (3.0 + shrinkage_lambda * p0) / (4.0 + shrinkage_lambda)
    actual = float(
        lookup.set_index(["delta_E", "delta_A", "delta_S", "delta_D"])
        .loc[(2, 0, 0, 0), "p_M2"]
    )
    assert np.isclose(actual, expected)


def test_m2_large_shrinkage_moves_probability_close_to_prior():
    prior_model = _simple_m1_prior()
    observed_counts = pd.DataFrame(
        {
            "delta_E": [2],
            "delta_A": [0],
            "delta_S": [0],
            "delta_D": [0],
            "k_c": [3.0],
            "n_c": [4.0],
        }
    )
    lookup = build_m2_lookup_table(
        observed_counts=observed_counts,
        prior_model=prior_model,
        shrinkage_lambda=1_000_000.0,
    )
    row = pd.DataFrame({"delta_E": [2], "delta_A": [0], "delta_S": [0], "delta_D": [0]})
    p0 = float(prior_model.predict_proba(row, exclude_b1=False).iloc[0])
    p_m2 = float(
        lookup.set_index(["delta_E", "delta_A", "delta_S", "delta_D"])
        .loc[(2, 0, 0, 0), "p_M2"]
    )
    assert abs(p_m2 - p0) < 1e-5


def test_m1_driver_equivalence_matches_contributions_for_actor_side():
    df = pd.DataFrame(
        {
            "trial_id": ["t"],
            "family_id": ["f"],
            "config_id": ["c"],
            "block": ["B3"],
            "delta_E": [2],
            "delta_A": [1],
            "delta_S": [-1],
            "delta_D": [0],
            "successes": [1],
            "trials": [1],
        }
    )
    design = build_behavioral_design_matrix(df, behavioral_model="m1")
    model = BehavioralModel(
        behavioral_model="m1",
        params=pd.Series(
            {
                "Intercept": 0.0,
                "z_E_1": 0.0,
                "z_E_2": 1.5,
                "z_A_1": 0.75,
                "z_A_2": 0.0,
                "z_S_1": -0.5,
                "z_S_2": 0.0,
                "z_D_1": 0.0,
                "z_D_2": 0.0,
            }
        ).reindex(design.X.columns, fill_value=0.0),
        feature_columns=list(design.X.columns),
        feature_info=design.feature_info,
    )
    contrib = model.attribute_contributions(df)
    driver_a = model.revealed_driver(df, pd.Series(["A"], index=df.index))
    driver_b = model.revealed_driver(df, pd.Series(["B"], index=df.index))
    for attr in ("E", "A", "S", "D"):
        assert np.isclose(driver_a[f"influence_{attr}"].iloc[0], contrib[f"contrib_{attr}"].iloc[0])
        assert np.isclose(driver_b[f"influence_{attr}"].iloc[0], -contrib[f"contrib_{attr}"].iloc[0])


def test_m2_contributions_match_revealed_driver_logic():
    lookup = all_behavioral_cells().copy()
    eta = (
        0.5 * lookup["delta_E"].to_numpy(dtype=float)
        + 0.3 * lookup["delta_A"].to_numpy(dtype=float)
        - 0.25 * lookup["delta_S"].to_numpy(dtype=float)
        + 0.1 * lookup["delta_D"].to_numpy(dtype=float)
    )
    lookup["p0"] = 1.0 / (1.0 + np.exp(-eta))
    lookup["k_c"] = 0.0
    lookup["n_c"] = 0.0
    lookup["p_M2"] = lookup["p0"]
    model = BehavioralModel(
        behavioral_model="m2",
        params=None,
        feature_columns=["delta_E", "delta_A", "delta_S", "delta_D"],
        feature_info={"canonical_attributes": ["E", "A", "S", "D"]},
        lookup_table=lookup,
        selected_shrinkage_lambda=10.0,
    )
    df = pd.DataFrame(
        {
            "trial_id": ["t"],
            "family_id": ["f"],
            "config_id": ["c"],
            "block": ["B3"],
            "delta_E": [2],
            "delta_A": [1],
            "delta_S": [-1],
            "delta_D": [2],
            "successes": [1],
            "trials": [1],
        }
    )
    contrib = model.attribute_contributions(df)
    driver_a = model.revealed_driver(df, pd.Series(["A"], index=df.index))
    driver_b = model.revealed_driver(df, pd.Series(["B"], index=df.index))
    assert np.isfinite(model.linear_predictor(df, exclude_b1=False)).all()
    for attr in ("E", "A", "S", "D"):
        assert np.isfinite(contrib[f"contrib_{attr}"]).all()
        assert np.isclose(driver_a[f"influence_{attr}"].iloc[0], contrib[f"contrib_{attr}"].iloc[0])
        assert np.isclose(driver_b[f"influence_{attr}"].iloc[0], -contrib[f"contrib_{attr}"].iloc[0])


def test_ridge_selection_prefers_smallest_lambda_within_tolerance():
    selected = select_ridge_lambda({"0.0": 1.0, "1e-06": 1.0 + 1e-9, "0.0001": 1.1})
    assert selected == 0.0


def test_end_to_end_smoke_on_real_condition():
    out_root = output_root()
    theme = "drugs"
    model_tag = "mini_low"

    train_dataset = dataset_dir(theme, "train", base=out_root)
    test_dataset = dataset_dir(theme, "test", base=out_root)
    train_run = resolve_run_dir(run_prefix(theme, "train", model_tag, "actor", base=out_root))
    test_run = resolve_run_dir(run_prefix(theme, "test", model_tag, "tau", base=out_root))
    m0_summary_path = stagea_dir(theme, model_tag, base=out_root) / "stageA_summary.json"
    if not train_dataset.exists() or not test_dataset.exists() or train_run is None or test_run is None or not m0_summary_path.exists():
        raise AssertionError("Smoke-test assets are missing from outputs/final_same_order")

    train_trials = pd.read_parquet(train_dataset / "dataset_trials.parquet")
    test_trials = pd.read_parquet(test_dataset / "dataset_trials.parquet")
    train_responses = load_responses(train_run / "responses.jsonl")
    test_responses = load_responses(test_run / "responses.jsonl")

    train_choice_ok = train_responses[train_responses["choice_ok"]].groupby("trial_id").agg(successes=("choice", lambda s: (s == "A").sum()), trials=("choice", "size")).reset_index()
    train_stagea = train_trials.merge(train_choice_ok, on="trial_id", how="left").fillna({"successes": 0, "trials": 0})
    train_stagea["successes"] = train_stagea["successes"].astype(int)
    train_stagea["trials"] = train_stagea["trials"].astype(int)

    m1_model, _, m1_summary = fit_behavioral_model(train_stagea, behavioral_model="m1")
    m2_model, _, m2_summary = fit_behavioral_model(train_stagea, behavioral_model="m2")
    m0_model, m0_summary = load_behavioral_model_summary(str(m0_summary_path))

    m0_rows = build_test_row_predictions(
        trials_df=test_trials,
        responses_df=test_responses,
        model=m0_model,
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "low", "model_tag": model_tag},
    )
    m1_rows = build_test_row_predictions(
        trials_df=test_trials,
        responses_df=test_responses,
        model=m1_model,
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "low", "model_tag": model_tag},
    )
    m2_rows = build_test_row_predictions(
        trials_df=test_trials,
        responses_df=test_responses,
        model=m2_model,
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "low", "model_tag": model_tag},
    )
    per_draw = build_per_draw_driver_table(
        trials_df=test_trials,
        responses_df=test_responses,
        row_predictions={"m0": m0_rows, "m1": m1_rows, "m2": m2_rows},
        models={"m0": m0_model, "m1": m1_model, "m2": m2_model},
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "low", "model_tag": model_tag},
    )
    comparison = build_condition_comparison(
        theme=theme,
        row_predictions={"m0": m0_rows, "m1": m1_rows, "m2": m2_rows},
        per_draw_df=per_draw,
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "low", "model_tag": model_tag},
        model_summaries={"m0": m0_summary, "m1": m1_summary, "m2": m2_summary},
    )

    assert len(m1_rows) > 0
    assert len(m2_rows) > 0
    assert len(m2_model.lookup_table) == 625
    assert np.isfinite(m2_summary["selected_shrinkage_lambda"])
    assert np.isfinite(m1_rows["eta"]).all()
    assert np.isfinite(m1_rows["p_choose_A"]).all()
    assert np.isfinite(m2_rows["eta"]).all()
    assert np.isfinite(m2_rows["p_choose_A"]).all()
    assert "m0_heldout_nll" in comparison
    assert "m1_heldout_choice_accuracy" in comparison
    assert "m0_judge_choice_accuracy" in comparison
    assert "m0_judge_driver_matches_rate" in comparison
    assert "m2_heldout_nll" in comparison
    assert "m0_m2_driver_agreement" in comparison


def test_m0_robustness_metrics_match_main_paper_definitions_on_real_condition():
    out_root = output_root()
    theme = "drugs"
    model_tag = "mini_min"

    test_dataset = dataset_dir(theme, "test", base=out_root)
    test_run = resolve_run_dir(run_prefix(theme, "test", model_tag, "tau", base=out_root))
    m0_summary_path = stagea_dir(theme, model_tag, base=out_root) / "stageA_summary.json"
    if not test_dataset.exists() or test_run is None or not m0_summary_path.exists():
        raise AssertionError("Paper-compatibility smoke-test assets are missing from outputs/final_same_order")

    trials_df = pd.read_parquet(test_dataset / "dataset_trials.parquet")
    responses_df = load_responses(test_run / "responses.jsonl")
    m0_model, m0_summary = load_behavioral_model_summary(str(m0_summary_path))

    row_predictions = {
        "m0": build_test_row_predictions(
            trials_df=trials_df,
            responses_df=responses_df,
            model=m0_model,
            condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "minimal", "model_tag": model_tag},
        )
    }
    per_draw = build_per_draw_driver_table(
        trials_df=trials_df,
        responses_df=responses_df,
        row_predictions=row_predictions,
        models={"m0": m0_model},
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "minimal", "model_tag": model_tag},
    )
    comparison = build_condition_comparison(
        theme=theme,
        row_predictions=row_predictions,
        per_draw_df=per_draw,
        condition_keys={"theme": theme, "family": "GPT-5-mini", "effort": "minimal", "model_tag": model_tag},
        model_summaries={"m0": m0_summary},
    )

    eval_df, _, _ = build_balanced_eval_frame(
        dataset_dir=test_dataset,
        responses_path=test_run / "responses.jsonl",
        stagea_summary_path=m0_summary_path,
    )

    def _ok(values: pd.Series) -> pd.Series:
        return values.fillna(False).astype(bool)

    def _match(left: pd.Series, right: pd.Series) -> pd.Series:
        return (left == right).fillna(False).astype(bool)

    paper_choice = float((_ok(eval_df["choice_ok"]) & _match(eval_df["linear_model_pred_choice"], eval_df["choice"])).mean())
    paper_actor_attr = float((_ok(eval_df["premise_ok"]) & _match(eval_df["premise_attr"], eval_df["linear_model_factor"])).mean())
    paper_judge_choice = float((_ok(eval_df["tau_ok"]) & _match(eval_df["tau_pred_choice"], eval_df["choice"])).mean())
    paper_judge_attr = float((_ok(eval_df["tau_ok"]) & _match(eval_df["tau_driver"], eval_df["linear_model_factor"])).mean())

    assert np.isclose(comparison["m0_heldout_choice_accuracy"], paper_choice)
    assert np.isclose(comparison["m0_driver_matches_stated_factor_rate"], paper_actor_attr)
    assert np.isclose(comparison["m0_judge_choice_accuracy"], paper_judge_choice)
    assert np.isclose(comparison["m0_judge_driver_matches_rate"], paper_judge_attr)
