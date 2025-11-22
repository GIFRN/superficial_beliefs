import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.stageA import build_design_matrix, per_trial_contributions


def test_build_design_matrix_columns():
    df = pd.DataFrame(
        {
            "trial_id": ["t1", "t2"],
            "config_id": ["c1", "c2"],
            "delta_E": [2, -2],
            "delta_A": [0, 0],
            "delta_S": [1, -1],
            "delta_D": [0, 0],
            "delta_pos_E": [0, 0],
            "delta_pos_A": [0, 0],
            "delta_pos_S": [0, 0],
            "delta_pos_D": [0, 0],
            "successes": [10, 2],
            "trials": [10, 10],
        }
    )
    design = build_design_matrix(df)
    assert "diff_E" in design.X.columns
    assert design.weights.sum() == 20


def test_per_trial_contributions_main_effect():
    df = pd.DataFrame(
        {
            "delta_E": [2],
            "delta_A": [0],
            "delta_S": [0],
            "delta_D": [0],
        },
        index=["t1"],
    )
    dummy_model = type(
        "Dummy",
        (),
        {
            "params": {"diff_E": 0.5, "diff_A": 0.0, "diff_S": 0.0, "diff_D": 0.0},
            "feature_info": {"main": {"E": "diff_E", "A": "diff_A", "S": "diff_S", "D": "diff_D"}, "interactions": {}},
        },
    )()
    contrib = per_trial_contributions(df, dummy_model)
    assert np.isclose(contrib.loc["t1", "C_E"], 1.0)
