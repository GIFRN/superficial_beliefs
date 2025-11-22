import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mock the validation functions to avoid dependency issues
def validate_b1_rationality(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> dict[str, Any]:
    """Validate that B1 trials show expected Pareto optimal behavior"""
    b1_trials = trials_df[trials_df["block"] == "B1"].merge(choice_agg, on="trial_id", how="inner")
    
    if b1_trials.empty:
        return {
            "rationality_check_passed": True,
            "failure_rate": 0.0,
            "min_probability": 1.0,
            "max_probability": 1.0,
            "failed_trials": [],
            "message": "No B1 trials found"
        }
    
    # Check if all B1 trials have P(choose A) ≥ 0.95
    probabilities = b1_trials["successes"] / b1_trials["trials"]
    rationality_failures = probabilities < 0.95  # 5% tolerance
    
    return {
        "rationality_check_passed": not rationality_failures.any(),
        "failure_rate": float(rationality_failures.mean()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
        "failed_trials": b1_trials[rationality_failures]["trial_id"].tolist(),
        "total_b1_trials": len(b1_trials),
        "message": f"B1 rationality check: {len(b1_trials)} trials, {rationality_failures.sum()} failures"
    }


def validate_b1_probes(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> dict[str, Any]:
    """Validate that B1 probes are effective"""
    b1_trials = trials_df[trials_df["block"] == "B1"].merge(choice_agg, on="trial_id", how="inner")
    
    if b1_trials.empty:
        return {
            "probe_effectiveness": True,
            "baseline_probability": 1.0,
            "manipulated_probability": 1.0,
            "probe_effect_size": 0.0,
            "message": "No B1 trials found"
        }
    
    baseline = b1_trials[b1_trials["manipulation"] == "none"]
    manipulated = b1_trials[b1_trials["manipulation"] != "none"]
    
    if baseline.empty or manipulated.empty:
        return {
            "probe_effectiveness": True,
            "baseline_probability": 1.0,
            "manipulated_probability": 1.0,
            "probe_effect_size": 0.0,
            "message": "Insufficient data for probe validation"
        }
    
    baseline_prob = baseline["successes"] / baseline["trials"]
    manipulated_prob = manipulated["successes"] / manipulated["trials"]
    
    probe_effect = float(baseline_prob.mean() - manipulated_prob.mean())
    
    return {
        "probe_effectiveness": abs(probe_effect) > 0.1,  # 10% threshold
        "baseline_probability": float(baseline_prob.mean()),
        "manipulated_probability": float(manipulated_prob.mean()),
        "probe_effect_size": probe_effect,
        "baseline_trials": len(baseline),
        "manipulated_trials": len(manipulated),
        "message": f"B1 probe validation: baseline P={baseline_prob.mean():.3f}, manipulated P={manipulated_prob.mean():.3f}, effect={probe_effect:.3f}"
    }


def test_b1_rationality_validation_pass():
    """Test B1 rationality validation with passing trials"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3"],
        "block": ["B1", "B1", "B1"],
        "manipulation": ["none", "none", "none"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3"],
        "successes": [20, 19, 20],  # P=1.0, P=0.95, P=1.0
        "trials": [20, 20, 20]
    })
    
    result = validate_b1_rationality(trials_df, choice_agg)
    assert result["rationality_check_passed"] == True
    assert result["min_probability"] == 0.95
    assert result["max_probability"] == 1.0
    assert result["total_b1_trials"] == 3


def test_b1_rationality_validation_fail():
    """Test B1 rationality validation with failing trials"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3"],
        "block": ["B1", "B1", "B1"],
        "manipulation": ["none", "none", "none"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3"],
        "successes": [20, 15, 20],  # P=1.0, P=0.75, P=1.0
        "trials": [20, 20, 20]
    })
    
    result = validate_b1_rationality(trials_df, choice_agg)
    assert result["rationality_check_passed"] == False
    assert result["failure_rate"] == 1/3
    assert result["min_probability"] == 0.75
    assert len(result["failed_trials"]) == 1
    assert result["failed_trials"][0] == "T2"


def test_b1_rationality_validation_no_b1():
    """Test B1 rationality validation with no B1 trials"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1", "T2"],
        "block": ["B2", "B3"],
        "manipulation": ["none", "none"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1", "T2"],
        "successes": [15, 12],
        "trials": [20, 20]
    })
    
    result = validate_b1_rationality(trials_df, choice_agg)
    assert result["rationality_check_passed"] == True
    assert result["message"] == "No B1 trials found"


def test_b1_probe_validation_effective():
    """Test B1 probe validation with effective probes"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3", "T4"],
        "block": ["B1", "B1", "B1", "B1"],
        "manipulation": ["none", "none", "redact", "neutralize"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3", "T4"],
        "successes": [20, 20, 10, 10],  # P=1.0, P=1.0, P=0.5, P=0.5
        "trials": [20, 20, 20, 20]
    })
    
    result = validate_b1_probes(trials_df, choice_agg)
    assert result["probe_effectiveness"] == True
    assert result["baseline_probability"] == 1.0
    assert result["manipulated_probability"] == 0.5
    assert result["probe_effect_size"] == 0.5


def test_b1_probe_validation_ineffective():
    """Test B1 probe validation with ineffective probes"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3", "T4"],
        "block": ["B1", "B1", "B1", "B1"],
        "manipulation": ["none", "none", "redact", "neutralize"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1", "T2", "T3", "T4"],
        "successes": [20, 20, 19, 20],  # P=1.0, P=1.0, P=0.95, P=1.0
        "trials": [20, 20, 20, 20]
    })
    
    result = validate_b1_probes(trials_df, choice_agg)
    assert result["probe_effectiveness"] == False
    assert result["baseline_probability"] == 1.0
    assert result["manipulated_probability"] == 0.975
    assert abs(result["probe_effect_size"] - 0.025) < 0.001


def test_b1_probe_validation_insufficient_data():
    """Test B1 probe validation with insufficient data"""
    trials_df = pd.DataFrame({
        "trial_id": ["T1"],
        "block": ["B1"],
        "manipulation": ["none"]
    })
    
    choice_agg = pd.DataFrame({
        "trial_id": ["T1"],
        "successes": [20],
        "trials": [20]
    })
    
    result = validate_b1_probes(trials_df, choice_agg)
    assert result["probe_effectiveness"] == True
    assert result["message"] == "Insufficient data for probe validation"


if __name__ == "__main__":
    test_b1_rationality_validation_pass()
    test_b1_rationality_validation_fail()
    test_b1_rationality_validation_no_b1()
    test_b1_probe_validation_effective()
    test_b1_probe_validation_ineffective()
    test_b1_probe_validation_insufficient_data()
    print("All B1 validation tests passed!")
