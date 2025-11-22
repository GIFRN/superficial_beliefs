import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import Profile
from src.llm.prompts import conversation_plan
from src.llm.types import TrialSpec


def make_trial(variant: str, manipulation: str) -> TrialSpec:
    profile_levels = {"E": "High", "A": "Medium", "S": "Low", "D": "Medium"}
    profile_a = Profile(profile_levels)
    profile_b = Profile({"E": "Medium", "A": "Medium", "S": "Medium", "D": "Low"})
    order = ("E", "S", "D", "A")
    return TrialSpec(
        trial_id="T1",
        config_id="C1",
        block="B1",
        profile_a=profile_a,
        profile_b=profile_b,
        order_a=order,
        order_b=order,
        paraphrase_id=0,
        manipulation=manipulation,
        attribute_target=None,
        inject_offset=0,
        variant=variant,
        seed=0,
        metadata={},
    )


def test_short_reason_plan():
    plan = conversation_plan(make_trial("short_reason", "short_reason"))
    assert plan.system_prompt
    assert len(plan.steps) == 3
    assert plan.steps[0].name == "choice"
    assert plan.steps[1].expects == "sentence"
    assert not plan.steps[1].reset_context


def test_premise_first_plan():
    plan = conversation_plan(make_trial("premise_first", "premise_first"))
    assert plan.steps[0].name == "premise"


def test_split_reason_plan():
    plan = conversation_plan(make_trial("split_reason", "split_reason"))
    assert plan.steps[0].name == "choice"
    assert plan.steps[1].reset_context
