import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.harness import classify_premise_open_text, parse_choice, parse_structured_premise


def test_parse_choice_strict():
    result = parse_choice("A")
    assert result["ok"] and result["choice"] == "A"


def test_parse_choice_invalid():
    result = parse_choice("Option A")
    assert not result["ok"]


def test_parse_choice_with_prefix():
    text = "Drug B. Higher efficacy predicts better outcomes."
    result = parse_choice(text)
    assert result["ok"] and result["choice"] == "B"


def test_parse_structured_premise():
    text = 'PremiseAttribute = Safety\nPremiseText = "fewer adverse events"'
    result = parse_structured_premise(text)
    assert result["ok"]
    assert result["attr"] == "S"
    assert result["text"] == "fewer adverse events"


def test_classify_premise_open_text():
    result = classify_premise_open_text("The better adherence keeps patients on therapy")
    assert result["attr"] == "A"
