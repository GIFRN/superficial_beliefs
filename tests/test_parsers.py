import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.harness import (
    classify_premise_open_text,
    parse_choice,
    parse_pairwise1,
    parse_pairwise6,
    parse_score1,
    parse_scores4,
    parse_structured_premise,
)


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


def test_parse_scores4_lines():
    text = "E=0.7 / A=0.2 / S=0.9 / D=0.1"
    result = parse_scores4(text)
    assert result["ok"]
    assert result["tau"]["E"] == 0.7
    assert result["tau"]["D"] == 0.1


def test_parse_score1():
    result = parse_score1("tau=0.73")
    assert result["ok"]
    assert result["tau"] == 0.73


def test_parse_pairwise6():
    text = "EA=E\nES=tie\nED=D\nAS=A\nAD=A\nSD=S"
    result = parse_pairwise6(text)
    assert result["ok"]
    assert result["pairs"]["EA"] == "E"
    assert result["pairs"]["ES"] == "tie"


def test_parse_pairwise1():
    result = parse_pairwise1("winner=A")
    assert result["ok"]
    assert result["winner"] == "A"
