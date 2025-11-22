from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_responses(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            for response in payload.get("responses", []):
                record = {
                    "trial_id": payload.get("trial_id"),
                    "config_id": payload.get("config_id"),
                    "block": payload.get("block"),
                    "manipulation": payload.get("manipulation"),
                    "variant": payload.get("variant"),
                    "seed": response.get("seed"),
                }
                steps = {step["name"]: step for step in response.get("steps", [])}
                choice_step = steps.get("choice")
                premise_step = steps.get("premise")
                sentence_step = steps.get("sentence")
                if choice_step:
                    parsed = choice_step.get("parsed", {})
                    record["choice_ok"] = bool(parsed.get("ok"))
                    record["choice"] = parsed.get("choice")
                    record["choice_raw"] = choice_step.get("content", "")
                else:
                    record["choice_ok"] = False
                    record["choice"] = None
                    record["choice_raw"] = ""
                if premise_step:
                    parsed = premise_step.get("parsed", {})
                    record["premise_ok"] = bool(parsed.get("ok"))
                    record["premise_attr"] = parsed.get("attr")
                    record["premise_text"] = parsed.get("text")
                    record["premise_raw"] = premise_step.get("content", "")
                else:
                    record["premise_ok"] = False
                    record["premise_attr"] = None
                    record["premise_text"] = ""
                    record["premise_raw"] = ""
                if sentence_step:
                    parsed = sentence_step.get("parsed", {})
                    record["sentence_ok"] = bool(parsed.get("ok", True))
                    record["sentence_text"] = parsed.get("text", sentence_step.get("content", ""))
                else:
                    record["sentence_ok"] = False
                    record["sentence_text"] = ""
                records.append(record)
    return pd.DataFrame(records)


def aggregate_choices(responses_df: pd.DataFrame) -> pd.DataFrame:
    valid = responses_df[responses_df["choice_ok"]].copy()
    valid["is_A"] = valid["choice"].eq("A")
    grouped = (
        valid.groupby("trial_id")
        .agg(
            successes=("is_A", "sum"),
            trials=("is_A", "count"),
        )
        .reset_index()
    )
    return grouped


def aggregate_premises(responses_df: pd.DataFrame) -> pd.DataFrame:
    valid = responses_df[responses_df["premise_ok"]].copy()
    grouped = (
        valid.groupby(["trial_id", "premise_attr"])
        .size()
        .reset_index(name="count")
    )
    return grouped


def prepare_stageA_data(trials_df: pd.DataFrame, choice_agg: pd.DataFrame) -> pd.DataFrame:
    merged = trials_df.merge(choice_agg, on="trial_id", how="left")
    merged["successes"] = merged["successes"].fillna(0).astype(int)
    merged["trials"] = merged["trials"].fillna(0).astype(int)
    return merged
