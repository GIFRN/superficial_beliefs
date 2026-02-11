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
                tau_values = {"E": None, "A": None, "S": None, "D": None}
                tau_ok = False
                tau_missing: list[str] = []
                pair_values = {"EA": None, "ES": None, "ED": None, "AS": None, "AD": None, "SD": None}
                pairwise_ok = False
                pairwise_missing: list[str] = []
                record = {
                    "trial_id": payload.get("trial_id"),
                    "config_id": payload.get("config_id"),
                    "block": payload.get("block"),
                    "manipulation": payload.get("manipulation"),
                    "variant": payload.get("variant"),
                    "seed": response.get("seed"),
                }
                steps_list = response.get("steps", [])
                steps = {step["name"]: step for step in steps_list}
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

                for step in steps_list:
                    name = step.get("name", "")
                    parsed = step.get("parsed", {})
                    if "tau" in parsed:
                        tau_ok = tau_ok or bool(parsed.get("ok", False))
                        tau_payload = parsed.get("tau")
                        if isinstance(tau_payload, dict):
                            for key, value in tau_payload.items():
                                if key in tau_values:
                                    tau_values[key] = value
                            for missing in parsed.get("missing", []):
                                if missing not in tau_missing:
                                    tau_missing.append(missing)
                        else:
                            match = None
                            if name.startswith("judge_score_") and len(name) >= len("judge_score_E"):
                                match = name.split("judge_score_", 1)[-1]
                            if match in tau_values:
                                tau_values[match] = tau_payload
                    if "pairs" in parsed:
                        pairwise_ok = pairwise_ok or bool(parsed.get("ok", False))
                        pairs = parsed.get("pairs", {})
                        if isinstance(pairs, dict):
                            for key, value in pairs.items():
                                if key in pair_values:
                                    pair_values[key] = value
                        for missing in parsed.get("missing", []):
                            if missing not in pairwise_missing:
                                pairwise_missing.append(missing)
                    if "winner" in parsed and name.startswith("judge_pair_"):
                        pairwise_ok = pairwise_ok or bool(parsed.get("ok", False))
                        pair = name.split("judge_pair_", 1)[-1]
                        if pair in pair_values:
                            pair_values[pair] = parsed.get("winner")

                for attr, value in tau_values.items():
                    record[f"tau_{attr}"] = value
                record["tau_ok"] = tau_ok
                record["tau_missing"] = tau_missing
                for pair, value in pair_values.items():
                    record[f"pair_{pair}"] = value
                record["pairwise_ok"] = pairwise_ok
                record["pairwise_missing"] = pairwise_missing
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
