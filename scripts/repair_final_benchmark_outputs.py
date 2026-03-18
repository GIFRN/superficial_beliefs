#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.themes import theme_from_dict
from src.llm.harness import parse_step_response
from src.llm.types import ConversationStep


@dataclass
class RunRepairStats:
    run_dir: Path
    total_trials: int = 0
    kept_trials: int = 0
    dropped_trials: int = 0
    choice_repairs: int = 0
    judge_repairs: int = 0


BRACKET_PAIR_RE = re.compile(r"=\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")


def _reparse_choice_step(step: dict[str, Any], theme_config) -> tuple[dict[str, Any], bool]:
    parsed = parse_step_response(
        ConversationStep(name="choice", prompt="", expects="choice_attr"),
        step.get("content", ""),
        theme_config,
    )
    changed = parsed != step.get("parsed", {})
    return parsed, changed


def _reparse_judge_step(step: dict[str, Any], theme_config, prompt: str = "") -> tuple[dict[str, Any], bool]:
    parsed = parse_step_response(
        ConversationStep(name="judge_scores_joint", prompt=prompt, expects="scores4"),
        step.get("content", ""),
        theme_config,
    )
    changed = parsed != step.get("parsed", {})
    return parsed, changed


def _step_prompts(response: dict[str, Any]) -> list[str]:
    conversation = response.get("conversation") or []
    return [str(msg.get("content", "")) for msg in conversation if msg.get("role") == "user"]


def _trial_is_valid(record: dict[str, Any], *, drop_judge_bracket_pairs: bool) -> bool:
    for response in record.get("responses", []):
        for step in response.get("steps", []):
            name = step.get("name")
            parsed = step.get("parsed", {})
            if name == "choice":
                if not parsed.get("choice_ok", False):
                    return False
                if not parsed.get("premise_ok", False):
                    return False
            elif name == "judge_scores_joint":
                if not parsed.get("ok", False):
                    return False
                if drop_judge_bracket_pairs and BRACKET_PAIR_RE.search(step.get("content", "")):
                    return False
    return True


def _repair_run_dir(run_dir: Path, *, dry_run: bool, drop_judge_bracket_pairs: bool) -> RunRepairStats:
    responses_path = run_dir / "responses.jsonl"
    manifest_path = run_dir / "MANIFEST.json"
    if not responses_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing responses/manifest in {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    theme_config = theme_from_dict(manifest["theme"]) if manifest.get("theme") else None
    stats = RunRepairStats(run_dir=run_dir)

    repaired_records: list[dict[str, Any]] = []
    dropped_trial_ids: list[str] = []

    with responses_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            stats.total_trials += 1
            for response in record.get("responses", []):
                prompts = _step_prompts(response)
                for step_idx, step in enumerate(response.get("steps", [])):
                    if step.get("name") == "choice":
                        parsed, changed = _reparse_choice_step(step, theme_config)
                        if changed:
                            stats.choice_repairs += 1
                            step["parsed"] = parsed
                    elif step.get("name") == "judge_scores_joint":
                        prompt = prompts[step_idx] if step_idx < len(prompts) else ""
                        parsed, changed = _reparse_judge_step(step, theme_config, prompt=prompt)
                        if changed:
                            stats.judge_repairs += 1
                            step["parsed"] = parsed
            if _trial_is_valid(record, drop_judge_bracket_pairs=drop_judge_bracket_pairs):
                repaired_records.append(record)
                stats.kept_trials += 1
            else:
                dropped_trial_ids.append(str(record.get("trial_id")))
                stats.dropped_trials += 1

    repair_summary = {
        "run_dir": str(run_dir),
        "total_trials_seen": stats.total_trials,
        "kept_trials": stats.kept_trials,
        "dropped_trials": stats.dropped_trials,
        "choice_repairs": stats.choice_repairs,
        "judge_repairs": stats.judge_repairs,
        "dropped_trial_ids": dropped_trial_ids,
    }

    summary_path = run_dir / "REPAIR_SUMMARY.json"
    if dry_run:
        print(json.dumps(repair_summary, indent=2))
        return stats

    backup_path = run_dir / "responses.pre_repair.jsonl"
    if not backup_path.exists():
        shutil.copy2(responses_path, backup_path)

    tmp_path = run_dir / "responses.jsonl.tmp"
    with tmp_path.open("w", encoding="utf-8") as out_fh:
        for record in repaired_records:
            out_fh.write(json.dumps(record))
            out_fh.write("\n")
    tmp_path.replace(responses_path)

    summary_path.write_text(json.dumps(repair_summary, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparse and repair final benchmark outputs in place.")
    parser.add_argument("--out-root", default="outputs/final_same_order")
    parser.add_argument("--run-dir", action="append", default=None, help="Specific run dir(s) to repair")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--drop-judge-bracket-pairs",
        action="store_true",
        help="Drop any judge trial record containing raw bracket-pair forms like label=[x,y].",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(path) for path in args.run_dir]
    else:
        run_dirs = sorted((Path(args.out_root) / "runs").rglob("*"))
        run_dirs = [path for path in run_dirs if path.is_dir() and (path / "responses.jsonl").exists() and (path / "MANIFEST.json").exists()]

    total_trials = kept_trials = dropped_trials = 0
    repaired_runs = 0
    for run_dir in run_dirs:
        stats = _repair_run_dir(
            run_dir,
            dry_run=args.dry_run,
            drop_judge_bracket_pairs=args.drop_judge_bracket_pairs,
        )
        total_trials += stats.total_trials
        kept_trials += stats.kept_trials
        dropped_trials += stats.dropped_trials
        repaired_runs += 1
        if stats.dropped_trials:
            print(
                f"{run_dir}\tkept={stats.kept_trials}\tdropped={stats.dropped_trials}"
                f"\tchoice_repairs={stats.choice_repairs}\tjudge_repairs={stats.judge_repairs}"
            )

    print(
        f"repaired_runs={repaired_runs} total_trials={total_trials} "
        f"kept_trials={kept_trials} dropped_trials={dropped_trials}"
    )


if __name__ == "__main__":
    main()
