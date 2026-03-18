#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_index_by_trial: dict[str, int] = {}
    for idx, record in enumerate(records):
        trial_id = str(record.get("trial_id"))
        last_index_by_trial[trial_id] = idx
    keep_indices = sorted(last_index_by_trial.values())
    return [records[idx] for idx in keep_indices]


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    tmp_path.replace(path)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate responses.jsonl by trial_id, keeping the last occurrence.")
    parser.add_argument("--run-dir", action="append", default=None, help="Run directory containing responses.jsonl")
    parser.add_argument("--out-root", default=None, help="Benchmark output root containing runs/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    if args.out_root:
        runs_root = Path(args.out_root) / "runs"
        run_dirs.extend(
            sorted(
                path.parent
                for path in runs_root.rglob("responses.jsonl")
                if (path.parent / "MANIFEST.json").exists()
            )
        )
    if args.run_dir:
        run_dirs.extend(Path(raw_run_dir) for raw_run_dir in args.run_dir)
    if not run_dirs:
        raise SystemExit("Provide --run-dir and/or --out-root")

    seen: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        responses_path = run_dir / "responses.jsonl"
        manifest_path = run_dir / "MANIFEST.json"
        if not responses_path.exists():
            raise SystemExit(f"Missing {responses_path}")

        records = _load_records(responses_path)
        deduped = _dedupe_records(records)
        before = len(records)
        after = len(deduped)
        removed = before - after
        print(f"{run_dir}\tbefore={before}\tafter={after}\tremoved={removed}")
        if args.dry_run or removed <= 0:
            continue

        backup_path = run_dir / "responses.pre_dedupe.jsonl"
        if not backup_path.exists():
            shutil.copy2(responses_path, backup_path)
        _rewrite(responses_path, deduped)

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["n_trials_completed_total"] = _count_lines(responses_path)
            manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
