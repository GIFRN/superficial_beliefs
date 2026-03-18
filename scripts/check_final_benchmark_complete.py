#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict completeness check for the final benchmark.")
    parser.add_argument("--out-root", default="outputs/final_same_order")
    args = parser.parse_args()

    runs_root = Path(args.out_root) / "runs"
    bad: list[tuple[str, int, int, str | None]] = []
    total = 0
    for manifest_path in sorted(runs_root.rglob("MANIFEST.json")):
        manifest = json.loads(manifest_path.read_text())
        total += 1
        run_dir = manifest_path.parent
        responses_path = run_dir / "responses.jsonl"
        actual = _count_lines(responses_path)
        expected = int(manifest.get("n_trials_total", 0))
        status = manifest.get("status")
        if actual != expected:
            bad.append((str(run_dir), actual, expected, status))

    print(f"runs_checked={total}")
    print(f"incomplete_runs={len(bad)}")
    for run_dir, actual, expected, status in bad:
        print(f"{run_dir}\tactual={actual}\texpected={expected}\tstatus={status}")

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
