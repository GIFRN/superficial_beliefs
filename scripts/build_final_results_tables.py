#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_balanced import (
    build_balanced_eval_frame,
    compute_main_metrics,
    compute_placebo_metrics,
)
from src.analysis.final_benchmark import (
    MAIN_THEMES,
    MODEL_SPECS,
    PLACEBO_THEMES,
    THEME_CONFIGS,
    dataset_dir,
    output_root,
    reports_root,
    resolve_run_dir,
    run_prefix,
    stagea_dir,
)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _write_table(
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
    payload: dict[str, Any],
    out_prefix: Path,
    title: str,
    summary_columns: list[str],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# {title}",
        "",
        f"- Expected rows: {payload['expected_rows']}",
        f"- Actual rows: {payload['actual_rows']}",
        f"- Missing rows: {len(payload['missing_rows'])}",
        "",
        "## Summary",
        _markdown_table(rows, summary_columns),
    ]
    if payload["missing_rows"]:
        md_lines.extend(
            [
                "",
                "## Missing Rows",
                _markdown_table(payload["missing_rows"], ["theme", "family", "effort", "model_tag", "missing"]),
            ]
        )
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines))
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.json')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


def _placebo_variant_label(theme: str) -> str:
    payload = yaml.safe_load(THEME_CONFIGS[theme].read_text())
    return str(payload["attributes"]["D"]["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final same-order point-estimate result tables.")
    parser.add_argument("--out-root", default=str(output_root()))
    parser.add_argument(
        "--main-out-prefix",
        default=None,
        help="Optional output prefix for the main table. Defaults to <out-root>/reports/FINAL_MAIN_RESULTS",
    )
    parser.add_argument(
        "--placebo-out-prefix",
        default=None,
        help="Optional output prefix for the placebo table. Defaults to <out-root>/reports/FINAL_PLACEBO_RESULTS",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    out_root = output_root(args.out_root)
    main_out_prefix = Path(args.main_out_prefix).resolve() if args.main_out_prefix else reports_root(out_root) / "FINAL_MAIN_RESULTS"
    placebo_out_prefix = Path(args.placebo_out_prefix).resolve() if args.placebo_out_prefix else reports_root(out_root) / "FINAL_PLACEBO_RESULTS"

    main_rows: list[dict[str, Any]] = []
    main_missing: list[dict[str, str]] = []
    for theme in MAIN_THEMES:
        test_dataset = dataset_dir(theme, "test", base=out_root)
        for spec in MODEL_SPECS:
            stagea_summary_path = stagea_dir(theme, spec.tag, base=out_root) / "stageA_summary.json"
            run_dir = resolve_run_dir(run_prefix(theme, "test", spec.tag, "tau", base=out_root))
            responses_path = run_dir / "responses.jsonl" if run_dir else None
            missing = []
            if not test_dataset.exists():
                missing.append("dataset")
            if not stagea_summary_path.exists():
                missing.append("stageA")
            if responses_path is None or not responses_path.exists():
                missing.append("responses")
            if missing:
                main_missing.append(
                    {
                        "theme": theme,
                        "family": spec.family,
                        "effort": spec.effort,
                        "model_tag": spec.tag,
                        "missing": ",".join(missing),
                    }
                )
                continue
            eval_df, _, _ = build_balanced_eval_frame(
                dataset_dir=test_dataset,
                responses_path=responses_path,
                stagea_summary_path=stagea_summary_path,
            )
            metrics = compute_main_metrics(eval_df)
            main_rows.append(
                {
                    "theme": theme,
                    "family": spec.family,
                    "effort": spec.effort,
                    "model_tag": spec.tag,
                    **metrics,
                    "dataset_relpath": _rel(test_dataset),
                    "responses_relpath": _rel(responses_path),
                    "stageA_summary_relpath": _rel(stagea_summary_path),
                }
            )

    if main_missing and not args.allow_missing:
        desc = "; ".join(f"{row['theme']}:{row['model_tag']}[{row['missing']}]" for row in main_missing)
        raise SystemExit(f"Missing inputs for {len(main_missing)} main rows: {desc}")

    main_rows = sorted(main_rows, key=lambda row: (row["theme"], row["family"], row["effort"]))
    main_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": _rel(out_root),
        "expected_rows": len(MAIN_THEMES) * len(MODEL_SPECS),
        "actual_rows": len(main_rows),
        "missing_rows": main_missing,
        "rows": main_rows,
    }
    _write_table(
        rows=main_rows,
        columns=[
            "theme",
            "family",
            "effort",
            "linear_model_predicts_actor_choice",
            "judge_predicts_actor_choice",
            "linear_model_factor_matches_stated_factor",
            "judge_factor_matches_stated_factor",
            "n_choice_ok",
            "n_judge_choice_ok",
            "n_premise_ok",
            "n_judge_premise_ok",
            "model_tag",
            "dataset_relpath",
            "responses_relpath",
            "stageA_summary_relpath",
        ],
        payload=main_payload,
        out_prefix=main_out_prefix,
        title="Final Main Results",
        summary_columns=[
            "theme",
            "family",
            "effort",
            "linear_model_predicts_actor_choice",
            "judge_predicts_actor_choice",
            "linear_model_factor_matches_stated_factor",
            "judge_factor_matches_stated_factor",
            "n_choice_ok",
            "n_judge_choice_ok",
            "n_premise_ok",
            "n_judge_premise_ok",
        ],
    )

    placebo_rows: list[dict[str, Any]] = []
    placebo_missing: list[dict[str, str]] = []
    for theme in PLACEBO_THEMES:
        test_dataset = dataset_dir(theme, "test", base=out_root)
        placebo_variant = _placebo_variant_label(theme)
        for spec in MODEL_SPECS:
            stagea_summary_path = stagea_dir(theme, spec.tag, base=out_root) / "stageA_summary.json"
            run_dir = resolve_run_dir(run_prefix(theme, "test", spec.tag, "tau", base=out_root))
            responses_path = run_dir / "responses.jsonl" if run_dir else None
            missing = []
            if not test_dataset.exists():
                missing.append("dataset")
            if not stagea_summary_path.exists():
                missing.append("stageA")
            if responses_path is None or not responses_path.exists():
                missing.append("responses")
            if missing:
                placebo_missing.append(
                    {
                        "theme": theme,
                        "family": spec.family,
                        "effort": spec.effort,
                        "model_tag": spec.tag,
                        "missing": ",".join(missing),
                    }
                )
                continue
            eval_df, _, _ = build_balanced_eval_frame(
                dataset_dir=test_dataset,
                responses_path=responses_path,
                stagea_summary_path=stagea_summary_path,
            )
            metrics = compute_placebo_metrics(eval_df, placebo_attr="D")
            placebo_rows.append(
                {
                    "theme": theme,
                    "placebo_variant": placebo_variant,
                    "family": spec.family,
                    "effort": spec.effort,
                    "model_tag": spec.tag,
                    **metrics,
                    "dataset_relpath": _rel(test_dataset),
                    "responses_relpath": _rel(responses_path),
                    "stageA_summary_relpath": _rel(stagea_summary_path),
                }
            )

    if placebo_missing and not args.allow_missing:
        desc = "; ".join(f"{row['theme']}:{row['model_tag']}[{row['missing']}]" for row in placebo_missing)
        raise SystemExit(f"Missing inputs for {len(placebo_missing)} placebo rows: {desc}")

    placebo_rows = sorted(placebo_rows, key=lambda row: (row["theme"], row["family"], row["effort"]))
    placebo_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": _rel(out_root),
        "expected_rows": len(PLACEBO_THEMES) * len(MODEL_SPECS),
        "actual_rows": len(placebo_rows),
        "missing_rows": placebo_missing,
        "rows": placebo_rows,
    }
    _write_table(
        rows=placebo_rows,
        columns=[
            "theme",
            "placebo_variant",
            "family",
            "effort",
            "actor_states_placebo_as_key_factor",
            "linear_model_factor_is_placebo",
            "judge_factor_is_placebo",
            "n_premise_ok",
            "n_choice_ok",
            "n_tau_ok",
            "model_tag",
            "dataset_relpath",
            "responses_relpath",
            "stageA_summary_relpath",
        ],
        payload=placebo_payload,
        out_prefix=placebo_out_prefix,
        title="Final Placebo Results",
        summary_columns=[
            "placebo_variant",
            "family",
            "effort",
            "actor_states_placebo_as_key_factor",
            "linear_model_factor_is_placebo",
            "judge_factor_is_placebo",
            "n_premise_ok",
            "n_choice_ok",
            "n_tau_ok",
        ],
    )


if __name__ == "__main__":
    main()
