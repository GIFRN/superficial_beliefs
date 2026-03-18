#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _weight_order(weights: dict[str, float]) -> str:
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return ">".join(attr for attr, _ in ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper-facing occlusion suite bundle from the completed drugs-only run.")
    parser.add_argument("--out-root", default=str(ROOT / "outputs/occlusion_suite_drugs_mini_min"))
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stagea = json.loads((out_root / "results/stage_A/stageA_summary.json").read_text())
    stageb = json.loads((out_root / "results/stage_B/stageB_summary.json").read_text())
    judge = json.loads((out_root / "reports/judge_baselines/judge_baselines_summary.json").read_text())
    current = json.loads((out_root / "reports/occlusion_suite_current_run/methodc_current_run_analysis.json").read_text())
    dataset_manifest = json.loads((ROOT / "data/occlusion_suite/test/MANIFEST.json").read_text())

    baseline_weights = current["baseline_only_stageA"]["weights"]
    baseline_order = current["baseline_weight_order"]
    row = {
        "model_family": "GPT-5-mini",
        "effort": "minimal",
        "judge": "tau",
        "full_stageA_eval_acc": stagea["evaluation"]["accuracy"],
        "full_stageA_cv_acc": stagea["cv"]["accuracy"],
        "stageB_driver": stageb["alignment"]["ECRB_top1_driver"],
        "stageB_weights": stageb["alignment"]["ECRB_top1_weights"],
        "stageB_rank_corr": stageb["alignment"]["rank_corr"],
        "judge_tau_vs_choice": judge["agreement"]["tau_vs_choice"],
        "judge_tau_driver_vs_premise": judge["alignment"]["tau_driver_vs_premise"],
        "tau_ok_rate": judge["tau_ok_rate"],
        "baseline_only_stageA_acc": current["baseline_only_stageA"]["evaluation"]["accuracy"],
        "baseline_weight_order": ">".join(baseline_order),
        "occlude_drop_effect_order": ">".join(current["intervention_rankings"]["occlude_drop"]["rank_order"]),
        "occlude_equalize_effect_order": ">".join(current["intervention_rankings"]["occlude_equalize"]["rank_order"]),
    }
    for attr in ["E", "D", "S", "A"]:
        row[f"baseline_weight_{attr}"] = baseline_weights[attr]
        row[f"drop_directional_{attr}"] = current["directional_summaries"]["occlude_drop"]["by_attribute"][attr]["delta_favored_mean"]
        row[f"equalize_directional_{attr}"] = current["directional_summaries"]["occlude_equalize"]["by_attribute"][attr]["delta_favored_mean"]
        row[f"drop_choice_flip_{attr}"] = current["directional_summaries"]["occlude_drop"]["by_attribute"][attr]["choice_flip_rate"]
        row[f"equalize_choice_flip_{attr}"] = current["directional_summaries"]["occlude_equalize"]["by_attribute"][attr]["choice_flip_rate"]
        row[f"drop_premise_flip_{attr}"] = current["directional_summaries"]["occlude_drop"]["by_attribute"][attr]["premise_flip_rate"]
        row[f"equalize_premise_flip_{attr}"] = current["directional_summaries"]["occlude_equalize"]["by_attribute"][attr]["premise_flip_rate"]

    csv_path = reports_dir / "OCCLUSION_SUITE_RESULTS.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    objective = "5-year overall patient outcome"
    summary_lines = [
        "# Occlusion Suite Results Summary",
        "",
        "This file is the single paper-facing summary for the completed drugs occlusion suite.",
        "",
        "## Scope",
        "",
        "- model family: `GPT-5-mini`",
        "- effort: `minimal`",
        "- judge: `tau`",
        "",
        "## Dataset Structure And Construction",
        "",
        f"The analyzed occlusion-suite dataset is `data/occlusion_suite/themes/drugs/test`.",
        "",
        "It was constructed from the final same-order full test split, then expanded into matched intervention families.",
        "",
        f"- objective: `{objective}`",
        f"- total trials: `{dataset_manifest['n_trials']}`",
        f"- matched base trials: `{dataset_manifest['n_base_trials']}`",
        f"- base configurations: `{dataset_manifest['n_configs']}`",
        "- block composition: `B3` only",
        "",
        "Each matched family contains `9` rows:",
        "",
        "- `1` baseline `short_reason` row",
        "- `4` `occlude_equalize` rows, one per attribute",
        "- `4` `occlude_drop` rows, one per attribute",
        "",
        "The intervention target counts are balanced across attributes, and the released suite does not include `occlude_swap`.",
        "",
        "## Headline Results",
        "",
        f"- baseline-only Stage A accuracy: `{row['baseline_only_stageA_acc']:.3f}`",
        f"- full Stage A evaluation accuracy: `{row['full_stageA_eval_acc']:.3f}`",
        f"- Stage B top-driver alignment: `{row['stageB_driver']:.3f}`",
        f"- Stage B weight alignment: `{row['stageB_weights']:.3f}`",
        f"- Stage B rank correlation: `{row['stageB_rank_corr']:.3f}`",
        f"- judge choice agreement: `{row['judge_tau_vs_choice']:.3f}`",
        f"- judge driver-vs-premise alignment: `{row['judge_tau_driver_vs_premise']:.3f}`",
        "",
        "## Intervention Pattern",
        "",
        f"- baseline ranking: `{row['baseline_weight_order'].replace('>', ' > ')}`",
        f"- `occlude_drop` ranking: `{row['occlude_drop_effect_order'].replace('>', ' > ')}`",
        f"- `occlude_equalize` ranking: `{row['occlude_equalize_effect_order'].replace('>', ' > ')}`",
        "",
        "Choice-flip rates:",
        "",
        f"- `occlude_drop`: `E {row['drop_choice_flip_E']:.3f}`, `D {row['drop_choice_flip_D']:.3f}`, `S {row['drop_choice_flip_S']:.3f}`, `A {row['drop_choice_flip_A']:.3f}`",
        f"- `occlude_equalize`: `E {row['equalize_choice_flip_E']:.3f}`, `D {row['equalize_choice_flip_D']:.3f}`, `S {row['equalize_choice_flip_S']:.3f}`, `A {row['equalize_choice_flip_A']:.3f}`",
        "",
        "Premise-flip rates:",
        "",
        f"- `occlude_drop`: `E {row['drop_premise_flip_E']:.3f}`, `D {row['drop_premise_flip_D']:.3f}`, `S {row['drop_premise_flip_S']:.3f}`, `A {row['drop_premise_flip_A']:.3f}`",
        f"- `occlude_equalize`: `E {row['equalize_premise_flip_E']:.3f}`, `D {row['equalize_premise_flip_D']:.3f}`, `S {row['equalize_premise_flip_S']:.3f}`, `A {row['equalize_premise_flip_A']:.3f}`",
        "",
        "## File",
        "",
        f"- `{csv_path.relative_to(ROOT)}`",
    ]
    (reports_dir / "OCCLUSION_SUITE_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    _run(
        [
            sys.executable,
            "scripts/plot_methodc_results.py",
            "--input",
            str(csv_path),
            "--out-prefix",
            str(reports_dir / "OCCLUSION_SUITE_RANKING"),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/plot_methodc_results.py",
            "--input",
            str(csv_path),
            "--out-prefix",
            str(reports_dir / "OCCLUSION_SUITE_RANKING_NOTITLE"),
            "--no-header",
        ]
    )

    print(f"wrote {csv_path}")
    print(f"wrote {reports_dir / 'OCCLUSION_SUITE_RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
