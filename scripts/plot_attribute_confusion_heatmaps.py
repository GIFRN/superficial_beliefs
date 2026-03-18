#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


THEMES = ["drugs", "policy", "software"]
ATTR_ORDER = ["E", "A", "S", "D"]
METHODS = {
    "linear": {
        "comparison": "actor_vs_linear",
        "label": "Choice model",
        "rate_key": "linear_matches_actor_rate",
    },
    "judge": {
        "comparison": "actor_vs_judge",
        "label": "Judge",
        "rate_key": "judge_matches_actor_rate",
    },
}
EXPECTED_TOP_OFFDIAG = {
    ("drugs", "linear"): ("E", "S"),
    ("drugs", "judge"): ("D", "E"),
    ("policy", "linear"): ("A", "E"),
    ("policy", "judge"): ("E", "S"),
    ("software", "linear"): ("E", "S"),
    ("software", "judge"): ("D", "S"),
}

THEME_CONFIGS = {
    "drugs": ROOT / "data/themes/drugs.yml",
    "policy": ROOT / "data/themes/policy.yml",
    "software": ROOT / "data/themes/software.yml",
}


def discover_first(paths: Iterable[Path], glob_pattern: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    matches = sorted(ROOT.rglob(glob_pattern))
    if not matches:
        raise SystemExit(f"Could not find {glob_pattern}")
    return matches[0]


def load_theme_labels() -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for theme, path in THEME_CONFIGS.items():
        conf = yaml.safe_load(path.read_text())
        payload[theme] = {code: conf["attributes"][code]["label"] for code in ATTR_ORDER}
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def build_matrices(
    conditionals: list[dict[str, str]],
    mismatches: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, str], list[list[float]]],
    dict[tuple[str, str], list[list[float]]],
    dict[tuple[str, str], float],
    dict[tuple[str, str], list[float]],
    list[str],
]:
    warnings: list[str] = []
    attr_idx = {code: idx for idx, code in enumerate(ATTR_ORDER)}

    mismatch_lookup: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    offdiag_by_row: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    for row in mismatches:
        theme = row["theme"]
        if theme not in THEMES:
            continue
        comparison = row["comparison"]
        family = row["family"]
        effort = row["effort"]
        from_attr = row["from_attr_code"]
        to_attr = row["to_attr_code"]
        count = float(row["count"])
        mismatch_lookup[(theme, comparison, family, effort, from_attr, to_attr)] += count
        offdiag_by_row[(theme, comparison, family, effort, from_attr)] += count

    count_mats = {
        (theme, method): [[0.0 for _ in ATTR_ORDER] for _ in ATTR_ORDER]
        for theme in THEMES
        for method in METHODS
    }
    panel_totals = {(theme, method): 0.0 for theme in THEMES for method in METHODS}

    for row in conditionals:
        theme = row["theme"]
        if theme not in THEMES:
            continue
        family = row["family"]
        effort = row["effort"]
        from_attr = row["attr_code"]
        i = attr_idx[from_attr]

        # Linear model panel
        linear_total = float(row["n_actor_attr"])
        linear_comp = METHODS["linear"]["comparison"]
        linear_offdiag = 0.0
        for to_attr in ATTR_ORDER:
            if to_attr == from_attr:
                continue
            count = mismatch_lookup[(theme, linear_comp, family, effort, from_attr, to_attr)]
            count_mats[(theme, "linear")][i][attr_idx[to_attr]] += count
            linear_offdiag += count
        linear_diag = max(0.0, linear_total - linear_offdiag)
        count_mats[(theme, "linear")][i][i] += linear_diag
        panel_totals[(theme, "linear")] += linear_total
        if linear_offdiag - linear_total > 1e-6:
            warnings.append(
                f"Linear row overfull for {theme}/{family}/{effort}/{from_attr}: offdiag={linear_offdiag:.3f} total={linear_total:.3f}"
            )

        # Judge panel
        judge_comp = METHODS["judge"]["comparison"]
        judge_rate = float(row["judge_matches_actor_rate"])
        judge_offdiag = offdiag_by_row[(theme, judge_comp, family, effort, from_attr)]
        if judge_rate >= 1.0 - 1e-9:
            judge_total = judge_offdiag if judge_offdiag > 0 else float(row["n_actor_attr"])
        else:
            judge_total = judge_offdiag / max(1e-9, (1.0 - judge_rate))
        for to_attr in ATTR_ORDER:
            if to_attr == from_attr:
                continue
            count = mismatch_lookup[(theme, judge_comp, family, effort, from_attr, to_attr)]
            count_mats[(theme, "judge")][i][attr_idx[to_attr]] += count
        judge_diag = max(0.0, judge_total - judge_offdiag)
        count_mats[(theme, "judge")][i][i] += judge_diag
        panel_totals[(theme, "judge")] += judge_total
        if judge_offdiag - judge_total > 1e-6:
            warnings.append(
                f"Judge row overfull for {theme}/{family}/{effort}/{from_attr}: offdiag={judge_offdiag:.3f} total={judge_total:.3f}"
            )

    norm_mats = {
        key: [[0.0 for _ in ATTR_ORDER] for _ in ATTR_ORDER]
        for key in count_mats
    }
    row_sums: dict[tuple[str, str], list[float]] = {key: [] for key in count_mats}
    for key, matrix in count_mats.items():
        for i, row in enumerate(matrix):
            row_sum = sum(row)
            if row_sum > 0:
                norm_mats[key][i] = [value / row_sum for value in row]
            row_sum_norm = sum(norm_mats[key][i])
            row_sums[key].append(row_sum_norm)
            if row_sum and abs(row_sum_norm - 1.0) > 1e-6:
                warnings.append(f"Row sum check failed for {key} row {ATTR_ORDER[i]}: {row_sum_norm:.6f}")

    return count_mats, norm_mats, panel_totals, row_sums, warnings


def largest_offdiag(matrix: list[list[float]]) -> tuple[str, str, float]:
    best = ("", "", -1.0)
    for i, from_attr in enumerate(ATTR_ORDER):
        for j, to_attr in enumerate(ATTR_ORDER):
            if i == j:
                continue
            value = matrix[i][j]
            if value > best[2]:
                best = (from_attr, to_attr, value)
    return best


def write_sidecar(
    path: Path,
    mismatch_path: Path,
    conditional_path: Path,
    panel_totals: dict[tuple[str, str], float],
    norm_mats: dict[tuple[str, str], list[list[float]]],
    row_sums: dict[tuple[str, str], list[float]],
    theme_labels: dict[str, dict[str, str]],
    warnings: list[str],
    expected_checks_enabled: bool,
) -> None:
    lines = [
        "# Attribute Confusion Heatmaps",
        "",
        "## Inputs",
        f"- mismatch file: `{mismatch_path}`",
        f"- conditional file: `{conditional_path}`",
        "",
        "## Normalization",
        "- Each panel is row-normalized so every actor-stated attribute row sums to 1.",
        "- Rows and columns follow canonical internal order `E, A, S, D`, displayed with theme-local labels.",
        "",
        "## Total N Per Theme",
    ]
    for theme in THEMES:
        lines.append(
            f"- {theme}: choice model N = {panel_totals[(theme, 'linear')]:.1f}; judge N = {panel_totals[(theme, 'judge')]:.1f}"
        )

    lines.extend(["", "## Largest Off-Diagonal Cells"])
    for theme in THEMES:
        for method in ["linear", "judge"]:
            from_attr, to_attr, value = largest_offdiag(norm_mats[(theme, method)])
            lines.append(
                f"- {theme.capitalize()} / {METHODS[method]['label']}: "
                f"{theme_labels[theme][from_attr]} -> {theme_labels[theme][to_attr]} = {value:.2f}"
            )

    lines.extend(["", "## Validation"])
    max_row_dev = max(abs(value - 1.0) for values in row_sums.values() for value in values)
    lines.append(f"- Max row-sum deviation after normalization: {max_row_dev:.6f}")
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
        lines.append("- Note: these off-diagonal warnings compare row-normalized panels against earlier raw-mismatch summaries, so shifts toward rarer but more diagnostic rows are possible.")
    elif not expected_checks_enabled:
        lines.append("- Qualitative expected-top-mismatch checks were disabled for this run; only normalization checks were enforced.")
    else:
        lines.append("- Expected top off-diagonal substitutions matched the qualitative checks.")

    path.write_text("\n".join(lines) + "\n")


def plot_heatmaps(
    out_pdf: Path,
    out_png: Path,
    norm_mats: dict[tuple[str, str], list[list[float]]],
    theme_labels: dict[str, dict[str, str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    def wrap_label(label: str) -> str:
        return label.replace(" ", "\n") if " " in label else label

    vmax = max(max(max(row) for row in matrix) for matrix in norm_mats.values())
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8))

    for col, theme in enumerate(THEMES):
        for row_idx, method in enumerate(["linear", "judge"]):
            ax = axes[row_idx, col]
            matrix = norm_mats[(theme, method)]
            im = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="Blues", aspect="equal")

            labels = [wrap_label(theme_labels[theme][code]) for code in ATTR_ORDER]
            ax.set_xticks(range(len(ATTR_ORDER)))
            ax.set_yticks(range(len(ATTR_ORDER)))
            ax.set_xticklabels(labels, rotation=18, ha="right", rotation_mode="anchor")
            ax.set_yticklabels(labels)
            if row_idx == 0:
                ax.set_title(theme.capitalize(), fontsize=13)
            ax.tick_params(axis="x", labelsize=8.5, pad=4)
            ax.tick_params(axis="y", labelsize=8.5, pad=2)

            for i in range(len(ATTR_ORDER)):
                for j in range(len(ATTR_ORDER)):
                    value = matrix[i][j]
                    text_color = "white" if value > vmax * 0.55 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

            for idx in range(len(ATTR_ORDER)):
                ax.add_patch(
                    patches.Rectangle(
                        (idx - 0.5, idx - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#444444",
                        linewidth=1.2,
                    )
                )
            ax.set_xticks([x - 0.5 for x in range(1, len(ATTR_ORDER))], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, len(ATTR_ORDER))], minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.19, top=0.90, wspace=0.34, hspace=0.22)
    fig.text(0.035, 0.68, "Choice model", rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.035, 0.30, "Judge", rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    cax = fig.add_axes([0.90, 0.17, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized proportion", fontsize=11)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main attribute confusion heatmap figure.")
    parser.add_argument(
        "--mismatches",
        default="",
        help="Path to ATTRIBUTE_LEVEL mismatch CSV. If empty, discover automatically.",
    )
    parser.add_argument(
        "--conditionals",
        default="",
        help="Path to ATTRIBUTE_LEVEL conditionals CSV. If empty, discover automatically.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Optional output prefix. If set, write <prefix>.md/.pdf/.png instead of the default filenames in --reports-dir.",
    )
    parser.add_argument(
        "--disable-expected-checks",
        action="store_true",
        help="Skip the qualitative expected top off-diagonal checks.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    preferred_mismatch = ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_RESULTS_20260315_MISMATCHES.csv"
    preferred_conditionals = ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_RESULTS_20260315_CONDITIONALS.csv"
    mismatch_path = discover_first(
        ([Path(args.mismatches)] if args.mismatches else []) + [preferred_mismatch],
        "*ATTRIBUTE_LEVEL*_MISMATCHES*.csv",
    )
    conditional_path = discover_first(
        ([Path(args.conditionals)] if args.conditionals else []) + [preferred_conditionals],
        "*ATTRIBUTE_LEVEL*_CONDITIONALS*.csv",
    )

    mismatches = load_csv(mismatch_path)
    conditionals = load_csv(conditional_path)
    theme_labels = load_theme_labels()
    count_mats, norm_mats, panel_totals, row_sums, warnings = build_matrices(conditionals, mismatches)

    if not args.disable_expected_checks:
        for theme in THEMES:
            for method in ["linear", "judge"]:
                from_attr, to_attr, value = largest_offdiag(norm_mats[(theme, method)])
                expected = EXPECTED_TOP_OFFDIAG[(theme, method)]
                if (from_attr, to_attr) != expected:
                    warnings.append(
                        f"Top off-diagonal mismatch for {theme}/{method}: "
                        f"expected {expected[0]}->{expected[1]}, got {from_attr}->{to_attr} ({value:.3f})"
                    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        note_path = out_prefix.with_suffix(".md")
        out_pdf = out_prefix.with_suffix(".pdf")
        out_png = out_prefix.with_suffix(".png")
    else:
        note_path = reports_dir / "attribute_confusion_heatmaps_main.md"
        out_pdf = reports_dir / "attribute_confusion_heatmaps_main.pdf"
        out_png = reports_dir / "attribute_confusion_heatmaps_main.png"
    write_sidecar(
        note_path,
        mismatch_path,
        conditional_path,
        panel_totals,
        norm_mats,
        row_sums,
        theme_labels,
        warnings,
        expected_checks_enabled=not args.disable_expected_checks,
    )
    print(f"wrote {note_path}")
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")

    try:
        plot_heatmaps(out_pdf, out_png, norm_mats, theme_labels)
    except ModuleNotFoundError as exc:
        print(
            "matplotlib is not available in this environment, so the PDF/PNG figure could not be rendered. "
            "The script and sidecar note were still generated."
        )
        print(f"Missing module: {exc}")
        return

    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
