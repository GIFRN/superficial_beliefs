#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ATTR_CODES = ["E", "D", "S", "A"]
DROP_COLOR = "#c26d2d"
EQUALIZE_COLOR = "#5d89b3"
BASELINE_COLOR = "#888888"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#222222"


def load_methodc_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly 1 Method C row in {csv_path}, found {len(rows)}")
    return rows[0]


def load_drugs_theme_meta() -> tuple[str, dict[str, str]]:
    theme_path = ROOT / "data/themes/drugs.yml"
    payload = yaml.safe_load(theme_path.read_text())
    return payload["objective"], {code: payload["attributes"][code]["label"] for code in ["E", "A", "S", "D"]}


def values_for(row: dict[str, str], prefix: str, attrs: list[str]) -> list[float]:
    return [float(row[f"{prefix}_{attr}"]) for attr in attrs]


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" fill="{TEXT_COLOR}">'
        f"{escape(text)}</text>"
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, color: str = GRID_COLOR, width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def _svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )


def write_svg_figure(
    out_path: Path,
    model_family: str,
    effort: str,
    objective: str,
    baseline_order: list[str],
    attr_labels: dict[str, str],
    baseline_weights: list[float],
    drop_choice: list[float],
    equalize_choice: list[float],
    drop_premise: list[float],
    equalize_premise: list[float],
    include_header: bool,
) -> None:
    width = 1450
    height = 520 if include_header else 430
    top = 82 if include_header else 28
    bottom = 88
    panel_height = 320
    panel_top = top + 28
    left_a = 80
    gap = 45
    panel_a_w = 280
    panel_bc_w = 430
    left_b = left_a + panel_a_w + gap
    left_c = left_b + panel_bc_w + gap
    y_max = min(0.7, math.ceil((max(drop_choice + equalize_choice + drop_premise + equalize_premise) + 0.05) * 20) / 20)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]
    if include_header:
        parts.extend(
            [
                _svg_text(width / 2, 34, "Method C: targeted attribute occlusions recover the baseline revealed-preference ranking", size=22, anchor="middle", weight="bold"),
                _svg_text(width / 2, 58, f"{model_family} ({effort} effort); drugs objective: {objective}; baseline order = {' > '.join(baseline_order)}", size=13, anchor="middle"),
            ]
        )

    # Panel A
    parts.append(_svg_text(left_a, top, "A. Baseline revealed ranking", size=15, weight="bold"))
    max_weight = max(baseline_weights)
    a_bar_left = left_a + 110
    a_bar_right = left_a + panel_a_w - 40
    a_scale = (a_bar_right - a_bar_left) / max_weight
    row_h = panel_height / len(baseline_order)
    for i, (code, weight) in enumerate(zip(baseline_order, baseline_weights)):
        y = panel_top + i * row_h + 10
        parts.append(_svg_text(left_a, y + 18, attr_labels[code], size=12))
        parts.append(_svg_rect(a_bar_left, y, weight * a_scale, 24, BASELINE_COLOR))
        parts.append(_svg_text(a_bar_left + weight * a_scale + 8, y + 18, f"{weight:.2f}", size=11))
    parts.append(_svg_text(left_a + panel_a_w / 2, panel_top + panel_height + 34, "Baseline weight", size=12, anchor="middle"))

    # Panels B and C helpers
    def draw_grouped_panel(left: float, title: str, ylabel: str, vals_1: list[float], vals_2: list[float]) -> None:
        axis_left = left + 40
        axis_right = left + panel_bc_w - 18
        axis_bottom = panel_top + panel_height
        axis_top = panel_top
        plot_w = axis_right - axis_left
        plot_h = axis_bottom - axis_top
        parts.append(_svg_text(left, top, title, size=15, weight="bold"))
        # grid + ticks
        ticks = [0.0, 0.2, 0.4, 0.6]
        if y_max > 0.6:
            ticks.append(y_max)
        ticks = sorted(set(ticks))
        for tick in ticks:
            y = axis_bottom - (tick / y_max) * plot_h
            parts.append(_svg_line(axis_left, y, axis_right, y))
            parts.append(_svg_text(axis_left - 10, y + 4, f"{tick:.1f}", size=10, anchor="end"))
        parts.append(_svg_line(axis_left, axis_top, axis_left, axis_bottom, color="#777777"))
        parts.append(_svg_line(axis_left, axis_bottom, axis_right, axis_bottom, color="#777777"))
        # axis label
        parts.append(
            f'<text x="{left + 4:.1f}" y="{panel_top + panel_height / 2:.1f}" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="{TEXT_COLOR}" '
            f'transform="rotate(-90 {left + 4:.1f},{panel_top + panel_height / 2:.1f})">{escape(ylabel)}</text>'
        )
        # bars
        centers = np.linspace(axis_left + 58, axis_right - 50, len(baseline_order))
        bar_w = 24
        for center, code, v1, v2 in zip(centers, baseline_order, vals_1, vals_2):
            h1 = (v1 / y_max) * plot_h
            h2 = (v2 / y_max) * plot_h
            x1 = center - bar_w - 3
            x2 = center + 3
            y1 = axis_bottom - h1
            y2 = axis_bottom - h2
            parts.append(_svg_rect(x1, y1, bar_w, h1, DROP_COLOR))
            parts.append(_svg_rect(x2, y2, bar_w, h2, EQUALIZE_COLOR))
            parts.append(_svg_text(x1 + bar_w / 2, y1 - 6, f"{v1:.2f}", size=10, anchor="middle"))
            parts.append(_svg_text(x2 + bar_w / 2, y2 - 6, f"{v2:.2f}", size=10, anchor="middle"))
            parts.append(_svg_text(center, axis_bottom + 26, attr_labels[code], size=11, anchor="middle"))

    draw_grouped_panel(left_b, "B. Choice disruption under targeted occlusion", "Choice-flip rate", drop_choice, equalize_choice)
    draw_grouped_panel(left_c, "C. Premise disruption under targeted occlusion", "Premise-flip rate", drop_premise, equalize_premise)

    # shared legend
    legend_y = height - 8
    legend_x = width / 2 - 95
    parts.append(_svg_rect(legend_x, legend_y - 12, 16, 12, DROP_COLOR))
    parts.append(_svg_text(legend_x + 22, legend_y - 1, "Drop", size=11))
    parts.append(_svg_rect(legend_x + 88, legend_y - 12, 16, 12, EQUALIZE_COLOR))
    parts.append(_svg_text(legend_x + 110, legend_y - 1, "Equalize", size=11))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the paper-facing Method C three-panel figure.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/METHODC_RESULTS_20260315.csv"),
        help="Single-row Method C CSV.",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/METHODC_OCCLUSION_RANKING_20260316"),
        help="Output prefix for figure files.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress the title and subtitle.",
    )
    args = parser.parse_args()

    row = load_methodc_row(Path(args.input))
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    objective, attr_labels = load_drugs_theme_meta()

    baseline_order = row["baseline_weight_order"].split(">")
    if sorted(baseline_order) != sorted(ATTR_CODES):
        raise SystemExit(f"Unexpected baseline weight order: {row['baseline_weight_order']}")

    x_attr_codes = baseline_order
    x_attr_labels = [attr_labels[code] for code in x_attr_codes]

    baseline_weights = values_for(row, "baseline_weight", x_attr_codes)
    drop_choice = values_for(row, "drop_choice_flip", x_attr_codes)
    equalize_choice = values_for(row, "equalize_choice_flip", x_attr_codes)
    drop_premise = values_for(row, "drop_premise_flip", x_attr_codes)
    equalize_premise = values_for(row, "equalize_premise_flip", x_attr_codes)
    svg_path = out_prefix.with_suffix(".svg")
    write_svg_figure(
        out_path=svg_path,
        model_family=row["model_family"],
        effort=row["effort"],
        objective=objective,
        baseline_order=x_attr_codes,
        attr_labels=attr_labels,
        baseline_weights=baseline_weights,
        drop_choice=drop_choice,
        equalize_choice=equalize_choice,
        drop_premise=drop_premise,
        equalize_premise=equalize_premise,
        include_header=not args.no_header,
    )


if __name__ == "__main__":
    main()
