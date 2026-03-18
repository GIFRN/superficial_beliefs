#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


THEMES = ["drugs", "policy", "software"]
ATTR_ORDER = ["E", "A", "S", "D"]
LINEAR_COLOR = "#4b78a8"
JUDGE_COLOR = "#c26d2d"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#222222"

THEME_CONFIGS = {
    "drugs": ROOT / "data/themes/drugs.yml",
    "policy": ROOT / "data/themes/policy.yml",
    "software": ROOT / "data/themes/software.yml",
}


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


def load_theme_meta() -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for theme, path in THEME_CONFIGS.items():
        payload = yaml.safe_load(path.read_text())
        meta[theme] = {
            "objective": payload["objective"],
            **{code: payload["attributes"][code]["label"] for code in ATTR_ORDER},
        }
    return meta


def aggregate_rows(csv_path: Path) -> dict[str, list[dict[str, float | str | int]]]:
    aggregates: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "linear_match_num": 0.0, "judge_match_num": 0.0}
    )
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            theme = row["theme"]
            if theme not in THEMES:
                continue
            attr_code = row["attr_code"]
            n = float(row["n_actor_attr"])
            aggregates[(theme, attr_code)]["n"] += n
            aggregates[(theme, attr_code)]["linear_match_num"] += n * float(row["linear_matches_actor_rate"])
            aggregates[(theme, attr_code)]["judge_match_num"] += n * float(row["judge_matches_actor_rate"])

    results: dict[str, list[dict[str, float | str | int]]] = {}
    for theme in THEMES:
        rows = []
        for attr_code in ATTR_ORDER:
            agg = aggregates[(theme, attr_code)]
            n = agg["n"]
            rows.append(
                {
                    "attr_code": attr_code,
                    "n": int(n),
                    "linear_match_rate": agg["linear_match_num"] / n if n else 0.0,
                    "judge_match_rate": agg["judge_match_num"] / n if n else 0.0,
                }
            )
        results[theme] = rows
    return results


def write_svg(
    out_path: Path,
    theme_meta: dict[str, dict[str, str]],
    aggregated: dict[str, list[dict[str, float | str | int]]],
    include_header: bool,
) -> None:
    width = 1480
    height = 520 if include_header else 435
    top = 82 if include_header else 30
    panel_top = top + 28
    panel_height = 300
    gap = 38
    left = 62
    panel_w = 430
    legend_y = height - 10

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    if include_header:
        parts.extend(
            [
                _svg_text(width / 2, 34, "Attribute recovery depends strongly on which attribute the actor states", size=22, anchor="middle", weight="bold"),
                _svg_text(width / 2, 58, "Weighted across 5 model families and 2 effort settings on the final short-reason test split", size=13, anchor="middle"),
            ]
        )

    for idx, theme in enumerate(THEMES):
        panel_left = left + idx * (panel_w + gap)
        axis_left = panel_left + 42
        axis_right = panel_left + panel_w - 12
        axis_top = panel_top
        axis_bottom = panel_top + panel_height
        plot_w = axis_right - axis_left
        plot_h = axis_bottom - axis_top
        center = panel_left + panel_w / 2

        title = theme.capitalize()
        parts.append(_svg_text(panel_left, top, title, size=15, weight="bold"))

        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        for tick in ticks:
            y = axis_bottom - tick * plot_h
            parts.append(_svg_line(axis_left, y, axis_right, y))
            parts.append(_svg_text(axis_left - 8, y + 4, f"{tick:.2f}", size=10, anchor="end"))

        parts.append(_svg_line(axis_left, axis_top, axis_left, axis_bottom, color="#777777"))
        parts.append(_svg_line(axis_left, axis_bottom, axis_right, axis_bottom, color="#777777"))
        parts.append(
            f'<text x="{panel_left + 6:.1f}" y="{panel_top + panel_height / 2:.1f}" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="{TEXT_COLOR}" '
            f'transform="rotate(-90 {panel_left + 6:.1f},{panel_top + panel_height / 2:.1f})">Conditional match rate</text>'
        )

        centers = [axis_left + plot_w * frac for frac in [0.14, 0.39, 0.64, 0.89]]
        bar_w = 24
        for x_center, row in zip(centers, aggregated[theme]):
            attr_code = str(row["attr_code"])
            linear = float(row["linear_match_rate"])
            judge = float(row["judge_match_rate"])
            n = int(row["n"])
            label = str(theme_meta[theme][attr_code])

            linear_h = linear * plot_h
            judge_h = judge * plot_h
            x1 = x_center - bar_w - 3
            x2 = x_center + 3
            y1 = axis_bottom - linear_h
            y2 = axis_bottom - judge_h

            parts.append(_svg_rect(x1, y1, bar_w, linear_h, LINEAR_COLOR))
            parts.append(_svg_rect(x2, y2, bar_w, judge_h, JUDGE_COLOR))
            parts.append(_svg_text(x1 + bar_w / 2, y1 - 6, f"{linear:.2f}", size=10, anchor="middle"))
            parts.append(_svg_text(x2 + bar_w / 2, y2 - 6, f"{judge:.2f}", size=10, anchor="middle"))
            parts.append(_svg_text(x_center, axis_bottom + 24, label, size=10, anchor="middle"))
            parts.append(_svg_text(x_center, axis_bottom + 40, f"(n={n})", size=9, anchor="middle"))

    legend_x = width / 2 - 115
    parts.append(_svg_rect(legend_x, legend_y - 12, 16, 12, LINEAR_COLOR))
    parts.append(_svg_text(legend_x + 22, legend_y - 1, "Linear model", size=11))
    parts.append(_svg_rect(legend_x + 118, legend_y - 12, 16, 12, JUDGE_COLOR))
    parts.append(_svg_text(legend_x + 140, legend_y - 1, "Judge", size=11))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot conditional attribute match rates by theme.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_RESULTS_20260315_CONDITIONALS.csv"),
        help="Attribute-level conditional CSV.",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(ROOT / "artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_MATCH_RATES_20260316"),
        help="Output prefix for the SVG figure.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress the title and subtitle.",
    )
    args = parser.parse_args()

    aggregated = aggregate_rows(Path(args.input))
    theme_meta = load_theme_meta()
    out_path = Path(args.out_prefix).with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_svg(out_path, theme_meta, aggregated, include_header=not args.no_header)


if __name__ == "__main__":
    main()
