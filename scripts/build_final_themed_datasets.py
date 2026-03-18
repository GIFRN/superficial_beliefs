#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_benchmark import ALL_THEMES, THEME_CONFIGS, datasets_root, logs_root
from src.data.apply_theme import transform_dataset_from_paths
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final themed train/test datasets from data/{train,test}.")
    parser.add_argument("--data-root", default="data", help="Root containing train/ and test/ source datasets.")
    parser.add_argument(
        "--out-root",
        default="outputs/final_same_order",
        help="Output root for the final benchmark.",
    )
    parser.add_argument(
        "--themes",
        default=",".join(ALL_THEMES),
        help="Comma-separated theme ids to build.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    selected_themes = [theme.strip() for theme in args.themes.split(",") if theme.strip()]

    built: list[dict[str, str]] = []
    for split in ["train", "test"]:
        source_dir = data_root / split
        if not source_dir.exists():
            raise SystemExit(f"Missing source split directory: {source_dir}")
        for theme in selected_themes:
            if theme not in THEME_CONFIGS:
                raise SystemExit(f"Unknown theme: {theme}")
            out_dir = datasets_root(out_root) / theme / split
            ensure_dir(out_dir)
            transform_dataset_from_paths(
                source_dir=source_dir,
                theme_name_or_path=str(THEME_CONFIGS[theme]),
                output_dir=out_dir,
                source_theme_name_or_path=str(THEME_CONFIGS["drugs"]),
                configs_filename="dataset_configs.parquet",
                trials_filename="full_trials.parquet",
            )
            built.append(
                {
                    "theme": theme,
                    "split": split,
                    "source": str(source_dir.relative_to(ROOT)),
                    "out_dir": str(out_dir.relative_to(ROOT)),
                }
            )
            print(f"built {out_dir}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.relative_to(ROOT)),
        "out_root": str(out_root.relative_to(ROOT)),
        "trials_source_filename": "full_trials.parquet",
        "configs_source_filename": "dataset_configs.parquet",
        "themes": selected_themes,
        "built": built,
    }
    manifest_path = logs_root(out_root) / "build_final_themed_datasets_manifest.json"
    ensure_dir(manifest_path.parent)
    write_json(manifest, manifest_path)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
