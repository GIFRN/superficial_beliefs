#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make import make_dataset_from_yaml
from src.data.themes import get_theme
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset for superficial beliefs study")
    parser.add_argument("--config", required=True, help="Path to dataset configuration YAML")
    parser.add_argument("--out", required=True, help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=None, help="Override global seed")
    parser.add_argument("--theme", default=None, help="Theme name (drugs, restaurants, candidates) or path to theme YAML")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    
    # Load theme if specified
    theme_config = None
    if args.theme:
        theme_config = get_theme(args.theme)
    
    result = make_dataset_from_yaml(args.config, seed=args.seed, output_dir=out_dir, theme_config=theme_config)
    
    # The MANIFEST is already written by make_dataset_from_yaml with theme info if provided


if __name__ == "__main__":
    main()
