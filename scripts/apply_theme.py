#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.apply_theme import transform_dataset_from_paths
from src.data.themes import get_theme, ThemeConfig, AttributeMapping, theme_from_dict


def parse_attribute_mapping(mapping_str: str) -> tuple[str, str, str]:
    """
    Parse attribute mapping string in format 'E:Quality:Food Quality'.
    
    Returns:
        Tuple of (source_attr, short_name, label)
    """
    parts = mapping_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid attribute mapping format: {mapping_str}. "
                        f"Expected format: 'E:Quality:Food Quality'")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a theme to an existing dataset, transforming entities, objectives, and attributes"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source dataset directory (containing dataset_configs.parquet and dataset_trials.parquet)"
    )
    parser.add_argument(
        "--theme",
        help="Theme name (drugs, restaurants, candidates) or path to theme YAML file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for transformed dataset"
    )
    parser.add_argument(
        "--source-theme",
        default=None,
        help="Source theme name or path (default: drugs)"
    )
    
    # Override options
    parser.add_argument(
        "--entity-a",
        help="Override entity A name (e.g., 'Restaurant A')"
    )
    parser.add_argument(
        "--entity-b",
        help="Override entity B name (e.g., 'Restaurant B')"
    )
    parser.add_argument(
        "--objective",
        help="Override objective text (e.g., 'overall customer satisfaction')"
    )
    parser.add_argument(
        "--map-attr",
        action="append",
        help="Map attribute in format 'E:Quality:Food Quality' (can be used multiple times)"
    )
    parser.add_argument(
        "--name",
        help="Theme name for the output (default: derived from theme)"
    )
    
    args = parser.parse_args()
    
    # Start with base theme if provided
    if args.theme:
        theme = get_theme(args.theme)
        theme_dict = theme.to_dict()
    else:
        # Create a minimal theme if only overrides provided
        if not (args.entity_a or args.entity_b or args.objective or args.map_attr):
            parser.error("Must provide either --theme or at least one override option")
        theme_dict = {
            "name": args.name or "custom",
            "entities": ["Entity A", "Entity B"],
            "objective": "the outcome",
            "attributes": {}
        }
    
    # Apply overrides
    if args.entity_a or args.entity_b:
        entities = list(theme_dict["entities"])
        if args.entity_a:
            entities[0] = args.entity_a
        if args.entity_b:
            entities[1] = args.entity_b
        theme_dict["entities"] = entities
    
    if args.objective:
        theme_dict["objective"] = args.objective
    
    if args.map_attr:
        # Parse attribute mappings
        attributes = {}
        for mapping_str in args.map_attr:
            source_attr, short_name, label = parse_attribute_mapping(mapping_str)
            attributes[source_attr] = {"name": short_name, "label": label}
        theme_dict["attributes"] = attributes
    
    if args.name:
        theme_dict["name"] = args.name
    
    # Create final theme config
    final_theme = theme_from_dict(theme_dict)
    
    # Transform the dataset
    print(f"Transforming dataset from {args.source}")
    print(f"Using theme: {final_theme.name}")
    print(f"  Entities: {final_theme.entity_a} and {final_theme.entity_b}")
    print(f"  Objective: {final_theme.objective}")
    print(f"  Attributes: {len(final_theme.attributes)}")
    for attr, mapping in final_theme.attributes.items():
        print(f"    {attr} -> {mapping.label} ({mapping.name})")
    
    # Use direct transformation with our final theme
    from src.data.apply_theme import transform_dataset
    from src.data.themes import DRUGS_THEME
    
    source_theme_obj = get_theme(args.source_theme) if args.source_theme else DRUGS_THEME
    
    result = transform_dataset(
        source_dir=args.source,
        theme_config=final_theme,
        output_dir=args.output,
        source_theme=source_theme_obj
    )
    
    print(f"\n✓ Transformation complete!")
    print(f"  Configs: {len(result['configs'])}")
    print(f"  Trials: {len(result['trials'])}")
    print(f"  Attributes kept: {result['attrs_kept']}")
    if result['attrs_dropped']:
        print(f"  Attributes dropped: {result['attrs_dropped']}")
    print(f"  Output: {args.output}")
    print(f"  Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()

