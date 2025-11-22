from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import Attribute
from .themes import ThemeConfig, DRUGS_THEME
from ..utils.io import ensure_dir, write_json


def transform_dataset(
    source_dir: str | Path,
    theme_config: ThemeConfig,
    output_dir: str | Path,
    source_theme: ThemeConfig | None = None
) -> dict[str, Any]:
    """
    Transform a dataset from one theme to another.
    
    Args:
        source_dir: Directory containing source dataset (with dataset_configs.parquet, dataset_trials.parquet)
        theme_config: Target theme configuration
        output_dir: Directory to save transformed dataset
        source_theme: Source theme (defaults to DRUGS_THEME if None)
    
    Returns:
        Dictionary with transformation results and paths
    """
    source_path = Path(source_dir)
    output_path = ensure_dir(output_dir)
    
    if source_theme is None:
        source_theme = DRUGS_THEME
    
    # Load source data
    configs_df = pd.read_parquet(source_path / "dataset_configs.parquet")
    trials_df = pd.read_parquet(source_path / "dataset_trials.parquet")
    
    # Get attribute mapping
    source_attrs = source_theme.get_mapped_attributes()
    target_attrs = theme_config.get_mapped_attributes()
    
    # Identify which attributes to keep (intersection)
    attrs_to_keep = [attr for attr in source_attrs if attr in target_attrs]
    attrs_to_drop = [attr for attr in source_attrs if attr not in target_attrs]
    
    if not attrs_to_keep:
        raise ValueError("No common attributes between source and target themes")
    
    # Transform configs
    transformed_configs = _transform_configs(configs_df, attrs_to_keep, attrs_to_drop)
    
    # Transform trials
    transformed_trials = _transform_trials(
        trials_df, 
        source_attrs, 
        target_attrs,
        attrs_to_keep,
        attrs_to_drop
    )
    
    # Save transformed data
    configs_path = output_path / "dataset_configs.parquet"
    trials_path = output_path / "dataset_trials.parquet"
    
    transformed_configs.to_parquet(configs_path, index=False)
    transformed_trials.to_parquet(trials_path, index=False)
    
    # Create MANIFEST with theme metadata
    manifest = {
        "n_configs": len(transformed_configs),
        "n_trials": len(transformed_trials),
        "target_total": len(transformed_trials),
        "actual_total": len(transformed_trials),
        "blocks": transformed_trials["block"].value_counts().to_dict(),
        "theme": theme_config.to_dict(),
        "source_theme": source_theme.to_dict(),
        "source_dataset": str(source_path),
    }
    
    write_json(manifest, output_path / "MANIFEST.json")
    
    return {
        "configs": transformed_configs,
        "trials": transformed_trials,
        "configs_path": str(configs_path),
        "trials_path": str(trials_path),
        "manifest_path": str(output_path / "MANIFEST.json"),
        "attrs_kept": attrs_to_keep,
        "attrs_dropped": attrs_to_drop,
    }


def _transform_configs(
    configs_df: pd.DataFrame,
    attrs_to_keep: list[Attribute],
    attrs_to_drop: list[Attribute]
) -> pd.DataFrame:
    """
    Transform the configs dataframe by filtering attributes.
    
    Args:
        configs_df: Source configs dataframe
        attrs_to_keep: Attributes to keep
        attrs_to_drop: Attributes to drop
    
    Returns:
        Transformed configs dataframe
    """
    df = configs_df.copy()
    
    # Transform the levels_left and levels_right JSON columns
    def filter_levels(levels_json: str) -> str:
        levels = json.loads(levels_json)
        # Keep only the attributes we want
        filtered = {k: v for k, v in levels.items() if k in attrs_to_keep}
        return json.dumps(filtered)
    
    df["levels_left"] = df["levels_left"].apply(filter_levels)
    df["levels_right"] = df["levels_right"].apply(filter_levels)
    
    return df


def _transform_trials(
    trials_df: pd.DataFrame,
    source_attrs: list[Attribute],
    target_attrs: list[Attribute],
    attrs_to_keep: list[Attribute],
    attrs_to_drop: list[Attribute]
) -> pd.DataFrame:
    """
    Transform the trials dataframe by renaming and filtering columns.
    
    Args:
        trials_df: Source trials dataframe
        source_attrs: Source theme attributes
        target_attrs: Target theme attributes
        attrs_to_keep: Attributes to keep
        attrs_to_drop: Attributes to drop
    
    Returns:
        Transformed trials dataframe
    """
    df = trials_df.copy()
    
    # Build list of columns to drop
    cols_to_drop = []
    
    for attr in attrs_to_drop:
        # Drop delta, position, and delta_pos columns for dropped attributes
        cols_to_drop.extend([
            f"delta_{attr}",
            f"posA_{attr}",
            f"posB_{attr}",
            f"delta_pos_{attr}",
        ])
    
    # Remove columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


def transform_dataset_from_paths(
    source_dir: str | Path,
    theme_name_or_path: str,
    output_dir: str | Path,
    source_theme_name_or_path: str | None = None
) -> dict[str, Any]:
    """
    Transform a dataset using theme names or paths.
    
    Args:
        source_dir: Directory containing source dataset
        theme_name_or_path: Target theme name (builtin) or path to YAML
        output_dir: Directory to save transformed dataset
        source_theme_name_or_path: Source theme name or path (defaults to "drugs")
    
    Returns:
        Dictionary with transformation results
    """
    from .themes import get_theme
    
    target_theme = get_theme(theme_name_or_path)
    
    if source_theme_name_or_path:
        source_theme = get_theme(source_theme_name_or_path)
    else:
        source_theme = DRUGS_THEME
    
    return transform_dataset(source_dir, target_theme, output_dir, source_theme)

