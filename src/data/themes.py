from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import yaml

from .schema import Attribute, ATTR_LABELS


@dataclass(frozen=True)
class AttributeMapping:
    """Maps a source attribute to a themed attribute."""
    name: str  # Short name for the attribute (e.g., "Q" for Quality)
    label: str  # Full label for display (e.g., "Food Quality")


@dataclass(frozen=True)
class ThemeConfig:
    """Configuration for a dataset theme."""
    name: str
    entities: tuple[str, str]  # Exactly 2 entities (e.g., "Drug A", "Drug B")
    objective: str  # The objective being optimized
    attributes: dict[Attribute, AttributeMapping]  # Mapping from source attrs to themed attrs

    def __post_init__(self):
        """Validate the theme configuration."""
        if len(self.entities) != 2:
            raise ValueError(f"Theme must have exactly 2 entities, got {len(self.entities)}")
        
        if not self.name:
            raise ValueError("Theme name cannot be empty")
        
        if not self.objective:
            raise ValueError("Theme objective cannot be empty")
        
        if not self.attributes:
            raise ValueError("Theme must have at least one attribute mapping")
        
        # Validate that all attribute keys are valid
        valid_attrs = set(ATTR_LABELS.keys())
        for attr in self.attributes.keys():
            if attr not in valid_attrs:
                raise ValueError(f"Invalid attribute key: {attr}. Must be one of {valid_attrs}")

    @property
    def entity_a(self) -> str:
        return self.entities[0]
    
    @property
    def entity_b(self) -> str:
        return self.entities[1]
    
    def get_attribute_label(self, attr: Attribute) -> str:
        """Get the themed label for an attribute."""
        if attr in self.attributes:
            return self.attributes[attr].label
        # Fall back to original label if not mapped
        return ATTR_LABELS.get(attr, attr)
    
    def get_attribute_name(self, attr: Attribute) -> str:
        """Get the themed short name for an attribute."""
        if attr in self.attributes:
            return self.attributes[attr].name
        # Fall back to original attribute key if not mapped
        return attr
    
    def get_mapped_attributes(self) -> list[Attribute]:
        """Get list of source attributes that are mapped in this theme."""
        return list(self.attributes.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "entities": list(self.entities),
            "objective": self.objective,
            "attributes": {
                attr: {"name": mapping.name, "label": mapping.label}
                for attr, mapping in self.attributes.items()
            }
        }


def load_theme_from_yaml(path: str | Path) -> ThemeConfig:
    """Load a theme configuration from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return theme_from_dict(data)


def theme_from_dict(data: dict[str, Any]) -> ThemeConfig:
    """Create a ThemeConfig from a dictionary."""
    name = data.get("name", "")
    entities = tuple(data.get("entities", []))
    objective = data.get("objective", "")
    
    attributes_data = data.get("attributes", {})
    attributes = {}
    for attr_key, attr_info in attributes_data.items():
        if not isinstance(attr_info, dict):
            raise ValueError(f"Attribute {attr_key} must be a dict with 'name' and 'label'")
        attributes[attr_key] = AttributeMapping(
            name=attr_info.get("name", ""),
            label=attr_info.get("label", "")
        )
    
    return ThemeConfig(
        name=name,
        entities=entities,
        objective=objective,
        attributes=attributes
    )


# Built-in theme definitions
DRUGS_THEME = ThemeConfig(
    name="drugs",
    entities=("Drug A", "Drug B"),
    objective="5-year overall patient outcome",
    attributes={
        "E": AttributeMapping(name="E", label="Efficacy"),
        "A": AttributeMapping(name="A", label="Adherence"),
        "S": AttributeMapping(name="S", label="Safety"),
        "D": AttributeMapping(name="D", label="Durability"),
    }
)

RESTAURANTS_THEME = ThemeConfig(
    name="restaurants",
    entities=("Restaurant A", "Restaurant B"),
    objective="overall customer satisfaction",
    attributes={
        "E": AttributeMapping(name="Q", label="Food Quality"),
        "A": AttributeMapping(name="V", label="Value for Money"),
        "S": AttributeMapping(name="S", label="Service"),
        "D": AttributeMapping(name="A", label="Ambiance"),
    }
)

CANDIDATES_THEME = ThemeConfig(
    name="candidates",
    entities=("Candidate A", "Candidate B"),
    objective="hiring decision for long-term success",
    attributes={
        "E": AttributeMapping(name="X", label="Experience"),
        "A": AttributeMapping(name="F", label="Culture Fit"),
        "S": AttributeMapping(name="S", label="Technical Skills"),
        "D": AttributeMapping(name="C", label="Communication"),
    }
)

BUILTIN_THEMES: dict[str, ThemeConfig] = {
    "drugs": DRUGS_THEME,
    "restaurants": RESTAURANTS_THEME,
    "candidates": CANDIDATES_THEME,
}


def get_theme(theme_name_or_path: str) -> ThemeConfig:
    """
    Get a theme by name or path.
    
    Args:
        theme_name_or_path: Either a built-in theme name or path to a YAML file
    
    Returns:
        ThemeConfig object
    """
    # Check if it's a built-in theme
    if theme_name_or_path in BUILTIN_THEMES:
        return BUILTIN_THEMES[theme_name_or_path]
    
    # Otherwise, try to load from file
    path = Path(theme_name_or_path)
    if path.exists():
        return load_theme_from_yaml(path)
    
    raise ValueError(f"Theme not found: {theme_name_or_path}. "
                     f"Available built-in themes: {list(BUILTIN_THEMES.keys())}")

