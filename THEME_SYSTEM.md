# Dataset Theme System

This document describes the dataset theming system that allows you to create variations of datasets with different entities, objectives, and attributes while preserving the underlying attribute strength configurations.

## Overview

The theme system transforms datasets to use different domains (e.g., drugs → restaurants → job candidates) while keeping the exact same attribute strength patterns (Low/Medium/High configurations) for all samples.

### Components

A **theme** consists of three parts:

1. **Entities**: The two options being compared (e.g., "Drug A" and "Drug B", or "Restaurant A" and "Restaurant B")
2. **Objective**: What you're optimizing for (e.g., "5-year overall patient outcome" or "overall customer satisfaction")
3. **Attributes**: The dimensions used for comparison with mappings from source attributes (e.g., E: Efficacy → Food Quality)

## Built-in Themes

Three themes are included:

### 1. Drugs (default)
- Entities: Drug A, Drug B
- Objective: 5-year overall patient outcome
- Attributes: Efficacy, Adherence, Safety, Durability

### 2. Restaurants
- Entities: Restaurant A, Restaurant B
- Objective: overall customer satisfaction
- Attributes: Food Quality, Price, Service, Ambiance

### 3. Candidates
- Entities: Candidate A, Candidate B
- Objective: hiring decision for long-term success
- Attributes: Experience, Culture Fit, Skills, Leadership

## Usage

### Transforming an Existing Dataset

Use the `apply_theme.py` script to transform an existing dataset:

```bash
# Using a built-in theme
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme restaurants \
  --output data/generated/v1_short_restaurants

# Using a custom theme YAML file
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme configs/themes/my_custom_theme.yml \
  --output data/generated/v1_short_custom
```

### Creating a Custom Theme with CLI Overrides

You can override theme properties via command line:

```bash
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme restaurants \
  --entity-a "Cafe A" \
  --entity-b "Cafe B" \
  --objective "customer return rate" \
  --output data/generated/v1_short_cafes
```

### Using Fewer Attributes

Map only a subset of attributes to exclude some:

```bash
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --output data/generated/v1_short_simple \
  --entity-a "Option A" \
  --entity-b "Option B" \
  --objective "best outcome" \
  --map-attr "E:Quality:Overall Quality" \
  --map-attr "A:Cost:Total Cost" \
  --map-attr "S:Speed:Response Speed" \
  --name "simple3attr"
```

This will keep only 3 attributes (E, A, S) and drop D (Durability).

### Creating a New Theme YAML File

Create a theme configuration file in `configs/themes/`:

```yaml
name: "products"
entities:
  - "Product A"
  - "Product B"
objective: "purchase decision"
attributes:
  E:
    name: "Q"
    label: "Quality"
  A:
    name: "P"
    label: "Price"
  S:
    name: "R"
    label: "Reliability"
  D:
    name: "D"
    label: "Design"
```

Then use it:

```bash
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme configs/themes/products.yml \
  --output data/generated/v1_short_products
```

### Generating a New Dataset with a Theme

You can also apply a theme when creating a dataset from scratch:

```bash
python scripts/make_dataset.py \
  --config configs/default.yml \
  --out data/generated/v2_restaurants \
  --theme restaurants
```

## What Gets Preserved

When transforming a dataset:

- ✅ All attribute strength configurations (Low/Medium/High patterns)
- ✅ All trial metadata (block, manipulation, seeds, etc.)
- ✅ Number of configs and trials
- ✅ Order permutations and paraphrase assignments

## What Gets Changed

- 🔄 Entity names in prompts (Drug A → Restaurant A)
- 🔄 Objective description in prompts
- 🔄 Attribute labels in prompts (Efficacy → Food Quality)
- 🔄 System prompt (clinical decision → restaurant recommendation)
- 🔄 Manipulation instructions (e.g., "drugs" → "restaurants")
- 🔄 Theme metadata in MANIFEST.json

## Python API

You can also use the theme system programmatically:

```python
from src.data.themes import get_theme, ThemeConfig
from src.data.apply_theme import transform_dataset

# Load a theme
theme = get_theme("restaurants")

# Transform a dataset
result = transform_dataset(
    source_dir="data/generated/v1_short",
    theme_config=theme,
    output_dir="data/generated/v1_short_restaurants"
)

print(f"Transformed {len(result['trials'])} trials")
```

## Files Created

### New Files
- `src/data/themes.py` - Theme configuration schema and built-in themes
- `src/data/apply_theme.py` - Dataset transformation logic
- `scripts/apply_theme.py` - CLI tool for applying themes
- `configs/themes/drugs.yml` - Drug theme (explicit default)
- `configs/themes/restaurants.yml` - Restaurant theme
- `configs/themes/candidates.yml` - Job candidate theme

### Modified Files
- `src/llm/prompts.py` - Updated to use theme configs
- `src/llm/types.py` - Added `theme_config` field to `TrialSpec`
- `src/data/paraphrases.py` - Updated `render_profile` to accept theme
- `src/data/make.py` - Added optional theme support during generation
- `scripts/make_dataset.py` - Added `--theme` option

## Example Workflow

1. Create a baseline dataset:
```bash
python scripts/make_dataset.py \
  --config configs/default.yml \
  --out data/generated/v1_short
```

2. Create themed variations:
```bash
# Restaurant theme
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme restaurants \
  --output data/generated/v1_short_restaurants

# Job candidate theme
python scripts/apply_theme.py \
  --source data/generated/v1_short \
  --theme candidates \
  --output data/generated/v1_short_candidates
```

3. Run trials with different themes to test domain transfer and superficial belief persistence.

