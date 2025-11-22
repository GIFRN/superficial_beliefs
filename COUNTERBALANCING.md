# Counterbalancing Implementation

## Overview

The `compare_reasoning_efforts.py` script now implements **presentation order counterbalancing** to control for position bias in LLM decision-making tasks.

## What is Counterbalancing?

Counterbalancing ensures that each trial configuration is tested with both possible label-to-profile mappings:

- **Original orientation**: Drug A = left profile, Drug B = right profile
- **Reversed orientation**: Drug A = right profile, Drug B = left profile

**Key principle**: Drug A is always mentioned first in the prompt, Drug B second. What changes is which underlying profile configuration gets labeled as "Drug A" vs "Drug B".

## Why is This Important?

1. **Controls for Position Bias**: LLMs may have preferences based on whether an option appears first/second in the prompt
2. **Controls for Label Bias**: LLMs may have preferences for "A" vs "B" labels
3. **Improves Validity**: By counterbalancing, any systematic biases average out across replicates
4. **Standard Practice**: This is a methodological best practice in experimental psychology

## Implementation Details

### Changes to `run_trial_streamlined()`

The function now:
1. Takes a `counterbalance` parameter (default: `True`)
2. Requires `S` (number of replicates) to be even when counterbalancing is enabled
3. Runs `S/2` replicates with the original orientation
4. Runs `S/2` replicates with the reversed orientation (swapping `profile_a` ↔ `profile_b`)
5. Records the orientation (`"original"` or `"reversed"`) in the response metadata

### Command-Line Arguments

- `--replicates`: Now defaults to **6** (was 5) to ensure even number
- `--no-counterbalance`: Optional flag to disable counterbalancing (not recommended)

### Example Usage

```bash
# Standard usage (counterbalancing enabled, 6 replicates)
python3 scripts/compare_reasoning_efforts.py

# Custom number of replicates (must be even)
python3 scripts/compare_reasoning_efforts.py --replicates 10

# Disable counterbalancing (not recommended)
python3 scripts/compare_reasoning_efforts.py --no-counterbalance --replicates 5
```

## How It Works

For a trial with `--replicates 6`:

### Original Orientation (3 replicates)
- Drug A gets `profile_a` (e.g., left profile from dataset)
- Drug B gets `profile_b` (e.g., right profile from dataset)
- Prompt shows: "Drug A ... Drug B ..."

### Reversed Orientation (3 replicates)
- Drug A gets `profile_b` (e.g., right profile from dataset)
- Drug B gets `profile_a` (e.g., left profile from dataset)  
- Prompt shows: "Drug A ... Drug B ..." (same order!)

## Impact on Analysis

The counterbalancing should:
- ✅ **Reduce systematic bias** in attribute weights and choice patterns
- ✅ **Improve robustness** of alignment metrics
- ✅ **Increase validity** of cross-reasoning-effort comparisons
- ⚠️  **May reduce apparent effects** that were driven by position/label bias

## Output Files

Response files (`.jsonl`) now include an `"orientation"` field for each replicate:
```json
{
  "seed": 12345,
  "orientation": "original",  // or "reversed"
  "steps": [...],
  "conversation": [...]
}
```

The comparison summary (`comparison_summary.json`) includes:
```json
{
  "counterbalanced": true,
  "replicates": 6,
  ...
}
```

## Validation

To verify counterbalancing is working:
1. Check that each trial has equal numbers of `"original"` and `"reversed"` orientations
2. Verify that choice proportions are balanced across orientations
3. Compare results with and without counterbalancing using `--no-counterbalance`

## Recommendations

- ✅ **Always use counterbalancing** for production experiments
- ✅ Use **6 replicates** as default (3 per orientation)
- ✅ For higher-powered studies, use 10+ replicates (5+ per orientation)
- ⚠️  Only disable counterbalancing for debugging or when explicitly testing for position effects

## References

This implementation follows standard practices from:
- Experimental psychology (Latin square designs)
- Survey methodology (question order effects)
- Clinical trials (randomization and counterbalancing)

