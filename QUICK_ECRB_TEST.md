# Quick ECRB Test Script

## Purpose

Test if increasing GPT-5's `reasoning_effort` from "minimal" to "medium" improves ECRB alignment metrics using a small sample of trials for fast iteration.

## Usage

### Basic Usage (50 trials, medium reasoning)
```bash
python3 scripts/quick_ecrb_test.py
```

### Compare Different Reasoning Efforts

**Minimal (baseline):**
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort minimal
# Auto-saves to: results/quick_ecrb_test_minimal/
```

**Medium:**
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort medium
# Auto-saves to: results/quick_ecrb_test_medium/
```

**High:**
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort high
# Auto-saves to: results/quick_ecrb_test_high/
```

### Custom Sample Size

For faster testing:
```bash
python3 scripts/quick_ecrb_test.py --sample-size 20 --replicates 3
```

For more reliable estimates:
```bash
python3 scripts/quick_ecrb_test.py --sample-size 100 --replicates 10
```

## Parameters

- `--reasoning-effort {minimal,low,medium,high}`: GPT-5 reasoning effort level (default: medium)
- `--sample-size N`: Number of non-B1 trials to sample (default: 50)
- `--replicates N`: Samples per trial (default: 5)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--out DIR`: Output directory (auto-generated as `results/quick_ecrb_test_{reasoning_effort}` if not specified)

## What It Does

1. **Samples** N non-B1 trials randomly from the dataset
2. **Runs** those trials with the specified reasoning effort
3. **Fits** a Stage A logistic regression model on the results
4. **Calculates** ECRB alignment metrics:
   - **ECRB_top1_driver**: % where stated driver matches fitted driver
   - **ECRB_top1_weights**: % where stated driver is the top-weighted attribute
   - **Rank correlation**: Spearman correlation between stated and fitted rankings

## Output

The script prints:
- Fitted attribute weights
- ECRB metrics
- Total valid attributions

And saves:
- `ecrb_test_summary.json`: All metrics and parameters
- `responses.jsonl`: Raw model responses for further analysis

## Example Output

```
================================================================================
QUICK ECRB TEST
================================================================================
Reasoning effort: medium
Sample size: 50 trials
Replicates per trial: 5
Random seed: 42

Sampled 50 trials from 429 non-B1 trials
Block distribution: {'B2': 25, 'B3': 20, 'DOM': 5}

Running 50 trials with reasoning_effort=medium...
Trials: 100%|██████████████████████| 50/50 [02:30<00:00,  3.01s/it]

Completed 50 trials

Fitting Stage A model...
Valid choices: 245 / 250

Fitted attribute weights:
  E: 38.2% (β = 2.945)
  D: 25.1% (β = 1.934)
  S: 22.8% (β = 1.756)
  A: 13.9% (β = 1.074)

Calculating ECRB...

================================================================================
RESULTS
================================================================================
ECRB_top1_driver:  52.3%
ECRB_top1_weights: 48.7%
Rank correlation:  1.000
Total attributions: 241
================================================================================

Results saved to: results/quick_ecrb_test
```

## Comparison Workflow

1. Run with minimal reasoning (baseline):
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort minimal --seed 42
# Saves to: results/quick_ecrb_test_minimal/
```

2. Run with medium reasoning:
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort medium --seed 42
# Saves to: results/quick_ecrb_test_medium/
```

3. Compare results:
```bash
# View minimal results
cat results/quick_ecrb_test_minimal/ecrb_test_summary.json | jq '.ecrb'

# View medium results
cat results/quick_ecrb_test_medium/ecrb_test_summary.json | jq '.ecrb'
```

## Notes

- Uses the **same random sample** with the same seed for fair comparison
- **Non-B1 trials only** (excludes dominant-choice rationality checks)
- **Self-contained**: All ECRB calculation logic is in the script
- **Fast**: 50 trials × 5 replicates ≈ 2-3 minutes runtime
- Uses **gpt-5-mini** model by default

## Expected Impact

Higher reasoning effort *might* improve ECRB by:
- More careful consideration of which attribute drives the decision
- Better premise statements that match the actual decision logic
- More consistent reasoning across replicates

But it also increases:
- Cost (more reasoning tokens)
- Latency (slower responses)

This script helps you quantify the tradeoff before running a full experiment.

