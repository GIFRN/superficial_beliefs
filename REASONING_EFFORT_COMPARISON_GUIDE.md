# Reasoning Effort Comparison Script - Usage Guide

## Overview

`scripts/compare_reasoning_efforts.py` runs a controlled comparison of GPT-5-mini across **four reasoning effort levels** (minimal, low, medium, high) using the **same 100 B3 trials** with **5 replicates each**.

### Key Features

✅ **Streamlined prompting** - Removes redundant "repeat" step for efficiency  
✅ **Live progress** - Prints results for each tranche as it completes  
✅ **Controlled comparison** - Same trials across all conditions  
✅ **Comprehensive metrics** - Weights, alignment, choice patterns, variance  
✅ **Saves all data** - Raw responses + analyzed results for each level  

## Usage

### Basic Usage

```bash
cd /Users/anna/Desktop/PhD/superficial_beliefs
source .venv/bin/activate
python3 scripts/compare_reasoning_efforts.py
```

This will:
1. Sample 100 B3 trials from `data/generated/v1_short`
2. Run each through minimal, low, medium, high reasoning effort
3. Use 5 replicates per trial per condition (400 total API calls)
4. Save results to `results/reasoning_effort_comparison/`

### Custom Options

```bash
python3 scripts/compare_reasoning_efforts.py \
  --dataset data/generated/my_dataset \
  --n-samples 50 \
  --replicates 3 \
  --model gpt-5-mini \
  --seed 42 \
  --out results/my_comparison
```

**Arguments:**
- `--dataset`: Path to dataset directory (default: `data/generated/v1_short`)
- `--n-samples`: Number of B3 trials to sample (default: 100)
- `--replicates`: Replicates per trial (default: 5)
- `--model`: Model name (default: `gpt-5-mini`)
- `--seed`: Random seed for sampling (default: 42)
- `--out`: Output directory (default: `results/reasoning_effort_comparison`)

## What It Does

### 1. Sampling Phase
```
📦 Sampling B3 trials...
✅ Sampled 100 trials
```

Randomly samples 100 trials from B3 block (complex multi-attribute trade-offs).

### 2. Execution Phase (for each reasoning level)

```
================================================================================
RUNNING: MINIMAL REASONING EFFORT
================================================================================

Running 100 trials...
minimal effort: 100%|██████████████████████████| 100/100 [02:15<00:00,  1.35s/it]
✅ Saved responses to results/reasoning_effort_comparison/responses_minimal.jsonl

🔬 Analyzing results...
```

**What happens:**
- Runs 100 trials × 5 replicates = 500 API calls
- Streamlined prompting (2 steps instead of 3):
  1. **Choice + reason** (combined in one prompt)
  2. **Premise extraction** (attribute + text)
- Saves raw responses immediately
- Analyzes and prints results

### 3. Results Display (after each level)

```
================================================================================
RESULTS: MINIMAL REASONING EFFORT
================================================================================

📊 Data Collection:
  Trials: 100
  Responses: 500

⚖️  Attribute Weights:
  E: 0.358 (β = 2.832)
  A: 0.157 (β = 1.243)
  S: 0.237 (β = 1.874)
  D: 0.248 (β = 1.958)
  CV (differentiation): 0.286

📈 Choice Patterns:
  All chose A: 47 (47.0%)
  All chose B: 33 (33.0%)
  Mixed: 20 (20.0%)
  Choice variance: 0.2118

🎯 Alignment Metrics:
  ECRB_top1_driver: 0.444
  ECRB_top1_weights: 0.419
  Rank correlation: 1.000
```

### 4. Final Comparison Table

After all four levels complete:

```
================================================================================
FINAL COMPARISON TABLE
================================================================================

Metric                    Minimal      Low          Medium       High        
-------------------------------------------------------------------------------------
E weight                  0.358        0.345        0.298        0.293       
A weight                  0.157        0.180        0.253        0.246       
S weight                  0.237        0.245        0.287        0.276       
D weight                  0.248        0.230        0.162        0.184       
CV (differentiation)      0.286        0.265        0.215        0.166       
Choice variance           0.2118       0.2085       0.2050       0.2049      
Extreme choice rate       0.800        0.775        0.730        0.714       
ECRB_top1_driver          0.444        0.495        0.572        0.438       
ECRB_top1_weights         0.419        0.405        0.388        0.303       
rank_corr                 1.000        0.950        0.800        0.800       
```

## Output Files

All saved to `results/reasoning_effort_comparison/`:

### Per-Level Files

1. **`responses_{effort}.jsonl`** - Raw API responses
   ```json
   {"trial_id": "T-00123", "responses": [...], "config_id": "B3-0045", ...}
   ```

2. **`results_{effort}.json`** - Analyzed metrics
   ```json
   {
     "weights": {"E": 0.358, "A": 0.157, "S": 0.237, "D": 0.248},
     "beta": {"E": 2.832, "A": 1.243, "S": 1.874, "D": 1.958},
     "alignment": {"ECRB_top1_driver": 0.444, ...},
     "choice_variance": 0.2118,
     "extreme_choices": {"all_a": 47, "all_b": 33, "mixed": 20}
   }
   ```

### Summary File

**`comparison_summary.json`** - Complete comparison
```json
{
  "effort_levels": ["minimal", "low", "medium", "high"],
  "results": {
    "minimal": {...},
    "low": {...},
    "medium": {...},
    "high": {...}
  },
  "n_samples": 100,
  "replicates": 5,
  "model": "gpt-5-mini"
}
```

## Streamlined Prompting

### Original (3 steps)
```
1. "State option and reason"
   → "Drug A. Efficacy is most important."

2. "Repeat only the reason"  ← REDUNDANT
   → "Efficacy is most important."

3. "Extract attribute"
   → "PremiseAttribute = Efficacy"
```

### Streamlined (2 steps)
```
1. "State option and reason"
   → "Drug A. Efficacy is most important."

2. "Extract attribute"
   → "PremiseAttribute = Efficacy"
```

**Benefits:**
- 33% fewer API calls
- Faster execution
- Less token usage
- Same information collected

## Expected Runtime

**For 100 trials × 5 replicates × 4 levels = 2,000 API calls:**

| Reasoning Effort | Avg Time/Call | Total Time |
|------------------|---------------|------------|
| Minimal | ~1.5s | ~12 minutes |
| Low | ~2.0s | ~17 minutes |
| Medium | ~2.5s | ~21 minutes |
| High | ~3.5s | ~29 minutes |

**Total estimated runtime: ~80 minutes** (1 hour 20 min)

*Note: Actual times vary based on API load*

## Cost Estimate

**For GPT-5-mini:**
- Input: ~200 tokens/call × 2 steps = 400 tokens
- Output: ~50 tokens/call × 2 steps = 100 tokens
- Total: ~500 tokens/call

**2,000 calls × 500 tokens = 1,000,000 tokens (~1M)**

At GPT-5-mini pricing (~$0.10/1M tokens):
- **Estimated cost: ~$0.10** (very affordable!)

## Monitoring Progress

The script prints real-time updates:

```
minimal effort: 45%|████████▌         | 45/100 [01:02<01:13,  1.34s/it]
```

You can:
- ✅ See progress bars for each level
- ✅ Watch trial-by-trial completion
- ✅ View results immediately after each level
- ✅ Stop anytime (Ctrl+C) - already-completed levels are saved

## Interpreting Results

### Weight Differentiation (CV)
- **High CV** (0.3+) = Strong preferences, clear hierarchy
- **Medium CV** (0.2-0.3) = Balanced preferences
- **Low CV** (<0.2) = Flat preferences, everything equal

### Choice Variance
- **High variance** (0.25+) = Diverse choices across replicates
- **Low variance** (<0.15) = Consistent, deterministic choices

### Extreme Choice Rate
- **High rate** (80%+) = Decisive (all A or all B)
- **Low rate** (<70%) = More uncertainty, mixed decisions

### Alignment (ECRB)
- **Top1_driver**: Premise matches trial-specific driver
- **Top1_weights**: Premise matches globally top-weighted attribute
- **Rank_corr**: Premise frequencies correlate with weights

## Troubleshooting

### "No valid choices collected"
- Check API key is set: `echo $OPENAI_API_KEY`
- Verify model access: `gpt-5-mini` requires special access

### "Error on trial X"
- Usually API rate limits or transient errors
- Script continues with remaining trials
- Check error messages in output

### Slow execution
- Normal! High reasoning effort takes longer
- Consider reducing `--n-samples` for testing
- Use `--replicates 3` instead of 5 for faster runs

### Different results than previous runs
- Expected! Sampling is random (use same `--seed` for reproducibility)
- Temperature=1.0 means responses vary across replicates

## Next Steps

After running the comparison:

1. **Analyze trends** - Look for monotonic or non-monotonic patterns
2. **Visualize** - Create plots using `scripts/visualize_reasoning_effort.py`
3. **Statistical testing** - Compare weights between levels
4. **Qualitative analysis** - Read actual responses in `responses_*.jsonl`
5. **Iterate** - Try different sample sizes or models

## Example Session

```bash
$ source .venv/bin/activate
$ python3 scripts/compare_reasoning_efforts.py --n-samples 50 --replicates 3

================================================================================
REASONING EFFORT COMPARISON
================================================================================
Model: gpt-5-mini
Samples: 50 B3 trials
Replicates: 3 per trial
Dataset: data/generated/v1_short

📦 Sampling B3 trials...
✅ Sampled 50 trials

================================================================================
RUNNING: MINIMAL REASONING EFFORT
================================================================================

Running 50 trials...
minimal effort: 100%|██████████████████| 50/50 [01:05<00:00,  1.31s/it]
✅ Saved responses to results/reasoning_effort_comparison/responses_minimal.jsonl

🔬 Analyzing results...

================================================================================
RESULTS: MINIMAL REASONING EFFORT
================================================================================
[... detailed results ...]

[Process repeats for low, medium, high...]

================================================================================
FINAL COMPARISON TABLE
================================================================================
[... comparison table ...]

✅ Complete! Results saved to results/reasoning_effort_comparison
```

## FAQ

**Q: Can I interrupt and resume?**
A: No resume capability, but completed levels are saved. You can manually skip completed levels by commenting out in the script.

**Q: Why 100 trials?**
A: Balance between statistical power and API costs. 50 is minimum for reliable estimates.

**Q: Can I use a different model?**
A: Yes, use `--model gpt-4o` or similar, but reasoning_effort only works with GPT-5/O-series models.

**Q: How do I get just minimal vs high?**
A: Edit `effort_levels = ["minimal", "high"]` in the script (line 326).

**Q: Can I add more replicates?**
A: Yes, use `--replicates 10` for more stable estimates (but 2x API calls).

