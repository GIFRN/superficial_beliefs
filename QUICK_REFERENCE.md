# Quick Reference: compare_reasoning_efforts.py

## Common Commands

### **Standard Run (First Time or Resume)**
```bash
python3 scripts/compare_reasoning_efforts.py
```
- Detects existing results and skips completed work
- Resumes from interruption automatically
- Default: 200 train + 50 test per effort level

### **Custom Sample Sizes**
```bash
python3 scripts/compare_reasoning_efforts.py --n-train 500 --n-test 100
```

### **Expand Existing Dataset (Continuation Mode)**
```bash
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 250 \
    --n-test 75
```
- Adds MORE samples to existing data
- Excludes trials already used
- Creates new result files with sample counts

### **Force Complete Rerun**
```bash
python3 scripts/compare_reasoning_efforts.py --force
```
- Overwrites all existing results
- Use with caution!

### **Run Without Counterbalancing**
```bash
python3 scripts/compare_reasoning_efforts.py --no-counterbalance --replicates 5
```
- Not recommended unless testing

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-train` | 200 | Number of training trials |
| `--n-test` | 50 | Number of test trials |
| `--b2-fraction` | 0.2 | Fraction of training from B2 block |
| `--replicates` | 6 | Replicates per trial (must be even) |
| `--model` | gpt-5-mini | OpenAI model name |
| `--seed` | 42 | Random seed for reproducibility |
| `--out` | results/reasoning_effort_comparison_1.1 | Output directory |
| `--no-counterbalance` | False | Disable counterbalancing |
| `--force` | False | Force rerun everything |
| `--continue-from-existing` | False | Expand existing dataset |

## File Outputs

### **Response Files (JSONL)**
```
responses_minimal_train.jsonl    # LLM responses (training)
responses_minimal_test.jsonl     # LLM responses (test)
responses_low_train.jsonl
responses_low_test.jsonl
responses_medium_train.jsonl
responses_medium_test.jsonl
responses_high_train.jsonl
responses_high_test.jsonl
```

### **Result Files (JSON)**

**Regular mode:**
```
results_minimal_train.json       # Analysis results
results_minimal_test.json
...
```

**Continuation mode:**
```
results_minimal_train_250.json   # With sample count
results_minimal_test_75.json
...
```

### **Summary File**
```
comparison_summary.json          # Cross-effort comparison
```

## Output Structure

Each result file contains:
```json
{
  "weights": {"E": 0.35, "A": 0.15, "S": 0.24, "D": 0.26},
  "beta": {"E": 2.13, "A": 1.32, "S": 1.77, "D": 1.94},
  "alignment": {
    "ECRB_top1_driver": 0.456,
    "ECRB_top1_weights": 0.452,
    "rank_corr": 1.000
  },
  "prediction": {  // Test set only
    "mae": 0.148,
    "rmse": 0.230,
    "correlation": 0.832,
    "accuracy": 0.800
  },
  "n_trials": 200,
  "n_responses": 1198,
  "n_train_samples": 200  // In continuation mode
}
```

## Modes of Operation

### **Mode 1: First Run**
```bash
python3 scripts/compare_reasoning_efforts.py
```
- Runs all 4 effort levels (minimal, low, medium, high)
- 200 train + 50 test per level
- Total: ~1000 trials × 6 replicates = 6000 LLM calls

### **Mode 2: Resume from Interruption**
```bash
# Same command after interruption
python3 scripts/compare_reasoning_efforts.py
```
- Detects completed trials automatically
- Skips finished work
- Continues from last incomplete trial

### **Mode 3: Expand Dataset**
```bash
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 300
```
- Keeps existing 200 train samples
- Adds 100 NEW train samples (different trial IDs)
- Creates results_*_train_300.json files
- Original results_*_train.json preserved

### **Mode 4: Force Rerun**
```bash
python3 scripts/compare_reasoning_efforts.py --force
```
- Deletes/overwrites all existing data
- Runs everything from scratch
- Use when you've changed prompts or found a bug

## Typical Workflow

### **Phase 1: Initial Study**
```bash
# Run pilot with modest sample size
python3 scripts/compare_reasoning_efforts.py \
    --n-train 100 \
    --n-test 25
```
**Time**: ~3-4 hours  
**Output**: results_*_train.json (100 samples)

### **Phase 2: Expand for Preliminary Results**
```bash
# Add more samples
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 200 \
    --n-test 50
```
**Time**: ~2-3 hours (only runs additional 100+25)  
**Output**: results_*_train_200.json (200 samples)

### **Phase 3: Full Dataset for Publication**
```bash
# Expand to publication-ready size
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 500 \
    --n-test 100
```
**Time**: ~6-8 hours (only runs additional 300+50)  
**Output**: results_*_train_500.json (500 samples)

## Checking Status

### **What's been completed?**
```python
from pathlib import Path
import json

out_dir = Path("results/reasoning_effort_comparison_1.1")

for effort in ["minimal", "low", "medium", "high"]:
    train = out_dir / f"responses_{effort}_train.jsonl"
    test = out_dir / f"responses_{effort}_test.jsonl"
    
    n_train = sum(1 for _ in open(train)) if train.exists() else 0
    n_test = sum(1 for _ in open(test)) if test.exists() else 0
    
    print(f"{effort}: {n_train} train, {n_test} test")
```

### **What trials have been used?**
```python
from scripts.compare_reasoning_efforts import collect_all_existing_trial_ids

out_dir = Path("results/reasoning_effort_comparison_1.1")
effort_levels = ["minimal", "low", "medium", "high"]

used_ids = collect_all_existing_trial_ids(out_dir, effort_levels)
print(f"Total unique trials used: {len(used_ids)}")
```

## Troubleshooting

### **Script won't continue after interruption**
```bash
# Check last line of response file
tail -1 results/reasoning_effort_comparison_1.1/responses_minimal_train.jsonl

# If it's incomplete JSON, remove it
head -n -1 responses_minimal_train.jsonl > temp.jsonl
mv temp.jsonl responses_minimal_train.jsonl
```

### **"Not enough trials available" error**
- You've used most of the dataset
- Check available: `wc -l data/generated/v1_short/dataset_trials.parquet`
- Reduce --n-train or --n-test

### **Results don't match expected sample count**
- Delete result files: `rm results_*_train_*.json`
- Re-run analysis (won't re-run trials)

### **Want to restart completely**
```bash
# Backup current work
mv results/reasoning_effort_comparison_1.1 results/reasoning_effort_comparison_1.1.backup

# Start fresh
python3 scripts/compare_reasoning_efforts.py
```

## Performance Notes

### **Timing Estimates**
- **Per trial**: ~20-30 seconds (depends on reasoning effort and model speed)
- **Per replicate**: ~3-5 seconds
- **200 trials × 6 replicates**: ~1-1.5 hours per effort level
- **Full run (4 efforts)**: ~4-6 hours total

### **Token Usage**
- **Per trial**: ~2000-4000 tokens (prompt + response × 6 replicates)
- **200 trials**: ~400k-800k tokens per effort level
- **Full run**: ~1.6M-3.2M tokens across all effort levels

### **Cost Estimates (GPT-5-mini)**
- Varies by pricing tier
- Recommend checking OpenAI usage dashboard

## Quick Commands Cheat Sheet

```bash
# Standard run (first time or resume)
python3 scripts/compare_reasoning_efforts.py

# Larger dataset
python3 scripts/compare_reasoning_efforts.py --n-train 500 --n-test 100

# Expand existing
python3 scripts/compare_reasoning_efforts.py --continue-from-existing --n-train 300

# Force rerun
python3 scripts/compare_reasoning_efforts.py --force

# Check help
python3 scripts/compare_reasoning_efforts.py --help

# View results
cat results/reasoning_effort_comparison_1.1/comparison_summary.json | python3 -m json.tool

# Count samples
wc -l results/reasoning_effort_comparison_1.1/responses_*.jsonl

# Check progress during run
watch -n 60 'wc -l results/reasoning_effort_comparison_1.1/responses_*.jsonl'
```

## Related Documentation

- **INCREMENTAL_EXECUTION.md**: Detailed info on resume capability
- **CONTINUATION_MODE.md**: Complete guide to expanding datasets
- **COUNTERBALANCING.md**: Understanding counterbalancing
- **TRAIN_TEST_SPLIT.md**: Train/test split methodology
- **README.md**: Project overview

---

**Need help?** Check the full documentation in the related .md files or examine the script help: `python3 scripts/compare_reasoning_efforts.py --help`

