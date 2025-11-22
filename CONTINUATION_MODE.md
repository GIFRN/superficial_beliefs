# Continuation Mode: Expanding Existing Datasets

## Overview

The `compare_reasoning_efforts.py` script supports **continuation mode**, which allows you to expand existing datasets by adding more samples without re-running completed trials.

## Use Case

**Scenario**: You have 200 training + 50 test trials for each reasoning effort level, but now want 250 training + 75 test trials.

**Without continuation**: Would need to re-run everything from scratch

**With continuation**: Runs only the additional 50 training + 25 test trials and appends to existing files

## How It Works

### **Key Features**

1. **Excludes already-used trials**: Collects trial IDs from ALL existing response files (across all effort levels, both train and test)
2. **Samples new trials only**: Ensures no overlap between existing and new samples
3. **Appends to existing files**: New responses added to existing `.jsonl` files
4. **Creates new result files**: Analysis results saved with sample count in filename

### **Important Behavior**

- **Global exclusion**: A trial used in ANY effort level (train OR test) is excluded from ALL new sampling
- **Independent per effort**: Each effort level gets its own NEW set of trials
- **Sample count in filename**: Results include actual number of samples (e.g., `results_minimal_train_250.json`)

## Usage

### **Basic Continuation Example**

```bash
# Current state: 200 train + 50 test per effort level
# Goal: Expand to 250 train + 75 test per effort level

python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 250 \
    --n-test 75
```

**What happens:**
1. Scans all existing response files
2. Finds 250 unique trial IDs already used (200 train + 50 test per effort level, but same trials used across efforts)
3. Samples 250 NEW train trials (avoiding the 250 already used)
4. Samples 75 NEW test trials (avoiding the 250 already used, plus any train trials)
5. Runs only the new trials (50 train + 25 test per effort level)
6. Appends to existing `.jsonl` files
7. Creates `results_*_train_250.json` and `results_*_test_75.json`

### **Starting Fresh (No Continuation)**

```bash
# Default behavior: checks for existing work and resumes
python3 scripts/compare_reasoning_efforts.py

# Force complete rerun: overwrites everything
python3 scripts/compare_reasoning_efforts.py --force
```

## File Structure

### **Response Files (Appended)**

```
responses_minimal_train.jsonl  # Now contains 250 trials (was 200)
responses_minimal_test.jsonl   # Now contains 75 trials (was 50)
responses_low_train.jsonl      # Now contains 250 trials (was 200)
...
```

**Format**: One JSON object per line (JSONL)

### **Result Files (New Files Created)**

```
results_minimal_train.json       # Original results (200 samples)
results_minimal_train_250.json   # NEW results (250 samples)
results_minimal_test.json        # Original results (50 samples)
results_minimal_test_75.json     # NEW results (75 samples)
...
```

**Benefits of separate files:**
- Compare results at different sample sizes
- Track how metrics change with more data
- Preserve original analysis

## Workflow Example

### **Iteration 1: Initial Study**

```bash
# Run with 200 train + 50 test
python3 scripts/compare_reasoning_efforts.py \
    --n-train 200 \
    --n-test 50
```

**Result:** 
- Files: `responses_*_train.jsonl` (200 trials each)
- Files: `responses_*_test.jsonl` (50 trials each)
- Files: `results_*_train.json` (analysis on 200)
- Files: `results_*_test.json` (analysis on 50)

### **Iteration 2: Expand Dataset**

```bash
# Add 50 more train + 25 more test
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 250 \
    --n-test 75
```

**Result:**
- Files: `responses_*_train.jsonl` (now 250 trials) ← **Appended**
- Files: `responses_*_test.jsonl` (now 75 trials) ← **Appended**
- Files: `results_*_train_250.json` (analysis on 250) ← **New**
- Files: `results_*_test_75.json` (analysis on 75) ← **New**
- Files: `results_*_train.json` (still there) ← **Preserved**
- Files: `results_*_test.json` (still there) ← **Preserved**

### **Iteration 3: Further Expansion**

```bash
# Expand to 500 train + 100 test
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 500 \
    --n-test 100
```

**Result:**
- Adds 250 more train + 25 more test
- Creates `results_*_train_500.json` and `results_*_test_100.json`
- All previous result files still preserved

## Technical Details

### **Trial ID Exclusion Logic**

```python
def collect_all_existing_trial_ids(out_dir, effort_levels):
    """Collect trial IDs from ALL response files (all efforts, train + test)."""
    all_trial_ids = set()
    
    for effort in ["minimal", "low", "medium", "high"]:
        for split in ["train", "test"]:
            path = out_dir / f"responses_{effort}_{split}.jsonl"
            if path.exists():
                # Read each line and extract trial_id
                for line in open(path):
                    resp = json.loads(line)
                    all_trial_ids.add(resp["trial_id"])
    
    return all_trial_ids
```

**Why exclude from ALL efforts:**
- Ensures clean separation between existing and new data
- Prevents any trial from appearing in both old and new portions
- Maintains independence for comparative analysis

### **Sampling Logic with Exclusion**

```python
def sample_b3_trials(..., exclude_trial_ids=None):
    # Filter out already-used trials
    if exclude_trial_ids:
        trials_df = trials_df[~trials_df["trial_id"].isin(exclude_trial_ids)]
    
    # Sample from remaining trials
    train_trials = sample_from_remaining(...)
    test_trials = sample_from_remaining(...)
    
    return train_trials, test_trials
```

### **Result Filename Logic**

```python
# In continuation mode
n_train_actual = len(train_responses)  # e.g., 250
results_path = out_dir / f"results_{effort}_train_{n_train_actual}.json"

# In regular mode
results_path = out_dir / f"results_{effort}_train.json"
```

## Common Patterns

### **Pattern 1: Doubling Sample Size**

```bash
# Start: 100 train + 25 test
python3 scripts/compare_reasoning_efforts.py --n-train 100 --n-test 25

# Double it: 200 train + 50 test
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 200 \
    --n-test 50
```

### **Pattern 2: Adding Only Training Samples**

```bash
# Expand training but keep test size same
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing \
    --n-train 500 \
    --n-test 50  # Same as before
```

**Note**: Test trials won't be re-run, but NEW test result files will be created with 50 samples (to match the new training results)

### **Pattern 3: Progressive Growth**

```bash
# Wave 1: Quick pilot
python3 scripts/compare_reasoning_efforts.py --n-train 50 --n-test 10

# Wave 2: Expand for preliminary results
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing --n-train 200 --n-test 50

# Wave 3: Full dataset for publication
python3 scripts/compare_reasoning_efforts.py \
    --continue-from-existing --n-train 1000 --n-test 200
```

## Comparison: Regular vs Continuation Mode

| Aspect | Regular Mode | Continuation Mode |
|--------|-------------|-------------------|
| **Purpose** | Resume interrupted runs | Expand existing datasets |
| **Trial selection** | Same trial IDs | NEW trial IDs |
| **Response files** | Append missing trials | Append additional trials |
| **Result files** | Same filename | New filename with sample count |
| **Sample count** | Fixed (as specified) | Cumulative (adds to existing) |
| **Use when** | Script was interrupted | Want more data |

## Troubleshooting

### **Issue: "Not enough trials available"**

**Cause**: You've used most of the B3 block and can't sample more

**Solution**: 
- Check how many B3 trials exist: `wc -l data/generated/v1_short/dataset_trials.parquet`
- Reduce target sample size
- Or generate a larger B3 block

### **Issue: Accidentally overwrote old results**

**Cause**: Ran without `--continue-from-existing` flag

**Solution**:
- Old result files (`results_*_train.json`) should still exist
- If using continuation mode, new files have sample count in name
- Check if `.json` backup files exist

### **Issue: Trial overlap detected**

**Cause**: Manual file manipulation or running multiple instances simultaneously

**Solution**:
```python
# Check for duplicate trial IDs
import json
from pathlib import Path

path = Path("results/reasoning_effort_comparison_1.1/responses_minimal_train.jsonl")
trial_ids = []

with path.open() as f:
    for line in f:
        if line.strip():
            resp = json.loads(line)
            trial_ids.append(resp["trial_id"])

# Check for duplicates
if len(trial_ids) != len(set(trial_ids)):
    print("⚠️  Warning: Duplicate trial IDs found!")
    print(f"Total: {len(trial_ids)}, Unique: {len(set(trial_ids))}")
else:
    print("✅ No duplicates")
```

### **Issue: Results don't match response count**

**Cause**: Analysis run on old data before new responses added

**Solution**:
- Delete result files with sample counts in name
- Re-run with `--continue-from-existing` (will regenerate analysis)

## Best Practices

1. **Always specify both --n-train and --n-test**: Even if one matches existing, specify both for clarity

2. **Use consistent seeds**: Use same `--seed` value across iterations for reproducibility

3. **Backup before large expansions**:
   ```bash
   tar -czf backup_$(date +%Y%m%d).tar.gz results/reasoning_effort_comparison_1.1/
   ```

4. **Track your iterations**: Keep a log of sample sizes
   ```bash
   # In a notebook or log file
   # 2025-01-15: Initial run with 200 train + 50 test
   # 2025-01-20: Expanded to 250 train + 75 test
   # 2025-01-25: Expanded to 500 train + 100 test
   ```

5. **Compare results across sample sizes**: Analyze how metrics change
   ```python
   import json
   
   results_200 = json.load(open("results_minimal_train.json"))
   results_250 = json.load(open("results_minimal_train_250.json"))
   results_500 = json.load(open("results_minimal_train_500.json"))
   
   print("Efficacy weight over time:")
   print(f"  200 samples: {results_200['weights']['E']}")
   print(f"  250 samples: {results_250['weights']['E']}")
   print(f"  500 samples: {results_500['weights']['E']}")
   ```

6. **Don't mix --force and --continue-from-existing**: They're incompatible
   - `--force`: Starts fresh (overwrites)
   - `--continue-from-existing`: Expands existing

## Validation

After running in continuation mode, verify:

```python
from pathlib import Path
import json

out_dir = Path("results/reasoning_effort_comparison_1.1")

for effort in ["minimal", "low", "medium", "high"]:
    train_path = out_dir / f"responses_{effort}_train.jsonl"
    
    # Count lines
    with train_path.open() as f:
        n_lines = sum(1 for line in f if line.strip())
    
    # Check result file
    result_path = out_dir / f"results_{effort}_train_{n_lines}.json"
    if result_path.exists():
        with result_path.open() as f:
            results = json.load(f)
            n_samples = results.get("n_train_samples", 0)
            
        print(f"{effort}: {n_lines} lines, {n_samples} samples in results")
        assert n_lines == n_samples, f"Mismatch for {effort}!"
    else:
        print(f"{effort}: {n_lines} lines, but result file not found")
```

---

**TL;DR**: Use `--continue-from-existing` to expand your dataset. New samples are added to existing files, and results are saved with sample counts in filenames so you can track changes over time. Perfect for iterative research where you start with a pilot and progressively expand your dataset!

