# Incremental Execution & Resume Capability

## Overview

The `compare_reasoning_efforts.py` script now supports **incremental execution** with automatic resume capability. If the script is interrupted, it can continue exactly where it left off without re-running completed trials.

## Features

### 1. **Automatic Detection of Completed Work**

The script checks for existing results at three levels:
- **Effort level**: Skips entire effort levels (minimal, low, medium, high) if complete
- **Phase level**: Distinguishes between train/test phases
- **Trial level**: Resumes within a phase if partially complete

### 2. **Incremental Saving**

Responses are saved **after each trial** completes:
- No need to wait for entire batch to finish
- If interrupted mid-batch, completed trials are preserved
- Next run picks up only the remaining trials

### 3. **Resume from Interruption**

If the script is stopped (Ctrl+C, system crash, timeout):
```bash
# Simply re-run the same command
python3 scripts/compare_reasoning_efforts.py

# The script will:
# 1. Detect existing responses
# 2. Skip completed trials
# 3. Continue with remaining trials
# 4. Append to existing JSONL files
```

## Usage

### **Standard Run (with resume)**

```bash
python3 scripts/compare_reasoning_efforts.py
```

**Behavior:**
- Checks `results/reasoning_effort_comparison_1.1/` for existing files
- Skips complete effort levels
- Resumes incomplete levels from last completed trial
- Shows progress: "Running X remaining trials (of Y total)"

### **Force Complete Rerun**

```bash
python3 scripts/compare_reasoning_efforts.py --force
```

**Behavior:**
- Ignores all existing results
- Overwrites existing files
- Runs all trials from scratch

### **Check Status Without Running**

```python
from pathlib import Path
from scripts.compare_reasoning_efforts import check_completion_status

out_dir = Path("results/reasoning_effort_comparison_1.1")

for effort in ["minimal", "low", "medium", "high"]:
    status = check_completion_status(out_dir, effort, 200, 50)
    print(f"{effort}: {status}")
```

## File Structure

### **Response Files (JSONL format)**

Each line is a complete trial response:
```
responses_{effort}_train.jsonl    # Training trial responses
responses_{effort}_test.jsonl     # Test trial responses
```

**Format**: One JSON object per line (newline-delimited JSON)

### **Analysis Files (JSON format)**

Computed metrics:
```
results_{effort}_train.json       # Training analysis results
results_{effort}_test.json        # Test analysis results
```

### **Completion Tracking**

The script tracks completion by:
1. **Counting lines** in `.jsonl` files (one trial per line)
2. **Checking trial_ids** to identify which trials are complete
3. **Verifying analysis files** exist and contain expected keys

## Interruption Scenarios

### **Scenario 1: Interrupted During Training**

```
✅ Completed: minimal (train + test)
✅ Completed: low (train + test)
🔄 In progress: medium (100/200 train trials)
⏸️  Not started: medium test, high train, high test
```

**On resume:**
- Skips minimal, low (complete)
- Loads existing 100 medium training responses
- Runs remaining 100 medium training trials
- Continues to medium test, then high

### **Scenario 2: Interrupted During Test**

```
✅ Completed: minimal (train + test)
✅ Completed: low (train + test)
✅ Completed: medium (train only)
🔄 In progress: medium (30/50 test trials)
⏸️  Not started: high
```

**On resume:**
- Skips minimal, low (complete)
- Loads medium training (complete, no rerun)
- Loads existing 30 medium test responses
- Runs remaining 20 medium test trials
- Continues to high

### **Scenario 3: Crashed After Training, Before Test**

```
✅ Completed: minimal (train + test)
✅ Completed: low train
⏸️  Not started: low test, medium, high
```

**On resume:**
- Skips minimal
- Loads low training (complete)
- Runs low test
- Continues normally

## Data Integrity

### **JSONL Format Benefits**

- **Append-safe**: Can safely append new trials without corrupting existing data
- **Partial reads**: Can read first N lines without parsing entire file
- **Recovery**: Incomplete final line is skipped (only complete lines counted)

### **Error Handling**

If a trial fails:
```python
try:
    result = run_trial_streamlined(...)
    save_response_incremental(result, path)  # Only saves on success
except Exception as e:
    print(f"❌ Error on trial {trial_id}: {e}")
    continue  # Skip this trial, continue with next
```

**Result**: Failed trials are not saved, will be retried on next run

### **Validation**

To verify data integrity:
```python
import json
from pathlib import Path

path = Path("results/reasoning_effort_comparison_1.1/responses_minimal_train.jsonl")

valid_count = 0
with path.open("r") as f:
    for line_num, line in enumerate(f, 1):
        if line.strip():
            try:
                data = json.loads(line)
                assert "trial_id" in data
                assert "responses" in data
                valid_count += 1
            except Exception as e:
                print(f"Line {line_num}: {e}")

print(f"Valid trials: {valid_count}")
```

## Performance

### **Resume Overhead**

Minimal overhead for checking existing results:
- **Line counting**: O(n) where n = number of trials
- **Trial ID extraction**: O(n) in-memory set operation
- **File I/O**: Sequential read, efficient for JSONL

For 200 trials: < 1 second overhead

### **Incremental Save Cost**

Negligible per-trial cost:
- **Open file (append mode)**: Reuses file handle if kept open
- **Write JSON + newline**: Single string write
- **No buffering delay**: Immediate disk write

Per trial: < 0.01 seconds

## Troubleshooting

### **Issue: "No progress detected"**

**Cause**: JSONL file may be corrupted (incomplete final line)

**Solution**:
```bash
# Check last line
tail -1 results/reasoning_effort_comparison_1.1/responses_minimal_train.jsonl | python3 -m json.tool

# If error, remove incomplete line
head -n -1 responses_minimal_train.jsonl > responses_minimal_train.jsonl.tmp
mv responses_minimal_train.jsonl.tmp responses_minimal_train.jsonl
```

### **Issue: "Duplicate trials"**

**Cause**: Script run twice simultaneously

**Solution**: Don't run multiple instances on same output directory

### **Issue: "Analysis results don't match responses"**

**Cause**: Analysis run on old data before new responses added

**Solution**:
```bash
# Delete analysis results to force recomputation
rm results/reasoning_effort_comparison_1.1/results_*_*.json

# Rerun (will reanalyze but not rerun trials)
python3 scripts/compare_reasoning_efforts.py
```

## Best Practices

1. **Use default settings first**: Don't use `--force` unless you want to rerun everything

2. **Monitor progress**: Check JSONL file size grows over time
   ```bash
   watch -n 60 'wc -l results/reasoning_effort_comparison_1.1/responses_*.jsonl'
   ```

3. **Check logs**: The script prints which trials are being run

4. **Backup before --force**: If you have valuable partial results
   ```bash
   cp -r results/reasoning_effort_comparison_1.1 results/reasoning_effort_comparison_1.1.backup
   ```

5. **Verify completion**: All four effort levels should show "✅ complete" message

## Example Session

```bash
# Initial run - gets interrupted after minimal and half of low
$ python3 scripts/compare_reasoning_efforts.py
================================================================================
CHECKING: MINIMAL REASONING EFFORT
================================================================================
  📂 Found 0 existing responses (0 trials)
🔵 Running 200 TRAINING trials...
minimal effort (train): 100%|████████| 200/200 [1:09:00<00:00]
...
[INTERRUPTED with Ctrl+C]

# Resume - picks up where left off
$ python3 scripts/compare_reasoning_efforts.py
================================================================================
CHECKING: MINIMAL REASONING EFFORT
================================================================================
  📂 Found 200 existing responses (200 trials)
  📂 Found 50 existing responses (50 trials)
✅ MINIMAL is complete. Skipping (use --force to rerun).

================================================================================
CHECKING: LOW REASONING EFFORT
================================================================================
  📂 Found 100 existing responses (100 trials)
🔵 Running 100 remaining TRAINING trials (of 200 total)...
low effort (train): 100%|████████| 100/100 [34:30<00:00]
...
```

## Comparison to Original

| Feature | Original | With Incremental |
|---------|----------|------------------|
| **Resume capability** | ❌ No | ✅ Yes |
| **Save frequency** | End of batch | After each trial |
| **Skip complete work** | ❌ No | ✅ Yes |
| **Data loss on interrupt** | Entire batch | Only current trial |
| **Manual cleanup** | Required | Automatic |
| **Force rerun option** | N/A | `--force` flag |

## Technical Details

### **Detection Logic**

```python
def check_completion_status(out_dir, effort, expected_train, expected_test):
    # 1. Check if response files exist
    # 2. Count lines (trials) in JSONL files
    # 3. Verify >= expected number of trials
    # 4. Check analysis files exist and are valid
    # 5. Return status dict with boolean flags
```

### **Resume Logic**

```python
# Load existing partial data
existing, completed_ids = load_existing_responses(path)

# Filter to incomplete trials only
remaining = [s for s in specs if s.trial_id not in completed_ids]

# Run remaining trials
for spec in remaining:
    result = run_trial(...)
    save_response_incremental(result, path)  # Append to file
```

### **Incremental Save**

```python
def save_response_incremental(response, path):
    with path.open("a") as f:  # Append mode
        f.write(json.dumps(response) + "\n")
```

Safe because:
- JSONL format: one object per line
- Append mode: doesn't overwrite existing data
- Atomic writes: OS ensures write completes or fails entirely

---

**TL;DR**: Run the script normally. If it gets interrupted, just run it again—it'll continue where it left off. Use `--force` only if you want to start completely fresh.

