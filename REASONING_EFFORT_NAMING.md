# Reasoning Effort in Output Directory Names

## Summary

All scripts now automatically include the `reasoning_effort` level in output directory names when applicable. This makes it easy to compare results across different reasoning effort settings.

## Changes Made

### 1. `run_trials.py` ✅
- Reads `reasoning_effort` from backend specification in `models.yml`
- Includes it in output directory name if present
- Saves it to `MANIFEST.json` for downstream scripts to use

**Example:**
```bash
# With reasoning_effort: minimal in models.yml
python3 scripts/run_trials.py
# → Saves to: data/runs/v1_short_openai_gpt5nano_minimal/

# Without reasoning_effort
python3 scripts/run_trials.py
# → Saves to: data/runs/v1_short_openai_gpt5nano/
```

### 2. `fit_stageA.py` ✅
- Reads `reasoning_effort` from `MANIFEST.json` in responses directory
- Includes it in output directory name if present
- Saves it to `stageA_summary.json`

**Example:**
```bash
python3 scripts/fit_stageA.py \
  --responses data/runs/v1_short_openai_gpt5nano_minimal/responses.jsonl
# → Saves to: results/stage_A_openai_gpt5nano_minimal/

# With interactions
python3 scripts/fit_stageA.py \
  --responses data/runs/v1_short_openai_gpt5nano_medium/responses.jsonl \
  --interactions
# → Saves to: results/stage_A_openai_gpt5nano_medium_interactions/
```

### 3. `stageB_alignment.py` ✅
- Reads `reasoning_effort` from `MANIFEST.json`
- Includes it in output directory name if present
- Saves it to `stageB_summary.json`

**Example:**
```bash
python3 scripts/stageB_alignment.py \
  --responses data/runs/v1_short_openai_gpt5nano_high/responses.jsonl \
  --stageA results/stage_A_openai_gpt5nano_high
# → Saves to: results/stage_B_openai_gpt5nano_high/
```

### 4. `summarize.py` ✅
- Reads `reasoning_effort` from Stage A summary
- Auto-detects all directory names with reasoning effort included

**Example:**
```bash
python3 scripts/summarize.py \
  --stageA results/stage_A_openai_gpt5nano_medium
# → Auto-detects:
#   Stage B: results/stage_B_openai_gpt5nano_medium/
#   Run dir: data/runs/v1_short_openai_gpt5nano_medium/
#   Report:  results/v1_short_openai_gpt5nano_medium/
```

### 5. `quick_ecrb_test.py` ✅
- Already implemented in previous change
- Includes reasoning effort in directory name by default

**Example:**
```bash
python3 scripts/quick_ecrb_test.py --reasoning-effort medium
# → Saves to: results/quick_ecrb_test_medium/
```

## Directory Naming Convention

### Format
```
{base_name}_{model}_{reasoning_effort}{suffix}
```

### Components
- **base_name**: e.g., `v1_short`, `stage_A`, `stage_B`, `quick_ecrb_test`
- **model**: Backend name from `models.yml` (e.g., `openai_gpt5nano`)
- **reasoning_effort**: If present (e.g., `minimal`, `medium`, `high`)
- **suffix**: Optional modifiers (e.g., `_interactions`)

### Examples

**Without reasoning effort:**
- `data/runs/v1_short_openai_gpt5nano/`
- `results/stage_A_openai_gpt5nano/`
- `results/stage_B_openai_gpt5nano_interactions/`

**With reasoning effort:**
- `data/runs/v1_short_openai_gpt5nano_minimal/`
- `results/stage_A_openai_gpt5nano_medium/`
- `results/stage_B_openai_gpt5nano_high_interactions/`
- `results/quick_ecrb_test_medium/`

## Configuration in models.yml

To use reasoning effort, add it to the backend specification:

```yaml
backends:
  openai_gpt5nano:
    type: openai
    model: gpt-5-nano
    temperature: 1.0
    max_tokens: 4000
    reasoning_effort: minimal  # ← Add this line
  
  openai_gpt5mini:
    type: openai
    model: gpt-5-mini
    temperature: 1.0
    max_tokens: 4000
    reasoning_effort: medium  # ← Or this
```

## Backward Compatibility

- ✅ If `reasoning_effort` is not specified, it is omitted from directory names
- ✅ All scripts work with both old and new directory structures
- ✅ Manual `--out` override still works in all scripts

## Complete Workflow Example

```bash
# 1. Set up models.yml with reasoning effort
# Edit configs/models.yml to add reasoning_effort: medium

# 2. Run trials
python3 scripts/run_trials.py
# → Saves to: data/runs/v1_short_openai_gpt5mini_medium/

# 3. Fit Stage A
python3 scripts/fit_stageA.py \
  --responses data/runs/v1_short_openai_gpt5mini_medium/responses.jsonl
# → Saves to: results/stage_A_openai_gpt5mini_medium/

# 4. Run Stage B
python3 scripts/stageB_alignment.py \
  --responses data/runs/v1_short_openai_gpt5mini_medium/responses.jsonl \
  --stageA results/stage_A_openai_gpt5mini_medium
# → Saves to: results/stage_B_openai_gpt5mini_medium/

# 5. Generate report
python3 scripts/summarize.py \
  --stageA results/stage_A_openai_gpt5mini_medium
# → Saves to: results/v1_short_openai_gpt5mini_medium/
```

## Comparison Workflow

```bash
# Run with minimal reasoning
# (Set reasoning_effort: minimal in models.yml)
python3 scripts/run_trials.py
python3 scripts/fit_stageA.py
python3 scripts/stageB_alignment.py
python3 scripts/summarize.py --stageA results/stage_A_openai_gpt5mini_minimal

# Run with medium reasoning
# (Change to reasoning_effort: medium in models.yml)
python3 scripts/run_trials.py
python3 scripts/fit_stageA.py
python3 scripts/stageB_alignment.py
python3 scripts/summarize.py --stageA results/stage_A_openai_gpt5mini_medium

# Compare results
cat results/v1_short_openai_gpt5mini_minimal/report.md
cat results/v1_short_openai_gpt5mini_medium/report.md
```

## Files Modified

1. `scripts/run_trials.py` - Add reasoning effort to output dir and MANIFEST
2. `scripts/fit_stageA.py` - Read from MANIFEST, add to output dir and summary
3. `scripts/stageB_alignment.py` - Read from MANIFEST, add to output dir and summary
4. `scripts/summarize.py` - Read from Stage A summary, auto-detect all dirs
5. `scripts/quick_ecrb_test.py` - Already had this feature

All changes maintain backward compatibility with existing code and data.

