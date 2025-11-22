# Bug Analysis: B1 Validation and Data Encoding

## Summary

The **B1 rationality validation is fundamentally flawed**, but the **core Stage A/B analysis is actually correct**. The issue is that B1 validation doesn't account for trial orientation.

## The Encoding System (CORRECT)

The codebase uses a consistent encoding:
- **Deltas**: Always computed as `A - B` (where A/B are the labeled options)
- **Successes**: Count of times option A was chosen
- **Model**: Fits P(choose A | deltas)

### Example from Data
```
Trial T-00001: labelA='A', delta_E=-1
  → A has LOWER efficacy than B
  → B is dominant
  → Model should choose B (P(A) ≈ 0)

Trial T-00004: labelA='B', delta_E=+1  
  → A has HIGHER efficacy than B
  → A is dominant
  → Model should choose A (P(A) ≈ 1)
```

## The Bug: B1 Validation

### Problem Code (`src/analysis/diagnostics.py:37-63`)

```python
def validate_b1_rationality(trials_df, choice_agg):
    # Check if all B1 trials have P(choose A) ≥ 0.95
    probabilities = b1_trials["successes"] / b1_trials["trials"]
    rationality_failures = probabilities < 0.95  # ← BUG!
```

**The bug**: Assumes P(choose A) should always be ≥ 0.95, regardless of which option is dominant.

### Correct Logic Should Be:

```python
# For each trial, determine which option is dominant
for each trial:
    if all deltas >= 0:  # A is dominant
        expect P(choose A) >= 0.95
    elif all deltas <= 0:  # B is dominant
        expect P(choose A) <= 0.05
    else:  # Mixed dominance
        skip validation
```

## Impact Assessment

### 1. ✅ Stage A Model Fitting - **CORRECT**
- Excludes B1 trials (`exclude_b1=True`)
- Fits P(choose A) as function of deltas
- Positive deltas → higher P(A)
- Negative deltas → lower P(A)
- **NO BUG HERE**

### 2. ❌ B1 Rationality Check - **WRONG**
- Reports 50% failure rate
- **Actually**: Model has 100% accuracy on dominant choices
- All "failures" are cases where B is dominant and model correctly chose B

### 3. ⚠️ B1 Probe Validation - **POTENTIALLY AFFECTED**
(`src/analysis/diagnostics.py:66-104`)

```python
baseline_prob = baseline["successes"] / baseline["trials"]
manipulated_prob = manipulated["successes"] / manipulated["trials"]
probe_effect = float(baseline_prob.mean() - manipulated_prob.mean())
```

**Issue**: Averages P(choose A) across trials with different orientations
- Half have A dominant (should choose A)
- Half have B dominant (should choose B)
- The average of [1.0, 1.0, 0.0, 0.0] = 0.5, which is meaningless

**Result**: Reports "insufficient data" because baseline and manipulated groups likely have different orientation distributions, making the probe effect calculation unreliable.

### 4. ✅ Stage B Alignment - **CORRECT**
- Uses premise attribution (not choice correctness)
- Compares which attribute model cites vs. what fitted model shows
- **NO ORIENTATION ISSUES**

### 5. ✅ Probe Deltas - **CORRECT**
- Refits models on manipulation subsets
- Compares beta coefficients
- Doesn't depend on orientation
- **NO ISSUES**

## Required Fixes

### High Priority

1. **Fix `validate_b1_rationality`**
   - Check dominant option for each trial
   - Validate P(choose dominant) >= 0.95

2. **Fix `validate_b1_probes`**
   - Compute dominance for each trial
   - Compare P(choose dominant) between baseline/manipulated
   - Or stratify by orientation

### Medium Priority

3. **Add orientation-aware metrics to reports**
   - Report P(correct dominant choice) instead of P(choose A)
   - Add B1 accuracy by attribute and manipulation type

### Low Priority

4. **Documentation**
   - Clarify delta encoding in docstrings
   - Document that "successes" means "chose A" not "chose correct"

## Files Requiring Changes

1. `src/analysis/diagnostics.py` - Both validation functions
2. `scripts/inspect_b1_failures.py` - Already shows dominant option (GOOD!)
3. `src/analysis/reporting.py` - Update metrics interpretation
4. Documentation/README - Clarify encoding

## Current Status

The GPT-5-mini model shows:
- **100% accuracy** on B1 dominant-choice tasks (not 50%)
- **86% accuracy** on non-B1 trials (Stage A)
- **44% alignment** between stated and revealed preferences (Stage B)

The core findings are valid, but B1 validation is completely wrong.

