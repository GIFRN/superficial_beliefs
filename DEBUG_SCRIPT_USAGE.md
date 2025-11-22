# Debug Script: All Variations Pipeline Test

## Overview

`scripts/debug_all_variations.py` is a comprehensive debugging tool that tests the entire experimental pipeline from trial generation through Stage A and Stage B analysis.

## What It Does

The script:

1. **Creates 3 diverse B3 trial configurations** with different attribute profiles
2. **Tests ALL manipulation types** (16 total variations):
   - `short_reason` - baseline format
   - `split_reason` - separate context for reasoning
   - `premise_first` - premise before choice
   - `redact` - for each attribute (E, A, S, D)
   - `neutralize` - for each attribute (E, A, S, D)  
   - `inject` - with offsets (-1, 0, +1) for attribute A
3. **Collects responses** using an enhanced mock backend that provides varied, format-appropriate responses
4. **Performs Stage A analysis** - estimates attribute weights from choice patterns
5. **Performs Stage B analysis** - measures alignment between stated premises and actual decision drivers
6. **Prints comprehensive results** - no files saved, all output to stdout

## Usage

```bash
# From project root with venv activated
python3 scripts/debug_all_variations.py
```

No arguments required. The script runs in ~1 second and produces detailed output.

## Output Sections

### 1. Trial Configurations
Shows the 3 B3 trial profiles used for testing

### 2. Generated Variations
Lists all 16 trial×manipulation combinations

### 3. Responses Summary
For each trial, shows:
- ✅/❌ status for each conversation step (choice, sentence, premise)
- Parsed values (choice letter, premise attribute)

### 4. Stage A Analysis
- **Design Matrix**: Number of observations and features
- **Estimated Weights**: Normalized weights for E, A, S, D
- **Beta Coefficients**: Raw GLM coefficients
- **CV Metrics**: Cross-validation results (may be skipped for small samples)
- **Per-Trial Contributions**: Which attribute drove each trial's prediction

### 5. Stage B Analysis
- **ECRB Metrics**:
  - `ECRB_top1_driver`: Fraction where premise matches trial-specific driver
  - `ECRB_top1_weights`: Fraction where premise matches globally top-weighted attribute
  - `rank_corr`: Spearman correlation between premise frequencies and weights
- **Premise Distribution**: Count and percentage for each attribute

## Example Output

```
================================================================================
  STAGE A: WEIGHT ESTIMATION
================================================================================

Estimated Weights:
  E: 0.000 (β = -0.202)
  A: 0.000 (β = -7.983)
  S: 0.000 (β = -2.061)
  D: 1.000 (β = 3.550)

================================================================================
  STAGE B: ALIGNMENT METRICS
================================================================================

Alignment Metrics:
  ECRB (top1 driver):  0.062
  ECRB (top1 weights): 0.938
  Rank correlation:    0.816

Premise Distribution:
  E:  1 (  6.2%)
  A:  0 (  0.0%)
  S:  0 (  0.0%)
  D: 15 ( 93.8%)
```

## Use Cases

- **Pipeline Verification**: Confirm all manipulation types work end-to-end
- **Format Testing**: Verify prompt formats and parsing logic
- **Quick Sanity Check**: Test changes to analysis code without running full experiments
- **Development**: Debug new features in a controlled environment

## Limitations

- Uses mock backend (not real LLM responses)
- Small sample size (16 trials)
- Cross-validation may be skipped due to singular matrices
- Results are for testing only, not scientific analysis

## Customization

To modify the script:

1. **Add more trials**: Edit `create_sample_b3_trials()` to add more configurations
2. **Change response patterns**: Modify `EnhancedMockBackend.complete()` to alter mock responses
3. **Add manipulations**: Update `build_all_variations()` to include additional manipulation types
4. **Save output**: Add file writing code in the analysis sections

## Related Scripts

- `scripts/run_trials.py` - Full trial execution with real LLMs
- `scripts/fit_stageA.py` - Stage A analysis on collected data
- `scripts/stageB_alignment.py` - Stage B analysis on collected data
- `scripts/summarize.py` - Generate markdown reports

