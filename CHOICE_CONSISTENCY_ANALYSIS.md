# Choice Consistency Analysis: How Necessary Are Multiple Samples?

## Summary

Analysis of **50 trials with 5 replicates each** (250 total samples) from `quick_ecrb_test` using `gpt-5-mini` with `reasoning_effort=medium`.

### Key Findings

**🎯 Models are highly consistent across samples:**
- **83.7%** of trials (36/43) show **perfect consistency** - all 5 samples chose the same option
- **16.3%** of trials (7/43) showed any choice variation
- **Mean consistency rate: 95.8%** (how often the dominant choice was selected)

### Detailed Statistics

| Metric | Value |
|--------|-------|
| Total trials analyzed | 43 |
| Trials with 100% consistency | 36 (83.7%) |
| Trials with choice switches | 7 (16.3%) |
| Mean consistency rate | 0.958 |
| Median consistency rate | 1.000 |
| Minimum consistency rate | 0.600 (3/5 choices) |

### Consistency Rate Distribution

| Consistency Rate | Number of Trials | Percentage |
|------------------|------------------|------------|
| 60% (3/5) | 2 | 4.7% |
| 80% (4/5) | 5 | 11.6% |
| 100% (5/5) | 36 | 83.7% |

### Examples of Inconsistent Trials

Only **2 trials** showed substantial variation (60% consistency):

**Trial T-00267:**
- 3 chose B, 2 chose A
- Consistency: 60%

**Trial T-00293:**
- 3 chose B, 2 chose A  
- Consistency: 60%

Five additional trials showed minor variation (80% consistency):
- T-00148, T-00149, T-00151, T-00189, T-00281
- Pattern: 4 chose one option, 1 chose the other

## Implications

### 1. **High Baseline Consistency**

With temperature=1.0 and reasoning_effort=medium, gpt-5-mini is remarkably consistent:
- 83.7% of trials produce identical choices across all samples
- When variation exists, it's typically 4/5 agreement, not random

### 2. **When Multiple Samples Help**

Multiple samples are most valuable when:
- **Estimating choice probabilities**: Even highly consistent models benefit from multiple samples to distinguish 95% vs 100% preference
- **Detecting weak preferences**: The 16.3% of cases with switches likely represent genuinely close decisions
- **Statistical power**: With only 1 sample, you can't distinguish determinism from high consistency

### 3. **Sample Size Recommendations**

Based on these results:

| Use Case | Recommended S | Rationale |
|----------|---------------|-----------|
| **Binary classification** | S=1-3 | High consistency means single samples are often sufficient |
| **Probability estimation** | S=5-10 | Need to distinguish 80%, 90%, 95%, 100% |
| **Close decisions detection** | S=10-20 | Better resolution for marginal cases |
| **Research/validation** | S=20+ | Gold standard for full characterization |

### 4. **Cost-Benefit Analysis**

With 83.7% perfect consistency:
- **Going from S=1 to S=5**: Captures the 16.3% variable cases, useful for probability estimation
- **Going from S=5 to S=10**: Diminishing returns - mostly refining estimates
- **Going from S=10 to S=20**: Primarily for statistical rigor, unlikely to change conclusions

## Current Configuration Review

### Existing Runs

| Run | S | Backend | Status |
|-----|---|---------|--------|
| `v1_short` | 1 | gpt-5-mini | Complete |
| `v1_short_openai_gpt5mini` | 10 | gpt-5-mini | Incomplete |
| `v1_short_openai_gpt5mini_high` | 10 | gpt-5-mini | Incomplete |
| `v1_short_openai_gpt5nano` | 3 | gpt-5-nano | Incomplete |
| `quick_ecrb_test` | 5 | gpt-5-mini (medium) | Complete |

### Recommendation

**For your research goals:**

Given the high consistency (95.8% mean), and that your analysis focuses on:
1. **Attribute weights** (Stage A) - benefits from probability estimation
2. **ECRB metrics** (Stage B) - needs to assess reasoning-choice alignment

**Suggested approach:**
- **S=10** is a good balance for your current analysis
  - Provides reliable probability estimates
  - Captures the 16.3% of variable cases
  - Sufficient statistical power for comparing conditions
  
- **Alternative: S=5** for initial exploration
  - Cuts API costs in half
  - Still captures most variation (as shown in this analysis)
  - Consider for preliminary runs or large-scale experiments

## Technical Details

### Analysis Method

Script: `scripts/analyze_choice_consistency.py`

Metrics computed:
- **Consistency rate**: Fraction of samples that chose the dominant option
- **Entropy**: Information-theoretic measure of randomness (0 = perfect consistency)
- **Distribution**: Full count of each option chosen

### Data Source

- **File**: `results/quick_ecrb_test/responses.jsonl`
- **Model**: gpt-5-mini
- **Reasoning effort**: medium
- **Temperature**: 1.0
- **Trials**: 50 non-B1 samples
- **Replicates per trial**: 5

### Limitations

1. **Small sample**: Only 50 trials analyzed (though 250 total choices)
2. **Single model**: Only tested gpt-5-mini with medium reasoning effort
3. **Non-B1 trials**: B1 trials (Pareto-optimal) would likely show even higher consistency
4. **Single temperature**: Temperature=1.0 tested; lower temperatures would increase consistency

## Reproducibility

To run this analysis on new data:

```bash
# Run trials with desired S value
python scripts/run_trials.py --dataset data/generated/v1_short_400

# Analyze choice consistency  
python scripts/analyze_choice_consistency.py \
  --responses data/runs/[run_dir]/responses.jsonl \
  --examples 20 \
  --output consistency_results.csv
```

## Conclusion

**Multiple samples are valuable but show diminishing returns:**

- The model is highly consistent (95.8% mean consistency)
- Most variation occurs in a small subset of trials (16.3%)
- **S=5-10** provides a good balance between cost and information
- **S=1-3** may suffice for binary classification tasks
- **S=20+** is overkill unless you need very precise probability estimates

For your current research with 400 trials, **S=10 is recommended** as it provides:
- Reliable probability estimation for attribute weights
- Adequate power to detect reasoning-choice misalignments
- Reasonable API costs (4,000 total samples vs 8,000 for S=20)

