# Train/Test Split Implementation

## Overview

The `compare_reasoning_efforts.py` script now implements **train/test splitting** for robust validation of Stage A (behavioral model) and Stage B (enthymeme alignment).

## What Changed?

### **Default Configuration**
- **Training set**: 200 trials (40 from B2, 160 from B3)
- **Test set**: 50 trials (from B3 only)
- **Total**: 250 trials per reasoning effort level

### **Key Features**

1. **Separate Training & Testing**: Model is fitted on training trials, then evaluated on held-out test trials
2. **Block Mixing**: Training set includes trials from both B2 (20%) and B3 (80%) for diversity
3. **No Overlap**: Test trials are guaranteed to be distinct from training trials
4. **Dual Analysis**: Both in-sample (training) and out-of-sample (test) metrics reported

## Command-Line Arguments

### **New Arguments**

```bash
--n-train N           # Number of training trials (default: 200)
--n-test N            # Number of test trials (default: 50, set 0 to disable split)
--b2-fraction F       # Fraction of training from B2 (default: 0.2 = 20%)
```

### **Examples**

```bash
# Standard usage (200 train, 50 test)
python3 scripts/compare_reasoning_efforts.py

# Larger training set
python3 scripts/compare_reasoning_efforts.py --n-train 400 --n-test 100

# No train/test split (backward compatible)
python3 scripts/compare_reasoning_efforts.py --n-train 250 --n-test 0

# Custom block mixing (50% B2, 50% B3)
python3 scripts/compare_reasoning_efforts.py --b2-fraction 0.5
```

## What Gets Tested?

### **Training Set Analysis**
1. **Stage A**: Fit GLM on 200 trials → estimate attribute weights
2. **Stage B**: Compute alignment metrics on same 200 trials (in-sample)
3. **Choice patterns**: Variance, extreme choice rates

### **Test Set Analysis** (NEW)
1. **Prediction Accuracy**: How well does the trained model predict held-out choices?
   - **MAE** (Mean Absolute Error): Average difference between predicted and actual P(choose A)
   - **RMSE** (Root Mean Square Error): Penalizes large errors
   - **Correlation**: Pred vs. actual choice probabilities
   - **Accuracy**: Discrete prediction (>0.5 → A) vs. actual majority choice

2. **Out-of-Sample Alignment**: Do stated premises align with behavioral model in NEW trials?
   - **ECRB_top1_driver**: % matches to trial-specific top driver
   - **ECRB_top1_weights**: % matches to global top weighted attribute
   - **Rank correlation**: Premise frequency ↔ attribute weights

## Interpretation Guide

### **What Good Results Look Like**

✅ **Prediction Metrics (Test Set)**
- MAE < 0.15: Model predicts choices well
- RMSE < 0.20: Errors are not extreme
- Correlation > 0.80: Strong linear relationship
- Accuracy > 0.85: Discrete predictions mostly correct

✅ **Alignment Metrics**
- **In-sample ≈ Out-of-sample**: Alignment is stable across trials
- Both > 0.40: Moderate-to-strong alignment
- If in-sample >> out-of-sample: Possible overfitting or trial-specific effects

### **What to Watch For**

⚠️ **Poor Prediction**
- MAE > 0.25 or Correlation < 0.60 → GLM doesn't generalize
- Possible causes: Overfitting, model misspecification, noisy data

⚠️ **Alignment Divergence**
- In-sample = 0.60, out-of-sample = 0.30 → Alignment is trial-specific, not systematic
- Suggests stated reasoning varies more than behavioral patterns

⚠️ **Poor Accuracy but Good Alignment**
- Model can't predict choices well, but premises align with weights
- Suggests: Behavioral model is weak, but LLM still states consistent reasons

## Technical Details

### **Sampling Logic**

```python
# Training: Mix B2 and B3
n_b2 = int(n_train * b2_fraction)  # 200 * 0.2 = 40
n_b3_train = n_train - n_b2         # 200 - 40 = 160

# Test: Only from B3 (held-out)
n_test = 50  # From remaining B3 trials
```

**Why mix B2 and B3 in training?**
- B2 has different trial characteristics (difficulty, attribute ranges)
- Mixing increases model robustness and generalizability
- B3-only test ensures clean held-out evaluation

### **Prediction Implementation**

For each test trial:
1. Build design matrix with trial features (delta_E, delta_A, delta_S, delta_D)
2. Apply trained β coefficients: `linear_pred = X @ β`
3. Convert to probability: `P(choose A) = 1 / (1 + exp(-linear_pred))`
4. Compare to actual: `P_actual = successes / trials`

### **File Organization**

**Training outputs:**
- `responses_{effort}_train.jsonl`: Raw LLM responses (training)
- `results_{effort}_train.json`: Training metrics

**Test outputs:**
- `responses_{effort}_test.jsonl`: Raw LLM responses (test)
- `results_{effort}_test.json`: Test metrics + predictions

**Summary:**
- `comparison_summary.json`: Combined results across all effort levels

## Research Implications

### **Advantages of Train/Test Split**

1. **Stronger Validity**: Out-of-sample validation is gold standard
2. **Generalization Test**: Shows if alignment is systematic, not spurious
3. **Model Diagnosis**: Can separate model quality from alignment quality
4. **Publication Ready**: Meets reviewer expectations for ML-based claims

### **What This Tests**

**Research Question**: "Can behavioral patterns predict what LLMs will state as their reasoning?"

- **In-sample alignment**: Within-trial consistency ("Do they state what they did?")
- **Out-of-sample alignment**: Cross-trial generalizability ("Can we predict what they'll say from what they do elsewhere?")

The second question is **theoretically stronger** - if behavioral patterns in trials 1-200 predict stated reasoning in trials 201-250, this suggests alignment is a stable property of the model, not just trial-specific post-hoc rationalization.

## Comparison to Previous Version

| Aspect | Previous (v1) | Current (v2) |
|--------|--------------|--------------|
| **Trials** | 100 from B3 | 200 train (B2+B3) + 50 test (B3) |
| **Analysis** | In-sample only | In-sample + out-of-sample |
| **Metrics** | Alignment only | Alignment + prediction accuracy |
| **Validation** | None | Train/test split |
| **Interpretation** | "States match behavior" | "Behavior predicts statements" |

## Backward Compatibility

To replicate the old behavior:

```bash
python3 scripts/compare_reasoning_efforts.py --n-train 100 --n-test 0 --b2-fraction 0.0
```

This runs 100 B3 trials with no train/test split.

## Recommendations

### **For Exploratory Analysis**
- Use default settings (200/50 split)
- Check both train and test metrics
- If test metrics are much worse, increase training size

### **For Publication**
- Use 400 train / 100 test for higher power
- Report both in-sample and out-of-sample
- Include confidence intervals (future enhancement: bootstrap)

### **For Debugging**
- Use small samples: `--n-train 20 --n-test 5`
- Set `--n-test 0` to disable split temporarily
- Check prediction accuracy first - if poor, investigate GLM before alignment

## Future Enhancements

Potential additions:
- [ ] Cross-validation (k-fold) for more robust estimates
- [ ] Bootstrap confidence intervals on alignment metrics
- [ ] Stratified sampling (balance trial difficulty in train/test)
- [ ] Cross-reasoning-effort generalization (train on minimal, test on high)

---

**TL;DR**: The script now trains behavioral models on 200 trials and tests on 50 held-out trials, providing robust out-of-sample validation of both choice prediction and enthymeme alignment. This strengthens the evidence that stated reasoning aligns with behavioral patterns systematically, not just coincidentally.

