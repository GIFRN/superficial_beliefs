# Reasoning Effort Analysis: Why High Effort Leads to Lower Performance

## Executive Summary

High reasoning effort in GPT-5-mini produces **more balanced but less predictable** decision-making:
- **42% flatter** preference weights (less differentiation between attributes)
- **24% weaker** contribution magnitudes (less decisive)
- **137% more ambiguous** trials (33.2% vs 14.0% with close calls)
- **20% worse** model fit (log loss 0.42 vs 0.35)
- **4.4% worse** accuracy (82.2% vs 86.5%)

**The core finding**: More reasoning leads to weighing multiple factors more equally, making behavior harder to model with simple additive weights.

---

## Detailed Findings

### 1. **Preference Flattening** (Weight Distribution)

| Attribute | Minimal | High | Change |
|-----------|---------|------|--------|
| **Efficacy (E)** | 35.8% | 29.3% | **-6.5%** ⬇️ |
| **Adherence (A)** | 15.7% | 24.6% | **+8.9%** ⬆️ |
| **Safety (S)** | 23.7% | 27.6% | **+3.9%** ⬆️ |
| **Durability (D)** | 24.8% | 18.4% | **-6.3%** ⬇️ |

**Coefficient of Variation (CV = std/mean):**
- Minimal: 0.286 (strong differentiation)
- High: 0.166 (weak differentiation)
- **→ 42.1% flatter preferences with high effort**

**Interpretation**: Minimal effort shows a clear hierarchy (E >> others), while high effort spreads importance more evenly across attributes.

---

### 2. **Reduced Sensitivity** (Beta Coefficients)

Beta coefficients measure how much a 1-unit attribute difference affects log-odds of choosing option A:

| Attribute | Minimal β | High β | Ratio |
|-----------|-----------|--------|-------|
| **Efficacy (E)** | 2.832 | 1.859 | **0.66x** |
| **Adherence (A)** | 1.243 | 1.561 | **1.26x** |
| **Safety (S)** | 1.874 | 1.751 | **0.93x** |
| **Durability (D)** | 1.958 | 1.170 | **0.60x** |

**Average Magnitude:**
- Minimal: 1.977
- High: 1.585
- **→ 19.8% less sensitive to attribute differences**

**Interpretation**: High reasoning effort dampens the impact of any single attribute difference, requiring larger differences to sway decisions.

---

### 3. **Increased Ambiguity** (Contribution Patterns)

Analyzing how clearly one attribute "drives" each decision:

**Maximum Contribution Magnitude (decisiveness):**
- Minimal: mean = 3.561
- High: mean = 2.718
- **→ 23.7% lower decisiveness**

**Second-best / Best Contribution Ratio:**
- Minimal: mean = 0.513 (second attribute contributes ~51% as much as top)
- High: mean = 0.563 (second attribute contributes ~56% as much as top)
- **→ 9.7% more ambiguous**

**Very Ambiguous Trials (second-best >80% of best):**
- Minimal: 70 trials (14.0%)
- High: 166 trials (33.2%)
- **→ 137% more ambiguous trials**

**Interpretation**: High effort produces more trials where multiple attributes contribute nearly equally, making the "driver" less clear-cut.

---

### 4. **Predictive Performance Impact**

| Metric | Minimal | High | Degradation |
|--------|---------|------|-------------|
| **Log Loss** | 0.349 | 0.421 | **+20.6%** ⬆️ |
| **Brier Score** | 0.073 | 0.084 | **+14.0%** ⬆️ |
| **Accuracy** | 86.5% | 82.2% | **-4.4%** ⬇️ |

**Interpretation**: The flatter, more ambiguous decision-making is harder for a simple linear model to predict.

---

### 5. **Choice Patterns**

**Extreme vs Mixed Decisions:**

| Type | Minimal | High | Change |
|------|---------|------|--------|
| **All chose A** (≥95%) | 236 (47.2%) | 186 (37.2%) | -10.0% |
| **All chose B** (≤5%) | 164 (32.8%) | 171 (34.2%) | +1.4% |
| **Mixed** (5-95%) | 100 (20.0%) | 143 (28.6%) | **+8.6%** |

**Extreme Rate:**
- Minimal: 80.0% (very decisive)
- High: 71.4% (less decisive)

**Interpretation**: High effort leads to **43% more mixed decisions** (100 → 143), suggesting more uncertainty or consideration of trade-offs.

---

### 6. **Close Call Performance**

On trials with weak net preference (|sum of deltas| ≤ 1):

|  | Minimal | High |
|--|---------|------|
| **Sample size** | 324 | 324 |
| **Mean P(choose A)** | 0.594 | 0.511 |
| **Std P(choose A)** | 0.445 | 0.435 |

**Interpretation**: 
- Minimal effort shows a **bias** (59.4% choose A vs expected 50%)
- High effort is more **balanced** (51.1% ≈ 50%)
- This suggests high effort is more "rational" but also introduces more noise

---

## Concrete Mechanisms: Why Does This Happen?

### Hypothesis 1: **Multi-Factor Integration**
**Minimal effort** → Fast heuristic → Latch onto 1-2 salient attributes
**High effort** → Deep analysis → Consider all attributes more equally

**Evidence**:
- Flatter weights (all attributes matter more equally)
- More ambiguous drivers (no single factor dominates)
- Higher entropy in driver distribution

### Hypothesis 2: **Diminishing Confidence in Trade-offs**
**Minimal effort** → Confident snap judgments → Decisive choices
**High effort** → Aware of complexity → Less confident → More 50/50 splits

**Evidence**:
- 43% more mixed decisions (not all A or all B)
- Lower max contribution magnitudes
- Better calibration on close calls (0.511 vs 0.594)

### Hypothesis 3: **Context Sensitivity**
**Minimal effort** → Apply fixed weights consistently
**High effort** → Weight attributes differently based on context

**Evidence**:
- Same trials, different weight distributions
- More trial-specific ambiguity
- Harder to fit with simple additive model

### Hypothesis 4: **Overthinking Paradox**
**Minimal effort** → System 1 (intuitive) → Consistent utility function
**High effort** → System 2 (deliberative) → Inconsistent, context-dependent reasoning

**Evidence**:
- Worse fit despite more "thinking"
- Less predictable despite being more "rational"
- Similar to human overthinking leading to decision paralysis

---

## Implications for Research

### 1. **Measurement Challenge**
Simple additive utility models (Stage A's GLM) work better for **fast, intuitive responses** than for **deliberative reasoning**. This is consistent with:
- Dual process theory (Kahneman's System 1 vs System 2)
- Bounded rationality (Simon)
- Ecological rationality (Gigerenzer)

### 2. **Alignment Paradox**
High reasoning effort shows:
- ✅ Better "rationality" (98.6% on B1 dominant choices)
- ✅ Better calibration (closer to 50% on close calls)
- ❌ Worse alignment (30.3% vs 41.9% cite top weight)
- ❌ Worse predictability (log loss 0.42 vs 0.35)

**Interpretation**: More reasoning doesn't necessarily mean more alignment between stated and revealed preferences—it may just mean **more complex preferences**.

### 3. **Probe Instability**
The redact probe shows numerical instability with high effort because:
- High effort is more deterministic on small subsets
- This causes perfect/quasi-perfect separation in GLM
- Beta coefficients explode to astronomical values

**Recommendation**: Probe analysis needs:
- Larger sample sizes (current: ~50 redact trials)
- Regularization (ridge/Lasso) to stabilize estimates
- Detection of separation before reporting deltas

---

## Recommendations

### For Analysis
1. **Report both metrics**: Include both fast and deliberative reasoning results
2. **Use regularization**: Apply L2 penalty to probe models to prevent coefficient explosion
3. **Stratify by ambiguity**: Report results separately for clear vs ambiguous trials
4. **Model complexity**: Consider non-linear or context-dependent models for high-effort data

### For Interpretation
1. **Don't assume high effort = better**: Higher accuracy with minimal effort is meaningful
2. **Recognize trade-offs**: Deliberation improves rationality but reduces consistency
3. **Context matters**: High effort may be context-dependent in ways minimal effort is not

### For Future Work
1. **Vary reasoning effort systematically**: Test low/medium/high/extreme to find inflection points
2. **Measure response times**: See if high effort actually takes longer (via API)
3. **Qualitative analysis**: Examine actual reasoning text to understand what changes
4. **Human comparison**: Do humans show similar patterns with Think-Aloud protocols?

---

## Bottom Line

**High reasoning effort makes GPT-5-mini more "human-like" in an interesting way**: 

It becomes:
- ✅ More balanced in considering multiple factors
- ✅ More calibrated on uncertain cases
- ✅ More rational on obvious choices
- ❌ Less consistent in weighting attributes
- ❌ Harder to model with simple utility functions
- ❌ More prone to context-dependent reasoning

This mirrors how human deliberation can improve rationality while reducing consistency—a fundamental tension in decision theory.

The "worse performance" of high effort isn't necessarily a failure—it may reflect **genuine complexity** that our simple linear model can't capture.

