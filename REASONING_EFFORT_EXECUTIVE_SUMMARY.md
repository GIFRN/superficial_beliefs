# Reasoning Effort Analysis: Executive Summary

## Key Finding

**Increasing reasoning effort makes GPT-5-mini's decisions more "human-like" but harder to predict.**

Medium reasoning effort appears to be a **sweet spot** for trial-specific alignment (57.2%), while high effort leads to flatter preferences and lower predictability.

---

## Quick Comparison Table

| Metric | Minimal | Medium | High | Trend |
|--------|---------|--------|------|-------|
| **Efficacy Weight** | 35.8% | 29.8% | 29.3% | ↓↓ Declining |
| **Adherence Weight** | 15.7% | 25.3% | 24.6% | ↑ Rising |
| **Weight CV** | 0.286 | 0.215 | 0.166 | ↓↓ Flattening |
| **Avg Beta Magnitude** | 1.977 | 1.415 | 1.585 | ↓ Weakening |
| **Accuracy** | 86.5% | — | 82.2% | ↓ Declining |
| **ECRB (driver)** | 44.4% | **57.2%** | 43.8% | ↑↓ Peaks at medium |
| **ECRB (weights)** | 41.9% | 38.8% | 30.3% | ↓↓ Declining |

---

## The Mechanisms: What Changes with More Reasoning?

### 1. **Preference Flattening** (-42% differentiation)
- **Minimal**: Clear hierarchy (E=36% >> A=16%)
- **High**: Much flatter (E=29% ≈ A=25%)
- More deliberation → more equal consideration of all factors

### 2. **Reduced Sensitivity** (-20% beta magnitude)
- **Minimal**: 1 unit difference → large impact (β̄=1.98)
- **High**: 1 unit difference → smaller impact (β̄=1.59)
- More thinking → requires bigger differences to sway decision

### 3. **Increased Ambiguity** (+137% ambiguous trials)
- **Minimal**: 14% of trials have close runner-up attributes
- **High**: 33% of trials have close runner-up attributes
- Deeper analysis → more attributes seem equally important

### 4. **Mixed Decisions** (+43% increase)
- **Minimal**: 80% extreme agreement (all A or all B)
- **High**: 71% extreme agreement, 29% mixed
- More uncertainty → less decisive responses

---

## Why Does Performance Degrade?

Our **simple additive model** (Stage A GLM) assumes:
```
P(choose A) = logit⁻¹(β_E·ΔE + β_A·ΔA + β_S·ΔS + β_D·ΔD)
```

This works well for **minimal effort** because:
- ✅ Preferences are stable and consistent
- ✅ Decisions follow clear attribute hierarchy
- ✅ Behavior matches simple utility maximization

This fails for **high effort** because:
- ❌ Preferences become context-dependent
- ❌ Multiple attributes matter equally
- ❌ Trade-offs are weighed case-by-case

**The model hasn't gotten worse—the behavior has gotten more complex.**

---

## The Sweet Spot: Medium Reasoning Effort

**Medium effort shows the best trial-specific alignment (57.2%)**:
- Still decisive enough to model (CV=0.215)
- Thoughtful enough to match drivers
- Balanced consideration without overthinking

This suggests there's an **optimal reasoning level** for alignment between stated and revealed preferences.

---

## Concrete Numbers

### Choice Decisiveness
| | Minimal | High | Change |
|---|---|---|---|
| **All chose A** | 236 (47%) | 186 (37%) | -10% |
| **All chose B** | 164 (33%) | 171 (34%) | +1% |
| **Mixed (5-95%)** | 100 (20%) | 143 (29%) | **+43%** |

### Contribution Patterns
| | Minimal | High | Change |
|---|---|---|---|
| **Max contribution** | 3.561 | 2.718 | -24% |
| **Ambiguity ratio** | 0.513 | 0.563 | +10% |
| **Very ambiguous trials** | 70 (14%) | 166 (33%) | **+137%** |

### Predictive Performance
| | Minimal | High | Change |
|---|---|---|---|
| **Log Loss** | 0.349 | 0.421 | +20.6% worse |
| **Brier Score** | 0.073 | 0.084 | +14.0% worse |
| **Accuracy** | 86.5% | 82.2% | -4.4% worse |

---

## Implications

### For AI Alignment Research
1. **Simple models may fail for deliberative AI**: As models get more sophisticated, their behavior may become too complex for additive utility models
2. **Alignment isn't monotonic with reasoning**: More thinking ≠ better alignment
3. **Context-dependence increases**: High effort decisions seem more case-by-case

### For LLM Application
1. **Fast responses may be more predictable**: For systems requiring consistent behavior, lower reasoning effort may work better
2. **Deliberation has trade-offs**: Better rationality but worse consistency
3. **Medium effort is a sweet spot**: For applications needing both consistency and thoughtfulness

### For Human Decision Research
This mirrors known patterns in human decision-making:
- **System 1 (fast)**: Consistent, predictable, heuristic-driven
- **System 2 (slow)**: Context-dependent, complex, harder to model
- **The deliberation paradox**: Thinking more can reduce consistency

---

## Visualizations Available

Generated visualizations show:
1. **Weight comparison** (`reasoning_effort_weights.png`)
   - Bar charts showing weight shifts
   - Difference plot highlighting changes

2. **Ambiguity distribution** (`reasoning_effort_ambiguity.png`)
   - Decisiveness histograms
   - Ambiguity ratio distributions

3. **Driver distribution** (`reasoning_effort_drivers.png`)
   - Frequency and percentage of each driver attribute
   - Comparison across conditions

All saved in `results/visualizations/`

---

## Bottom Line

**High reasoning effort doesn't make the model "worse"—it makes it more complex:**

| Aspect | Minimal Effort | High Effort |
|--------|----------------|-------------|
| **Type** | Fast, intuitive | Deliberative, analytical |
| **Preferences** | Clear hierarchy | Balanced consideration |
| **Consistency** | High (86.5% accuracy) | Lower (82.2% accuracy) |
| **Alignment** | Moderate (42% global) | Poor (30% global) |
| **Rationality** | Good (B1 check) | Excellent (98.6% B1) |
| **Predictability** | High (log loss 0.35) | Low (log loss 0.42) |

The "degraded performance" reflects **genuine increased complexity** that our simple model can't capture—not a failure of the LLM.

This is theoretically important: it suggests that as AI systems become more sophisticated and deliberative, we may need more sophisticated models to understand and align them.

---

## Files Generated

1. **`REASONING_EFFORT_ANALYSIS.md`** - Detailed technical analysis (6 sections, 400+ lines)
2. **`REASONING_EFFORT_EXECUTIVE_SUMMARY.md`** - This file (quick reference)
3. **`scripts/visualize_reasoning_effort.py`** - Visualization generation script
4. **`results/visualizations/*.png`** - Three comparison plots

## Next Steps

1. **Test intermediate levels**: Try reasoning_effort='low' to map the full curve
2. **Examine actual reasoning text**: Qualitative analysis of what changes
3. **Try different models**: Does this pattern hold for o1, GPT-4, Claude?
4. **Non-linear models**: Fit decision trees or neural nets to high-effort data
5. **Context features**: Add interaction terms to capture context-dependence

