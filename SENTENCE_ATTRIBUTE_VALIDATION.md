# Sentence-Attribute Consistency Validation

## Research Question

**Is it necessary to have the model write both:**
1. A sentence about why it made its choice
2. A single-word attribute identification

**Or can we just use one and infer the other?**

## Key Findings

### Overall Consistency: 84.4%

Out of 250 samples analyzed (50 trials × 5 replicates):
- **Consistent**: 211 samples (84.4%) - sentence keywords match stated attribute
- **Inconsistent**: 39 samples (15.6%) - sentence discusses different attribute(s)

### Critical Finding: Attribute Matters!

Consistency varies **dramatically** by attribute:

| Attribute | Consistency Rate | Observation |
|-----------|-----------------|-------------|
| **Safety (S)** | 96.2% (51/53) | ✅ Nearly perfect |
| **Efficacy (E)** | 94.9% (75/79) | ✅ Highly consistent |
| **Durability (D)** | 93.3% (28/30) | ✅ Highly consistent |
| **Adherence (A)** | **41.5% (22/53)** | ⚠️ **Major problem!** |

## The Adherence Problem

### Pattern of Inconsistency

When the model identifies **Adherence** as the key attribute, it frequently:
- **States**: "Adherence" (attribute A)
- **But writes**: Sentences about "effectiveness" or "efficacy"

### Examples:

**Inconsistent Case 1:**
```
Sentence: "High adherence ensures sustained therapeutic exposure, 
           maximizing long-term effectiveness."
Stated Attribute: A (Adherence)
Keywords Found: "effectiveness" → maps to E (Efficacy)
```

**Inconsistent Case 2:**
```
Sentence: "High adherence maximizes real-world effectiveness over five years."
Stated Attribute: A (Adherence)
Keywords Found: "effectiveness" → maps to E (Efficacy)
```

**Inconsistent Case 3:**
```
Sentence: "High adherence ensures sustained treatment effectiveness, 
           maximizing five-year patient outcomes."
Stated Attribute: A (Adherence)
Keywords Found: "effectiveness" → maps to E (Efficacy)
```

### Why This Happens

The model appears to understand the **causal chain**:
```
Adherence → Effectiveness → Outcomes
```

When asked for the "most important factor," it identifies **adherence** (the root cause), but when explaining *why*, it talks about **effectiveness** (the proximal benefit).

This is conceptually coherent but creates apparent "inconsistency" in simple keyword matching.

## What the Redundancy Reveals

### Information Gained from Having Both

Having both the sentence and attribute label actually captures two different aspects:

1. **Attribute label** (single word): The **root factor** the model prioritizes
2. **Sentence reasoning**: The **mechanism** or **downstream effect** the model emphasizes

For adherence cases:
- **Attribute**: Adherence (the decision factor)
- **Sentence**: Effectiveness (the reason adherence matters)

This is NOT redundant - it's capturing:
- **WHAT** the model valued (adherence)
- **WHY** the model valued it (because → effectiveness)

### Cases Where They're Redundant

For Safety (96.2%), Efficacy (94.9%), and Durability (93.3%), the sentence and attribute are highly consistent because:
- These attributes are more **directly** valued
- No intermediate causal chain needs to be explained
- The attribute *is* the benefit

Example (consistent):
```
Sentence: "High safety reduces long-term harm over five years."
Stated Attribute: S (Safety)
Keywords: "safety" → S ✅
```

## Implications for Experimental Design

### Option 1: Keep Both (Recommended)

**Advantages:**
- Captures richer information about reasoning chains
- Reveals when models prioritize mechanisms vs outcomes
- The 15.6% "inconsistent" cases are actually informative
- Helps distinguish direct valuation from causal reasoning

**Disadvantages:**
- More API calls
- More parsing complexity
- Requires interpretation of apparent inconsistencies

### Option 2: Drop the Attribute Label, Keep Sentence

**Advantages:**
- Single query step
- Reduces API costs
- Sentences contain the reasoning

**Disadvantages:**
- **Loses 58.5% of adherence attributions** (would be miscoded as "effectiveness")
- Can't distinguish "valued adherence because → effectiveness" from "valued effectiveness directly"
- Makes it harder to compare attribute weights across trials

### Option 3: Drop the Sentence, Keep Attribute

**Advantages:**
- Simple, unambiguous
- Clear attribute attribution
- Sufficient for basic weight estimation

**Disadvantages:**
- Loses mechanistic reasoning
- Can't validate that the model actually understands *why* the attribute matters
- Miss cases where model conflates related attributes

## Recommendation

**KEEP BOTH** for at least the initial analysis, because:

1. **The "inconsistencies" are informative**, not noise:
   - They reveal how the model reasons causally
   - 58.5% of adherence cases would be misattributed without the explicit label
   - This affects Stage A weight estimates significantly

2. **You need the sentence for Stage B anyway** (ECRB analysis):
   - Stage B alignment metrics depend on the reasoning text
   - So you're collecting it regardless
   - Might as well use it to validate Stage A

3. **The redundancy provides a validation check**:
   - High consistency (>90%) for S, E, D confirms reliable measurement
   - Low consistency (41.5%) for A flags a conceptual issue
   - This is valuable quality control

### Alternative: Simplified Version for Cost Reduction

If you need to cut costs, you could:
- **Pilot phase**: Use both to understand the consistency patterns
- **Main study**: Use attribute label only, with the understanding that:
  - Adherence estimates may conflate adherence → effectiveness reasoning
  - You lose some mechanistic insight
  - ECRB analysis would need to collect sentences anyway

## Statistical Impact

### Impact on Stage A Weights

If you drop the explicit attribute label and rely only on sentence keywords:

**Adherence weight would be underestimated by ~58.5%**

Current distribution (with labels):
- Adherence: 21.2% of attributions

Estimated distribution (sentence keywords only):
- Adherence: ~9% of attributions (losing most to "effectiveness")
- Efficacy: ~35% of attributions (gaining adherence → effectiveness cases)

This would substantially bias your conclusions about which attributes matter most!

## Validation Script

Created: `scripts/validate_sentence_attribute_consistency.py`

Usage:
```bash
python scripts/validate_sentence_attribute_consistency.py \
  --responses results/quick_ecrb_test/responses.jsonl \
  --output consistency_analysis.csv
```

## Conclusion

**The sentence and attribute label are NOT redundant for all attributes:**

- ✅ **Highly redundant** for Safety, Efficacy, Durability (>93% agreement)
- ⚠️ **Importantly different** for Adherence (41.5% agreement)

The divergence reveals that adherence is often valued **instrumentally** (as a means to effectiveness), not **terminally** (for its own sake). This is substantively interesting and would be lost if you only collected one piece of information.

**Recommended approach**: Keep both, and treat the adherence inconsistency as a feature, not a bug. It tells you something real about how the model reasons about adherence vs other attributes.

