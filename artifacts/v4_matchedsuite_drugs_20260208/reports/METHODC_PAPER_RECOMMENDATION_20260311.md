# Method C Paper Recommendation

Generated (UTC): 2026-03-11 15:15:30

## Recommendation

Method C should be framed as an intervention-based validation of the revealed-preference story, not as another benchmark row beside the main holdout results.

The strongest question it answers is:

> If the prompt-visible evidence for attribute `X` is selectively neutralized or removed, does the model's choice and stated premise move in the direction predicted by the latent preference ranking inferred from baseline choices?

That framing matches the actual construction of the occlusion suite and avoids overclaiming from a single analyzed model.

## What Is Justified Now

The completed `GPT-5-mini minimal tau` Method C run already supports a compact but real result.

### 1. The intervention ranking exactly matches the baseline revealed-preference ranking.

From the baseline-only `short_reason` rows, Stage A gives:

- weight order: `E > D > S > A`
- normalized weights: `E=0.327`, `D=0.255`, `S=0.238`, `A=0.180`
- baseline-only Stage A accuracy: `0.873`

The intervention effect ranking is identical for both manipulation types:

| manipulation | normalized effect order | Spearman vs baseline weights |
| --- | --- | --- |
| occlude_drop | `E > D > S > A` | `1.000` |
| occlude_equalize | `E > D > S > A` | `1.000` |

This is the cleanest Method C result.

### 2. Occluding a more important attribute produces larger behavioral disruption.

Orientation-corrected choice effects (`delta_favored_mean`) are monotone in the same order:

| manipulation | E | D | S | A |
| --- | --- | --- | --- | --- |
| occlude_drop | `-0.458` | `-0.354` | `-0.306` | `-0.232` |
| occlude_equalize | `-0.466` | `-0.355` | `-0.310` | `-0.238` |

Choice-flip rates show the same pattern:

| manipulation | E | D | S | A |
| --- | --- | --- | --- | --- |
| occlude_drop | `0.386` | `0.280` | `0.277` | `0.210` |
| occlude_equalize | `0.366` | `0.274` | `0.266` | `0.204` |

Interpretation: removing evidence about the most important attribute changes choices the most.

### 3. Premises move with the intervention, especially for high-weight attributes.

Premise-flip rates also track the latent ranking:

| manipulation | E | D | S | A |
| --- | --- | --- | --- | --- |
| occlude_drop | `0.559` | `0.376` | `0.356` | `0.296` |
| occlude_equalize | `0.528` | `0.371` | `0.323` | `0.276` |

For the targeted attribute, premise shifts are always away from the target and never toward it in the current summaries.

This is important because it shows the intervention is not only changing the final choice; it is changing the model's stated reason in the predicted direction.

### 4. Choice changes and premise changes are tightly coupled.

Across attributes and both manipulation types:

- `P(choice flip | premise flip)` is high: about `0.68` to `0.80`
- `P(choice flip | no premise flip)` is near zero: about `0.003` to `0.010`
- the share of choice flips accompanied by a premise flip is about `0.975` to `0.992`

This supports a strong "full-chain" statement at the descriptive level:

> attribute occlusion shifts stated premises, and those premise shifts almost entirely account for observed choice flips.

This should still be described as a mediation-style pattern rather than a formal identified mediation effect.

### 5. `occlude_drop` and `occlude_equalize` behave very similarly.

For every attribute, `equalize - drop` differences are small:

| attribute | directional effect | choice-flip rate | premise-flip rate |
| --- | --- | --- | --- |
| E | `-0.008` | `-0.020` | `-0.031` |
| D | `-0.001` | `-0.006` | `-0.004` |
| S | `-0.004` | `-0.011` | `-0.032` |
| A | `-0.006` | `-0.006` | `-0.020` |

So for Results writing, the two manipulation types can be discussed together unless you specifically want an appendix note about their near-equivalence.

### 6. Effect size scales with intervention magnitude.

Example for `E`:

- `occlude_drop`, magnitude `1`: `delta_favored_mean=-0.352`, `choice_flip_rate=0.505`
- `occlude_drop`, magnitude `2`: `delta_favored_mean=-0.683`, `choice_flip_rate=0.814`

The same qualitative pattern holds for the other attributes and for `occlude_equalize`.

This is useful because it is harder to dismiss than a binary "occluded vs not occluded" result.

## Suggested Paper Framing

Recommended framing for the main text:

> We next use an intervention-style occlusion suite as a causal stress test of the revealed-preference account. Starting from matched baseline prompts, we selectively remove or neutralize evidence for a target attribute while holding the rest of the item family fixed. If the inferred latent preference structure is meaningful, occluding more important attributes should induce larger shifts in both choices and stated premises.

Then report three compact claims:

1. baseline revealed-preference ranking is `E > D > S > A`
2. intervention effects recover exactly the same ranking
3. premise shifts co-occur with nearly all choice flips

That is enough for a short main-text robustness subsection.

## What Is Not Yet Justified

Current Method C evidence does **not** justify:

- model-family generalization
- reasoning-effort generalization
- tau vs pairwise conclusions for Method C
- a formal causal mediation claim
- claims that Method C is stronger than the main holdout result rather than a validation of it

Because only one analyzed model is complete, Method C should currently be described as a strong single-model intervention validation.

## Main Text Vs Supplement

Recommended main text:

- one short Method C subsection
- one figure or compact table with baseline weights, intervention ranking, flip rates, and one mediation-style statistic

Recommended supplement:

- full attribute-by-attribute tables
- separate `drop` and `equalize` breakdowns
- magnitude-response table
- premise transition destinations

## Files To Cite

- baseline/intervention analysis:
  `artifacts/v4_matchedsuite_drugs_20260208/reports/methodc_current_run_mini_min/methodc_current_run_analysis.md`
- machine-readable output:
  `artifacts/v4_matchedsuite_drugs_20260208/reports/methodc_current_run_mini_min/methodc_current_run_analysis.json`
- existing extra diagnostics:
  `artifacts/v4_matchedsuite_drugs_20260208/reports/methodc_extra_mini_min/methodc_extra_diagnostics.md`

## Rerun Command

The current report was generated with `--bootstrap 0` for a fast paper-facing pass.

For final uncertainty estimates, rerun:

```bash
source /vol/bitbucket/gif22/argllms_plus_plus/venv/bin/activate
python scripts/analyze_methodc_current_run.py \
  --dataset artifacts/v4_matchedsuite_drugs_20260208/data/occlusion_suite_methodC \
  --responses artifacts/v4_matchedsuite_drugs_20260208/runs/methodC/mini_min_joint__openai_gpt5mini_minimal_var-short_reason__judge_scores_joint_mt1024 \
  --full-stagea artifacts/v4_matchedsuite_drugs_20260208/results/stage_A_methodC_mini_min/stageA_summary.json \
  --out-dir artifacts/v4_matchedsuite_drugs_20260208/reports/methodc_current_run_mini_min \
  --bootstrap 400 \
  --seed 13
```

This is compute-heavy but local. The in-turn test pass showed that `400` matched-pair bootstrap resamples take several minutes in serial mode.
