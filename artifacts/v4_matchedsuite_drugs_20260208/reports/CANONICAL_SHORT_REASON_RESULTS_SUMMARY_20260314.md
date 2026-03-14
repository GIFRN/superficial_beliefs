# Canonical Short-Reason Results Summary

This document is the standalone paper-facing summary for the final canonical benchmark.

## Benchmark Definition

The final benchmark uses a slot-preserving mirror construction. For every original pairwise trial, a mirrored counterpart swaps the underlying profiles across slots while keeping the slot-specific order scaffolding fixed. This yields balanced `A`/`B` placement and sign-reversed visible differentials without changing the order template assigned to each slot.

The final canonical splits contain only `short_reason` prompts shown to the models:

- train: `762` trials = `381` original + `381` mirrored
- test: `194` trials = `97` original + `97` mirrored
- unique base configurations: `285` in train and `71` in test
- label balance: exactly `50/50` `A` vs `B` in both train and test
- split hygiene: zero train/test `config_id` overlap

The benchmark covers three substantive themes (`drugs`, `policy`, `software`) and two placebo themes (`Packaging Symmetry`, `Label Border Thickness`) across five model families and two effort settings.

Bootstrap intervals below use `500` config-level resamples.

## Main Findings

The fitted choice model continues to predict held-out model choices substantially better than the judge recovers them from textual attributions. This gap is especially clear on `software`, where the linear model remains strong while judge-based choice recovery is weaker.

The strongest linear-model choice rows are:

- `drugs`: `GPT-5-mini minimal` at `0.809` with `95% CI [0.746, 0.876]`
- `policy`: `GPT-5-nano low` at `0.887` with `95% CI [0.838, 0.924]`
- `software`: `GPT-5-mini low` at `0.845` with `95% CI [0.765, 0.910]`

The strongest judge-choice rows are lower:

- `drugs`: `Claude Haiku 4.5 low` at `0.737`
- `policy`: `GPT-5-mini low` at `0.810`
- `software`: `Claude Haiku 4.5 low` at `0.790`

Factor alignment is strongest on `policy` and weaker on `drugs` and `software`. The best rows are:

- linear-model factor match:
  - `drugs`: `GPT-5-mini minimal` at `0.776`
  - `policy`: `Claude Haiku 4.5 low` at `0.922`
  - `software`: `GPT-5-mini low` at `0.884`
- judge factor match:
  - `drugs`: `GPT-5-mini low` at `0.562`
  - `policy`: `Claude Haiku 4.5 low` at `0.769`
  - `software`: `Claude Haiku 4.5 low` at `0.620`

So the main empirical story is:

- revealed preference remains recoverable from held-out choices
- judge-based choice recovery is weaker than the fitted choice model
- textual factor alignment is materially noisier than behavioral choice prediction
- `policy` is the easiest theme for factor recovery, while `software` remains hard for judge-based choice recovery

## Placebo Findings

The placebo results remain clean.

- `Claude Haiku 4.5` and `GPT-5-mini` are effectively at zero placebo uptake across both placebo themes.
- `GPT-5-nano minimal` is the clearest placebo-sensitive condition:
  - `Packaging Symmetry`: judge placebo factor rate `0.076`, `95% CI [0.046, 0.108]`
  - `Label Border Thickness`: judge placebo factor rate `0.062`, `95% CI [0.041, 0.087]`
- `Qwen3.5-14B` shows small but nonzero placebo attribution in the judge output, around `0.025` to `0.031`.
- `Ministral-3-14B` shows very small placebo attribution in judge outputs, around `0.004` to `0.006`, though actor-stated placebo factors are somewhat higher than for the closed models.

Overall, the placebo themes support the same qualitative conclusion as before: stronger models largely avoid treating the irrelevant attribute as a key factor, while weaker settings show measurable but still bounded placebo uptake.

## Files

Numerical point estimates:

- [SHORT_REASON_BALANCED_MAIN_RESULTS_20260313.csv](/vol/bitbucket/gif22/superficial_beliefs/artifacts/v4_matchedsuite_drugs_20260208/reports/SHORT_REASON_BALANCED_MAIN_RESULTS_20260313.csv)
- [SHORT_REASON_BALANCED_PLACEBO_RESULTS_20260313.csv](/vol/bitbucket/gif22/superficial_beliefs/artifacts/v4_matchedsuite_drugs_20260208/reports/SHORT_REASON_BALANCED_PLACEBO_RESULTS_20260313.csv)

Bootstrap confidence intervals:

- [SHORT_REASON_BALANCED_MAIN_RESULTS_BOOTSTRAP_20260313.csv](/vol/bitbucket/gif22/superficial_beliefs/artifacts/v4_matchedsuite_drugs_20260208/reports/SHORT_REASON_BALANCED_MAIN_RESULTS_BOOTSTRAP_20260313.csv)
- [SHORT_REASON_BALANCED_PLACEBO_RESULTS_BOOTSTRAP_20260313.csv](/vol/bitbucket/gif22/superficial_beliefs/artifacts/v4_matchedsuite_drugs_20260208/reports/SHORT_REASON_BALANCED_PLACEBO_RESULTS_BOOTSTRAP_20260313.csv)
