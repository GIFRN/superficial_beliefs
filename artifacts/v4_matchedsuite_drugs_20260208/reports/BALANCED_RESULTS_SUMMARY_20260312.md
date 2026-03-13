# Balanced Results Summary

This summary treats the canonical balanced stage42 outputs as the paper-facing main results.

Source tables:

- `BALANCED_MAIN_RESULTS_20260312.*`
- `BALANCED_PLACEBO_RESULTS_20260312.*`
- `BALANCED_MAIN_RESULTS_BOOTSTRAP_20260312.*`
- `BALANCED_PLACEBO_RESULTS_BOOTSTRAP_20260312.*`
- `BALANCED_MIRROR_CONSISTENCY_20260312.*`

## Main Claims

1. Choice-derived preference models predict held-out actor choices strongly on the balanced test sets, with the clearest performance from GPT-5-mini.

- Drugs, GPT-5-mini minimal: linear-model choice prediction `0.890`, 95% bootstrap CI `[0.834, 0.927]`.
- Software, GPT-5-mini low: linear-model choice prediction `0.901`, 95% bootstrap CI `[0.828, 0.942]`.
- Policy, GPT-5-mini low: linear-model choice prediction `0.852`, 95% bootstrap CI `[0.782, 0.917]`.

2. Judge-based choice prediction is usually lower than the choice-derived linear model, especially on the harder software theme.

- Software, GPT-5-mini low: judge choice prediction `0.694`, 95% bootstrap CI `[0.605, 0.772]`, versus linear-model `0.901`.
- Software, GPT-5-mini minimal: judge choice prediction `0.641`, 95% bootstrap CI `[0.576, 0.707]`, versus linear-model `0.825`.
- Policy is the partial exception: GPT-5-mini low still shows relatively strong judge choice prediction `0.827`, 95% bootstrap CI `[0.761, 0.885]`.

3. Premise/factor alignment is consistently weaker and noisier than choice prediction.

- Drugs, GPT-5-mini minimal: linear-model factor match `0.862`, judge factor match `0.583`.
- Policy, GPT-5-nano low: linear-model factor match `0.884`, judge factor match `0.642`.
- Software, Claude Haiku 4.5 low: linear-model factor match `0.876`, judge factor match `0.649`.

4. Placebo uptake remains near zero for Claude Haiku 4.5 and GPT-5-mini, but not for GPT-5-nano minimal.

- Packaging Symmetry, GPT-5-mini low: actor placebo premise rate `0.000`, judge placebo rate `0.000`.
- Label Border Thickness, Claude Haiku 4.5 low: actor placebo premise rate `0.000`, judge placebo rate `0.000`.
- Packaging Symmetry, GPT-5-nano minimal: actor placebo premise rate `0.041`, 95% bootstrap CI `[0.017, 0.071]`; judge placebo rate `0.068`, 95% bootstrap CI `[0.042, 0.094]`.
- Label Border Thickness, GPT-5-nano minimal: actor placebo premise rate `0.004`, 95% bootstrap CI `[0.000, 0.010]`; judge placebo rate `0.054`, 95% bootstrap CI `[0.036, 0.075]`.

## Mirror Findings

The balanced mirror is a slot-preserving swap, not a pure order swap: the underlying profiles exchange slots while the slot-specific order scaffold stays fixed.

The main mirror result is that choice is more stable than the stated factor once comparisons are evaluated at the underlying-profile level rather than the raw A/B label.

- Across all five balanced test themes, GPT-5-mini low has the strongest mirror stability: expected same-profile choice agreement `0.891`, majority same-profile agreement `0.926`, expected premise-attribute agreement `0.811`.
- Claude Haiku 4.5 low is also high: expected same-profile choice agreement `0.854`, majority same-profile agreement `0.880`, expected premise-attribute agreement `0.759`.
- GPT-5-nano low is somewhat lower but still fairly stable on choice: expected same-profile choice agreement `0.825`, majority same-profile agreement `0.878`, expected premise-attribute agreement `0.692`.
- Qwen is the main low-stability case: expected same-profile choice agreement is about `0.630` in both low and minimal settings, with expected premise-attribute agreement about `0.560`.

On the non-placebo themes only, the same pattern holds:

- GPT-5-mini low: expected same-profile choice agreement `0.885`, expected premise-attribute agreement `0.780`.
- Claude Haiku 4.5 low: expected same-profile choice agreement `0.846`, expected premise-attribute agreement `0.748`.
- GPT-5-nano low: expected same-profile choice agreement `0.815`, expected premise-attribute agreement `0.669`.
- Qwen low: expected same-profile choice agreement `0.646`, expected premise-attribute agreement `0.549`.

Interpretation: for the stronger closed models, swapping slots under the fixed order scaffold usually preserves the chosen underlying profile, but the stated factor is less invariant. This reinforces the main paper story that revealed preferences are more stable than textual rationales.

## Recommended Results Framing

- Lead with the balanced held-out choice-prediction results and their bootstrap intervals.
- Then show that judge-based explanation recovery is materially weaker than behavior-based recovery, especially on software.
- Use the placebo table as the irrelevance stress test.
- Use the mirror analysis as a robustness result showing that underlying-profile choice is more stable than stated factor under the slot-preserving mirror construction.
