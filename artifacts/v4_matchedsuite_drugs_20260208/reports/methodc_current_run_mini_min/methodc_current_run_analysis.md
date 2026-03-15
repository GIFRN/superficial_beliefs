# Method C Current-Run Analysis

Generated (UTC): 2026-03-11 15:14:49

- Dataset: `artifacts/v4_matchedsuite_drugs_20260208/data/occlusion_suite_methodC`
- Responses: `artifacts/v4_matchedsuite_drugs_20260208/runs/methodC/mini_min_joint__openai_gpt5mini_minimal_var-short_reason__judge_scores_joint_mt1024/responses.jsonl`
- Bootstrap resamples: `0` by `base_trial_id` / matched pair

## Baseline-Only Stage A
| weight_order | weight_E | weight_A | weight_S | weight_D | beta_E | beta_A | beta_S | beta_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E > D > S > A | 0.327 | 0.180 | 0.238 | 0.255 | 3.332 | 1.833 | 2.422 | 2.599 |

## Baseline Vs Intervention Ranking
| manipulation | baseline_weight_order | intervention_order | spearman_vs_baseline | effect_E | effect_A | effect_S | effect_D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E > D > S > A | E > D > S > A | 1.000 | 0.340 | 0.172 | 0.227 | 0.262 |
| occlude_equalize | E > D > S > A | E > D > S > A | 1.000 | 0.341 | 0.174 | 0.226 | 0.259 |

## Directional Intervention Summary
| manipulation | attribute | n | delta_favored_mean | choice_flip_rate | premise_flip_rate | shift_away_from_target_rate |
| --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E | 900 | -0.458 | 0.386 | 0.559 | 0.459 |
| occlude_drop | A | 900 | -0.232 | 0.210 | 0.296 | 0.133 |
| occlude_drop | S | 900 | -0.306 | 0.277 | 0.356 | 0.142 |
| occlude_drop | D | 900 | -0.354 | 0.280 | 0.376 | 0.266 |
| occlude_equalize | E | 900 | -0.466 | 0.366 | 0.528 | 0.458 |
| occlude_equalize | A | 900 | -0.238 | 0.204 | 0.276 | 0.133 |
| occlude_equalize | S | 900 | -0.310 | 0.266 | 0.323 | 0.142 |
| occlude_equalize | D | 900 | -0.355 | 0.274 | 0.371 | 0.266 |

## Drop Vs Equalize Differences
| attribute | delta_equalize_minus_drop_directional | delta_equalize_minus_drop_choice_flip_rate | delta_equalize_minus_drop_premise_flip_rate |
| --- | --- | --- | --- |
| E | -0.008 | -0.020 | -0.031 |
| A | -0.006 | -0.006 | -0.020 |
| S | -0.004 | -0.011 | -0.032 |
| D | -0.001 | -0.006 | -0.004 |

## Premise Transition Destinations
| manipulation | attribute | n_shifted_from_target | top_destination | top_destination_rate_among_shifted |
| --- | --- | --- | --- | --- |
| occlude_drop | E | 413 | D | 0.458 |
| occlude_drop | A | 120 | D | 0.392 |
| occlude_drop | S | 128 | D | 0.391 |
| occlude_drop | D | 239 | E | 0.494 |
| occlude_equalize | E | 412 | D | 0.408 |
| occlude_equalize | A | 120 | S | 0.358 |
| occlude_equalize | S | 128 | E | 0.367 |
| occlude_equalize | D | 239 | E | 0.469 |

## Notes
- `delta_favored_mean` is orientation-corrected: negative values mean the intervention reduced support for the option favored by the targeted attribute.
- The baseline-only Stage A fit uses only `short_reason` rows as the reference preference model.
- The JSON output contains the full mediation-style and magnitude-response summaries.
