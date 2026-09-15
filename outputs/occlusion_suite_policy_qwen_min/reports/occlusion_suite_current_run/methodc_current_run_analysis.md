# Method C Current-Run Analysis

Generated (UTC): 2026-03-28 22:16:20

- Dataset: `data/occlusion_suite/themes/policy/test`
- Responses: `outputs/occlusion_suite_policy_qwen_min/runs/mini_min_joint__local_qwen3_14b_vllm_minimal_minimal_var-short_reason__judge_scores_joint/responses.jsonl`
- Bootstrap resamples: `500` by `base_trial_id` / matched pair

## Baseline-Only Stage A
| weight_order | weight_E | weight_A | weight_S | weight_D | beta_E | beta_A | beta_S | beta_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E > A > S > D | 0.399 | 0.223 | 0.216 | 0.162 | 4.536 | 2.536 | 2.452 | 1.846 |

## Baseline Vs Intervention Ranking
| manipulation | baseline_weight_order | intervention_order | spearman_vs_baseline | effect_E | effect_A | effect_S | effect_D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E > A > S > D | E > S > A > D | 0.800 | 0.433 | 0.183 | 0.207 | 0.176 |
| occlude_equalize | E > A > S > D | E > S > A > D | 0.800 | 0.446 | 0.192 | 0.201 | 0.161 |

## Directional Intervention Summary
| manipulation | attribute | n | delta_favored_mean | choice_flip_rate | premise_flip_rate | shift_away_from_target_rate |
| --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E | 400 | -0.567 [-0.621, -0.521] | 0.482 [0.435, 0.532] | 0.757 [0.719, 0.797] | 0.652 [0.609, 0.694] |
| occlude_drop | A | 400 | -0.240 [-0.284, -0.194] | 0.190 [0.152, 0.225] | 0.258 [0.215, 0.301] | 0.105 [0.079, 0.135] |
| occlude_drop | S | 400 | -0.271 [-0.318, -0.228] | 0.258 [0.217, 0.302] | 0.343 [0.299, 0.390] | 0.172 [0.136, 0.210] |
| occlude_drop | D | 400 | -0.231 [-0.283, -0.183] | 0.200 [0.159, 0.240] | 0.302 [0.260, 0.351] | 0.070 [0.045, 0.095] |
| occlude_equalize | E | 400 | -0.589 [-0.652, -0.539] | 0.460 [0.412, 0.509] | 0.512 [0.468, 0.560] | 0.453 [0.407, 0.500] |
| occlude_equalize | A | 400 | -0.253 [-0.302, -0.208] | 0.207 [0.169, 0.251] | 0.253 [0.215, 0.297] | 0.105 [0.077, 0.138] |
| occlude_equalize | S | 400 | -0.265 [-0.311, -0.220] | 0.245 [0.204, 0.292] | 0.282 [0.242, 0.328] | 0.172 [0.135, 0.206] |
| occlude_equalize | D | 400 | -0.212 [-0.262, -0.172] | 0.160 [0.128, 0.200] | 0.185 [0.150, 0.220] | 0.070 [0.048, 0.095] |

## Drop Vs Equalize Differences
| attribute | delta_equalize_minus_drop_directional | delta_equalize_minus_drop_choice_flip_rate | delta_equalize_minus_drop_premise_flip_rate |
| --- | --- | --- | --- |
| E | -0.022 [-0.061, 0.014] | -0.022 [-0.060, 0.015] | -0.245 [-0.288, -0.203] |
| A | -0.014 [-0.036, 0.005] | 0.017 [-0.003, 0.043] | -0.005 [-0.030, 0.018] |
| S | 0.006 [-0.013, 0.024] | -0.013 [-0.030, 0.013] | -0.060 [-0.090, -0.030] |
| D | 0.018 [-0.010, 0.046] | -0.040 [-0.067, -0.013] | -0.117 [-0.155, -0.080] |

## Premise Transition Destinations
| manipulation | attribute | n_shifted_from_target | top_destination | top_destination_rate_among_shifted |
| --- | --- | --- | --- | --- |
| occlude_drop | E | 261 | S | 0.609 |
| occlude_drop | A | 42 | S | 0.357 |
| occlude_drop | S | 69 | E | 0.522 |
| occlude_drop | D | 28 | E | 0.429 |
| occlude_equalize | E | 181 | S | 0.475 |
| occlude_equalize | A | 42 | D | 0.429 |
| occlude_equalize | S | 69 | E | 0.478 |
| occlude_equalize | D | 28 | E | 0.393 |

## Notes
- `delta_favored_mean` is orientation-corrected: negative values mean the intervention reduced support for the option favored by the targeted attribute.
- The baseline-only Stage A fit uses only `short_reason` rows as the reference preference model.
- The JSON output contains the full mediation-style and magnitude-response summaries.
