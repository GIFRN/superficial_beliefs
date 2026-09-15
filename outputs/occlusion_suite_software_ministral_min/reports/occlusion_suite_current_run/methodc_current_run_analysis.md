# Method C Current-Run Analysis

Generated (UTC): 2026-03-28 22:16:21

- Dataset: `data/occlusion_suite/themes/software/test`
- Responses: `outputs/occlusion_suite_software_ministral_min/runs/mini_min_joint__local_ministral3_14b_instruct_vllm_minimal_minimal_var-short_reason__judge_scores_joint/responses.jsonl`
- Bootstrap resamples: `500` by `base_trial_id` / matched pair

## Baseline-Only Stage A
| weight_order | weight_E | weight_A | weight_S | weight_D | beta_E | beta_A | beta_S | beta_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S > E > D > A | 0.282 | 0.139 | 0.384 | 0.195 | 1.022 | 0.505 | 1.396 | 0.709 |

## Baseline Vs Intervention Ranking
| manipulation | baseline_weight_order | intervention_order | spearman_vs_baseline | effect_E | effect_A | effect_S | effect_D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | S > E > D > A | S > E > D > A | 1.000 | 0.309 | 0.119 | 0.389 | 0.183 |
| occlude_equalize | S > E > D > A | S > E > D > A | 1.000 | 0.304 | 0.134 | 0.373 | 0.189 |

## Directional Intervention Summary
| manipulation | attribute | n | delta_favored_mean | choice_flip_rate | premise_flip_rate | shift_away_from_target_rate |
| --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E | 400 | -0.270 [-0.315, -0.228] | 0.335 [0.287, 0.383] | 0.482 [0.435, 0.531] | 0.247 [0.199, 0.290] |
| occlude_drop | A | 400 | -0.104 [-0.147, -0.064] | 0.242 [0.200, 0.282] | 0.410 [0.360, 0.460] | 0.125 [0.091, 0.158] |
| occlude_drop | S | 400 | -0.341 [-0.382, -0.301] | 0.388 [0.338, 0.436] | 0.660 [0.615, 0.714] | 0.460 [0.412, 0.510] |
| occlude_drop | D | 400 | -0.160 [-0.201, -0.121] | 0.268 [0.223, 0.315] | 0.445 [0.393, 0.494] | 0.168 [0.131, 0.205] |
| occlude_equalize | E | 400 | -0.256 [-0.302, -0.215] | 0.333 [0.289, 0.376] | 0.432 [0.388, 0.482] | 0.205 [0.164, 0.247] |
| occlude_equalize | A | 400 | -0.112 [-0.148, -0.075] | 0.245 [0.205, 0.287] | 0.427 [0.379, 0.472] | 0.107 [0.079, 0.140] |
| occlude_equalize | S | 400 | -0.314 [-0.354, -0.275] | 0.380 [0.328, 0.426] | 0.562 [0.515, 0.606] | 0.357 [0.312, 0.401] |
| occlude_equalize | D | 400 | -0.159 [-0.196, -0.121] | 0.273 [0.230, 0.314] | 0.385 [0.333, 0.432] | 0.163 [0.125, 0.200] |

## Drop Vs Equalize Differences
| attribute | delta_equalize_minus_drop_directional | delta_equalize_minus_drop_choice_flip_rate | delta_equalize_minus_drop_premise_flip_rate |
| --- | --- | --- | --- |
| E | 0.014 [-0.022, 0.051] | -0.003 [-0.045, 0.042] | -0.050 [-0.100, -0.001] |
| A | -0.008 [-0.041, 0.029] | 0.003 [-0.048, 0.050] | 0.018 [-0.033, 0.070] |
| S | 0.026 [-0.005, 0.058] | -0.008 [-0.052, 0.035] | -0.098 [-0.142, -0.047] |
| D | 0.001 [-0.030, 0.037] | 0.005 [-0.037, 0.042] | -0.060 [-0.106, -0.015] |

## Premise Transition Destinations
| manipulation | attribute | n_shifted_from_target | top_destination | top_destination_rate_among_shifted |
| --- | --- | --- | --- | --- |
| occlude_drop | E | 99 | S | 0.586 |
| occlude_drop | A | 50 | S | 0.500 |
| occlude_drop | S | 184 | E | 0.527 |
| occlude_drop | D | 67 | S | 0.552 |
| occlude_equalize | E | 82 | S | 0.549 |
| occlude_equalize | A | 43 | S | 0.465 |
| occlude_equalize | S | 143 | D | 0.385 |
| occlude_equalize | D | 65 | S | 0.477 |

## Notes
- `delta_favored_mean` is orientation-corrected: negative values mean the intervention reduced support for the option favored by the targeted attribute.
- The baseline-only Stage A fit uses only `short_reason` rows as the reference preference model.
- The JSON output contains the full mediation-style and magnitude-response summaries.
