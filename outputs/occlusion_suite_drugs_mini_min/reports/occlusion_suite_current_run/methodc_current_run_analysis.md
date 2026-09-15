# Method C Current-Run Analysis

Generated (UTC): 2026-03-19 10:15:18

- Dataset: `data/occlusion_suite/themes/drugs/test`
- Responses: `outputs/occlusion_suite_drugs_mini_min/runs/mini_min_joint__openai_gpt5mini_minimal_var-short_reason__judge_scores_joint/responses.jsonl`
- Bootstrap resamples: `500` by `base_trial_id` / matched pair

## Baseline-Only Stage A
| weight_order | weight_E | weight_A | weight_S | weight_D | beta_E | beta_A | beta_S | beta_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E > S > D > A | 0.426 | 0.161 | 0.218 | 0.195 | 5.536 | 2.085 | 2.835 | 2.530 |

## Baseline Vs Intervention Ranking
| manipulation | baseline_weight_order | intervention_order | spearman_vs_baseline | effect_E | effect_A | effect_S | effect_D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E > S > D > A | E > D > S > A | 0.800 | 0.496 | 0.103 | 0.189 | 0.211 |
| occlude_equalize | E > S > D > A | E > D > S > A | 0.800 | 0.508 | 0.100 | 0.196 | 0.196 |

## Directional Intervention Summary
| manipulation | attribute | n | delta_favored_mean | choice_flip_rate | premise_flip_rate | shift_away_from_target_rate |
| --- | --- | --- | --- | --- | --- | --- |
| occlude_drop | E | 400 | -0.637 [-0.685, -0.590] | 0.520 [0.470, 0.569] | 0.645 [0.593, 0.691] | 0.495 [0.446, 0.542] |
| occlude_drop | A | 400 | -0.132 [-0.163, -0.097] | 0.142 [0.113, 0.172] | 0.273 [0.231, 0.311] | 0.107 [0.075, 0.138] |
| occlude_drop | S | 400 | -0.243 [-0.286, -0.205] | 0.215 [0.175, 0.258] | 0.367 [0.315, 0.415] | 0.193 [0.152, 0.230] |
| occlude_drop | D | 400 | -0.271 [-0.312, -0.229] | 0.228 [0.188, 0.268] | 0.318 [0.273, 0.362] | 0.205 [0.165, 0.250] |
| occlude_equalize | E | 400 | -0.645 [-0.697, -0.596] | 0.515 [0.464, 0.569] | 0.625 [0.580, 0.677] | 0.492 [0.445, 0.547] |
| occlude_equalize | A | 400 | -0.127 [-0.160, -0.095] | 0.110 [0.080, 0.147] | 0.237 [0.201, 0.279] | 0.107 [0.080, 0.139] |
| occlude_equalize | S | 400 | -0.248 [-0.292, -0.205] | 0.228 [0.186, 0.270] | 0.328 [0.281, 0.375] | 0.193 [0.152, 0.235] |
| occlude_equalize | D | 400 | -0.249 [-0.300, -0.204] | 0.203 [0.165, 0.242] | 0.312 [0.268, 0.360] | 0.205 [0.165, 0.245] |

## Drop Vs Equalize Differences
| attribute | delta_equalize_minus_drop_directional | delta_equalize_minus_drop_choice_flip_rate | delta_equalize_minus_drop_premise_flip_rate |
| --- | --- | --- | --- |
| E | -0.008 [-0.040, 0.025] | -0.005 [-0.041, 0.030] | -0.020 [-0.050, 0.009] |
| A | 0.006 [-0.015, 0.027] | -0.032 [-0.063, -0.007] | -0.035 [-0.075, 0.003] |
| S | -0.005 [-0.025, 0.014] | 0.013 [-0.012, 0.038] | -0.040 [-0.065, -0.012] |
| D | 0.022 [0.002, 0.042] | -0.025 [-0.045, -0.003] | -0.005 [-0.028, 0.017] |

## Premise Transition Destinations
| manipulation | attribute | n_shifted_from_target | top_destination | top_destination_rate_among_shifted |
| --- | --- | --- | --- | --- |
| occlude_drop | E | 198 | D | 0.414 |
| occlude_drop | A | 43 | D | 0.512 |
| occlude_drop | S | 77 | D | 0.390 |
| occlude_drop | D | 82 | S | 0.402 |
| occlude_equalize | E | 197 | S | 0.391 |
| occlude_equalize | A | 43 | D | 0.395 |
| occlude_equalize | S | 77 | E | 0.351 |
| occlude_equalize | D | 82 | S | 0.402 |

## Notes
- `delta_favored_mean` is orientation-corrected: negative values mean the intervention reduced support for the option favored by the targeted attribute.
- The baseline-only Stage A fit uses only `short_reason` rows as the reference preference model.
- The JSON output contains the full mediation-style and magnitude-response summaries.
