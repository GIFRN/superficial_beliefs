# Paper Appendix Family Consistency Summary

- Consistency is computed at the base-family level over 12 draws: 4 prompt variants x 3 samples.
- Choice consistency is canonicalized to the underlying chosen profile rather than raw A/B labels.
- Consistency rates are conditional on all 12 draws being valid for that measure; completeness rates and modal-share diagnostics are reported in the CSV and JSON outputs.

| row_label | actor_choice_family_complete_rate | actor_choice_family_consistency_rate | actor_choice_family_modal_share | self_report_driver_family_complete_rate | self_report_driver_family_consistency_rate | self_report_driver_family_modal_share | score_judge_choice_family_complete_rate | score_judge_choice_family_consistency_rate | score_judge_choice_family_modal_share | score_judge_driver_family_complete_rate | score_judge_driver_family_consistency_rate | score_judge_driver_family_modal_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Drugs | 1.000 [1.000, 1.000] | 0.346 [0.319, 0.371] | 0.811 [0.801, 0.822] | 1.000 [1.000, 1.000] | 0.205 [0.178, 0.229] | 0.709 [0.698, 0.721] | 1.000 [1.000, 1.000] | 0.560 [0.526, 0.594] | 0.890 [0.880, 0.900] | 1.000 [1.000, 1.000] | 0.339 [0.311, 0.369] | 0.804 [0.792, 0.817] |
| Policy | 1.000 [1.000, 1.000] | 0.419 [0.390, 0.448] | 0.833 [0.821, 0.844] | 1.000 [1.000, 1.000] | 0.336 [0.309, 0.368] | 0.783 [0.770, 0.796] | 1.000 [1.000, 1.000] | 0.537 [0.507, 0.574] | 0.881 [0.872, 0.892] | 1.000 [1.000, 1.000] | 0.320 [0.290, 0.348] | 0.799 [0.787, 0.810] |
| Software | 1.000 [1.000, 1.000] | 0.300 [0.273, 0.326] | 0.785 [0.775, 0.796] | 1.000 [1.000, 1.000] | 0.175 [0.150, 0.199] | 0.706 [0.693, 0.719] | 1.000 [1.000, 1.000] | 0.585 [0.554, 0.617] | 0.897 [0.887, 0.907] | 1.000 [1.000, 1.000] | 0.206 [0.178, 0.234] | 0.748 [0.736, 0.762] |
| Pooled substantive | 1.000 [1.000, 1.000] | 0.355 [0.338, 0.372] | 0.810 [0.803, 0.816] | 1.000 [1.000, 1.000] | 0.239 [0.223, 0.254] | 0.733 [0.725, 0.740] | 1.000 [1.000, 1.000] | 0.561 [0.541, 0.580] | 0.889 [0.883, 0.896] | 1.000 [1.000, 1.000] | 0.288 [0.272, 0.305] | 0.784 [0.777, 0.792] |
