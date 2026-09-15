# Paper Appendix Family Majority Correctness Summary

- Majority correctness is computed at the base-family level over 12 draws: 4 prompt variants x 3 samples.
- A family counts as majority-correct when more than half of its valid draws are correct relative to the row-level latent or revealed target.
- For choice metrics, correctness uses canonicalized underlying profiles rather than raw A/B labels.

| row_label | actor_choice_family_majority_correct_rate | self_report_driver_family_majority_correct_rate | score_judge_choice_family_majority_correct_rate | score_judge_driver_family_majority_correct_rate |
| --- | --- | --- | --- | --- |
| Drugs | 0.836 [0.812, 0.861] | 0.527 [0.497, 0.560] | 0.771 [0.744, 0.801] | 0.583 [0.550, 0.616] |
| Policy | 0.895 [0.874, 0.916] | 0.681 [0.653, 0.710] | 0.759 [0.729, 0.786] | 0.635 [0.603, 0.665] |
| Software | 0.879 [0.856, 0.900] | 0.508 [0.476, 0.539] | 0.684 [0.651, 0.716] | 0.475 [0.443, 0.504] |
| Pooled substantive | 0.870 [0.858, 0.882] | 0.572 [0.554, 0.592] | 0.738 [0.720, 0.755] | 0.564 [0.545, 0.583] |
