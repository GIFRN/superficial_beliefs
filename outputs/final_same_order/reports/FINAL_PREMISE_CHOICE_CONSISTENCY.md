# Premise-Choice Consistency

This report checks whether the actor's stated premise attribute actually favors the chosen option on the final same-order benchmark test split.

- all themes: support on `0.950` of non-tied rows; contradiction on `0.050`; tie rate `0.051`
- substantive themes only: support on `0.952` of non-tied rows; contradiction on `0.048`; tie rate `0.051`

## Theme Aggregates
| theme | support_rate_non_tied | contradiction_rate_non_tied | tied_rate | worst_attr_label | worst_attr_contradiction_rate_non_tied |
| --- | --- | --- | --- | --- | --- |
| drugs | 0.954 | 0.046 | 0.041 | Durability | 0.069 |
| policy | 0.958 | 0.042 | 0.054 | Implementation Ease | 0.102 |
| software | 0.945 | 0.055 | 0.058 | Adoption Ease | 0.115 |

## Worst Attributes
| theme | worst_attr_label | worst_attr_contradiction_rate_non_tied | worst_attr_n_non_tied |
| --- | --- | --- | --- |
| drugs | Durability | 0.069 | 1906 |
| policy | Implementation Ease | 0.102 | 598 |
| software | Adoption Ease | 0.115 | 1545 |

## Weakest Theme/Model Cases
| theme | family | effort | contradiction_rate_non_tied | tied_rate | n_non_tied |
| --- | --- | --- | --- | --- | --- |
| software | GPT-5-nano | minimal | 0.267 | 0.122 | 1054 |
| drugs | GPT-5-nano | minimal | 0.160 | 0.093 | 1088 |
| policy | GPT-5-nano | minimal | 0.148 | 0.081 | 1103 |
| policy | Ministral-3-14B | minimal | 0.081 | 0.096 | 1085 |
| policy | Ministral-3-14B | low | 0.080 | 0.092 | 1090 |
| drugs | Ministral-3-14B | low | 0.071 | 0.069 | 1117 |

## File
- `outputs/final_same_order/reports/FINAL_PREMISE_CHOICE_CONSISTENCY.csv`
