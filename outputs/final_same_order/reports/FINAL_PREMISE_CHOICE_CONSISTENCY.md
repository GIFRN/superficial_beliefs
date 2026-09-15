# Premise-Choice Consistency

This report checks whether the actor's stated premise attribute actually favors the chosen option on the final same-order benchmark test split.

- all themes: support on `0.937` of non-tied rows; contradiction on `0.063`; tie rate `0.049`
- substantive themes only: support on `0.940` of non-tied rows; contradiction on `0.060`; tie rate `0.050`

## Theme Aggregates
| theme | support_rate_non_tied | contradiction_rate_non_tied | tied_rate | worst_attr_label | worst_attr_contradiction_rate_non_tied |
| --- | --- | --- | --- | --- | --- |
| drugs | 0.924 | 0.076 | 0.040 | Durability | 0.119 |
| policy | 0.955 | 0.045 | 0.055 | Implementation Ease | 0.126 |
| software | 0.941 | 0.059 | 0.056 | Adoption Ease | 0.120 |

## Worst Attributes
| theme | worst_attr_label | worst_attr_contradiction_rate_non_tied | worst_attr_n_non_tied |
| --- | --- | --- | --- |
| drugs | Durability | 0.119 | 1956 |
| policy | Implementation Ease | 0.126 | 570 |
| software | Adoption Ease | 0.120 | 1472 |

## Weakest Theme/Model Cases
| theme | family | effort | contradiction_rate_non_tied | tied_rate | n_non_tied |
| --- | --- | --- | --- | --- | --- |
| software | GPT-5-nano | minimal | 0.258 | 0.115 | 1062 |
| drugs | Qwen3-14B | minimal | 0.214 | 0.061 | 1127 |
| drugs | GPT-5-nano | minimal | 0.155 | 0.082 | 1102 |
| policy | GPT-5-nano | minimal | 0.133 | 0.089 | 1093 |
| drugs | Ministral-3-14B | minimal | 0.109 | 0.061 | 1127 |
| policy | Qwen3-14B | minimal | 0.087 | 0.064 | 1123 |

## File
- `outputs/final_same_order/reports/FINAL_PREMISE_CHOICE_CONSISTENCY.csv`
