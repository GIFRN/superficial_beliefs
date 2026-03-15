# Attribute-Level Results Summary

This report surfaces attribute-specific structure that is not visible in the headline score tables alone.

## Theme Metadata
| theme | objective | E_label | A_label | S_label | D_label |
| --- | --- | --- | --- | --- | --- |
| drugs | 5-year overall patient outcome | Efficacy | Adherence | Safety | Durability |
| policy | 5-year overall community outcome | Effectiveness | Compliance | Safety | Implementation Ease |
| software | 5-year overall production engineering outcome for a small team | Capability | Adoption Ease | Reliability | Maintainability |

## Theme-Level Hotspots
| theme | most_common_top_attr_label | top_actor_vs_linear_pair | top_actor_vs_judge_pair | linear_correct_judge_wrong_top_actor_attr | judge_correct_linear_wrong_top_actor_attr | worst_linear_match_attr | worst_linear_match_rate | worst_judge_match_attr | worst_judge_match_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | Efficacy | Efficacy -> Safety | Durability -> Efficacy | Durability | Efficacy | Adherence | 0.326 | Durability | 0.121 |
| policy | Effectiveness | Compliance -> Effectiveness | Effectiveness -> Safety | Effectiveness | Safety | Implementation Ease | 0.036 | Implementation Ease | 0.000 |
| software | Reliability | Capability -> Reliability | Maintainability -> Reliability | Maintainability | Capability | Adoption Ease | 0.222 | Maintainability | 0.078 |

## Stage A Weight Orders
| theme | family | effort | top_attr_label | weight_order | weight_E | weight_A | weight_S | weight_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | GPT-5-mini | minimal | Efficacy | E>D>S>A | 0.318 | 0.180 | 0.246 | 0.256 |
| drugs | GPT-5-mini | low | Safety | S>E>A>D | 0.287 | 0.254 | 0.290 | 0.169 |
| drugs | GPT-5-nano | minimal | Efficacy | E>D>S>A | 0.350 | 0.185 | 0.206 | 0.259 |
| drugs | GPT-5-nano | low | Safety | S>A>E>D | 0.248 | 0.257 | 0.285 | 0.211 |
| drugs | Claude Haiku 4.5 | minimal | Efficacy | E>A>S>D | 0.283 | 0.260 | 0.232 | 0.225 |
| drugs | Claude Haiku 4.5 | low | Safety | S>A>E>D | 0.234 | 0.289 | 0.299 | 0.178 |
| drugs | Qwen3.5-14B | minimal | Adherence | A>D>S>E | 0.229 | 0.284 | 0.240 | 0.248 |
| drugs | Qwen3.5-14B | low | Adherence | A>D>S>E | 0.230 | 0.285 | 0.240 | 0.244 |
| drugs | Ministral-3-14B | minimal | Efficacy | E>A>S>D | 0.326 | 0.234 | 0.221 | 0.219 |
| drugs | Ministral-3-14B | low | Efficacy | E>A>S>D | 0.327 | 0.234 | 0.220 | 0.219 |
| policy | GPT-5-mini | minimal | Effectiveness | E>S>A>D | 0.419 | 0.188 | 0.268 | 0.125 |
| policy | GPT-5-mini | low | Effectiveness | E>S>A>D | 0.397 | 0.247 | 0.262 | 0.094 |
| policy | GPT-5-nano | minimal | Effectiveness | E>S>A>D | 0.415 | 0.188 | 0.230 | 0.167 |
| policy | GPT-5-nano | low | Effectiveness | E>S>A>D | 0.436 | 0.168 | 0.302 | 0.093 |
| policy | Claude Haiku 4.5 | minimal | Effectiveness | E>S>A>D | 0.422 | 0.189 | 0.298 | 0.090 |
| policy | Claude Haiku 4.5 | low | Effectiveness | E>S>A>D | 0.412 | 0.205 | 0.337 | 0.046 |
| policy | Qwen3.5-14B | minimal | Effectiveness | E>S>A>D | 0.371 | 0.251 | 0.264 | 0.113 |
| policy | Qwen3.5-14B | low | Effectiveness | E>S>A>D | 0.372 | 0.252 | 0.264 | 0.112 |
| policy | Ministral-3-14B | minimal | Effectiveness | E>S>A>D | 0.436 | 0.191 | 0.267 | 0.106 |
| policy | Ministral-3-14B | low | Effectiveness | E>S>A>D | 0.437 | 0.190 | 0.267 | 0.106 |
| software | GPT-5-mini | minimal | Reliability | S>D>E>A | 0.169 | 0.157 | 0.338 | 0.336 |
| software | GPT-5-mini | low | Maintainability | D>S>A>E | 0.076 | 0.118 | 0.396 | 0.410 |
| software | GPT-5-nano | minimal | Reliability | S>D>E>A | 0.240 | 0.201 | 0.309 | 0.250 |
| software | GPT-5-nano | low | Reliability | S>D>A>E | 0.151 | 0.168 | 0.362 | 0.319 |
| software | Claude Haiku 4.5 | minimal | Reliability | S>D>E>A | 0.247 | 0.166 | 0.330 | 0.258 |
| software | Claude Haiku 4.5 | low | Reliability | S>D>E>A | 0.182 | 0.126 | 0.413 | 0.279 |
| software | Qwen3.5-14B | minimal | Maintainability | D>S>E>A | 0.227 | 0.178 | 0.292 | 0.303 |
| software | Qwen3.5-14B | low | Maintainability | D>S>E>A | 0.226 | 0.179 | 0.293 | 0.303 |
| software | Ministral-3-14B | minimal | Reliability | S>E>D>A | 0.269 | 0.155 | 0.330 | 0.246 |
| software | Ministral-3-14B | low | Reliability | S>E>D>A | 0.269 | 0.155 | 0.330 | 0.246 |

## Notes
- `top_actor_vs_linear_pair` is the most common attribute mismatch between the actor's stated factor and the linear-model trial factor.
- `top_actor_vs_judge_pair` is the most common mismatch between the actor's stated factor and the judge's trial factor.
- `linear_correct_judge_wrong_top_actor_attr` identifies which actor-stated attribute most often appears when the linear model predicts the actor's choice correctly but the judge does not.
- `worst_*_match_attr` is computed from conditional rates given the actor's stated attribute.

## Files
- `artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_RESULTS_20260315_CONDITIONALS.csv`
- `artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_RESULTS_20260315_MISMATCHES.csv`
- `artifacts/v4_matchedsuite_drugs_20260208/reports/ATTRIBUTE_LEVEL_EXEMPLARS_20260315.md`
