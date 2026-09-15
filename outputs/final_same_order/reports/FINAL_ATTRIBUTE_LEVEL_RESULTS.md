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
| drugs | Efficacy | Durability -> Efficacy | Durability -> Efficacy | Efficacy | Durability | Safety | 0.307 | Durability | 0.092 |
| policy | Effectiveness | Safety -> Effectiveness | Safety -> Effectiveness | Effectiveness | Safety | Implementation Ease | 0.333 | Implementation Ease | 0.055 |
| software | Reliability | Reliability -> Capability | Reliability -> Capability | Reliability | Reliability | Adoption Ease | 0.231 | Adoption Ease | 0.022 |

## Stage A Weight Orders
| theme | family | effort | top_attr_label | weight_order | weight_E | weight_A | weight_S | weight_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | GPT-5-mini | minimal | Efficacy | E>S>D>A | 0.396 | 0.173 | 0.245 | 0.187 |
| drugs | GPT-5-mini | low | Efficacy | E>S>A>D | 0.346 | 0.187 | 0.301 | 0.166 |
| drugs | GPT-5-nano | minimal | Efficacy | E>D>A>S | 0.412 | 0.191 | 0.181 | 0.216 |
| drugs | GPT-5-nano | low | Safety | S>E>A>D | 0.265 | 0.228 | 0.290 | 0.217 |
| drugs | Qwen3-14B | minimal | Efficacy | E>A>D>S | 0.329 | 0.281 | 0.190 | 0.200 |
| drugs | Qwen3-14B | low | Adherence | A>E>D>S | 0.291 | 0.307 | 0.184 | 0.218 |
| drugs | Ministral-3-14B | minimal | Efficacy | E>S>A>D | 0.390 | 0.180 | 0.274 | 0.155 |
| drugs | Ministral-3-14B | low | Efficacy | E>S>A>D | 0.390 | 0.205 | 0.227 | 0.177 |
| policy | GPT-5-mini | minimal | Effectiveness | E>S>A>D | 0.467 | 0.183 | 0.265 | 0.085 |
| policy | GPT-5-mini | low | Effectiveness | E>S>A>D | 0.445 | 0.227 | 0.272 | 0.056 |
| policy | GPT-5-nano | minimal | Effectiveness | E>S>A>D | 0.433 | 0.179 | 0.209 | 0.179 |
| policy | GPT-5-nano | low | Effectiveness | E>S>A>D | 0.445 | 0.184 | 0.286 | 0.085 |
| policy | Qwen3-14B | minimal | Effectiveness | E>A>S>D | 0.375 | 0.227 | 0.226 | 0.172 |
| policy | Qwen3-14B | low | Effectiveness | E>S>A>D | 0.468 | 0.201 | 0.234 | 0.097 |
| policy | Ministral-3-14B | minimal | Effectiveness | E>S>A>D | 0.449 | 0.154 | 0.294 | 0.104 |
| policy | Ministral-3-14B | low | Effectiveness | E>S>A>D | 0.475 | 0.165 | 0.230 | 0.130 |
| software | GPT-5-mini | minimal | Reliability | S>E>D>A | 0.259 | 0.155 | 0.336 | 0.251 |
| software | GPT-5-mini | low | Reliability | S>D>A>E | 0.074 | 0.101 | 0.427 | 0.398 |
| software | GPT-5-nano | minimal | Capability | E>S>A>D | 0.302 | 0.220 | 0.265 | 0.213 |
| software | GPT-5-nano | low | Reliability | S>D>E>A | 0.216 | 0.179 | 0.348 | 0.256 |
| software | Qwen3-14B | minimal | Capability | E>S>A>D | 0.293 | 0.218 | 0.274 | 0.216 |
| software | Qwen3-14B | low | Reliability | S>D>E>A | 0.217 | 0.131 | 0.354 | 0.299 |
| software | Ministral-3-14B | minimal | Reliability | S>E>D>A | 0.325 | 0.114 | 0.352 | 0.209 |
| software | Ministral-3-14B | low | Reliability | S>E>D>A | 0.285 | 0.153 | 0.340 | 0.222 |

## Files
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_CONDITIONALS.csv`
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_MISMATCHES.csv`
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_EXEMPLARS.md`
