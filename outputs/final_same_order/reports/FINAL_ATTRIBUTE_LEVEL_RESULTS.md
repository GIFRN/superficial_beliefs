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
| drugs | Efficacy | Durability -> Efficacy | Durability -> Efficacy | Durability | Adherence | Safety | 0.308 | Durability | 0.061 |
| policy | Effectiveness | Safety -> Effectiveness | Safety -> Effectiveness | Effectiveness | Safety | Implementation Ease | 0.222 | Implementation Ease | 0.040 |
| software | Reliability | Adoption Ease -> Capability | Reliability -> Capability | Maintainability | Adoption Ease | Adoption Ease | 0.238 | Adoption Ease | 0.071 |

## Stage A Weight Orders
| theme | family | effort | top_attr_label | weight_order | weight_E | weight_A | weight_S | weight_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | GPT-5-mini | minimal | Efficacy | E>S>D>A | 0.406 | 0.168 | 0.241 | 0.185 |
| drugs | GPT-5-mini | low | Efficacy | E>S>A>D | 0.340 | 0.197 | 0.299 | 0.164 |
| drugs | GPT-5-nano | minimal | Efficacy | E>D>A>S | 0.413 | 0.186 | 0.180 | 0.221 |
| drugs | GPT-5-nano | low | Safety | S>E>A>D | 0.268 | 0.225 | 0.288 | 0.219 |
| drugs | Qwen3.5-14B | minimal | Efficacy | E>D>S>A | 0.319 | 0.202 | 0.232 | 0.246 |
| drugs | Qwen3.5-14B | low | Efficacy | E>D>S>A | 0.318 | 0.202 | 0.233 | 0.247 |
| drugs | Ministral-3-14B | minimal | Efficacy | E>S>A>D | 0.390 | 0.205 | 0.227 | 0.177 |
| drugs | Ministral-3-14B | low | Efficacy | E>S>A>D | 0.390 | 0.205 | 0.227 | 0.177 |
| policy | GPT-5-mini | minimal | Effectiveness | E>S>A>D | 0.468 | 0.184 | 0.263 | 0.085 |
| policy | GPT-5-mini | low | Effectiveness | E>S>A>D | 0.449 | 0.220 | 0.276 | 0.054 |
| policy | GPT-5-nano | minimal | Effectiveness | E>S>A>D | 0.433 | 0.184 | 0.206 | 0.177 |
| policy | GPT-5-nano | low | Effectiveness | E>S>A>D | 0.449 | 0.185 | 0.288 | 0.078 |
| policy | Qwen3.5-14B | minimal | Effectiveness | E>S>A>D | 0.450 | 0.215 | 0.243 | 0.091 |
| policy | Qwen3.5-14B | low | Effectiveness | E>S>A>D | 0.450 | 0.216 | 0.242 | 0.092 |
| policy | Ministral-3-14B | minimal | Effectiveness | E>S>A>D | 0.475 | 0.165 | 0.230 | 0.130 |
| policy | Ministral-3-14B | low | Effectiveness | E>S>A>D | 0.475 | 0.165 | 0.230 | 0.130 |
| software | GPT-5-mini | minimal | Reliability | S>E>D>A | 0.255 | 0.152 | 0.340 | 0.253 |
| software | GPT-5-mini | low | Reliability | S>D>A>E | 0.071 | 0.100 | 0.426 | 0.404 |
| software | GPT-5-nano | minimal | Capability | E>S>A>D | 0.298 | 0.221 | 0.272 | 0.210 |
| software | GPT-5-nano | low | Reliability | S>D>E>A | 0.213 | 0.182 | 0.346 | 0.259 |
| software | Qwen3.5-14B | minimal | Capability | E>S>D>A | 0.328 | 0.160 | 0.272 | 0.241 |
| software | Qwen3.5-14B | low | Capability | E>S>D>A | 0.327 | 0.160 | 0.270 | 0.242 |
| software | Ministral-3-14B | minimal | Reliability | S>E>D>A | 0.285 | 0.153 | 0.340 | 0.222 |
| software | Ministral-3-14B | low | Reliability | S>E>D>A | 0.288 | 0.150 | 0.339 | 0.223 |

## Files
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_CONDITIONALS.csv`
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_MISMATCHES.csv`
- `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_EXEMPLARS.md`
