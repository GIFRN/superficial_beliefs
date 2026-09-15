# Attribute-Level Exemplars

This table gives concrete disagreement cases from the final same-order benchmark.

## Case Table
| theme | family | effort | trial_id | visible_deltas | actor | linear_model | judge | why_this_case_matters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | Qwen3-14B | minimal | test_0048__q_at_oa_vs_p_at_oa | Efficacy `+2`; Adherence `+0`; Safety `-1`; Durability `-2` | choice `A`; factor `Durability` | choice `A`; factor `Efficacy` | choice `B`; factor `Efficacy` | Linear model is correct while judge is wrong. This illustrates the common `Durability -> Efficacy` actor-vs-linear substitution. |
| drugs | Qwen3-14B | minimal | test_0029__q_at_ob_vs_p_at_ob | Efficacy `+2`; Adherence `-1`; Safety `-1`; Durability `+0` | choice `B`; factor `Durability` | choice `A`; factor `Adherence` | choice `B`; factor `Efficacy` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Efficacy` while the actor states `Durability`. |
| policy | Qwen3-14B | minimal | test_0074__p_at_oa_vs_q_at_oa | Effectiveness `-1`; Compliance `+0`; Safety `-1`; Implementation Ease `+2` | choice `B`; factor `Safety` | choice `B`; factor `Effectiveness` | choice `A`; factor `Implementation Ease` | Linear model is correct while judge is wrong. This illustrates the common `Safety -> Effectiveness` actor-vs-linear substitution. |
| policy | Qwen3-14B | low | test_0048__p_at_oa_vs_q_at_oa | Effectiveness `-2`; Compliance `+0`; Safety `+1`; Implementation Ease `+2` | choice `A`; factor `Safety` | choice `B`; factor `Safety` | choice `A`; factor `Effectiveness` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Effectiveness` while the actor states `Safety`. |
| software | Qwen3-14B | minimal | test_0078__q_at_ob_vs_p_at_ob | Capability `+2`; Adoption Ease `-1`; Reliability `+1`; Maintainability `-1` | choice `A`; factor `Reliability` | choice `A`; factor `Capability` | choice `B`; factor `Maintainability` | Linear model is correct while judge is wrong. This illustrates the common `Reliability -> Capability` actor-vs-linear substitution. |
| software | Ministral-3-14B | minimal | test_0013__p_at_oa_vs_q_at_oa | Capability `-1`; Adoption Ease `+1`; Reliability `+0`; Maintainability `+2` | choice `A`; factor `Reliability` | choice `B`; factor `Maintainability` | choice `A`; factor `Capability` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Capability` while the actor states `Reliability`. |
