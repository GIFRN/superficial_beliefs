# Attribute-Level Exemplars

This table gives concrete disagreement cases from the final same-order benchmark.

## Case Table
| theme | family | effort | trial_id | visible_deltas | actor | linear_model | judge | why_this_case_matters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | Qwen3.5-14B | minimal | test_0019__p_at_ob_vs_q_at_ob | Efficacy `-1`; Adherence `+2`; Safety `+1`; Durability `-1` | choice `B`; factor `Durability` | choice `B`; factor `Efficacy` | choice `A`; factor `Adherence` | Linear model is correct while judge is wrong. This illustrates the common `Durability -> Efficacy` actor-vs-linear substitution. |
| drugs | Qwen3.5-14B | minimal | test_0063__q_at_oa_vs_p_at_oa | Efficacy `+2`; Adherence `-2`; Safety `+1`; Durability `-1` | choice `B`; factor `Durability` | choice `A`; factor `Adherence` | choice `B`; factor `Efficacy` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Efficacy` while the actor states `Durability`. |
| policy | Ministral-3-14B | minimal | test_0075__q_at_ob_vs_p_at_ob | Effectiveness `+1`; Compliance `+0`; Safety `+1`; Implementation Ease `-2` | choice `A`; factor `Safety` | choice `A`; factor `Effectiveness` | choice `B`; factor `Implementation Ease` | Linear model is correct while judge is wrong. This illustrates the common `Safety -> Effectiveness` actor-vs-linear substitution. |
| policy | Qwen3.5-14B | minimal | test_0087__p_at_ob_vs_q_at_ob | Effectiveness `-1`; Compliance `+0`; Safety `+2`; Implementation Ease `+1` | choice `A`; factor `Safety` | choice `B`; factor `Safety` | choice `A`; factor `Effectiveness` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Effectiveness` while the actor states `Safety`. |
| software | Ministral-3-14B | minimal | test_0014__q_at_oa_vs_p_at_oa | Capability `+2`; Adoption Ease `+1`; Reliability `-1`; Maintainability `+0` | choice `A`; factor `Adoption Ease` | choice `A`; factor `Capability` | choice `B`; factor `Reliability` | Linear model is correct while judge is wrong. This illustrates the common `Adoption Ease -> Capability` actor-vs-linear substitution. |
| software | Qwen3.5-14B | minimal | test_0029__q_at_oa_vs_p_at_oa | Capability `+2`; Adoption Ease `-1`; Reliability `-1`; Maintainability `+0` | choice `B`; factor `Reliability` | choice `A`; factor `Reliability` | choice `B`; factor `Capability` | Judge is correct while linear model is wrong. This highlights how the judge shifts toward `Capability` while the actor states `Reliability`. |
