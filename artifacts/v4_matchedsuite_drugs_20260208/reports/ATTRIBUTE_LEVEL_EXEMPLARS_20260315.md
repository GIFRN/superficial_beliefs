# Attribute-Level Exemplars

This table gives concrete disagreement cases from the final canonical short-reason benchmark. Each row is a real test trial chosen to illustrate one of the dominant disagreement patterns in the aggregate attribute-level analysis.

## Case Table
| theme | family | effort | trial_id | visible_deltas | actor | linear_model | judge | why_this_case_matters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drugs | Qwen3.5-14B | low | T-00552 | Efficacy `+1`; Adherence `-1`; Safety `+2`; Durability `+0` | choice `A`; factor `Efficacy` | choice `A`; factor `Safety` | choice `B`; factor `Adherence` | Linear model is correct while judge is wrong. The visible Safety differential is largest, but the actor verbalizes Efficacy instead. |
| drugs | GPT-5-mini | low | T-00481__mirror | Efficacy `+2`; Adherence `-1`; Safety `+0`; Durability `-2` | choice `B`; factor `Durability` | choice `A`; factor `Durability` | choice `B`; factor `Efficacy` | Judge is correct while linear model is wrong. The actor and linear model agree on factor attribution, but that still does not recover the observed choice. |
| policy | Ministral-3-14B | minimal | T-01029__mirror | Effectiveness `+2`; Compliance `+2`; Safety `-2`; Implementation Ease `+0` | choice `A`; factor `Compliance` | choice `A`; factor `Effectiveness` | choice `B`; factor `Safety` | Linear model is correct while judge is wrong. This illustrates the common `Compliance -> Effectiveness` actor-vs-linear substitution. |
| policy | Qwen3.5-14B | minimal | T-00783 | Effectiveness `-1`; Compliance `+2`; Safety `-1`; Implementation Ease `+0` | choice `B`; factor `Effectiveness` | choice `A`; factor `Effectiveness` | choice `B`; factor `Safety` | Judge is correct while linear model is wrong. The judge often re-reads policy cases through Safety even when the actor names Effectiveness. |
| software | Qwen3.5-14B | minimal | T-00240 | Capability `+1`; Adoption Ease `-1`; Reliability `+1`; Maintainability `-1` | choice `A`; factor `Capability` | choice `A`; factor `Reliability` | choice `B`; factor `Maintainability` | Linear model is correct while judge is wrong. This captures the frequent `Capability -> Reliability` substitution in software. |
| software | GPT-5-nano | low | T-00732__mirror | Capability `-1`; Adoption Ease `+1`; Reliability `+1`; Maintainability `-1` | choice `B`; factor `Maintainability` | choice `A`; factor `Maintainability` | choice `B`; factor `Reliability` | Judge is correct while linear model is wrong. Software disagreements are often concentrated on Reliability versus Maintainability. |

## Notes
- `visible_deltas` are the prompt-visible attribute differences for option `A` relative to option `B`.
- `actor` reports the model's observed choice and stated premise attribute.
- `linear_model` reports the held-out Stage A predicted choice and the choice-conditional top-contribution attribute.
- `judge` reports the ArgLLM `tau`-style predicted choice and its top driver attribute.
