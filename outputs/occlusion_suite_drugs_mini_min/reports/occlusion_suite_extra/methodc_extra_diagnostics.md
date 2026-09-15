# Method-C Extra Diagnostics

- Dataset: `data/occlusion_suite/themes/drugs/test`
- Responses: `outputs/occlusion_suite_drugs_mini_min/runs/mini_min_joint__openai_gpt5mini_minimal_var-short_reason__judge_scores_joint/responses.jsonl`
- Parsed response rows: 10800
- Distinct response trials: 3600
- Trial-level rows: 3600

## Pairing
- occlude_drop: 1600 paired trials
- occlude_equalize: 1600 paired trials

## Availability
- directional_effects: yes
- magnitude_response: yes
- choice_flip_rates: yes
- premise_shift_rates: yes
- mediation_proxy: yes
- intervention_alignment_deltas: yes
- cross_model_causal_agreement: no (Need at least two --compare-run inputs to compute agreement)

## Not Currently Computable
- formal_causal_mediation_effects: Not identifiable from current saved artifacts without extra causal assumptions and an explicit mediation model
- path_specific_counterfactual_effects: Would require intervention-specific counterfactual labels not present in current outputs
