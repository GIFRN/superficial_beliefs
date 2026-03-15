# Method-C Extra Diagnostics

- Dataset: `artifacts/v4_matchedsuite_drugs_20260208/data/occlusion_suite_methodC`
- Responses: `artifacts/v4_matchedsuite_drugs_20260208/runs/methodC/mini_min_joint__openai_gpt5mini_minimal_var-short_reason__judge_scores_joint_mt1024/responses.jsonl`
- Parsed response rows: 40500
- Distinct response trials: 8100
- Trial-level rows: 8100

## Pairing
- occlude_drop: 3600 paired trials
- occlude_equalize: 3600 paired trials

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
