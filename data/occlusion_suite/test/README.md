# Occlusion Suite Test Dataset

This dataset is built directly from `data/test/full_trials.parquet`.

For each same-order source trial, it includes:
- 1 baseline `short_reason` row
- 4 `occlude_equalize` rows, one per attribute
- 4 `occlude_drop` rows, one per attribute

All members of a matched family share `order_id_A`, `order_id_B`, `paraphrase_id`, and `seed`.

- Base trials: `400`
- Total rows: `3600`

Use `base_trial_id` as the paired-family key.
