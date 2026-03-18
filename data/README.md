# Final Base Dataset

This directory contains the train/test source families and the derived same-order datasets for the final experiments.

- `train/dataset_trials.parquet` and `test/dataset_trials.parquet` are the canonical same-order base datasets.
- Each base row is the `p_at_oa_vs_q_at_oa` variant for one family.
- `train/full_trials.parquet` and `test/full_trials.parquet` contain the full four-variant expansion:
  - `p_at_oa_vs_q_at_oa`
  - `q_at_oa_vs_p_at_oa`
  - `p_at_ob_vs_q_at_ob`
  - `q_at_ob_vs_p_at_ob`
- `occlusion_suite/test/` contains the occlusion-suite alternative built from the final same-order full test set.
- `occlusion_suite/themes/` contains the theme-transformed occlusion-suite test datasets.
- `paraphrase_id` is fixed to `0`; profile rendering no longer varies by paraphrase template.
- `base_samples.csv` and `full_samples.csv` are the inspection-friendly CSV versions.
- `themes/` contains copied theme configs with simplified names.
- `models/` contains copied model configs with simplified names.
- `configs/dataset.yml` is the copied dataset-generation config.
