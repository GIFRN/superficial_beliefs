# Final Same-Order Results Summary

This document is the standalone paper-facing summary for the completed final same-order benchmark.

## Benchmark Definition

The final benchmark uses a same-order four-variant construction built from a base train/test split. For each base family with profiles `P` and `Q` and source orders `oA` and `oB`, the benchmark evaluates four prompts:

- `P@oA vs Q@oA`
- `Q@oA vs P@oA`
- `P@oB vs Q@oB`
- `Q@oB vs P@oB`

So every evaluated prompt is same-order within-row, and each source order is seen in both slot orientations.

- base families: `400` train and `100` test
- evaluated prompts per theme: `1600` train and `400` test
- themes: `3` substantive + `2` placebo
- model settings: `8` (`GPT-5-mini`, `GPT-5-nano`, `Qwen3.5-14B`, `Ministral-3-14B`, each at `minimal/low`)
- actor replicates per prompt: `S=3`
- bootstrap intervals use `500` resamples

## Main Findings

The fitted choice model still predicts held-out model choices better than the judge recovers them from textual factor attributions. Factor recovery remains noisier than behavioral choice prediction.

The strongest linear-model choice rows are:

- `drugs`: `GPT-5-mini minimal` at 0.873 with 95% CI [0.818, 0.923]
- `policy`: `Qwen3.5-14B minimal` at 0.887 with 95% CI [0.844, 0.918]
- `software`: `GPT-5-nano minimal` at 0.868 with 95% CI [0.830, 0.891]

The strongest judge-choice rows are:

- `drugs`: `GPT-5-mini minimal` at 0.795
- `policy`: `GPT-5-mini minimal` at 0.827
- `software`: `GPT-5-mini minimal` at 0.785

The strongest factor-alignment rows are:

- linear-model factor match:
  - `drugs`: `GPT-5-mini low` at 0.893
  - `policy`: `GPT-5-mini low` at 0.938
  - `software`: `GPT-5-mini low` at 0.891
- judge factor match:
  - `drugs`: `GPT-5-mini low` at 0.668
  - `policy`: `GPT-5-mini low` at 0.824
  - `software`: `GPT-5-mini low` at 0.608

## Placebo Findings

The placebo results remain bounded and interpretable rather than dominating the main story.

- lowest judge placebo row: `placebo_label_border` / `GPT-5-mini low` at `0.000`
- highest judge placebo row: `placebo_label_border` / `GPT-5-nano minimal` at `0.077`

## Files

- `outputs/final_same_order/reports/FINAL_MAIN_RESULTS.csv`
- `outputs/final_same_order/reports/FINAL_PLACEBO_RESULTS.csv`
- `outputs/final_same_order/reports/FINAL_MAIN_RESULTS_BOOTSTRAP.csv`
- `outputs/final_same_order/reports/FINAL_PLACEBO_RESULTS_BOOTSTRAP.csv`
