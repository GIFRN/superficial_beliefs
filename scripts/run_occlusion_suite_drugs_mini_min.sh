#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY="/vol/bitbucket/gif22/argllms_plus_plus/venv/bin/python"
if [[ -x "$DEFAULT_PY" ]]; then
  PY="${PY:-$DEFAULT_PY}"
else
  PY="${PY:-python}"
fi
CONFIG="${CONFIG:-$ROOT/data/configs/dataset.yml}"
DATASET="${DATASET:-$ROOT/data/occlusion_suite/themes/drugs/test}"
MODELS="${MODELS:-$ROOT/data/models/mini_min.yml}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/occlusion_suite_drugs_mini_min}"
TRIAL_CONCURRENCY="${TRIAL_CONCURRENCY:-12}"
BOOTSTRAP="${BOOTSTRAP:-500}"

if [[ ! -f "$DATASET/dataset_trials.parquet" ]]; then
  echo "Missing occlusion-suite dataset: $DATASET/dataset_trials.parquet" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/results" "$OUT_ROOT/reports"

"$PY" "$ROOT/scripts/run_trials.py" \
  --config "$CONFIG" \
  --models "$MODELS" \
  --dataset "$DATASET" \
  --out "$OUT_ROOT/runs/mini_min_joint" \
  --variant-override short_reason__judge_scores_joint \
  --resume any \
  --trial-concurrency "$TRIAL_CONCURRENCY"

run_dir="$(compgen -G "$OUT_ROOT/runs/mini_min_joint__*" | head -n1)"
if [[ -z "$run_dir" || ! -f "$run_dir/responses.jsonl" ]]; then
  echo "Could not resolve occlusion-suite run directory under $OUT_ROOT/runs" >&2
  exit 1
fi

"$PY" "$ROOT/scripts/fit_stageA.py" \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --responses "$run_dir/responses.jsonl" \
  --out "$OUT_ROOT/results/stage_A"

"$PY" "$ROOT/scripts/stageB_alignment.py" \
  --stageA "$OUT_ROOT/results/stage_A" \
  --model-dir "$OUT_ROOT/results/stage_A" \
  --dataset "$DATASET" \
  --responses "$run_dir/responses.jsonl" \
  --out "$OUT_ROOT/results/stage_B"

"$PY" "$ROOT/scripts/analyze_judge_baselines.py" \
  --dataset "$DATASET" \
  --responses "$run_dir/responses.jsonl" \
  --bootstrap "$BOOTSTRAP" \
  --out "$OUT_ROOT/reports/judge_baselines"

"$PY" "$ROOT/scripts/analyze_methodc_current_run.py" \
  --dataset "$DATASET" \
  --responses "$run_dir/responses.jsonl" \
  --out-dir "$OUT_ROOT/reports/occlusion_suite_current_run" \
  --bootstrap "$BOOTSTRAP"

"$PY" "$ROOT/scripts/methodc_extra_diagnostics.py" \
  --dataset "$DATASET" \
  --responses "$run_dir/responses.jsonl" \
  --out "$OUT_ROOT/reports/occlusion_suite_extra" \
  --stageA "$OUT_ROOT/results/stage_A"

echo "Occlusion suite complete: $OUT_ROOT"
