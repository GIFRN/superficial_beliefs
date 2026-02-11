#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?}"
: "${PY:?}"

DS="${DS:-$ROOT/data/occlusion_suite_methodC}"
BOOTSTRAP="${BOOTSTRAP:-500}"

if [[ ! -f "$DS/dataset_trials.parquet" ]]; then
  echo "Missing dataset: $DS/dataset_trials.parquet"
  exit 1
fi

analyze_key () {
  local key="$1"
  local run_dir
  run_dir="$(compgen -G "$ROOT/runs/methodC/${key}_joint__"* | head -n1)"
  if [[ -z "$run_dir" || ! -f "$run_dir/responses.jsonl" ]]; then
    echo "Missing run for key=$key"
    return 1
  fi

  local stagea="$ROOT/results/stage_A_methodC_${key}"
  "$PY" scripts/fit_stageA.py \
    --config configs/default.yml \
    --dataset "$DS" \
    --responses "$run_dir/responses.jsonl" \
    --out "$stagea"

  "$PY" scripts/stageB_alignment.py \
    --stageA "$stagea" \
    --model-dir "$stagea" \
    --responses "$run_dir/responses.jsonl" \
    --out "$ROOT/results/stage_B_methodC_${key}"

  "$PY" scripts/analyze_judge_baselines.py \
    --dataset "$DS" \
    --responses "$run_dir/responses.jsonl" \
    --bootstrap "$BOOTSTRAP" \
    --out "$ROOT/reports/judge_methodC_${key}"
}

analyze_key nano_min
analyze_key nano_low
analyze_key mini_min
analyze_key mini_low

