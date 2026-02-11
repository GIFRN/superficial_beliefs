#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?}"
: "${PY:?}"

TRAIN_DS="$ROOT/splits/holdout80/train"
TEST_DS="$ROOT/splits/holdout80/test"

analyze_key () {
  local key="$1"
  local train_run
  local test_run

  train_run="$(compgen -G "$ROOT/runs/holdout80_pairwise/train/${key}_pairwise_joint__"* | head -n1)"
  test_run="$(compgen -G "$ROOT/runs/holdout80_pairwise/test/${key}_pairwise_joint__"* | head -n1)"
  if [[ -z "$train_run" || -z "$test_run" ]]; then
    echo "Missing pairwise run dirs for key=$key"
    return 1
  fi

  local stagea="$ROOT/results/stage_A_holdout80_pairwise_train_${key}"
  "$PY" scripts/fit_stageA.py \
    --config configs/default.yml \
    --dataset "$TRAIN_DS" \
    --responses "$train_run/responses.jsonl" \
    --out "$stagea"

  "$PY" scripts/stageB_alignment.py \
    --stageA "$stagea" \
    --model-dir "$stagea" \
    --responses "$train_run/responses.jsonl" \
    --out "$ROOT/results/stage_B_holdout80_pairwise_train_${key}"

  "$PY" scripts/stageB_alignment.py \
    --stageA "$stagea" \
    --model-dir "$stagea" \
    --dataset "$TEST_DS" \
    --responses "$test_run/responses.jsonl" \
    --out "$ROOT/results/stage_B_holdout80_pairwise_test_${key}"

  "$PY" scripts/analyze_judge_baselines.py \
    --dataset "$TRAIN_DS" \
    --responses "$train_run/responses.jsonl" \
    --out "$ROOT/reports/judge_holdout80_pairwise_train_${key}"

  "$PY" scripts/analyze_judge_baselines.py \
    --dataset "$TEST_DS" \
    --responses "$test_run/responses.jsonl" \
    --out "$ROOT/reports/judge_holdout80_pairwise_test_${key}"
}

analyze_key nano_min
analyze_key nano_low
analyze_key mini_min
analyze_key mini_low

