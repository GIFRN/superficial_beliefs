#!/usr/bin/env bash
set -euo pipefail
: "${ROOT:?}"; : "${PY:?}"

TRAIN_DS="$ROOT/splits/holdout80/train"
TEST_DS="$ROOT/splits/holdout80/test"

run_one () {
  local models="$1"
  local key="$2"
  "$PY" scripts/run_trials.py \
    --config configs/default.yml \
    --models "$models" \
    --dataset "$TRAIN_DS" \
    --out "$ROOT/runs/holdout80/train/${key}_joint" \
    --variant-override short_reason__judge_scores_joint \
    --max-tokens 1024 \
    --max-tokens-ceiling 8000 \
    --resume any \
    --max-retries 6 \
    --retry-backoff 5
  "$PY" scripts/run_trials.py \
    --config configs/default.yml \
    --models "$models" \
    --dataset "$TEST_DS" \
    --out "$ROOT/runs/holdout80/test/${key}_joint" \
    --variant-override short_reason__judge_scores_joint \
    --max-tokens 1024 \
    --max-tokens-ceiling 8000 \
    --resume any \
    --max-retries 6 \
    --retry-backoff 5
}
run_one configs/models_gpt5nano_minimal.yml nano_min
run_one configs/models_gpt5nano_low.yml     nano_low
run_one configs/models_gpt5mini_minimal.yml mini_min
run_one configs/models_gpt5mini_low.yml     mini_low
