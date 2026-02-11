#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?}"
: "${PY:?}"

TRAIN_DS="$ROOT/splits/holdout80/train"
TEST_DS="$ROOT/splits/holdout80/test"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_TOKENS_CEILING="${MAX_TOKENS_CEILING:-8000}"

for ds in "$TRAIN_DS" "$TEST_DS"; do
  if [[ ! -f "$ds/dataset_trials.parquet" ]]; then
    echo "Missing dataset: $ds/dataset_trials.parquet"
    exit 1
  fi
done

run_one () {
  local models="$1"
  local key="$2"
  "$PY" scripts/run_trials.py \
    --config configs/default.yml \
    --models "$models" \
    --dataset "$TRAIN_DS" \
    --out "$ROOT/runs/holdout80_pairwise/train/${key}_pairwise_joint" \
    --variant-override short_reason__judge_pairwise_joint \
    --max-tokens "$MAX_TOKENS" \
    --max-tokens-ceiling "$MAX_TOKENS_CEILING" \
    --resume any \
    --max-retries 6 \
    --retry-backoff 5

  "$PY" scripts/run_trials.py \
    --config configs/default.yml \
    --models "$models" \
    --dataset "$TEST_DS" \
    --out "$ROOT/runs/holdout80_pairwise/test/${key}_pairwise_joint" \
    --variant-override short_reason__judge_pairwise_joint \
    --max-tokens "$MAX_TOKENS" \
    --max-tokens-ceiling "$MAX_TOKENS_CEILING" \
    --resume any \
    --max-retries 6 \
    --retry-backoff 5
}

run_one configs/models_gpt5nano_minimal.yml nano_min
run_one configs/models_gpt5nano_low.yml     nano_low
run_one configs/models_gpt5mini_minimal.yml mini_min
run_one configs/models_gpt5mini_low.yml     mini_low

