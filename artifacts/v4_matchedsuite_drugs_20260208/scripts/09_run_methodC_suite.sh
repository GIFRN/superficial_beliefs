#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?}"
: "${PY:?}"

DS="${DS:-$ROOT/data/occlusion_suite_methodC}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_TOKENS_CEILING="${MAX_TOKENS_CEILING:-8000}"

if [[ ! -f "$DS/dataset_trials.parquet" ]]; then
  echo "Missing dataset: $DS/dataset_trials.parquet"
  exit 1
fi

run_one () {
  local models="$1"
  local key="$2"
  "$PY" scripts/run_trials.py \
    --config configs/default.yml \
    --models "$models" \
    --dataset "$DS" \
    --out "$ROOT/runs/methodC/${key}_joint" \
    --variant-override short_reason__judge_scores_joint \
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

