#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/final_same_order}"
PY="${PY:-python}"
THEMES="${THEMES:-drugs,policy,software,placebo_packaging,placebo_label_border}"

RUN_OPENAI="${RUN_OPENAI:-1}"
RUN_LOCAL_QWEN="${RUN_LOCAL_QWEN:-1}"
RUN_LOCAL_MINISTRAL="${RUN_LOCAL_MINISTRAL:-1}"

OPENAI_TRIAL_CONCURRENCY="${OPENAI_TRIAL_CONCURRENCY:-12}"
QWEN_TRIAL_CONCURRENCY="${QWEN_TRIAL_CONCURRENCY:-1}"
MINISTRAL_TRIAL_CONCURRENCY="${MINISTRAL_TRIAL_CONCURRENCY:-1}"
LOCAL_SERVER_WAIT_TIMEOUT="${LOCAL_SERVER_WAIT_TIMEOUT:-900}"

wait_for_endpoint() {
  local name="$1"
  local url="$2"
  local timeout_s="$3"
  local start_ts
  start_ts="$(date +%s)"

  echo "waiting for ${name} endpoint: ${url}"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} endpoint is ready"
      return 0
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_s )); then
      echo "timed out waiting for ${name} endpoint: ${url}" >&2
      return 1
    fi
    sleep 5
  done
}

mkdir -p "$OUT_ROOT/logs"

"$PY" "$ROOT/scripts/build_final_themed_datasets.py" --out-root "$OUT_ROOT" --themes "$THEMES"

if [[ "$RUN_OPENAI" == "1" ]]; then
  nohup "$PY" "$ROOT/scripts/run_final_benchmark.py" \
    --out-root "$OUT_ROOT" \
    --themes "$THEMES" \
    --providers openai \
    --openai-concurrency "$OPENAI_TRIAL_CONCURRENCY" \
    > "$OUT_ROOT/logs/launcher_openai.log" 2>&1 &
fi

if [[ "$RUN_LOCAL_QWEN" == "1" ]]; then
  wait_for_endpoint "qwen" "http://127.0.0.1:8010/v1/models" "$LOCAL_SERVER_WAIT_TIMEOUT"
  nohup "$PY" "$ROOT/scripts/run_final_benchmark.py" \
    --out-root "$OUT_ROOT" \
    --themes "$THEMES" \
    --providers qwen \
    --qwen-concurrency "$QWEN_TRIAL_CONCURRENCY" \
    > "$OUT_ROOT/logs/launcher_qwen.log" 2>&1 &
fi

if [[ "$RUN_LOCAL_MINISTRAL" == "1" ]]; then
  wait_for_endpoint "ministral" "http://127.0.0.1:8020/v1/models" "$LOCAL_SERVER_WAIT_TIMEOUT"
  nohup "$PY" "$ROOT/scripts/run_final_benchmark.py" \
    --out-root "$OUT_ROOT" \
    --themes "$THEMES" \
    --providers ministral \
    --ministral-concurrency "$MINISTRAL_TRIAL_CONCURRENCY" \
    > "$OUT_ROOT/logs/launcher_ministral.log" 2>&1 &
fi

echo "launched final benchmark workers under $OUT_ROOT/logs"
