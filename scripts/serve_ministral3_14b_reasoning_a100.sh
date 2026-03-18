#!/usr/bin/env bash
set -euo pipefail

# Serve Ministral-3-14B-Reasoning-2512 on a single A100 via vLLM.
#
# Profiles:
# - PROFILE=auto (default): detect A100 80GB vs ~40GB by nvidia-smi memory.
# - PROFILE=a100_80gb: official BF16 weights defaults.
# - PROFILE=a100_40gb: non-quant fallback tuned for 40GB class GPUs.
#
# Examples:
#   nohup PROFILE=a100_80gb bash scripts/serve_ministral3_14b_reasoning_a100.sh > data/logs/serve_ministral14b.log 2>&1 &
#   nohup PROFILE=a100_40gb bash scripts/serve_ministral3_14b_reasoning_a100.sh > data/logs/serve_ministral14b_40gb.log 2>&1 &

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found in PATH. Install vLLM first." >&2
  exit 1
fi

PROFILE="${PROFILE:-auto}"
GPU_MEM_MB="${GPU_MEM_MB:-}"
if [[ -z "${GPU_MEM_MB}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)"
  if [[ ! "${GPU_MEM_MB}" =~ ^[0-9]+$ ]]; then
    GPU_MEM_MB=""
  fi
fi

if [[ "${PROFILE}" == "auto" ]]; then
  if [[ -n "${GPU_MEM_MB}" && "${GPU_MEM_MB}" -ge 70000 ]]; then
    PROFILE="a100_80gb"
  else
    PROFILE="a100_40gb"
  fi
fi

: "${HOST:=0.0.0.0}"
: "${TP_SIZE:=1}"
: "${MAX_NUM_SEQS:=16}"
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENFORCE_EAGER:=0}"

if [[ "${PROFILE}" == "a100_80gb" ]]; then
  : "${MODEL:=mistralai/Ministral-3-14B-Reasoning-2512}"
  : "${SERVED_MODEL_NAME:=ministral_3_14b_reasoning_local}"
  : "${PORT:=8020}"
  : "${DTYPE:=bfloat16}"
  : "${QUANTIZATION:=}"
  : "${USE_MISTRAL_LOAD_FORMAT:=1}"
  : "${MAX_MODEL_LEN:=32768}"
  : "${MAX_NUM_BATCHED_TOKENS:=8192}"
  : "${GPU_MEMORY_UTILIZATION:=0.92}"
elif [[ "${PROFILE}" == "a100_40gb" ]]; then
  # Keep served model name compatible with existing experiment configs.
  # On 40GB devices, prefer the official non-quantized checkpoint to avoid
  # quantization backend incompatibilities.
  : "${MODEL:=mistralai/Ministral-3-14B-Reasoning-2512}"
  : "${SERVED_MODEL_NAME:=ministral_3_14b_reasoning_local}"
  : "${PORT:=8020}"
  : "${DTYPE:=bfloat16}"
  : "${QUANTIZATION:=}"
  : "${USE_MISTRAL_LOAD_FORMAT:=1}"
  : "${MAX_MODEL_LEN:=8192}"
  : "${MAX_NUM_BATCHED_TOKENS:=2048}"
  : "${MAX_NUM_SEQS:=8}"
  : "${GPU_MEMORY_UTILIZATION:=0.88}"
  : "${ENFORCE_EAGER:=1}"
else
  echo "Unsupported PROFILE=${PROFILE}. Use auto|a100_80gb|a100_40gb." >&2
  exit 1
fi

echo "Starting Ministral-3-14B-Reasoning server with:"
echo "  PROFILE=${PROFILE}"
echo "  MODEL=${MODEL}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  PORT=${PORT}"
echo "  DTYPE=${DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION:-none}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  ENFORCE_EAGER=${ENFORCE_EAGER}"

ARGS=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --reasoning-parser mistral
  --tokenizer-mode mistral
  --config-format mistral
  --generation-config vllm
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ -n "${QUANTIZATION}" ]]; then
  ARGS+=(--quantization "${QUANTIZATION}")
elif [[ "${USE_MISTRAL_LOAD_FORMAT}" == "1" ]]; then
  ARGS+=(--load-format mistral)
fi

exec vllm "${ARGS[@]}"
