#!/usr/bin/env bash
set -euo pipefail

# Install runtime dependencies for local vLLM serving on a single A100.
#
# Usage:
#   bash scripts/setup_vllm_single_a100.sh
#   INSTALL_CHANNEL=stable bash scripts/setup_vllm_single_a100.sh
#
# Notes:
# - Qwen3.5 model card currently recommends vLLM nightly.
# - Ministral-3-14B-Reasoning-2512 requires vLLM >= 0.12.0 and mistral_common >= 1.8.6.

: "${PY:=python}"
: "${INSTALL_CHANNEL:=nightly}"  # nightly | stable
: "${WORK_TMP_ROOT:=/vol/bitbucket/${USER:-$(id -un)}/.pip_work}"

# Avoid tiny /tmp on cluster nodes during large wheel installs.
mkdir -p "${WORK_TMP_ROOT}/tmp" "${WORK_TMP_ROOT}/pip-cache"
export TMPDIR="${WORK_TMP_ROOT}/tmp"
export PIP_CACHE_DIR="${WORK_TMP_ROOT}/pip-cache"

"${PY}" -m pip install -U pip setuptools wheel

if [[ "${INSTALL_CHANNEL}" == "nightly" ]]; then
  # Prefer nightly wheels over source builds to avoid huge build deps.
  "${PY}" -m pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
else
  "${PY}" -m pip install -U "vllm>=0.12.0"
fi

"${PY}" -m pip install -U "mistral_common>=1.8.6"

echo "Done. Verify with:"
echo "  ${PY} -c 'import vllm; print(vllm.__version__)'"
