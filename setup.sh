#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

set -euo pipefail

ENV_NAME="${1:-}"
if [ -z "${ENV_NAME}" ]; then
    echo "Usage: bash setup.sh <env_name>" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    CONDA_SH_CANDIDATES=()
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_ROOT="$(cd -- "$(dirname -- "${CONDA_EXE}")/.." && pwd -P)"
        CONDA_SH_CANDIDATES+=("${CONDA_ROOT}/etc/profile.d/conda.sh")
    fi
    if [ -n "${HOME:-}" ]; then
        CONDA_SH_CANDIDATES+=(
            "${HOME}/miniconda3/etc/profile.d/conda.sh"
            "${HOME}/anaconda3/etc/profile.d/conda.sh"
        )
    fi
    CONDA_SH_CANDIDATES+=("/opt/conda/etc/profile.d/conda.sh")

    for CONDA_SH in "${CONDA_SH_CANDIDATES[@]}"; do
        if [ -f "${CONDA_SH}" ]; then
            # shellcheck disable=SC1090
            source "${CONDA_SH}"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available in PATH." >&2
    exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "Error: CUDA toolkit was not found at ${CUDA_HOME}." >&2
    echo "Set CUDA_HOME to a CUDA 12.x toolkit directory and retry." >&2
    exit 1
fi

CUDA_VERSION="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VERSION%%.*}"
if [ -z "${CUDA_VERSION}" ] || [ "${CUDA_MAJOR}" != "12" ]; then
    echo "Error: CUDA toolkit 12.x is required; found ${CUDA_VERSION:-unknown}." >&2
    exit 1
fi
export CUDA_HOME

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

echo "--- Creating Conda environment: ${ENV_NAME} ---"
conda env create --name "${ENV_NAME}" --file environment.yml

eval "$(conda shell.bash hook)"
export JAVA_HOME="${JAVA_HOME:-}"
export JAVA_LD_LIBRARY_PATH="${JAVA_LD_LIBRARY_PATH:-}"
conda activate "${ENV_NAME}"

echo "--- Installing Python dependencies ---"
python -m pip install --requirement requirements.txt --constraint constraints.txt

echo "--- Installing flash-attn 2.6.3 ---"
python -m pip install flash-attn==2.6.3 --no-build-isolation

echo "--- Installing segmentation evaluation extension ---"
python -m pip install "panopticapi @ https://github.com/cocodataset/panopticapi/archive/7bb4655548f98f3fedc07bf37e9040a992b054b0.zip" --constraint constraints.txt

echo "--- Installing detection evaluation extension ---"
python -m pip install -e tools/evaluation/detect/evaluation/fastevaluate

echo "--- Verifying installed dependencies ---"
if ! PIP_CHECK_OUTPUT="$(python -m pip check 2>&1)"; then
    echo "${PIP_CHECK_OUTPUT}" >&2
    PIP_CHECK_REMAINDER="$(printf '%s\n' "${PIP_CHECK_OUTPUT}" | grep -v '^decord 0\.6\.0 is not supported on this platform$' || true)"
    if [ -n "${PIP_CHECK_REMAINDER}" ]; then
        exit 1
    fi
    echo "Warning: ignoring known pip metadata warning for decord 0.6.0." >&2
else
    echo "${PIP_CHECK_OUTPUT}"
fi
python -c "import decord, fastevaluate, flash_attn, panopticapi, torch; print(f'decord={getattr(decord, \"__version__\", \"unknown\")} torch={torch.__version__} cuda={torch.version.cuda} flash_attn={flash_attn.__version__} panopticapi=ok fastevaluate=ok')"

echo "--- Environment '${ENV_NAME}' created successfully ---"
echo "To activate it, run: conda activate ${ENV_NAME}"
