#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

set -euo pipefail

config_name="${1:-data/configs/cv_unify/cv_unify_baseline_v9.yaml}"

repo_dir="${SENSENOVA_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
conda_env_path="${SENSENOVA_ENV_PATH:-}"
base_model_dir="${SENSENOVA_MODEL_DIR:?Set SENSENOVA_MODEL_DIR to a sensenova-vision checkpoint directory.}"
finetune_base_weight="${SENSENOVA_FINETUNE_FROM:-${base_model_dir}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${conda_env_path}" ]; then
  export PYTHONPATH="${conda_env_path}:${repo_dir}:${PYTHONPATH:-}"
  if [ "${CONDA_DEFAULT_ENV:-}" != "${conda_env_path}" ]; then
    . /usr/local/lib/miniconda3/bin/activate "${conda_env_path}"
  fi
else
  export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
fi

cd "${repo_dir}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${SENSENOVA_TEST_NPROC:-1}" \
  --master_addr="${SENSENOVA_TEST_MASTER_ADDR:-localhost}" \
  --master_port="${SENSENOVA_TEST_MASTER_PORT:-12348}" \
  train/test_dataset.py \
  --dataset_config_file "${config_name}" \
  --num_workers 0 \
  --layer_module Qwen2MoTDecoderLayer \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --save_every 100000 \
  --total_steps "${SENSENOVA_TEST_TOTAL_STEPS:-2000}" \
  --lr 2e-5 \
  --max_num_tokens 36864 \
  --expected_num_tokens 32768 \
  --max_num_tokens_per_sample 24064 \
  --prefer_buffer_before 15360 \
  --resume_from "${finetune_base_weight}" \
  --resume_from_hf "${base_model_dir}" \
  --results_dir "./results/${config_name%.yaml}/" \
  --checkpoint_dir "./results/${config_name%.yaml}/checkpoint" \
  --wandb_offline True \
  --max_latent_size 64
