#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

set -euo pipefail

config_name="${1:-data/configs/cv_unify/cv_unify_baseline_v9.yaml}"

repo_dir="${SENSENOVA_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
conda_env_path="${SENSENOVA_ENV_PATH:-}"
base_model_dir="${SENSENOVA_MODEL_DIR:?Set SENSENOVA_MODEL_DIR to a sensenova-vision checkpoint directory.}"
finetune_base_weight="${SENSENOVA_FINETUNE_FROM:-${base_model_dir}}"

num_shard=16
grad_acc=2

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

gpu_per_node="${SENSENOVA_GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
num_nodes="${WORLD_SIZE:-1}"
if [ -n "${num_shard}" ] && [ "${num_shard}" -gt 0 ]; then
  (( num_nodes * gpu_per_node % num_shard == 0 )) || {
    >&2 echo "Error: num_shard must divide num_nodes * gpu_per_node"
    exit 1
  }
  num_replicate=$((num_nodes * gpu_per_node / num_shard))
else
  num_shard=$((num_nodes * gpu_per_node))
  num_replicate=1
fi

echo "config = ${config_name}"
echo "repo_dir = ${repo_dir}"
echo "nproc_per_node = ${gpu_per_node}"
echo "num_shard = ${num_shard}"
echo "num_replicate = ${num_replicate}"
echo "grad_acc = ${grad_acc}"

py_cmd=(
  train/pretrain_unified_navit.py
  --dataset_config_file "${config_name}"
  --layer_module Qwen2MoTDecoderLayer
  --auto_resume True
  --resume-model-only True
  --finetune-from-ema True
  --log_every 10
  --save_every 1000
  --lr 2.5e-5
  --resume_from "${finetune_base_weight}"
  --resume_from_hf "${base_model_dir}"
  --results_dir "./results/${config_name%.yaml}/"
  --checkpoint_dir "./results/${config_name%.yaml}/checkpoint"
  --gradient_accumulation_steps "${grad_acc}"
  --split_vae_encode True
  --wandb_offline True
  --use_flex True
  --max_latent_size 64
  --ema 0.995
  --total_steps 200000
  --warmup_steps 500
  --timestep_shift 4.0
  --ce_weight 0.25
  --max_num_tokens 36864
  --expected_num_tokens 32768
  --max_num_tokens_per_sample 24064
  --prefer_buffer_before 15360
  --text_cond_dropout_prob 0.05
  --vit_cond_dropout_prob 0.1
  --vae_cond_dropout_prob 0.1
  --num_shard "${num_shard}"
  --num_replicate "${num_replicate}"
  --copy_init_moe False
  --data_seed 42
)

if [ "${WORLD_SIZE:-0}" -gt 1 ]; then
  torchrun --nnodes "${WORLD_SIZE}" \
    --nproc_per_node "${gpu_per_node}" \
    --node_rank "${RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${py_cmd[@]}"
else
  torchrun --nproc_per_node "${gpu_per_node}" "${py_cmd[@]}"
fi
