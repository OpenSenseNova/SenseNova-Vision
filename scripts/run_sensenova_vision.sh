#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' 1>&2
Usage:
  bash scripts/run_sensenova_vision.sh <command> [options]

Commands:
  example
      Run inference/example_visualize.py.

  inference [task] [query] [image_path] [extra inference_demo.py args]
      Run one inference_demo.py task. Positional arguments map to --task,
      --query, and --image_path. Defaults:
        task=raw_query
        query="What are the main objects in this scene and their relationships?"
        image_path=examples/images/1.jpg

  interactive [extra inference_demo.py args]
      Run inference_demo.py --interactive, initialized with Example 1 general
      understanding defaults.

  demo [extra inference/app.py args]
      Run the Gradio web demo via inference/app.py.

  benchmark [tasks] [sub_tasks] [extra inference_benchmark.sh args]
      Run scripts/inference_benchmark.sh.
      tasks supports all, seg, detection, depth, normal.
      sub_tasks supports all or dataset/sub-task names such as pan_coco_val.

  evaluate <output_dir> [tasks] [extra eval_all.sh args]
      Run tools/evaluation/eval_all.sh to compute benchmark metrics.

Common environment defaults:
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT

Examples:
  bash scripts/run_sensenova_vision.sh example
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT bash scripts/run_sensenova_vision.sh inference binary_seg "person" examples/images/2.jpg
  bash scripts/run_sensenova_vision.sh inference raw_query "Describe this image." examples/images/1.jpg --mode understanding --model_path /path/to/SenseNova-Vision-7B-MoT
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT bash scripts/run_sensenova_vision.sh interactive
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT bash scripts/run_sensenova_vision.sh demo --host 0.0.0.0 --port 9001
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT bash scripts/run_sensenova_vision.sh benchmark seg pan_coco_val --num_gpus 8 --tasks_per_gpu 2 --save_pred_masks
  MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT bash scripts/run_sensenova_vision.sh benchmark detection all --data_root /path/to/datas
  bash scripts/run_sensenova_vision.sh evaluate output/benchmark all
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${script_dir}/.." && pwd -P)"

model_path="${MODEL_PATH:-}"
model_path_args=()
if [ -n "${model_path}" ]; then
  model_path_args=(--model_path "${model_path}")
fi
python_cmd="python"

export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"

command="${1:-}"
if [ -z "${command}" ]; then
  usage
  exit 2
fi
shift

case "${command}" in
  example)
    cd "${repo_dir}"
    exec "${python_cmd}" inference/example_visualize.py "$@"
    ;;

  inference)
    task="${1:-raw_query}"
    query="${2:-What are the main objects in this scene and their relationships?}"
    image_path="${3:-examples/images/1.jpg}"
    if [ $# -gt 0 ]; then shift; fi
    if [ $# -gt 0 ]; then shift; fi
    if [ $# -gt 0 ]; then shift; fi
    if [ "${1:-}" = "--" ]; then shift; fi
    cd "${repo_dir}"
    exec "${python_cmd}" -m inference.inference_demo \
      "${model_path_args[@]}" \
      --task "${task}" \
      --query "${query}" \
      --image_path "${image_path}" \
      "$@"
    ;;

  interactive)
    if [ "${1:-}" = "--" ]; then shift; fi
    cd "${repo_dir}"
    exec "${python_cmd}" -m inference.inference_demo \
      "${model_path_args[@]}" \
      --task raw_query \
      --mode understanding \
      --query "What are the main objects in this scene and their relationships?" \
      --image_path "examples/images/1.jpg" \
      --interactive \
      "$@"
    ;;

  demo)
    if [ "${1:-}" = "--" ]; then shift; fi
    demo_host="${GRADIO_SERVER_NAME:-0.0.0.0}"
    demo_port="${GRADIO_SERVER_PORT:-9001}"
    demo_args=("$@")
    demo_arg_count="${#demo_args[@]}"
    demo_arg_index=0
    while [ "${demo_arg_index}" -lt "${demo_arg_count}" ]; do
      demo_arg="${demo_args[${demo_arg_index}]}"
      case "${demo_arg}" in
        --host=*)
          demo_host="${demo_arg#--host=}"
          ;;
        --host)
          if [ $((demo_arg_index + 1)) -lt "${demo_arg_count}" ]; then
            demo_arg_index=$((demo_arg_index + 1))
            demo_host="${demo_args[${demo_arg_index}]}"
          fi
          ;;
        --port=*)
          demo_port="${demo_arg#--port=}"
          ;;
        --port)
          if [ $((demo_arg_index + 1)) -lt "${demo_arg_count}" ]; then
            demo_arg_index=$((demo_arg_index + 1))
            demo_port="${demo_args[${demo_arg_index}]}"
          fi
          ;;
      esac
      demo_arg_index=$((demo_arg_index + 1))
    done
    demo_local_host="${demo_host}"
    if [ "${demo_host}" = "0.0.0.0" ] || [ "${demo_host}" = "::" ]; then
      demo_local_host="127.0.0.1"
    fi
    echo "[SenseNova-Vision] Web demo local URL: http://${demo_local_host}:${demo_port}/"
    if [ "${demo_host}" = "0.0.0.0" ] || [ "${demo_host}" = "::" ]; then
      echo "[SenseNova-Vision] Web demo remote URL: http://<server-ip>:${demo_port}/"
    fi
    if [ -n "${model_path}" ] && [ -z "${SENSENOVA_MODEL_PATH:-}" ]; then
      export SENSENOVA_MODEL_PATH="${model_path}"
    fi
    cd "${repo_dir}"
    exec "${python_cmd}" -m inference.app "${demo_args[@]}"
    ;;

  benchmark)
    if [ $# -gt 0 ] && [[ "${1}" != --* ]]; then
      tasks="$1"
      shift
    else
      tasks="all"
    fi
    if [ "${1:-}" = "--" ]; then
      sub_tasks="all"
      shift
    elif [ $# -gt 0 ] && [[ "${1}" != --* ]]; then
      sub_tasks="$1"
      shift
    else
      sub_tasks="all"
    fi
    if [ "${1:-}" = "--" ]; then
      if [ $# -gt 0 ]; then shift; fi
    fi
    cd "${repo_dir}"
    exec bash scripts/inference_benchmark.sh \
      "${model_path_args[@]}" \
      --tasks "${tasks}" \
      --sub_tasks "${sub_tasks}" \
      "$@"
    ;;

  evaluate)
    if [ $# -lt 1 ]; then
      echo "[ERROR] evaluate requires <output_dir>." 1>&2
      usage
      exit 2
    fi
    output_dir="$1"
    shift
    if [ "${1:-}" = "--" ]; then shift; fi
    cd "${repo_dir}"
    exec bash tools/evaluation/eval_all.sh \
      --output_dir "${output_dir}" \
      "$@"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "[ERROR] Unknown command: ${command}" 1>&2
    usage
    exit 2
    ;;
esac
