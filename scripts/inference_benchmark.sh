#!/usr/bin/env bash
set -u
set -o pipefail

# ============================================================
# Local benchmark inference entrypoint.
# Runs segmentation, depth, normal, and detection.
# ============================================================

usage() {
  cat <<USAGE 1>&2
Usage:
  bash $0 [options]

Options:
  --model_path PATH        SenseNova-Vision model directory or Hugging Face repo id.
                           Default: sensenova/SenseNova-Vision-7B-MoT, or MODEL_PATH if set.
  --output_dir DIR         Output directory. Default: ./output/benchmark
  --tasks TASKS            all | seg | depth | normal | detection | recon3d | camera_pose. Comma-separated values are supported.
  --sub_tasks TASKS        Dataset/sub-task selector. Default: all. Comma-separated values are supported.
                          Supported segment subtasks: pan_coco_val, ade20k_pan_val, gcg_val, gcg_test,
                          refcoco_val, refcocop_val, refcocog_val, reason_val, reason_test.
  --data_root DIR          Benchmark data directory. Default: ./datas
                           Pass the datas/ directory itself, for example /path/to/datas.
                           Keep the directory name datas for the provided segmentation JSONL files.
  --jsonl_root DIR         Benchmark JSONL directory. Default: ./jsonl_generate
                           Contains files such as seg_*.jsonl and detection/*.jsonl.
  --num_gpus N             Number of GPUs. Default: 8
  --tasks_per_gpu N        Concurrent jobs per GPU. Default: 1
  --total_test_length N    Optional sample limit per dataset/task for quick validation runs.
  --save_pred_masks        Save restored segmentation prediction masks.
  -h, --help               Show this help.

Backward-compatible positional form:
  bash $0 <model_path> [output_dir] [tasks] [num_gpus] [tasks_per_gpu]

Examples:
  bash $0 --output_dir /tmp/out --tasks seg --sub_tasks pan_coco_val --total_test_length 16
  bash $0 --model_path /path/to/SenseNova-Vision-7B-MoT --output_dir /tmp/out --tasks seg,depth,normal --num_gpus 4 --tasks_per_gpu 1
  bash $0 /path/to/SenseNova-Vision-7B-MoT /tmp/out seg,depth,normal 4 1

Environment overrides:
  REPO_DIR=/path/to/repo
USAGE
}

DEFAULT_MODEL_PATH="sensenova/SenseNova-Vision-7B-MoT"
MODEL_PATH_ARG="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MODEL_PATH_SET_BY_OPTION="0"
OUTPUT_DIR=""
data_root_arg=""
JSONL_ROOT_ARG=""
TASKS="all"
SUB_TASKS="all"
NUM_GPUS="8"
TASKS_PER_GPU="1"
TOTAL_TEST_LENGTH=""
SAVE_PRED_MASKS="0"
POSITIONAL_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model_path)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      MODEL_PATH_ARG="$2"; MODEL_PATH_SET_BY_OPTION="1"; shift 2 ;;
    --output_dir)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      OUTPUT_DIR="$2"; shift 2 ;;
    --tasks)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      TASKS="$2"; shift 2 ;;
    --sub_tasks)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      SUB_TASKS="$2"; shift 2 ;;
    --data_root)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      data_root_arg="$2"; shift 2 ;;
    --jsonl_root)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      JSONL_ROOT_ARG="$2"; shift 2 ;;
    --num_gpus)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      NUM_GPUS="$2"; shift 2 ;;
    --tasks_per_gpu)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      TASKS_PER_GPU="$2"; shift 2 ;;
    --total_test_length)
      if [ $# -lt 2 ]; then echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; fi
      TOTAL_TEST_LENGTH="$2"; shift 2 ;;
    --save_pred_masks)
      SAVE_PRED_MASKS="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      while [ $# -gt 0 ]; do POSITIONAL_ARGS+=("$1"); shift; done ;;
    --*)
      echo "[ERROR] Unknown option: $1" 1>&2
      usage
      exit 2 ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift ;;
  esac
done

if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
  if [ "${MODEL_PATH_SET_BY_OPTION}" = "1" ]; then
    echo "[ERROR] Do not mix positional model_path with --model_path." 1>&2
    usage
    exit 2
  fi
  if [ ${#POSITIONAL_ARGS[@]} -gt 5 ]; then
    echo "[ERROR] Too many positional arguments: ${#POSITIONAL_ARGS[@]}" 1>&2
    usage
    exit 2
  fi
  MODEL_PATH_ARG="${POSITIONAL_ARGS[0]}"
  OUTPUT_DIR="${POSITIONAL_ARGS[1]:-${OUTPUT_DIR}}"
  TASKS="${POSITIONAL_ARGS[2]:-${TASKS}}"
  NUM_GPUS="${POSITIONAL_ARGS[3]:-${NUM_GPUS}}"
  TASKS_PER_GPU="${POSITIONAL_ARGS[4]:-${TASKS_PER_GPU}}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
default_repo_dir="$(cd -- "${script_dir}/.." && pwd -P)"
repo_dir="${REPO_DIR:-${default_repo_dir}}"
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${repo_dir}/output/benchmark"
fi
model_path="${MODEL_PATH_ARG}"
data_root="${data_root_arg:-${repo_dir}/datas}"
geometry_data_root="${data_root%/}/geometry_data"
detection_data_root="${data_root%/}/detection_data"
multiview3d_data_root="${data_root%/}/multiview3d_data"
seg_jsonl_path_root="$(dirname -- "${data_root%/}")"
jsonl_root="${JSONL_ROOT_ARG:-${repo_dir}/jsonl_generate}"
detection_jsonl_dir="${jsonl_root}/detection"
python_cmd="python"
dry_run="${LOCAL_INFER_DRY_RUN:-0}"

is_positive_int() {
  local value="$1"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]]
}

validate_tasks() {
  local raw_tasks="$1"
  local task
  if [ "${raw_tasks}" = "all" ]; then
    return 0
  fi
  IFS=',' read -r -a task_items <<< "${raw_tasks}"
  if [ ${#task_items[@]} -eq 0 ]; then
    echo "[ERROR] Empty tasks value." 1>&2
    return 1
  fi
  for task in "${task_items[@]}"; do
    case "${task}" in
      seg|depth|normal|detection|recon3d|camera_pose) ;;
      "") echo "[ERROR] Empty task in tasks='${raw_tasks}'." 1>&2; return 1 ;;
      *) echo "[ERROR] Unknown task: ${task}. Supported: all, seg, depth, normal, detection, recon3d, camera_pose" 1>&2; return 1 ;;
    esac
  done
}

validate_sub_tasks() {
  local raw_tasks="$1"
  local task
  if [ "${raw_tasks}" = "all" ]; then
    return 0
  fi
  IFS=',' read -r -a task_items <<< "${raw_tasks}"
  if [ ${#task_items[@]} -eq 0 ]; then
    echo "[ERROR] Empty sub_tasks value." 1>&2
    return 1
  fi
  for task in "${task_items[@]}"; do
    case "${task}" in
      pan_coco_val|ade20k_pan_val|gcg_val|gcg_test|refcoco_val|refcocop_val|refcocog_val|reason_val|reason_test) ;;
      "") echo "[ERROR] Empty task in sub_tasks='${raw_tasks}'." 1>&2; return 1 ;;
      *) echo "[ERROR] Unknown dataset/sub-task: ${task}." 1>&2; return 1 ;;
    esac
  done
}

if ! is_positive_int "${NUM_GPUS}"; then
  echo "[ERROR] num_gpus must be a positive integer, got '${NUM_GPUS}'." 1>&2
  exit 2
fi
if ! is_positive_int "${TASKS_PER_GPU}"; then
  echo "[ERROR] tasks_per_gpu must be a positive integer, got '${TASKS_PER_GPU}'." 1>&2
  exit 2
fi
if [ -n "${TOTAL_TEST_LENGTH}" ] && ! is_positive_int "${TOTAL_TEST_LENGTH}"; then
  echo "[ERROR] total_test_length must be a positive integer, got '${TOTAL_TEST_LENGTH}'." 1>&2
  exit 2
fi
validate_tasks "${TASKS}" || exit 2
validate_sub_tasks "${SUB_TASKS}" || exit 2

want_task() {
  local name="$1"
  if [ "${TASKS}" = "all" ]; then
    return 0
  fi
  case ",${TASKS}," in
    *,"${name}",*) return 0 ;;
    *) return 1 ;;
  esac
}

want_sub_task() {
  local name="$1"
  if [ "${SUB_TASKS}" = "all" ]; then
    return 0
  fi
  case ",${SUB_TASKS}," in
    *,"${name}",*) return 0 ;;
    *) return 1 ;;
  esac
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "${path}" ]; then
    echo "[ERROR] ${label} directory not found: ${path}" 1>&2
    return 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] ${label} file not found: ${path}" 1>&2
    return 1
  fi
}

jsonl_prepare_hint() {
  echo "        Prepare benchmark JSONL files under: ${jsonl_root}" 1>&2
  echo "        See: ${repo_dir}/docs/data_prepare.md" 1>&2
}

require_jsonl_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] ${label} JSONL not found: ${path}" 1>&2
    jsonl_prepare_hint
    return 1
  fi
}

if [ ! -d "${repo_dir}" ]; then
  echo "[ERROR] repo_dir not found: ${repo_dir}" 1>&2
  exit 2
fi
if ! command -v "${python_cmd}" >/dev/null 2>&1; then
  echo "[ERROR] python is not available in the current environment." 1>&2
  echo "        Prepare and activate the runtime environment before launching benchmark." 1>&2
  exit 2
fi
for required_script in \
  inference/benchmark/batch_dense_geometry.py \
  inference/benchmark/batch_panoptic_segment.py \
  inference/benchmark/batch_gcg_segment.py \
  inference/benchmark/batch_binary_segment.py \
  inference/benchmark/batch_detect.py \
  inference/benchmark/batch_recon3d.py \
  inference/benchmark/batch_camera_pose.py; do
  if [ ! -f "${repo_dir}/${required_script}" ]; then
    echo "[ERROR] required script not found: ${repo_dir}/${required_script}" 1>&2
    exit 2
  fi
done

validate_selected_inputs() {
  local missing=0

  if want_task "depth" || want_task "normal"; then
    require_dir "${geometry_data_root}" "geometry_data_root" || missing=1
  fi

  if want_task "seg"; then
    if want_sub_task "pan_coco_val"; then
      require_file "${data_root}/gen_seg_data/coco2017/annotations/panoptic_val2017.json" "COCO panoptic annotation" || missing=1
    fi
    if want_sub_task "ade20k_pan_val"; then
      require_file "${data_root}/ov_seg_data/ade20k/ade20k_panoptic_val.json" "ADE20K panoptic annotation" || missing=1
    fi
  fi

  if want_task "detection"; then
    require_dir "${detection_data_root}" "detection_data_root" || missing=1
  fi

  if want_task "recon3d" || want_task "camera_pose"; then
    require_dir "${multiview3d_data_root}" "multiview3d_data_root" || missing=1
  fi

  return "${missing}"
}

validate_selected_inputs || exit 2

MAX_JOBS=$((NUM_GPUS * TASKS_PER_GPU))
LOG_DIR="${OUTPUT_DIR%/}/local_logs"
STATUS_DIR="${LOG_DIR}/.status_$(date +%Y%m%d_%H%M%S)_$$"

export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${STATUS_DIR}"

total_test_length_args=()
if [ -n "${TOTAL_TEST_LENGTH}" ]; then
  total_test_length_args=(--total_test_length "${TOTAL_TEST_LENGTH}")
fi

save_pred_masks_args=()
if [ "${SAVE_PRED_MASKS}" = "1" ]; then
  save_pred_masks_args=(--save_pred_masks)
fi

GPU_PIDS=()
for ((gpu_init = 0; gpu_init < NUM_GPUS; gpu_init++)); do
  GPU_PIDS[${gpu_init}]=""
done

prune_gpu_pids() {
  local gpu_id="$1"
  local pid kept_pids=""
  for pid in ${GPU_PIDS[${gpu_id}]:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kept_pids="${kept_pids} ${pid}"
    fi
  done
  GPU_PIDS[${gpu_id}]="${kept_pids# }"
}

count_gpu_pids() {
  local gpu_id="$1"
  local pid count=0
  prune_gpu_pids "${gpu_id}"
  for pid in ${GPU_PIDS[${gpu_id}]:-}; do
    count=$((count + 1))
  done
  echo "${count}"
}

count_running_jobs() {
  local gpu_id total=0
  for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
    total=$((total + $(count_gpu_pids "${gpu_id}")))
  done
  echo "${total}"
}

acquire_gpu_slot() {
  local gpu_id running_on_gpu
  while true; do
    for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
      running_on_gpu="$(count_gpu_pids "${gpu_id}")"
      if (( running_on_gpu < TASKS_PER_GPU )); then
        echo "${gpu_id}"
        return 0
      fi
    done
    echo "[$(date '+%H:%M:%S')] Waiting for available GPU slot... running=$(count_running_jobs)/${MAX_JOBS}" 1>&2
    sleep 20
  done
}

launch_job() {
  local job_name="$1"
  shift
  local log_file="$1"
  shift
  local gpu_id
  if [ "${dry_run}" = "1" ]; then
    gpu_id=$(( (JOB_IDX / TASKS_PER_GPU) % NUM_GPUS ))
  else
    gpu_id="$(acquire_gpu_slot)"
  fi

  mkdir -p "$(dirname "${log_file}")"
  echo "[+] Launching ${job_name} on GPU ${gpu_id}"
  if [ "${dry_run}" = "1" ]; then
    printf '[DRY-RUN] CUDA_VISIBLE_DEVICES=%s' "${gpu_id}"
    printf ' %q' "$@"
    printf ' > %q 2>&1\n' "${log_file}"
    echo "0" > "${STATUS_DIR}/${job_name}.status"
    JOB_IDX=$((JOB_IDX + 1))
    return 0
  fi

  (
    cd "${repo_dir}" || exit 1
    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" > "${log_file}" 2>&1
    rc=$?
    if [ ${rc} -eq 0 ]; then
      echo "0" > "${STATUS_DIR}/${job_name}.status"
      echo "[OK] ${job_name}"
    else
      echo "${rc}" > "${STATUS_DIR}/${job_name}.status"
      echo "[FAIL] ${job_name} exit=${rc}; log=${log_file}" 1>&2
    fi
    exit ${rc}
  ) &
  local pid=$!
  GPU_PIDS[${gpu_id}]="${GPU_PIDS[${gpu_id}]:-} ${pid}"

  JOB_IDX=$((JOB_IDX + 1))
}

run_depth() {
  local datasets=(nyudepth_v2 kitti eth3d scannet diode)
  local out_dir="${OUTPUT_DIR%/}"
  for test_dataset in "${datasets[@]}"; do
    local job_name="depth_${test_dataset}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" inference/benchmark/batch_dense_geometry.py \
        --model_path "${model_path}" \
        --dataset "${test_dataset}" \
        --test_mode Depth \
        --data_root "${geometry_data_root}" \
        --output_dir "${out_dir}" \
        "${total_test_length_args[@]}"
  done
}

run_normal() {
  local datasets=(nyu scannet ibims)
  local out_dir="${OUTPUT_DIR%/}"
  for test_dataset in "${datasets[@]}"; do
    local job_name="normal_${test_dataset}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" inference/benchmark/batch_dense_geometry.py \
        --model_path "${model_path}" \
        --dataset "${test_dataset}" \
        --test_mode Normal \
        --data_root "${geometry_data_root}" \
        --output_dir "${out_dir}" \
        "${total_test_length_args[@]}"
  done
}

run_segment_task() {
  local task_name="$1"
  local script_name input_jsonl out_subdir total_len
  local extra_args=()
  local total_split="${MAX_JOBS}"

  case "${task_name}" in
    pan_coco_val)
      script_name="inference/benchmark/batch_panoptic_segment.py"
      input_jsonl="${jsonl_root}/seg_panoptic_coco_val.jsonl"
      out_subdir="segmentation/pan/val"
      total_len=5000
      extra_args=(--coco_json "${data_root}/gen_seg_data/coco2017/annotations/panoptic_val2017.json")
      ;;
    ade20k_pan_val)
      script_name="inference/benchmark/batch_panoptic_segment.py"
      input_jsonl="${jsonl_root}/seg_ade20k_panoptic_val.jsonl"
      out_subdir="segmentation/ade/val"
      total_len=2000
      extra_args=(--coco_json "${data_root}/ov_seg_data/ade20k/ade20k_panoptic_val.json")
      ;;
    gcg_val)
      script_name="inference/benchmark/batch_gcg_segment.py"
      input_jsonl="${jsonl_root}/seg_gcg_val_gcgseg.jsonl"
      out_subdir="segmentation/gcg/val"
      total_len=2938
      ;;
    gcg_test)
      script_name="inference/benchmark/batch_gcg_segment.py"
      input_jsonl="${jsonl_root}/seg_gcg_test_gcgseg.jsonl"
      out_subdir="segmentation/gcg/test"
      total_len=5157
      ;;
    refcoco_val)
      script_name="inference/benchmark/batch_binary_segment.py"
      input_jsonl="${jsonl_root}/seg_refcoco_val_binary.jsonl"
      out_subdir="segmentation/ref/refcoco_val"
      total_len=10268
      ;;
    refcocop_val)
      script_name="inference/benchmark/batch_binary_segment.py"
      input_jsonl="${jsonl_root}/seg_refcocop_val_binary.jsonl"
      out_subdir="segmentation/ref/refcocop_val"
      total_len=10096
      ;;
    refcocog_val)
      script_name="inference/benchmark/batch_binary_segment.py"
      input_jsonl="${jsonl_root}/seg_refcocog_val_binary.jsonl"
      out_subdir="segmentation/ref/refcocog_val"
      total_len=4889
      ;;
    reason_val)
      script_name="inference/benchmark/batch_binary_segment.py"
      input_jsonl="${jsonl_root}/seg_reason_val.jsonl"
      out_subdir="segmentation/rea/val"
      total_len=340
      ;;
    reason_test)
      script_name="inference/benchmark/batch_binary_segment.py"
      input_jsonl="${jsonl_root}/seg_reason_test.jsonl"
      out_subdir="segmentation/rea/test"
      total_len=3391
      ;;
    *)
      echo "Unknown segment task: ${task_name}" 1>&2
      return 1
      ;;
  esac

  local final_out_dir="${OUTPUT_DIR%/}/${out_subdir}"
  local effective_total_len="${TOTAL_TEST_LENGTH:-${total_len}}"
  if [ "${effective_total_len}" -lt "${total_split}" ]; then
    total_split="${effective_total_len}"
  fi
  require_jsonl_file "${input_jsonl}" "${task_name}" || return 1
  mkdir -p "${final_out_dir}"
  for split_num in $(seq 0 $((total_split - 1))); do
    local job_name="${task_name}_s${split_num}of${total_split}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" "${script_name}" \
        --model_path "${model_path}" \
        --input_jsonl "${input_jsonl}" \
        --output_dir "${final_out_dir}" \
        --data_path "${seg_jsonl_path_root}" \
        --total_test_length "${effective_total_len}" \
        --total_split "${total_split}" \
        --split_num "${split_num}" \
        "${save_pred_masks_args[@]}" \
        "${extra_args[@]}"
  done
}

run_segment() {
  local tasks=(
    pan_coco_val
    ade20k_pan_val
    gcg_val
    gcg_test
    refcoco_val
    refcocop_val
    refcocog_val
    reason_val
    reason_test
  )
  for task_name in "${tasks[@]}"; do
    if want_sub_task "${task_name}"; then
      run_segment_task "${task_name}" || exit 2
    fi
  done
}

run_detection_dataset() {
  local input_jsonl="$1"
  local task_name="$2"
  local total_samples="$3"
  local max_samples_per_split="$4"
  local mode="$5"
  local final_out_dir="${OUTPUT_DIR%/}/detection"
  local dataset_name total_split split_num job_name

  dataset_name="$(basename "${input_jsonl}" .jsonl)"
  local effective_total_samples="${TOTAL_TEST_LENGTH:-${total_samples}}"
  if [ "${effective_total_samples}" -le "${max_samples_per_split}" ]; then
    total_split=1
  else
    total_split=$(( (effective_total_samples + max_samples_per_split - 1) / max_samples_per_split ))
  fi

  require_jsonl_file "${input_jsonl}" "${dataset_name}" || return 1
  mkdir -p "${final_out_dir}"
  for split_num in $(seq 0 $((total_split - 1))); do
    job_name="${dataset_name}_split${split_num}_of_${total_split}_${task_name}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" inference/benchmark/batch_detect.py \
        --model_path "${model_path}" \
        --input_jsonl "${input_jsonl}" \
        --output_dir "${final_out_dir}" \
        --data_root "${detection_data_root}" \
        --total_split "${total_split}" \
        --split_num "${split_num}" \
        --task_name "${task_name}" \
        --mode "${mode}" \
        "${total_test_length_args[@]}"
  done
}

run_detection() {
  local datasets=(
    "${detection_jsonl_dir}/SROIE.jsonl common_object_detection 360 1000 understanding"
    "${detection_jsonl_dir}/HierText.jsonl common_object_detection 1723 50 dense_OCR"
    "${detection_jsonl_dir}/LVIS.jsonl common_object_detection 19626 1000 understanding"
    "${detection_jsonl_dir}/point_eval/LVIS.jsonl pointing 19583 1000 understanding"
    "${detection_jsonl_dir}/visual_prompt_eval/LVIS.jsonl visual_prompt_detection 70139 5000 understanding"
    "${detection_jsonl_dir}/visual_prompt_eval/Dense200.jsonl visual_prompt_detection 200 5000 understanding"
    "${detection_jsonl_dir}/TotalText.jsonl common_object_detection 300 1000 understanding"
    "${detection_jsonl_dir}/VisDrone.jsonl common_object_detection 1610 50 dense_detection"
    "${detection_jsonl_dir}/Dense200.jsonl common_object_detection 200 1000 dense_detection"
    "${detection_jsonl_dir}/point_eval/VisDrone.jsonl pointing 1610 100 dense_detection"
    "${detection_jsonl_dir}/point_eval/Dense200.jsonl pointing 187 1000 dense_detection"
    "${detection_jsonl_dir}/IC15.jsonl common_object_detection 496 1000 understanding"
    "${detection_jsonl_dir}/COCO.jsonl common_object_detection 4952 1000 understanding"
    "${detection_jsonl_dir}/HumanRef.jsonl referring_object_detection 5000 1000 understanding"
    "${detection_jsonl_dir}/RefCOCOg_test.jsonl referring_object_detection 9577 2000 understanding"
    "${detection_jsonl_dir}/RefCOCOg_val.jsonl referring_object_detection 4889 2000 understanding"
    "${detection_jsonl_dir}/point_eval/COCO.jsonl pointing 4940 1000 understanding"
    "${detection_jsonl_dir}/point_eval/HumanRef.jsonl pointing_referring 4964 1000 understanding"
    "${detection_jsonl_dir}/point_eval/RefCOCOg_val.jsonl pointing_referring 4875 2000 understanding"
    "${detection_jsonl_dir}/point_eval/RefCOCOg_test.jsonl pointing_referring 9559 2000 understanding"
    "${detection_jsonl_dir}/visual_prompt_eval/COCO.jsonl visual_prompt_detection 14631 5000 understanding"
    "${detection_jsonl_dir}/visual_prompt_eval/FSCD_test.jsonl visual_prompt_detection 1190 400 understanding"
    "${detection_jsonl_dir}/DocLayNet.jsonl common_object_detection 6480 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_desktop_v2_icon.jsonl gui 140 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_desktop_v2_text.jsonl gui 194 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_mobile_v2_icon.jsonl gui 211 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_mobile_v2_text.jsonl gui 290 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_web_v2_icon.jsonl gui 203 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/screenspot_web_v2_text.jsonl gui 234 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_cad_icon.jsonl gui 64 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_cad_text.jsonl gui 197 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_creative_icon.jsonl gui 143 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_creative_text.jsonl gui 198 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_dev_icon.jsonl gui 145 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_dev_text.jsonl gui 154 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_office_icon.jsonl gui 53 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_office_text.jsonl gui 177 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_os_icon.jsonl gui 89 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_os_text.jsonl gui 107 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_sci_icon.jsonl gui 110 1000 understanding"
    "${detection_jsonl_dir}/gui_eval/ScreenSpotPro_sci_text.jsonl gui 144 1000 understanding"
    "${detection_jsonl_dir}/keypoint_eval/ap-10k.jsonl keypoint 1997 1000 understanding"
    "${detection_jsonl_dir}/keypoint_eval/coco.jsonl keypoint 2693 1000 understanding"
  )
  local dataset input_jsonl dataset_name missing=0
  for dataset in "${datasets[@]}"; do
    # shellcheck disable=SC2086
    set -- ${dataset}
    input_jsonl="$1"
    dataset_name="$(basename "${input_jsonl}" .jsonl)"
    require_jsonl_file "${input_jsonl}" "${dataset_name}" || missing=1
  done
  if [ "${missing}" != "0" ]; then
    exit 2
  fi

  for dataset in "${datasets[@]}"; do
    # shellcheck disable=SC2086
    run_detection_dataset ${dataset} || exit 2
  done
}

run_recon3d() {
  local datasets=(7scenes eth3d)
  local out_dir="${OUTPUT_DIR%/}"
  for test_dataset in "${datasets[@]}"; do
    local job_name="recon3d_${test_dataset}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" inference/benchmark/batch_recon3d.py \
        --model_path "${model_path}" \
        --dataset "${test_dataset}" \
        --data_root "${multiview3d_data_root}" \
        --output_dir "${out_dir}" \
        "${total_test_length_args[@]}"
  done
}

run_camera_pose() {
  local datasets=(re10k co3dv2)
  local out_dir="${OUTPUT_DIR%/}"
  for test_dataset in "${datasets[@]}"; do
    local job_name="camera_pose_${test_dataset}"
    launch_job "${job_name}" "${LOG_DIR}/${job_name}.log" \
      "${python_cmd}" inference/benchmark/batch_camera_pose.py \
        --model_path "${model_path}" \
        --dataset "${test_dataset}" \
        --data_root "${multiview3d_data_root}" \
        --output_dir "${out_dir}" \
        "${total_test_length_args[@]}"
  done
}

echo "============================================================"
echo " Launching local benchmark inference"
echo "------------------------------------------------------------"
echo " repo_dir:             ${repo_dir}"
echo " model_path:           ${model_path}"
echo " output_dir:           ${OUTPUT_DIR}"
echo " data_root:            ${data_root}"
echo " seg_jsonl_path_root:  ${seg_jsonl_path_root}"
echo " geometry_data_root:   ${geometry_data_root}"
echo " jsonl_root:           ${jsonl_root}"
echo " detection_data_root:  ${detection_data_root}"
echo " python:               ${python_cmd}"
echo " tasks:                ${TASKS}"
echo " sub_tasks:            ${SUB_TASKS}"
echo " total_test_length:    ${TOTAL_TEST_LENGTH:-<full>}"
echo " save_pred_masks:      ${SAVE_PRED_MASKS}"
echo " dry_run:              ${dry_run}"
echo " num_gpus:             ${NUM_GPUS}"
echo " tasks_per_gpu:        ${TASKS_PER_GPU}"
echo " max concurrency:      ${MAX_JOBS}"
echo "============================================================"

JOB_IDX=0

if want_task "seg"; then
  run_segment
fi
if want_task "depth"; then
  run_depth
fi
if want_task "normal"; then
  run_normal
fi
if want_task "detection"; then
  run_detection
fi
if want_task "recon3d"; then
  run_recon3d
fi
if want_task "camera_pose"; then
  run_camera_pose
fi

wait

rc_all=0
for status_file in "${STATUS_DIR}"/*.status; do
  [ -e "${status_file}" ] || continue
  rc="$(cat "${status_file}")"
  if [ "${rc}" != "0" ]; then
    echo "[FAIL] $(basename "${status_file}" .status) exit=${rc}" 1>&2
    rc_all=1
  fi
done

echo "============================================================"
if [ ${rc_all} -eq 0 ]; then
  echo "All local inference jobs completed."
else
  echo "Some local inference jobs failed. Please check logs under ${LOG_DIR}."
fi
echo "============================================================"

exit ${rc_all}
