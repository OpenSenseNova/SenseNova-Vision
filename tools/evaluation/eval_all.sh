#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CALLER_PWD="$(pwd)"
METRIC_LABEL="sensenova-vision"

usage() {
  cat <<USAGE 1>&2
Usage:
  bash $0 --output_dir DIR [TASKS] [options]
  bash $0 DIR [TASKS] [options]

Options:
  --output_dir DIR        Benchmark output directory to evaluate.
  --tasks TASKS           all | detection | depth | normal | segmentation.
                          Comma-separated values are supported. Default: all.
  --parallel              Run selected top-level metric tasks in parallel.
                          This is task-level parallelism over detection/depth/
                          normal/segmentation, not dataset- or split-level
                          parallelism inside an evaluator.
  -h, --help              Show this help.
USAGE
}

TMP_DIR=""
TASKS="all"
PARALLEL="0"
TASKS_SET="0"

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  TMP_DIR="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir|--tmp_dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; }
      TMP_DIR="$2"; shift 2 ;;
    --tasks)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value." 1>&2; usage; exit 2; }
      TASKS="$2"; TASKS_SET="1"; shift 2 ;;
    --parallel)
      PARALLEL="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --*)
      echo "[ERROR] Unknown argument: $1" 1>&2
      usage
      exit 2 ;;
    *)
      if [[ "$TASKS_SET" == "0" ]]; then
        TASKS="$1"
        TASKS_SET="1"
        shift
      else
        echo "[ERROR] Unexpected positional argument: $1" 1>&2
        usage
        exit 2
      fi ;;
  esac
done

if [[ -z "$TMP_DIR" ]]; then
  usage
  exit 2
fi
case "$TMP_DIR" in
  /*) ;;
  *) TMP_DIR="$CALLER_PWD/$TMP_DIR" ;;
esac
TMP_DIR="${TMP_DIR%/}"

normalize_tasks() {
  local raw_tasks="$1"
  local task normalized
  SELECTED_TASKS=()
  if [[ "$raw_tasks" == "all" ]]; then
    SELECTED_TASKS=(detect depth normal seg)
    return 0
  fi
  IFS=',' read -r -a task_items <<< "$raw_tasks"
  for task in "${task_items[@]}"; do
    case "$task" in
      detect|detection) normalized="detect" ;;
      depth) normalized="depth" ;;
      normal) normalized="normal" ;;
      seg|segmentation) normalized="seg" ;;
      "") echo "[ERROR] Empty task in --tasks '$raw_tasks'." 1>&2; return 1 ;;
      *) echo "[ERROR] Unknown task '$task'. Supported: all, detection, depth, normal, segmentation." 1>&2; return 1 ;;
    esac
    SELECTED_TASKS+=("$normalized")
  done
}

script_for_task() {
  case "$1" in
    detect) echo "$TOOL_ROOT/detect/evaluation/scripts/eval_detection.sh" ;;
    depth) echo "$TOOL_ROOT/Marigold_normal_depth/eval_depth.sh" ;;
    normal) echo "$TOOL_ROOT/Marigold_normal_depth/eval_normals.sh" ;;
    seg) echo "$TOOL_ROOT/segment/eval_segmentation.sh" ;;
  esac
}

set_default_eval_roots() {
  local datas_root="$REPO_ROOT/datas"

  export SEG_EVAL_DATA_ROOT="${SEG_EVAL_DATA_ROOT:-$datas_root}"
  export DETECTION_EVAL_DATA_ROOT="${DETECTION_EVAL_DATA_ROOT:-$datas_root/detection_data}"
  export DEPTH_EVAL_DATA_ROOT="${DEPTH_EVAL_DATA_ROOT:-$datas_root/geometry_data/evaluation_depth_dataset}"
  export NORMAL_EVAL_DATA_ROOT="${NORMAL_EVAL_DATA_ROOT:-$datas_root/geometry_data/evaluation_normal_dataset}"
}

normalize_tasks "$TASKS" || exit 2
set_default_eval_roots

for task in "${SELECTED_TASKS[@]}"; do
  task_script="$(script_for_task "$task")"
  [[ -f "$task_script" ]] || {
    echo "[ERROR] evaluation script not found for task '$task': $task_script" 1>&2
    exit 1
  }
done

LOG_DIR="$TMP_DIR/local_logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/eval_all_${METRIC_LABEL}_${ts}.log"

run_one () {
  local name="$1"
  local sh="$2"
  local log="$LOG_DIR/${name}_${METRIC_LABEL}_${ts}.log"

  {
    echo "========== [$name] START $(date) =========="
    echo "[CMD] bash $sh \"$TMP_DIR\" \"$METRIC_LABEL\""
    echo "[LOG] $log"
  } | tee -a "$RUN_LOG"

  if bash "$sh" "$TMP_DIR" "$METRIC_LABEL" 2>&1 | tee "$log"; then
    echo "========== [$name] OK $(date) ==========" | tee -a "$RUN_LOG"
    return 0
  else
    echo "========== [$name] FAIL $(date) ==========" | tee -a "$RUN_LOG"
    return 1
  fi
}

fail=0

echo "========== [evaluation] START $(date) ==========" | tee -a "$RUN_LOG"
echo "[OUTPUT_DIR] $TMP_DIR" | tee -a "$RUN_LOG"
echo "[METRIC_LABEL] $METRIC_LABEL" | tee -a "$RUN_LOG"
echo "[TOOL_ROOT] $TOOL_ROOT" | tee -a "$RUN_LOG"
echo "[REPO_ROOT] $REPO_ROOT" | tee -a "$RUN_LOG"
echo "[TASKS] ${SELECTED_TASKS[*]}" | tee -a "$RUN_LOG"
echo "[SEG_EVAL_DATA_ROOT] $SEG_EVAL_DATA_ROOT" | tee -a "$RUN_LOG"
echo "[DETECTION_EVAL_DATA_ROOT] $DETECTION_EVAL_DATA_ROOT" | tee -a "$RUN_LOG"
echo "[DEPTH_EVAL_DATA_ROOT] $DEPTH_EVAL_DATA_ROOT" | tee -a "$RUN_LOG"
echo "[NORMAL_EVAL_DATA_ROOT] $NORMAL_EVAL_DATA_ROOT" | tee -a "$RUN_LOG"
echo "[RUN_LOG] $RUN_LOG" | tee -a "$RUN_LOG"

if [[ "$PARALLEL" == "1" ]]; then
  # Parallel execution can increase the risk of file overwrite or resource conflicts.
  TASK_PIDS=()
  for task in "${SELECTED_TASKS[@]}"; do
    run_one "$task" "$(script_for_task "$task")" &
    TASK_PIDS+=("$!")
  done
  for pid in "${TASK_PIDS[@]}"; do
    wait "$pid" || fail=1
  done
else
  for task in "${SELECTED_TASKS[@]}"; do
    run_one "$task" "$(script_for_task "$task")" || fail=1
  done
fi

echo "========================================" | tee -a "$RUN_LOG"
if [[ $fail -eq 0 ]]; then
  echo "[ALL DONE] all evaluations succeeded." | tee -a "$RUN_LOG"
  echo "[RUN LOG] $RUN_LOG"
  exit 0
else
  echo "[ALL DONE] some evaluations failed. check logs: $LOG_DIR" | tee -a "$RUN_LOG"
  echo "[RUN LOG] $RUN_LOG"
  exit 1
fi
