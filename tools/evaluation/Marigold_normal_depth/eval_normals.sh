#!/usr/bin/env bash
set -euo pipefail
# set -x

# Usage:
#   bash eval_normals.sh <tmp_dir> <metric_label>
# Environment:
#   EVAL_PYTHON            Python executable. Default: python
#   NORMAL_EVAL_DATA_ROOT  Root containing normal evaluation datasets.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tmp_dir> <metric_label>" 1>&2
  exit 2
fi

tmp_dir="$1"
metric_label="$2"

predict_base="${tmp_dir}/normal"

# Script root directory (Marigold_normal_depth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

EVAL_PY="$SCRIPT_DIR/script/normals/eval.py"
CFG_DIR="$SCRIPT_DIR/config/dataset_normals"
OUT_DIR="${predict_base}/metrics"

PYTHON="${EVAL_PYTHON:-python}"

# Export normalized fact JSON from auto_table.csv.
EXPORT_PY="$SCRIPT_DIR/script/auto_table.py"

if [ ! -f "$EVAL_PY" ]; then
  echo "[ERROR] eval.py not found: $EVAL_PY" 1>&2
  exit 1
fi

resolve_pred_path () {
  local base="$1"
  local suffix="$2"   # e.g. nyu_normal

  # Directory layout: base/suffix
  if [ -d "${base}/${suffix}" ]; then
    echo "${base}/${suffix}"
    return 0
  fi
  # Dotted-prefix layout: base.suffix
  if [ -d "${base}.${suffix}" ]; then
    echo "${base}.${suffix}"
    return 0
  fi

  echo "[ERROR] Cannot find prediction dir for '${suffix}'. Tried:" 1>&2
  echo "  - ${base}/${suffix}" 1>&2
  echo "  - ${base}.${suffix}" 1>&2
  exit 1
}

NYU_PRED="$(resolve_pred_path "$predict_base" "nyu_normal")"
SCANNET_PRED="$(resolve_pred_path "$predict_base" "scannet_normal")"
IBIMS_PRED="$(resolve_pred_path "$predict_base" "ibims_normal")"

mkdir -p "$OUT_DIR"

# All three normal datasets share the same base_data_dir.
BASE_NORMAL="${NORMAL_EVAL_DATA_ROOT:-}"
[ -n "$BASE_NORMAL" ] || { echo "[ERROR] Set NORMAL_EVAL_DATA_ROOT." 1>&2; exit 2; }

# ---------- NYU normals ----------
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_NORMAL" \
  --dataset_config "$CFG_DIR/data_nyu_test.yaml" \
  --prediction_dir "$NYU_PRED" \
  --output_dir "$OUT_DIR/nyu_normals_test/eval_metric"

# ---------- ScanNet normals ----------
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_NORMAL" \
  --dataset_config "$CFG_DIR/data_scannet_test.yaml" \
  --prediction_dir "$SCANNET_PRED" \
  --output_dir "$OUT_DIR/scannet_normals_test/eval_metric"

# ---------- iBims normals ----------
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_NORMAL" \
  --dataset_config "$CFG_DIR/data_ibims_test.yaml" \
  --prediction_dir "$IBIMS_PRED" \
  --output_dir "$OUT_DIR/ibims_normals_test/eval_metric"

echo "[OK] Normal eval done. auto_table.csv should be under: $OUT_DIR/**/auto_table.csv"

# ---------- Export Feishu fact json (Normal) ----------
if [ ! -f "$EXPORT_PY" ]; then
  echo "[WARN] export script not found: $EXPORT_PY" 1>&2
  echo "       Skipping fact-json export." 1>&2
  exit 0
fi

$PYTHON "$EXPORT_PY" \
  --root_dir "$predict_base" \
  --task Normal \
  --model_name "$metric_label"

echo "[OK] Exported Normal fact json under: $predict_base/metrics/"
