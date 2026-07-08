#!/usr/bin/env bash
set -euo pipefail
# set -x

# Usage:
#   bash eval_depth.sh <tmp_dir> <metric_label>
# Environment:
#   EVAL_PYTHON              Python executable. Default: python
#   DEPTH_EVAL_DATA_ROOT     Root containing depth evaluation datasets.
#   DEPTH_NYU_ROOT           NYU base_data_dir. Default: $DEPTH_EVAL_DATA_ROOT/nyu_depth_v2
#   DEPTH_KITTI_ROOT         KITTI base_data_dir.
#   DEPTH_ETH3D_ROOT         ETH3D base_data_dir.
#   DEPTH_SCANNET_ROOT       ScanNet base_data_dir.
#   DEPTH_DIODE_ROOT         DIODE base_data_dir.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tmp_dir> <metric_label>" 1>&2
  exit 2
fi

tmp_dir="$1"
metric_label="$2"

predict_base="${tmp_dir}/depth"

# Script root directory (Marigold_normal_depth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

EVAL_PY="$SCRIPT_DIR/script/depth/eval.py"
CFG_DIR="$SCRIPT_DIR/config/dataset_depth"
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
  local suffix="$2"   # e.g. nyudepth_v2_depth

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

NYU_PRED="$(resolve_pred_path "$predict_base" "nyudepth_v2_depth")"
KITTI_PRED="$(resolve_pred_path "$predict_base" "kitti_depth")"
ETH3D_PRED="$(resolve_pred_path "$predict_base" "eth3d_depth")"
SCANNET_PRED="$(resolve_pred_path "$predict_base" "scannet_depth")"
DIODE_PRED="$(resolve_pred_path "$predict_base" "diode_depth")"

mkdir -p "$OUT_DIR"

# ---------- NYU ----------
DATA_ROOT="${DEPTH_EVAL_DATA_ROOT:-}"
BASE_NYU="${DEPTH_NYU_ROOT:-${DATA_ROOT:+$DATA_ROOT/nyuv2}}"
[ -n "$BASE_NYU" ] || { echo "[ERROR] Set DEPTH_NYU_ROOT or DEPTH_EVAL_DATA_ROOT." 1>&2; exit 2; }
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_NYU" \
  --dataset_config "$CFG_DIR/data_nyu_test.yaml" \
  --alignment least_square_disparity \
  --prediction_dir "$NYU_PRED" \
  --output_dir "$OUT_DIR/nyu_test/eval_metric"

# ---------- KITTI ----------
BASE_KITTI="${DEPTH_KITTI_ROOT:-${DATA_ROOT:+$DATA_ROOT/kitti}}"
[ -n "$BASE_KITTI" ] || { echo "[ERROR] Set DEPTH_KITTI_ROOT or DEPTH_EVAL_DATA_ROOT." 1>&2; exit 2; }
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_KITTI" \
  --dataset_config "$CFG_DIR/data_kitti_eigen_test.yaml" \
  --alignment least_square_disparity \
  --prediction_dir "$KITTI_PRED" \
  --output_dir "$OUT_DIR/kitti_eigen_test/eval_metric"

# ---------- ETH3D ----------
BASE_ETH3D="${DEPTH_ETH3D_ROOT:-${DATA_ROOT:+$DATA_ROOT/eth3d}}"
[ -n "$BASE_ETH3D" ] || { echo "[ERROR] Set DEPTH_ETH3D_ROOT or DEPTH_EVAL_DATA_ROOT." 1>&2; exit 2; }
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_ETH3D" \
  --dataset_config "$CFG_DIR/data_eth3d.yaml" \
  --alignment least_square_disparity \
  --prediction_dir "$ETH3D_PRED" \
  --output_dir "$OUT_DIR/eth3d/eval_metric" \
  --alignment_max_res 1024

# ---------- ScanNet ----------
BASE_SCANNET="${DEPTH_SCANNET_ROOT:-${DATA_ROOT:+$DATA_ROOT/scannet}}"
[ -n "$BASE_SCANNET" ] || { echo "[ERROR] Set DEPTH_SCANNET_ROOT or DEPTH_EVAL_DATA_ROOT." 1>&2; exit 2; }
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_SCANNET" \
  --dataset_config "$CFG_DIR/data_scannet_val.yaml" \
  --alignment least_square_disparity \
  --prediction_dir "$SCANNET_PRED" \
  --output_dir "$OUT_DIR/scannet/eval_metric"

# ---------- DIODE ----------
BASE_DIODE="${DEPTH_DIODE_ROOT:-${DATA_ROOT:+$DATA_ROOT/diode}}"
[ -n "$BASE_DIODE" ] || { echo "[ERROR] Set DEPTH_DIODE_ROOT or DEPTH_EVAL_DATA_ROOT." 1>&2; exit 2; }
$PYTHON "$EVAL_PY" \
  --base_data_dir "$BASE_DIODE" \
  --dataset_config "$CFG_DIR/data_diode_all.yaml" \
  --alignment least_square_disparity \
  --prediction_dir "$DIODE_PRED" \
  --output_dir "$OUT_DIR/diode/eval_metric"

echo "[OK] Depth eval done. auto_table.csv should be under: $OUT_DIR/**/auto_table.csv"

# ---------- Export Feishu fact json (Depth) ----------
if [ ! -f "$EXPORT_PY" ]; then
  echo "[WARN] export script not found: $EXPORT_PY" 1>&2
  echo "       Skipping fact-json export." 1>&2
  exit 0
fi

$PYTHON "$EXPORT_PY" \
  --root_dir "$predict_base" \
  --task Depth \
  --model_name "$metric_label"

echo "[OK] Exported Depth fact json under: $predict_base/metrics/"
