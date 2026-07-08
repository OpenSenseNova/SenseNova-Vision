#!/usr/bin/env bash
set -euo pipefail
# set -x

# Usage:
#   bash eval_segmentation.sh <tmp_dir> <metric_label>
# Environment:
#   EVAL_PYTHON          Python executable. Default: python
#   SEG_EVAL_DATA_ROOT   Directory containing segmentation evaluation data.
#   SEG_PQ_WORKERS       Worker count for panoptic PQ. Default: min(cpu_count, 16).
#
# Expected directory layout under $tmp_dir/segmentation:
#   $OUTPUT_ROOT/ade/val/
#   $OUTPUT_ROOT/gcg/val/
#   $OUTPUT_ROOT/gcg/test/
#   $OUTPUT_ROOT/pan/val/
#   $OUTPUT_ROOT/ref/refcoco_val
#   $OUTPUT_ROOT/ref/refcocop_val
#   $OUTPUT_ROOT/ref/refcocog_val
#   $OUTPUT_ROOT/rea/val/
#   $OUTPUT_ROOT/rea/test/

if [ "$#" -ne 2 ]; then
  echo "[ERROR] Usage: $0 <tmp_dir> <metric_label>"
  exit 2
fi

OUTPUT_ROOT="$1/segmentation"
metric_label="$2"
if [ -n "${EVAL_JAVA_HOME:-}" ]; then
  export JAVA_HOME="$EVAL_JAVA_HOME"
  export PATH="$JAVA_HOME/bin:$PATH"
fi
# Segment tool root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
cd "$SCRIPT_DIR"

PYTHON="${EVAL_PYTHON:-python}"

# -------- data root --------
SEG_DATA_ROOT="${SEG_EVAL_DATA_ROOT:-}"
if [ -z "$SEG_DATA_ROOT" ]; then
  if [ -e "$SCRIPT_DIR/datas" ]; then
    SEG_DATA_ROOT="$SCRIPT_DIR/datas"
  else
    SEG_DATA_ROOT="$REPO_ROOT/datas"
  fi
fi
if [ ! -d "$SEG_DATA_ROOT" ]; then
  echo "[ERROR] segmentation data root not found: $SEG_DATA_ROOT" 1>&2
  echo "Hint: prepare datas/ under the repository root, or set SEG_EVAL_DATA_ROOT=/path/to/datas" 1>&2
  exit 1
fi
SEG_DATA_ROOT="$(cd "$SEG_DATA_ROOT" && pwd)"

# -------- Tool script checks --------
for f in evaluate_panoptic_segmentation.py evaluate_grounded_caption_segmentation.py evaluate_region_segmentation.py; do
  if [ ! -f "$SCRIPT_DIR/$f" ]; then
    echo "[ERROR] missing $f under $SCRIPT_DIR" 1>&2
    exit 1
  fi
done

# -------- Unified output directories --------
METRICS_ROOT="$OUTPUT_ROOT/metrics"
PAN_VAL_DIR="$OUTPUT_ROOT/pan/val"
ADE_VAL_DIR="$OUTPUT_ROOT/ade/val"
GCG_VAL_DIR="$OUTPUT_ROOT/gcg/val"
GCG_TEST_DIR="$OUTPUT_ROOT/gcg/test"
mkdir -p "$METRICS_ROOT"

# -------- 1) GenSeg (pan) --------
if [ -d "$PAN_VAL_DIR" ]; then
  "$PYTHON" "$SCRIPT_DIR/evaluate_panoptic_segmentation.py" \
    --result_dir "$PAN_VAL_DIR/" \
    --gt_json "$SEG_DATA_ROOT/gen_seg_data/coco2017/annotations/panoptic_val2017.json" \
    --gt_folder "$SEG_DATA_ROOT/gen_seg_data/coco2017/panoptic_val2017/" \
    --gt_semseg_folder "$SEG_DATA_ROOT/gen_seg_data/coco2017/panoptic_semseg_val2017/" \
    --gt_instance_json "$SEG_DATA_ROOT/gen_seg_data/coco2017/annotations/instances_val2017.json" \
    --metrics_dir "$METRICS_ROOT/pan/val" \
    --dataset_name "GenSeg"
else
  echo "[WARN] skip GenSeg: not found $PAN_VAL_DIR" 1>&2
fi

# -------- 2) OVSeg (ADE20K) --------
if [ -d "$ADE_VAL_DIR" ]; then
  "$PYTHON" "$SCRIPT_DIR/evaluate_panoptic_segmentation.py" \
    --result_dir "$ADE_VAL_DIR/" \
    --gt_json "$SEG_DATA_ROOT/ov_seg_data/ade20k/ade20k_panoptic_val.json" \
    --gt_folder "$SEG_DATA_ROOT/ov_seg_data/ade20k/ade20k_panoptic_val/" \
    --gt_semseg_folder "$SEG_DATA_ROOT/ov_seg_data/ade20k/annotations_detectron2/validation/" \
    --gt_instance_json "$SEG_DATA_ROOT/ov_seg_data/ade20k/ade20k_instance_val.json" \
    --metrics_dir "$METRICS_ROOT/ade/val" \
    --dataset_name "ade_val"
else
  echo "[WARN] skip OVSeg(ADE): not found $ADE_VAL_DIR" 1>&2
fi

# -------- 3) GCGSeg (val) --------
if [ -d "$GCG_VAL_DIR" ]; then
  "$PYTHON" "$SCRIPT_DIR/evaluate_grounded_caption_segmentation.py" \
    --result_dir "$GCG_VAL_DIR/" \
    --gt_json "$SEG_DATA_ROOT/gcg_seg_data/annotations/val_test/val_gcg_coco_mask_gt.json" \
    --cap_gt_json "$SEG_DATA_ROOT/gcg_seg_data/annotations/val_test/val_gcg_coco_caption_gt.json" \
    --metrics_dir "$METRICS_ROOT/gcg/val" \
    --dataset_name "GCG val"
else
  echo "[WARN] skip GCG(val): not found $GCG_VAL_DIR" 1>&2
fi

# -------- 4) GCGSeg (test) --------
if [ -d "$GCG_TEST_DIR" ]; then
  "$PYTHON" "$SCRIPT_DIR/evaluate_grounded_caption_segmentation.py" \
    --result_dir "$GCG_TEST_DIR/" \
    --gt_json "$SEG_DATA_ROOT/gcg_seg_data/annotations/val_test/test_gcg_coco_mask_gt.json" \
    --cap_gt_json "$SEG_DATA_ROOT/gcg_seg_data/annotations/val_test/test_gcg_coco_caption_gt.json" \
    --metrics_dir "$METRICS_ROOT/gcg/test" \
    --dataset_name "GCG_test"
else
  echo "[WARN] skip GCG(test): not found $GCG_TEST_DIR" 1>&2
fi

# -------- 5) RefSeg/ReaSeg from saved JSON predictions --------
"$PYTHON" "$SCRIPT_DIR/evaluate_region_segmentation.py" \
  --segmentation_root "$OUTPUT_ROOT" \
  --repo_root "$REPO_ROOT" \
  --write_auto_tables

# -------- 6) Export segmentation metric JSONL and markdown summary --------
EXPORT_PY="$SCRIPT_DIR/segment_utils/auto_table.py"
OUT_JSONL="$METRICS_ROOT/${metric_label}_Segmentation.jsonl"

if [ -f "$EXPORT_PY" ]; then
  "$PYTHON" "$EXPORT_PY" \
    --tmp_dir "$METRICS_ROOT" \
    --model_name "$metric_label" \
    --out_jsonl "$OUT_JSONL"
  echo "[OK] Wrote segmentation metric JSONL: $OUT_JSONL"
else
  echo "[WARN] skip export: not found $EXPORT_PY" 1>&2
  echo "       Expected exporter path: $EXPORT_PY" 1>&2
fi
