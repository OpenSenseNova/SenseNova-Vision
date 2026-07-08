#!/bin/bash

# COCO Evaluation Pipeline - Simple Version
# This script runs the complete COCO evaluation pipeline with fixed parameters

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$EVAL_DIR:$EVAL_DIR/fastevaluate:$EVAL_DIR/fastevaluate/fastevaluate:${PYTHONPATH:-}"
PYTHON="${EVAL_PYTHON:-python}"

# Fixed parameters
PRED_JSONL="$1"
DATA_ROOT="${DETECTION_EVAL_DATA_ROOT:-}"
TEST_JSONL="${DETECTION_COCO_TEST_JSONL:-${DATA_ROOT:+$DATA_ROOT/annotations/box_eval/COCO.jsonl}}"
IMAGE_ROOT="${DETECTION_IMAGE_ROOT:-$DATA_ROOT}"
COCO_JSON="${DETECTION_COCO_JSON:-${DATA_ROOT:+$DATA_ROOT/coco/instances_val2017.json}}"
OUTPUT_DIR="$2"
START_IDX=0
END_IDX=-1
MAX_TOKENS=8192
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# Create output directory
mkdir -p "$OUTPUT_DIR"

[ -n "$COCO_JSON" ] && [ -f "$COCO_JSON" ] || { echo "[ERROR] Set DETECTION_COCO_JSON or DETECTION_EVAL_DATA_ROOT." 1>&2; exit 2; }

# Set file paths
# PRED_JSONL="$OUTPUT_DIR/answer_node4_26000.jsonl"
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval_coco.tsv"

echo "Starting COCO Evaluation Pipeline"
echo "======================================"
echo "Pred_jsonl: $PRED_JSONL"
echo "Test data: $TEST_JSONL"
echo "Output dir: $OUTPUT_DIR"
echo ""

echo "Step 2: Converting predictions to FastEval TSV format..."
$PYTHON evaluation/utils/convert_coco_lvis_to_standard_format.py \
    --our_pred_jsonl "$PRED_JSONL" \
    --coco_json "$COCO_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

echo "Step 3: Running FastEval evaluation..."
$PYTHON evaluation/metrics/coco_lvis_metric.py \
    --gt "$COCO_JSON" \
    --pred_tsv "$FASTEVAL_TSV"
