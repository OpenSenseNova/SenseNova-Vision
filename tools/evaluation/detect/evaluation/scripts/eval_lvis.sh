#!/bin/bash

# LVIS Evaluation Pipeline - Simple Version
# This script runs the complete LVIS evaluation pipeline with fixed parameters

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$EVAL_DIR:$EVAL_DIR/fastevaluate:$EVAL_DIR/fastevaluate/fastevaluate:${PYTHONPATH:-}"
PYTHON="${EVAL_PYTHON:-python}"

# Fixed parameters
PRED_JSONL="$1"
DATA_ROOT="${DETECTION_EVAL_DATA_ROOT:-}"
TEST_JSONL="${DETECTION_LVIS_TEST_JSONL:-${DATA_ROOT:+$DATA_ROOT/annotations/box_eval/LVIS.jsonl}}"
IMAGE_ROOT="${DETECTION_IMAGE_ROOT:-$DATA_ROOT}"
LVIS_JSON="${DETECTION_LVIS_JSON:-${DATA_ROOT:+$DATA_ROOT/coco/lvis_v1_val_with_filename2.json}}"
OUTPUT_DIR="$2"
START_IDX=0
END_IDX=-1
MAX_TOKENS=8192
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# Create output directory
mkdir -p "$OUTPUT_DIR"

[ -n "$LVIS_JSON" ] && [ -f "$LVIS_JSON" ] || { echo "[ERROR] Set DETECTION_LVIS_JSON or DETECTION_EVAL_DATA_ROOT." 1>&2; exit 2; }

# Set file paths
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval_lvis.tsv"

echo "Starting LVIS Evaluation Pipeline"
echo "======================================"
echo "Pred_jsonl: $PRED_JSONL"
echo "Test data: $TEST_JSONL"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 2: Convert to FastEval TSV format
echo "Step 2: Converting predictions to FastEval TSV format..."
$PYTHON evaluation/utils/convert_coco_lvis_to_standard_format.py \
    --our_pred_jsonl "$PRED_JSONL" \
    --coco_json "$LVIS_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

echo "Format conversion completed successfully."
echo ""

# Step 3: Evaluate using FastEval
echo "Step 3: Running FastEval evaluation..."
$PYTHON evaluation/metrics/coco_lvis_metric.py \
    --gt "$LVIS_JSON" \
    --pred_tsv "$FASTEVAL_TSV" \
    --eval_type "lvis"

echo "Evaluation completed successfully."
echo ""

echo "LVIS Evaluation Pipeline completed."
echo "======================================"
echo "Results saved to:"
echo "  - Predictions: $PRED_JSONL"
echo "  - FastEval TSV: $FASTEVAL_TSV"
