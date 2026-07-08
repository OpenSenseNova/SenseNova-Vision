#!/bin/bash
set -euo pipefail

# ===================== Argument validation =====================
if [ $# -ne 2 ]; then
    echo "[ERROR] Usage: $0 <tmp_dir> <metric_label>"
    exit 1
fi

# Core variables.
RESULT_DIR="$1/detection"
metric_label="$2"
TMP_DIR="$RESULT_DIR/output"
PYTHON="${EVAL_PYTHON:-python}"
# ===================== Preflight checks =====================
# Check whether the result directory exists.
if [ ! -d "$RESULT_DIR" ]; then
    echo "[ERROR] Result directory not found: $RESULT_DIR"
    exit 1
fi

# Switch to the detect tool root regardless of the caller's current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DETECT_DIR"

# Create the temporary output directory if needed.
mkdir -p "$TMP_DIR"

# ===================== Run evaluations =====================
echo "===== Starting detection evaluation batch ====="
echo "Result directory: $RESULT_DIR"
echo "Temporary directory: $TMP_DIR"
echo "============================"

# 0. Merge split JSONL files.
echo -e "Merging all JSONL shards"
if [ ! -f "merge_detection_jsonl.py" ]; then
    echo "[ERROR] Required merge script not found: $DETECT_DIR/merge_detection_jsonl.py"
    exit 1
fi
"$PYTHON" merge_detection_jsonl.py "$RESULT_DIR"

# 1. COCO evaluation.
echo -e "\n[1/11] Running COCO evaluation..."
bash evaluation/scripts/eval_coco.sh "$RESULT_DIR/COCO.jsonl" "$TMP_DIR"

# 2. LVIS evaluation.
echo -e "\n[2/11] Running LVIS evaluation..."
bash evaluation/scripts/eval_lvis.sh "$RESULT_DIR/LVIS.jsonl" "$TMP_DIR"

# 3. HumanRef metrics
echo -e "\n[3/11] Computing HumanRef metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/HumanRef.jsonl" --output_path "$TMP_DIR/HumanRef.jsonl"

# 4. RefCOCOg_val metrics
echo -e "\n[4/11] Computing RefCOCOg_val metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/RefCOCOg_val.jsonl" --output_path "$TMP_DIR/RefCOCOg_val.jsonl"

# 5. RefCOCOg_test metrics
echo -e "\n[5/11] Computing RefCOCOg_test metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/RefCOCOg_test.jsonl" --output_path "$TMP_DIR/RefCOCOg_test.jsonl"

# 6. Dense200 metrics
echo -e "\n[6/11] Computing Dense200 metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/Dense200.jsonl" --output_path "$TMP_DIR/Dense200.jsonl"

# 7. VisDrone metrics
echo -e "\n[7/11] Computing VisDrone metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/VisDrone.jsonl" --output_path "$TMP_DIR/VisDrone.jsonl"

# 8. HierText metrics
echo -e "\n[1/4] Computing HierText metrics..."
$PYTHON evaluation/metrics/other_metric.py --match_by_category --data_path "$RESULT_DIR/HierText.jsonl" --output_path "$TMP_DIR/HierText.jsonl"

# 9. IC15 metrics
echo -e "\n[2/4] Computing IC15 metrics..."
$PYTHON evaluation/metrics/other_metric.py --match_by_category --data_path "$RESULT_DIR/IC15.jsonl" --output_path "$TMP_DIR/IC15.jsonl"


# 10. TotalText metrics
echo -e "\n[3/4] Computing TotalText metrics..."
$PYTHON evaluation/metrics/other_metric.py --match_by_category --data_path "$RESULT_DIR/TotalText.jsonl" --output_path "$TMP_DIR/TotalText.jsonl"

# 11. SROIE metrics
echo -e "\n[4/4] Computing SROIE metrics..."

$PYTHON evaluation/metrics/other_metric.py --match_by_category --data_path "$RESULT_DIR/SROIE.jsonl" --output_path "$TMP_DIR/SROIE.jsonl"

# Pointing metrics; task_name is pointing
# 12. COCO pointing metrics
echo -e "\n[1/7] Computing COCO pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.COCO.jsonl" --output_path "$TMP_DIR/point.COCO.jsonl"

# 13. LVIS pointing metrics
echo -e "\n[2/7] Computing LVIS pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.LVIS.jsonl" --output_path "$TMP_DIR/point.LVIS.jsonl"

# 14. Dense200 pointing metrics
echo -e "\n[3/7] Computing Dense200 pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.Dense200.jsonl" --output_path "$TMP_DIR/point.Dense200.jsonl"

# 15. VisDrone pointing metrics
echo -e "\n[4/7] Computing VisDrone pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.VisDrone.jsonl" --output_path "$TMP_DIR/point.VisDrone.jsonl"

# 16. HumanRef pointing metrics
echo -e "\n[5/7] Computing HumanRef pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.HumanRef.jsonl" --output_path "$TMP_DIR/point.HumanRef.jsonl"

# 17. RefCOCOg_val pointing metrics
echo -e "\n[6/7] Computing RefCOCOg_val pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.RefCOCOg_val.jsonl" --output_path "$TMP_DIR/point.RefCOCOg_val.jsonl"

# 18. RefCOCOg_test pointing metrics
echo -e "\n[7/7] Computing RefCOCOg_test pointing metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/point.RefCOCOg_test.jsonl" --output_path "$TMP_DIR/point.RefCOCOg_test.jsonl"

# 19. COCO visual prompting metrics
echo -e "\n[1/4] Computing COCO visual prompting metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/visual.COCO.jsonl" --output_path "$TMP_DIR/visual.COCO.jsonl"

# 20. LVIS visual prompting metrics
echo -e "\n[1/4] Computing COCO visual prompting metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/visual.LVIS.jsonl" --output_path "$TMP_DIR/visual.LVIS.jsonl"

# 21. Dense200 visual prompting metrics
echo -e "\n[1/4] Computing COCO visual prompting metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/visual.Dense200.jsonl" --output_path "$TMP_DIR/visual.Dense200.jsonl"

# 22. FSCD_test visual prompting metrics
echo -e "\n[1/4] Computing COCO visual prompting metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/visual.FSCD_test.jsonl" --output_path "$TMP_DIR/visual.FSCD_test.jsonl"

# 23. DocLayNet metrics
echo -e "\n[1/1] Computing DocLayNet metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/DocLayNet.jsonl" --output_path "$TMP_DIR/DocLayNet.jsonl"

##### 25. ScreenSpot v2 metrics; six subcategories are evaluated separately and merged for dataset-level metrics
echo -e "\n[1/2] Computing screenspotv2 metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_desktop_v2_icon.jsonl" --output_path "$TMP_DIR/gui.screenspot_desktop_v2_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_desktop_v2_text.jsonl" --output_path "$TMP_DIR/gui.screenspot_desktop_v2_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_mobile_v2_icon.jsonl"  --output_path "$TMP_DIR/gui.screenspot_mobile_v2_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_mobile_v2_text.jsonl"  --output_path "$TMP_DIR/gui.screenspot_mobile_v2_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_web_v2_icon.jsonl"     --output_path "$TMP_DIR/gui.screenspot_web_v2_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspot_web_v2_text.jsonl"     --output_path "$TMP_DIR/gui.screenspot_web_v2_text.jsonl"

echo "Evaluation tasks completed. Merging screenspotv2 results..."
# Define the merged output path
MERGED_OUTPUT="$RESULT_DIR/gui.screenspotv2.jsonl"
# Merge the six result files under RESULT_DIR
cat "$RESULT_DIR/gui.screenspot_desktop_v2_icon.jsonl" \
    "$RESULT_DIR/gui.screenspot_desktop_v2_text.jsonl" \
    "$RESULT_DIR/gui.screenspot_mobile_v2_icon.jsonl" \
    "$RESULT_DIR/gui.screenspot_mobile_v2_text.jsonl" \
    "$RESULT_DIR/gui.screenspot_web_v2_icon.jsonl" \
    "$RESULT_DIR/gui.screenspot_web_v2_text.jsonl" \
    > "$MERGED_OUTPUT"
echo "All results merged into: $MERGED_OUTPUT"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.screenspotv2.jsonl" --output_path "$TMP_DIR/gui.screenspotv2.jsonl"

##### 26. ScreenSpotPro metrics; 12 subcategories are evaluated separately and merged for dataset-level metrics
echo -e "\n[2/2] Computing ScreenSpotPro metrics..."
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_cad_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_cad_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_cad_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_cad_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_creative_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_creative_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_creative_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_creative_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_dev_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_dev_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_dev_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_dev_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_office_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_office_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_office_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_office_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_os_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_os_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_os_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_os_text.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_sci_icon.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_sci_icon.jsonl"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro_sci_text.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro_sci_text.jsonl"

echo "Evaluation tasks completed. Merging ScreenSpotPro results..."
# Define the merged output path
MERGED_OUTPUT="$RESULT_DIR/gui.ScreenSpotPro.jsonl"
# Merge the 12 result files under RESULT_DIR
cat "$RESULT_DIR/gui.ScreenSpotPro_cad_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_cad_text.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_creative_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_creative_text.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_dev_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_dev_text.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_office_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_office_text.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_os_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_os_text.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_sci_icon.jsonl" \
    "$RESULT_DIR/gui.ScreenSpotPro_sci_text.jsonl" \
    > "$MERGED_OUTPUT"
echo "All results merged into: $MERGED_OUTPUT"
$PYTHON evaluation/metrics/other_metric.py --data_path "$RESULT_DIR/gui.ScreenSpotPro.jsonl" --output_path "$TMP_DIR/gui.ScreenSpotPro.jsonl"

# 27. ap-10k metrics
echo -e "\n[1/2] Computing ap-10k metrics..."
$PYTHON evaluation/metrics/other_metric.py --save_keypoint_metrics --data_path "$RESULT_DIR/keypoint.ap-10k.jsonl" --output_path "$TMP_DIR/keypoint.ap-10k.jsonl"
# 28. COCO metrics
echo -e "\n[2/2] Computing COCO metrics..."
$PYTHON evaluation/metrics/other_metric.py --save_keypoint_metrics --data_path "$RESULT_DIR/keypoint.coco.jsonl" --output_path "$TMP_DIR/keypoint.coco.jsonl"


echo -e "Exporting the summary table"
$PYTHON evaluation/utils/auto_table.py \
  --tmp_dir "$RESULT_DIR" \
  --model_name "$metric_label" \
  --out_json "$RESULT_DIR/metrics/${metric_label}_Detection.jsonl"
# ===================== completed =====================
echo -e "\n===== Batch evaluation completed ====="
echo "All results have been written to the temporary directory: $TMP_DIR"
echo "============================"
