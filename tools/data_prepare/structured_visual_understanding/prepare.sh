#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
CONVERTER_DIR="$SCRIPT_DIR/converters"
SOURCE_CONVERTER="$CONVERTER_DIR/bbox_source_to_jsonl.py"
VISUAL_CONVERTER="$CONVERTER_DIR/visual_prompt_to_jsonl.py"
KEYPOINT_SOURCE_CONVERTER="$CONVERTER_DIR/keypoint_source_to_jsonl.py"
KEYPOINT_INTERMEDIATE_CONVERTER="$CONVERTER_DIR/keypoint_intermediate_to_jsonl.py"
OCR_SOURCE_CONVERTER="$CONVERTER_DIR/ocr_source_to_jsonl.py"
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/datas/train_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/jsonl_generate/train_jsonls/structure_view_understanding"}
PYTHON_BIN=${PYTHON_BIN:-python}
LIMIT=${LIMIT:-}

ALL_CASES=(
  aptv2
  blood-cell-bbox
  blood-cell-visual
  fsc147
  coco2017-keypoint
  crowdpose-keypoint
  sroie
  icdar2013-word-bbox
  icdar2015-word-bbox
  icdar2015-word-poly
  cdla-layout
  doclaynet-core-layout
  publaynet-layout
  tablebank-layout
  tabrecset-layout
  os-atlas-desktop-gui
  os-atlas-mobile-gui
  os-atlas-rico-gui
  os-atlas-web-gui
  showui-desktop-gui
  ui-refexp-gui
  paco-lvis-bbox
  sku110k-bbox
  humanref-bbox
  rexverse-onesentence-bbox
  rexverse-referring-bbox
  egoobjects-bbox v3det-ovd-bbox owdod-bbox
  objects365-bbox objects365-visual
  lvis-fruit-vegetable-bbox sheep-bbox football-bbox industrial-safety-bbox
  lvis-fruit-vegetable-visual sheep-visual football-visual
  industrial-safety-visual owdod-visual sku110k-visual
  ap-10k-keypoint apt36k-keypoint human-art-keypoint
  macaquepose-keypoint mpii-keypoint ochuman-keypoint
)

declare -A EXTRA_BBOX_CASES=(
  [egoobjects-bbox]="egoobjects-bbox|EgoObjects|EgoObjectsV1_unified_train.json|EgoObjects/json/processed_bbox_trainv2.jsonl|.|EgoObjects_processed_bbox_train.jsonl|.|78895"
  [v3det-ovd-bbox]="v3det-ovd-bbox|V3Det___V3Det/raw|v3det_2023_v1_train_ovd_base.json|V3Det___V3Det/raw/V3Det/json/processed_bbox_trainv2.jsonl|V3Det___V3Det/raw/V3Det|V3Det_ovd_processed_bbox_train.jsonl|V3Det___V3Det/raw/V3Det|116182"
  [owdod-bbox]="owdod-bbox|owdod|data/train-00000-of-00004.parquet|owdod/json/processed_bbox_trainv2.jsonl|.|owdod_processed_bbox_train.jsonl|.|8001"
  [objects365-bbox]="objects365-bbox|object365|data/train-*.parquet|object365/json/processed_bbox_trainv2.jsonl|object365|Objects365_processed_bbox_train.jsonl|object365|1742289"
  [lvis-fruit-vegetable-bbox]="lvis-fruit-vegetable-bbox|LVIS_Fruits_And_Vegetables|data.yaml|LVIS_Fruits_And_Vegetables/json/processed_bbox_trainv2.jsonl|LVIS_Fruits_And_Vegetables|LVIS_Fruits_And_Vegetables_processed_bbox_train.jsonl|LVIS_Fruits_And_Vegetables/images/train|6721"
  [sheep-bbox]="sheep-bbox|aerial-sheep-object-detection|data/_annotations.coco.json|aerial-sheep-object-detection/json/processed_bbox_trainv2.jsonl|aerial-sheep-object-detection|sheep_processed_bbox_train.jsonl|aerial-sheep-object-detection|3602"
  [football-bbox]="football-bbox|football-object-detection|data/_annotations.coco.json|football-object-detection/json/processed_bbox_trainv2.jsonl|football-object-detection|football_processed_bbox_train.jsonl|football-object-detection|858"
  [industrial-safety-bbox]="industrial-safety-bbox|Industrial-Site-Safety-Detection-v1-DATASET|6thdataset/data.yaml|Industrial-Site-Safety-Detection-v1-DATASET/json/processed_bbox_trainv2.jsonl|Industrial-Site-Safety-Detection-v1-DATASET|Industrial_Site_Safety_processed_bbox_train.jsonl|Industrial-Site-Safety-Detection-v1-DATASET|327"
)

declare -A EXTRA_KEYPOINT_CASES=(
  [ap-10k-keypoint]="ap-10k-keypoint|keypoints/ap-10k/annotations/ap10k-train-split1.json|RexOmni/keypoints/ap-10k/json/processed_train.jsonl|ap_10k_processed_keypoints_train.jsonl|9741"
  [apt36k-keypoint]="apt36k-keypoint|keypoints/APT36k/apt36k_annotations.json|RexOmni/keypoints/APT36k/json/processed_train.jsonl|APT36k_processed_keypoints_train.jsonl|35402"
  [human-art-keypoint]="human-art-keypoint|keypoints/Human-Art/annotations/training_humanart.json|RexOmni/keypoints/Human-Art/json/processed_train.jsonl|Human_Art_processed_keypoints_train.jsonl|33249"
  [macaquepose-keypoint]="macaquepose-keypoint|keypoints/macaquepose_v1/annotations.csv|RexOmni/keypoints/macaquepose_v1/json/processed_train.jsonl|macaquepose_v1_processed_keypoints_train.jsonl|1301"
  [mpii-keypoint]="mpii-keypoint|keypoints/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat|RexOmni/keypoints/mpii/json/processed_train.jsonl|mpii_processed_keypoints_train.jsonl|17408"
  [ochuman-keypoint]="ochuman-keypoint|keypoints/ochuman/ochuman.json|RexOmni/keypoints/ochuman/json/processed_train.jsonl|ochuman_processed_keypoints_train.jsonl|3293"
)

declare -A EXTRA_OCR_CASES=(
  [sroie]="sroie-text-bbox|sroie-text-bbox|sroie-datasetv2|versions/4/SROIE2019/train/box|sroie-datasetv2/json/processed_text_line_trainv2.jsonl|SROIE_processed_ocr_train.jsonl|.|626"
  [icdar2013-word-bbox]="icdar2013-word-bbox|icdar2013-word-bbox|OCR/icdar2013|Challenge2_Training_Task1_GT|OCR/icdar2013/json/processed_word_trainv2.jsonl|icdar2013_processed_ocr_word_bbox_train.jsonl|.|229"
  [icdar2015-word-bbox]="icdar2015-word|icdar2015-word-bbox|OCR/icdar2015|ch4_training_localization_transcription_gt|OCR/icdar2015/json/processed_word_trainv2.jsonl|icdar2015_processed_ocr_word_bbox_train.jsonl|.|979"
  [icdar2015-word-poly]="icdar2015-word|icdar2015-word-poly|OCR/icdar2015|ch4_training_localization_transcription_gt|OCR/icdar2015/json/processed_word_trainv2.jsonl|icdar2015_processed_ocr_word_poly_train.jsonl|.|979"
)

declare -A EXTRA_VISUAL_CASES=(
  [lvis-fruit-vegetable-visual]="lvis-fruit-vegetable-bbox|lvis-fruit-vegetable-visual|LVIS_Fruits_And_Vegetables|data.yaml|LVIS_Fruits_And_Vegetables/json/processed_bbox_trainv2.jsonl|LVIS_Fruits_And_Vegetables_processed_visual_train.jsonl|LVIS_Fruits_And_Vegetables/images/train|11138"
  [sheep-visual]="sheep-bbox|sheep-visual|aerial-sheep-object-detection|data/_annotations.coco.json|aerial-sheep-object-detection/json/processed_bbox_trainv2.jsonl|sheep_processed_visual_train.jsonl|aerial-sheep-object-detection|3602"
  [football-visual]="football-bbox|football-visual|football-object-detection|data/_annotations.coco.json|football-object-detection/json/processed_bbox_trainv2.jsonl|football_processed_visual_train.jsonl|football-object-detection|1140"
  [industrial-safety-visual]="industrial-safety-bbox|industrial-safety-visual|Industrial-Site-Safety-Detection-v1-DATASET|6thdataset/data.yaml|Industrial-Site-Safety-Detection-v1-DATASET/json/processed_bbox_trainv2.jsonl|Industrial_Site_Safety_processed_visual_train.jsonl|Industrial-Site-Safety-Detection-v1-DATASET|126"
  [owdod-visual]="owdod-bbox|owdod-visual|owdod|data/train-00000-of-00004.parquet|owdod/json/processed_bbox_trainv2.jsonl|owdod_processed_visual_train.jsonl|.|8969"
  [objects365-visual]="objects365-bbox|objects365-visual|object365|data/train-*.parquet|object365/json/processed_bbox_trainv2.jsonl|Objects365_processed_visual_train.jsonl|object365|2268587"
  [sku110k-visual]="sku110k-bbox|sku110k-visual|SKU110k|SKU/labels/train|SKU110k/json/processed_bbox_trainv2.jsonl|SKU110k_processed_visual_train.jsonl|SKU110k|28264"
)

usage() {
  cat <<'EOF'
Usage:
  prepare.sh [case ...]

Cases:
  aptv2 blood-cell-bbox blood-cell-visual fsc147
  coco2017-keypoint crowdpose-keypoint sroie
  icdar2013-word-bbox icdar2015-word-bbox icdar2015-word-poly
  cdla-layout doclaynet-core-layout publaynet-layout tablebank-layout
  tabrecset-layout os-atlas-desktop-gui os-atlas-mobile-gui
  os-atlas-rico-gui os-atlas-web-gui showui-desktop-gui ui-refexp-gui
  paco-lvis-bbox sku110k-bbox humanref-bbox
  rexverse-onesentence-bbox rexverse-referring-bbox
  egoobjects-bbox v3det-ovd-bbox owdod-bbox
  objects365-bbox objects365-visual
  lvis-fruit-vegetable-bbox sheep-bbox football-bbox industrial-safety-bbox
  lvis-fruit-vegetable-visual sheep-visual football-visual
  industrial-safety-visual owdod-visual sku110k-visual
  ap-10k-keypoint apt36k-keypoint human-art-keypoint
  macaquepose-keypoint mpii-keypoint ochuman-keypoint

With no case arguments, all supported conversions run. Environment overrides:
  DATA_ROOT, OUTPUT_DIR, PYTHON_BIN, LIMIT

Set LIMIT=1 for a smoke conversion. Without LIMIT, complete JSONLs are written.
Each output is checked for minimum row count, first-row schema, and first-image
decoding. Layout/GUI and OCR cases run a raw-source adapter first when
configured; PubLayNet and UI RefExp require an existing common bbox JSONL.
Source data must already follow docs/train_data_prepare.md.
EOF
}

require_path() {
  if ! compgen -G "$1" >/dev/null; then
    echo "Missing required source path: $1" >&2
    exit 1
  fi
}

limit_args=()
if [[ -n "$LIMIT" ]]; then
  if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit "$LIMIT")
fi

validate_output() {
  local jsonl_path=$1
  local media_root=$2
  local configured_count=$3

  "$PYTHON_BIN" - "$jsonl_path" "$media_root" "$configured_count" "$LIMIT" <<'PY'
import json
import os
import sys

from PIL import Image


jsonl_path, media_root, configured_count, limit = sys.argv[1:]
configured_count = int(configured_count)
minimum_count = min(configured_count, int(limit)) if limit else configured_count

with open(jsonl_path, encoding="utf-8") as handle:
    first_line = handle.readline()
    if not first_line:
        raise ValueError(f"empty JSONL: {jsonl_path}")
    first = json.loads(first_line)
    line_count = 1 + sum(1 for line in handle if line.strip())

missing = {"id", "image", "conversations"}.difference(first)
if missing:
    raise ValueError(f"first row missing fields {sorted(missing)}: {jsonl_path}")
if line_count < minimum_count:
    raise ValueError(
        f"{jsonl_path}: got {line_count} rows, expected at least {minimum_count}"
    )

image_value = first["image"]
if isinstance(image_value, list):
    if not image_value:
        raise ValueError(f"first row has an empty image list: {jsonl_path}")
    image_value = image_value[0]
image_path = image_value if os.path.isabs(image_value) else os.path.join(media_root, image_value)
with Image.open(image_path) as image:
    image.verify()
    image_size = image.size
    image_mode = image.mode

print(
    f"validated rows={line_count} first_image={image_value} "
    f"size={image_size} mode={image_mode}"
)
PY
}

run_case() {
  local name=$1
  local output media_root configured_count
  local command=()
  local source_command=()

  if [[ -n ${EXTRA_VISUAL_CASES[$name]-} ]]; then
    local source_preset visual_preset source_root required source_output output_name media_rel
    IFS='|' read -r source_preset visual_preset source_root required source_output output_name media_rel configured_count <<<"${EXTRA_VISUAL_CASES[$name]}"
    require_path "$DATA_ROOT/$source_root/$required"
    output="$OUTPUT_DIR/$output_name"
    media_root="$DATA_ROOT/$media_rel"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" "$SOURCE_CONVERTER" --dataset "$source_preset" \
      --input "$DATA_ROOT/$source_root" --output "$DATA_ROOT/$source_output" \
      "${limit_args[@]}"
    "$PYTHON_BIN" "$VISUAL_CONVERTER" --dataset "$visual_preset" \
      --input "$DATA_ROOT/$source_output" --output "$output" "${limit_args[@]}"
    validate_output "$output" "$media_root" "$configured_count"
    return
  fi

  if [[ -n ${EXTRA_BBOX_CASES[$name]-} ]]; then
    local preset source_root required source_output final_root output_name media_rel
    IFS='|' read -r preset source_root required source_output final_root output_name media_rel configured_count <<<"${EXTRA_BBOX_CASES[$name]}"
    require_path "$DATA_ROOT/$source_root/$required"
    output="$OUTPUT_DIR/$output_name"
    media_root="$DATA_ROOT/$media_rel"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" "$SOURCE_CONVERTER" --dataset "$preset" \
      --input "$DATA_ROOT/$source_root" --output "$DATA_ROOT/$source_output" \
      "${limit_args[@]}"
    "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py" --dataset "$preset" \
      --input "$DATA_ROOT/$final_root" --output "$output" "${limit_args[@]}"
    validate_output "$output" "$media_root" "$configured_count"
    return
  fi

  if [[ -n ${EXTRA_KEYPOINT_CASES[$name]-} ]]; then
    local preset required intermediate output_name
    IFS='|' read -r preset required intermediate output_name configured_count <<<"${EXTRA_KEYPOINT_CASES[$name]}"
    require_path "$DATA_ROOT/$required"
    output="$OUTPUT_DIR/$output_name"
    media_root="$DATA_ROOT"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" "$KEYPOINT_SOURCE_CONVERTER" --dataset "$preset" \
      --input "$DATA_ROOT" --output "$DATA_ROOT/$intermediate" "${limit_args[@]}"
    "$PYTHON_BIN" "$KEYPOINT_INTERMEDIATE_CONVERTER" --dataset "$preset" \
      --input "$DATA_ROOT/$intermediate" --output "$output" "${limit_args[@]}"
    validate_output "$output" "$media_root" "$configured_count"
    return
  fi

  if [[ -n ${EXTRA_OCR_CASES[$name]-} ]]; then
    local source_preset final_preset source_root required intermediate output_name media_rel
    IFS='|' read -r source_preset final_preset source_root required intermediate output_name media_rel configured_count <<<"${EXTRA_OCR_CASES[$name]}"
    require_path "$DATA_ROOT/$source_root/$required"
    output="$OUTPUT_DIR/$output_name"
    media_root="$DATA_ROOT/$media_rel"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" "$OCR_SOURCE_CONVERTER" --dataset "$source_preset" \
      --input "$DATA_ROOT/$source_root" --output "$DATA_ROOT/$intermediate" "${limit_args[@]}"
    "$PYTHON_BIN" "$CONVERTER_DIR/ocr_to_jsonl.py" --dataset "$final_preset" \
      --input "$DATA_ROOT/$intermediate" --output "$output" "${limit_args[@]}"
    validate_output "$output" "$media_root" "$configured_count"
    return
  fi

  case "$name" in
    aptv2)
      require_path "$DATA_ROOT/APTv2/annotations/train_annotations.json"
      output="$OUTPUT_DIR/APTv2_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=28471
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset aptv2
        --input "$DATA_ROOT/APTv2"
        --output "$output"
      )
      ;;
    blood-cell-bbox)
      require_path "$DATA_ROOT/Blood Cell Detection/train/_annotations.coco.json"
      require_path "$DATA_ROOT/Blood Cell Detection/train"
      output="$OUTPUT_DIR/blood_cell_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT/Blood Cell Detection/train"
      configured_count=255
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset blood-cell
        --input "$DATA_ROOT/Blood Cell Detection"
        --output "$output"
      )
      ;;
    blood-cell-visual)
      require_path "$DATA_ROOT/Blood Cell Detection/train/_annotations.coco.json"
      require_path "$DATA_ROOT/Blood Cell Detection/train"
      output="$OUTPUT_DIR/blood_cell_processed_visual_train.jsonl"
      media_root="$DATA_ROOT/Blood Cell Detection/train"
      configured_count=255
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/visual_prompt_to_jsonl.py"
        --input "$DATA_ROOT/Blood Cell Detection"
        --output "$output"
      )
      ;;
    fsc147)
      require_path "$DATA_ROOT/FSC147/annotation_FSC147_384.json"
      require_path "$DATA_ROOT/FSC147/Train_Test_Val_FSC_147.json"
      require_path "$DATA_ROOT/FSC147/ImageClasses_FSC147.txt"
      output="$OUTPUT_DIR/FSC147_processed_point_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=3659
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/point_to_jsonl.py"
        --input "$DATA_ROOT/FSC147"
        --output "$output"
      )
      ;;
    coco2017-keypoint)
      require_path "$DATA_ROOT/coco2017/annotations/person_keypoints_train2017.json"
      output="$OUTPUT_DIR/coco2017_processed_keypoints_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=56599
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/keypoint_to_jsonl.py"
        --input "$DATA_ROOT/coco2017"
        --output "$output"
      )
      ;;
    crowdpose-keypoint)
      require_path "$DATA_ROOT/keypoints/crowdpose/crowdpose_train.json"
      require_path "$DATA_ROOT/keypoints/crowdpose/images"
      output="$OUTPUT_DIR/crowdpose_processed_keypoints_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=10000
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/keypoint_to_jsonl.py"
        --input "$DATA_ROOT/keypoints/crowdpose"
        --annotation crowdpose_train.json
        --image-dir images
        --image-prefix keypoints/crowdpose/images
        --output "$output"
      )
      ;;
    cdla-layout)
      require_path "$DATA_ROOT/Layout/CDLA_DATASET/train"
      output="$OUTPUT_DIR/CDLA_processed_layout_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=5000
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset cdla-layout --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/Layout/CDLA/json/processed_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset cdla-layout --input "$DATA_ROOT" --output "$output"
      )
      ;;
    doclaynet-core-layout)
      require_path "$DATA_ROOT/Layout/DocLayNet_core/COCO/train.json"
      output="$OUTPUT_DIR/DocLayNet_core_processed_layout_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=69103
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset doclaynet-core-layout --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/Layout/DocLayNet_core/json/processed_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset doclaynet-core-layout --input "$DATA_ROOT" --output "$output"
      )
      ;;
    publaynet-layout)
      require_path "$DATA_ROOT/RexOmni/Layout/publaynet/json/processed_trainv2.jsonl"
      output="$OUTPUT_DIR/publaynet_processed_layout_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=335703
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset publaynet-layout --input "$DATA_ROOT" --output "$output"
      )
      ;;
    tablebank-layout)
      require_path "$DATA_ROOT/Layout/TableBank/TableBank/Detection/annotations/tablebank_latex_train.json"
      require_path "$DATA_ROOT/Layout/TableBank/TableBank/Detection/annotations/tablebank_word_train.json"
      output="$OUTPUT_DIR/TableBank_processed_layout_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=260582
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset tablebank-layout --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/Layout/TableBank/json/processed_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset tablebank-layout --input "$DATA_ROOT" --output "$output"
      )
      ;;
    tabrecset-layout)
      require_path "$DATA_ROOT/Layout/TabRecSet/TabRecSet(CurveTabSet)/TD annotation"
      require_path "$DATA_ROOT/Layout/TabRecSet/TabRecSet(CurveTabSet)/image"
      output="$OUTPUT_DIR/TabRecSet_processed_layout_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=32072
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset tabrecset-layout --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/Layout/TabRecSet/json/processed_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset tabrecset-layout --input "$DATA_ROOT" --output "$output"
      )
      ;;
    os-atlas-desktop-gui)
      require_path "$DATA_ROOT/GUI/OS-Atlas-data/desktop_domain/linux_splited.json"
      output="$OUTPUT_DIR/OS-Atlas-data_processed_desktop_domain_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=1136719
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset os-atlas-desktop-gui --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/GUI/OS-Atlas-data/json/processed_desktop_domain_train.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset os-atlas-desktop-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    os-atlas-mobile-gui)
      require_path "$DATA_ROOT/GUI/OS-Atlas-data/mobile_domain/amex_raw.json"
      output="$OUTPUT_DIR/OS-Atlas-data_processed_mobile_domain_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=1309167
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset os-atlas-mobile-gui --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/GUI/OS-Atlas-data/json/processed_mobile_domain_train.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset os-atlas-mobile-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    os-atlas-rico-gui)
      require_path "$DATA_ROOT/GUI/OS-Atlas-data/mobile_domain/widget_captioning.json"
      output="$OUTPUT_DIR/OS-Atlas-data_processed_rico_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=101426
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset os-atlas-rico-gui --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/GUI/OS-Atlas-data/json/processed_rico_train.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset os-atlas-rico-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    os-atlas-web-gui)
      require_path "$DATA_ROOT/GUI/OS-Atlas-data/web_domain/fineweb_3m.json"
      output="$OUTPUT_DIR/OS-Atlas-data_processed_web_domain_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=12962897
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset os-atlas-web-gui --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/GUI/OS-Atlas-data/json/processed_web_domain_train.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset os-atlas-web-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    showui-desktop-gui)
      require_path "$DATA_ROOT/GUI/ShowUI-desktop/data/train-00000-of-00034.parquet"
      output="$OUTPUT_DIR/ShowUI-desktop_processed_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=7496
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset showui-desktop-gui --input "$DATA_ROOT"
        --output "$DATA_ROOT/RexOmni/GUI/ShowUI-desktop/json/processed_train.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset showui-desktop-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    ui-refexp-gui)
      require_path "$DATA_ROOT/RexOmni/GUI/ui_refexp/json/processed_train.jsonl"
      output="$OUTPUT_DIR/ui_refexp_processed_gui_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=15624
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset ui-refexp-gui --input "$DATA_ROOT" --output "$output"
      )
      ;;
    paco-lvis-bbox)
      require_path "$DATA_ROOT/PACO/paco_lvis_v1_train.json"
      require_path "$DATA_ROOT/PACO/train2017"
      output="$OUTPUT_DIR/PACO_LVIS_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT"
      configured_count=45790
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset paco-lvis-bbox --input "$DATA_ROOT/PACO"
        --output "$DATA_ROOT/PACO/json/processed_bbox_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset paco-lvis-bbox --input "$DATA_ROOT/PACO" --output "$output"
      )
      ;;
    sku110k-bbox)
      require_path "$DATA_ROOT/SKU110k/SKU/labels/train"
      require_path "$DATA_ROOT/SKU110k/SKU+gooreal/labels/train"
      require_path "$DATA_ROOT/SKU110k/gooreal/labels/train"
      output="$OUTPUT_DIR/SKU110k_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT/SKU110k"
      configured_count=28264
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset sku110k-bbox --input "$DATA_ROOT/SKU110k"
        --output "$DATA_ROOT/SKU110k/json/processed_bbox_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset sku110k-bbox --input "$DATA_ROOT/SKU110k" --output "$output"
      )
      ;;
    humanref-bbox)
      require_path "$DATA_ROOT/humanref_cot_45k_converted/all_annotations.json"
      require_path "$DATA_ROOT/humanref_cot_45k_converted/images"
      output="$OUTPUT_DIR/humanref_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT/humanref_cot_45k_converted"
      configured_count=37057
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset humanref-bbox
        --input "$DATA_ROOT/humanref_cot_45k_converted"
        --output "$DATA_ROOT/humanref_cot_45k_converted/json/processed_bbox_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset humanref-bbox
        --input "$DATA_ROOT/humanref_cot_45k_converted" --output "$output"
      )
      ;;
    rexverse-onesentence-bbox)
      require_path "$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense/annotations.jsonl"
      require_path "$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense/images"
      output="$OUTPUT_DIR/rexverse_onesentense_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense"
      configured_count=415050
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset rexverse-onesentence-bbox
        --input "$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense"
        --output "$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense/json/processed_bbox_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset rexverse-onesentence-bbox
        --input "$DATA_ROOT/RexVerse-2M/rexverse2M_onesentense" --output "$output"
      )
      ;;
    rexverse-referring-bbox)
      require_path "$DATA_ROOT/RexVerse-2M/rexverse2M_referring/annotations.jsonl"
      require_path "$DATA_ROOT/RexVerse-2M/rexverse2M_referring/images"
      output="$OUTPUT_DIR/rexverse_referring_processed_bbox_train.jsonl"
      media_root="$DATA_ROOT/RexVerse-2M/rexverse2M_referring"
      configured_count=415050
      source_command=(
        "$PYTHON_BIN" "$SOURCE_CONVERTER"
        --dataset rexverse-referring-bbox
        --input "$DATA_ROOT/RexVerse-2M/rexverse2M_referring"
        --output "$DATA_ROOT/RexVerse-2M/rexverse2M_referring/json/processed_bbox_trainv2.jsonl"
      )
      command=(
        "$PYTHON_BIN" "$CONVERTER_DIR/bbox_to_jsonl.py"
        --dataset rexverse-referring-bbox
        --input "$DATA_ROOT/RexVerse-2M/rexverse2M_referring" --output "$output"
      )
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown case: $name" >&2
      usage >&2
      exit 2
      ;;
  esac

  mkdir -p "$OUTPUT_DIR"
  echo "==> $name"
  if ((${#source_command[@]})); then
    "${source_command[@]}" "${limit_args[@]}"
  fi
  "${command[@]}" "${limit_args[@]}"
  validate_output "$output" "$media_root" "$configured_count"
}

if (($# == 0)); then
  cases=("${ALL_CASES[@]}")
else
  cases=("$@")
fi

for case_name in "${cases[@]}"; do
  run_case "$case_name"
done
