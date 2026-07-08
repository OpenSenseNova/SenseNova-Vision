# SenseNova-Vision Detection Evaluation

This directory is based on the Rex-Omni evaluation toolkit:

https://github.com/IDEA-Research/Rex-Omni/tree/master/evaluation

SenseNova-Vision keeps the Rex-Omni evaluation layout under `evaluation/` and adds a wrapper for evaluating already generated benchmark predictions.

# Rex-Omni Evaluation Guide

This guide shows how to download evaluation data, unpack images, and run evaluations across datasets and task types.

### 1 Install FastEvaluate (required for COCO/LVIS metrics)

```bash
cd evaluation/fastevaluate
pip install -e .
```

### 2 Download datasets

- Source: `https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData`
- In this repository, place the downloaded files directly under `datas/detection_data/`.
- After downloading, the directory layout should look like this, with images packaged as `.tar.gz` files:

```
datas/detection_data
  *.tar.gz               # per-dataset image archives (e.g., coco.tar.gz, hiertext.tar.gz, ...)
  _annotations/          # JSONL annotations (multiple eval types)
  _rex_omni_eval_results # The evaluation results of Rex-Omni
```

Unpack the image archives before running:

```bash
cd datas/detection_data
for f in *.tar.gz; do
  echo "Extracting $f" && tar -xzf "$f"
done
```
- Place the downloaded `missing_annotaitons/lvis_v1_val_with_filename2.json` file at `datas/detection_data/coco/lvis_v1_val_with_filename2.json`.

### 3 Evaluation
The evaluation is separated into two categories:
1. COCO/LVIS text-prompt evaluation
2. Other datasets (box/point/visual-prompt)

#### COCO/LVIS text-prompt evaluation in box format
For text prompt evaluation on the COCO and LVIS datasets (box format), run the
standalone scripts from `tools/evaluation/detect/` so their relative paths
resolve correctly.

- For COCO evaluation

```bash
cd sensenova-vision
export DETECTION_EVAL_DATA_ROOT="$PWD/datas/detection_data"
cd tools/evaluation/detect
bash evaluation/scripts/eval_coco.sh \
    YOUR_PREDICTION_PATH/COCO.jsonl \
    YOUR_METRIC_OUTPUT_PATH/box_eval/COCO
```

- For LVIS evaluation

```bash
cd sensenova-vision
export DETECTION_EVAL_DATA_ROOT="$PWD/datas/detection_data"
cd tools/evaluation/detect
bash evaluation/scripts/eval_lvis.sh \
    YOUR_PREDICTION_PATH/LVIS.jsonl \
    YOUR_METRIC_OUTPUT_PATH/box_eval/LVIS
```

#### Other datasets and tasks

- For text prompt tasks (output box, standard box matching)

Supported datasets: `HumanRef`, `RefCOCOg_val`, `RefCOCOg_test`, `Dense200`, `VisDrone`, and `DocLayNet`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --data_path YOUR_PREDICTION_PATH/Dense200.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/box_eval/Dense200.jsonl
```

- For OCR/text detection tasks (output box, category-aware matching)

Supported datasets: `HierText`, `IC15`, `TotalText`, and `SROIE`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --match_by_category \
    --data_path YOUR_PREDICTION_PATH/HierText.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/box_eval/HierText.jsonl
```

- For text prompt tasks (output point)

Supported datasets: `COCO`, `LVIS`, `Dense200`, `VisDrone`, `HumanRef`, `RefCOCOg_val`, and `RefCOCOg_test`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --data_path YOUR_PREDICTION_PATH/point.COCO.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/point_eval/point.COCO.jsonl
```

- For visual prompt tasks

Supported datasets: `COCO`, `LVIS`, `Dense200`, and `FSCD_test`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --data_path YOUR_PREDICTION_PATH/visual.COCO.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/visual_prompt_eval/visual.COCO.jsonl
```

- For GUI grounding tasks

Supported datasets: `gui.screenspot_desktop_v2_icon`, `gui.screenspot_desktop_v2_text`, `gui.screenspot_mobile_v2_icon`, `gui.screenspot_mobile_v2_text`, `gui.screenspot_web_v2_icon`, `gui.screenspot_web_v2_text`, `gui.ScreenSpotPro_cad_icon`, `gui.ScreenSpotPro_cad_text`, `gui.ScreenSpotPro_creative_icon`, `gui.ScreenSpotPro_creative_text`, `gui.ScreenSpotPro_dev_icon`, `gui.ScreenSpotPro_dev_text`, `gui.ScreenSpotPro_office_icon`, `gui.ScreenSpotPro_office_text`, `gui.ScreenSpotPro_os_icon`, `gui.ScreenSpotPro_os_text`, `gui.ScreenSpotPro_sci_icon`, and `gui.ScreenSpotPro_sci_text`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --data_path YOUR_PREDICTION_PATH/gui.screenspot_desktop_v2_icon.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/gui_eval/gui.screenspot_desktop_v2_icon.jsonl
```

- For keypoint tasks

Supported datasets: `keypoint.ap-10k` and `keypoint.coco`.

```bash
cd sensenova-vision
python tools/evaluation/detect/evaluation/metrics/other_metric.py \
    --save_keypoint_metrics \
    --data_path YOUR_PREDICTION_PATH/keypoint.ap-10k.jsonl \
    --output_path YOUR_METRIC_OUTPUT_PATH/keypoint_eval/keypoint.ap-10k.jsonl
```

## SenseNova-Vision Wrapper

Use this wrapper if you already have benchmark prediction files and only want
to compute detection metrics.

Run from the repository root. Benchmark prediction files should be under:

```text
output/benchmark/detection/
```

For the recommended repository-level entrypoint below, if your detection
evaluation data is stored in the default repository layout under
`datas/detection_data/`, no extra environment variable is needed. Only set
`DETECTION_EVAL_DATA_ROOT` when the data lives elsewhere:

```bash
export DETECTION_EVAL_DATA_ROOT=/absolute/path/to/detection_data
```

Recommended command:

```bash
bash scripts/run_sensenova_vision.sh evaluate output/benchmark detection
```

If you want to call the detection wrapper directly:

```bash
export DETECTION_EVAL_DATA_ROOT="$PWD/datas/detection_data"
bash tools/evaluation/detect/evaluation/scripts/eval_detection.sh \
  output/benchmark sensenova-vision-7b
```

Outputs are written to:

```text
output/benchmark/detection/output/
output/benchmark/detection/metrics/
```

The final summary file is:

```text
output/benchmark/detection/metrics/<metric_label>_Detection.jsonl
```
