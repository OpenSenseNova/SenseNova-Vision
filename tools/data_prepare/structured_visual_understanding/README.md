# Structured Visual Understanding Dataset Conversion Examples

This directory contains a small set of standalone example scripts for
converting public vision datasets into training-ready `jsonl` files for
multimodal structured visual understanding tasks.

At a high level, each converter follows the same workflow:

- Download the original public dataset.
- Reorganize the files into a local folder layout expected by the converter.
- Parse the original annotations and map them into a unified training `jsonl` format.
- Adjust prompt templates, category text, coordinate conversion, and output fields as needed.

For bbox datasets with non-COCO raw annotations, the flow is explicitly split
into two reusable stages:

1. `bbox_source_to_jsonl.py` maps COCO, LabelMe, JSON-array, Parquet, unified
   bbox JSON/JSONL, or YOLO source annotations to a common
   `image_name/image_path + image_info + annotation.boxes` JSONL.
2. `bbox_to_jsonl.py` maps either COCO or that common bbox JSONL to the final
   conversation-style training JSONL.

Dataset paths and schema choices live in `configs/bbox_source_datasets.json`,
`configs/bbox_datasets.json`, `configs/keypoint_datasets.json`, and
`configs/point_datasets.json`; visual-prompt settings live in
`configs/visual_prompt_datasets.json`; OCR settings live in
`configs/ocr_source_datasets.json` and `configs/ocr_datasets.json`. They are
not hard-coded in the converters.

These scripts are meant as practical references rather than universal dataset
loaders. Public datasets often differ substantially in storage layout,
annotation schema, category definitions, split files, and licensing terms, so
in real projects you will usually need to adapt the parser, path mapping, field
names, and output formatting to match your own data organization and training
setup. If you are working with other public datasets, you can treat the scripts
here as templates and extend them for your own supervision target.

For closely related grounding-style tasks such as Referring Detection, Layout
Analysis, and GUI Grounding, the input is likewise an image plus a category or
expression to localize, and the output is still a bounding box. In practice,
the training data conversion flow for these tasks can usually follow the same
recipe as the bbox detection examples in this directory, with only light prompt
or annotation-schema adjustments.

The generated data follows these rules:

- Each line in the output file is one independent JSON object, so the final file is standard `jsonl`.
- A typical record contains:

```json
{
  "id": 0,
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>prompt text ..."
    },
    {
      "from": "gpt",
      "value": "structured target text ..."
    }
  ]
}
```

- `id` is the sample index inside the converted file.
- `image` stores the relative image path expected by the downstream training pipeline. It does not have to be identical to the raw download folder name, as long as your loader resolves it consistently.
- `conversations` contains the instruction-response pair.
- The human prompt starts with `<image>` and then directly continues with the prompt text.
- The assistant response is plain text, but it uses lightweight structure markers to encode labels and coordinates.
- Normalized coordinates are clipped into `0.000-0.999` and written with three decimal places.

Common markers used in structured visual understanding outputs:

- `<p>...</p>`: phrase span, such as object category text, referring text, or other semantic labels.
- `<bbox>...</bbox>`: normalized bounding box coordinates, usually in `xyxy` order.
- `<point>...</point>`: normalized point coordinates.
- `<kpt>...</kpt>`: keypoint coordinates or keypoint visibility placeholders such as `unvisible`.
- `<ins>...</ins>`: one object instance block, often used to group one bbox and its keypoints together.
- `<polygon>...</polygon>`: polygon coordinates for tasks that need region outlines.

In other words, the image itself stays external as an image file, while the `jsonl` stores:

- which image to read,
- what instruction to ask about that image,
- and how the structured answer should be represented in text.

## Example Scripts

- `bbox_source_to_jsonl.py`
  raw Layout/GUI/other bbox annotations to the common bbox JSONL
- `bbox_to_jsonl.py`
  COCO or common bbox JSONL to bbox training `jsonl`
- `point_to_jsonl.py`
  FSC147 to point training `jsonl`
- `visual_prompt_to_jsonl.py`
  COCO or common bbox JSONL to visual-prompt training `jsonl`
- `keypoint_to_jsonl.py`
  COCO-style COCO 2017 or CrowdPose keypoints to training `jsonl`
- `keypoint_source_to_jsonl.py`
  dataset-specific animal/human pose annotations to a common keypoint JSONL
- `keypoint_intermediate_to_jsonl.py`
  common keypoint JSONL to keypoint training `jsonl`
- `ocr_source_to_jsonl.py`
  raw OCR annotations to a common OCR JSONL
- `ocr_to_jsonl.py`
  common OCR JSONL to bbox/polygon OCR training `jsonl`

## Download Raw Datasets

Please download the original datasets from their source pages and follow the corresponding licenses:

- APTv2:
  `https://huggingface.co/datasets/DenisKochetov/APTv2`
- Objects365:
  `https://huggingface.co/datasets/jxu124/objects365`
- Blood Cell Detection:
  `https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu`
- FSC147 / Learning to Count Everything:
  `https://github.com/cvlab-stonybrook/LearningToCountEverything`
- COCO 2017:
  `https://cocodataset.org/#download`
- CrowdPose:
  `https://github.com/Jeff-sjtu/CrowdPose`
- SROIE 2019:
  `https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data`
- ICDAR 2013:
  `https://rrc.cvc.uab.es/?ch=2&com=downloads`
- ICDAR 2015:
  `https://rrc.cvc.uab.es/?ch=4&com=downloads`



## Example Folder Layout

For the example scripts in this directory, place the raw data under the repository `data/` directory like this:

```text
YOUR_DATASET_PATH/
├── README.md
├── converters/
│   ├── bbox_source_to_jsonl.py
│   ├── bbox_to_jsonl.py
│   ├── point_to_jsonl.py
│   ├── visual_prompt_to_jsonl.py
│   ├── keypoint_to_jsonl.py
│   └── ...
└── data/
    ├── Blood Cell Detection/
    │   ├── train/
    │   │   ├── _annotations.coco.json
    │   │   └── *.jpg
    │   ├── valid/
    │   └── test/
    ├── APTv2/
    │   ├── annotations/
    │   │   └── train_annotations.json
    │   └── data/
    │       ├── easy/
    │       └── hard/
    ├── FSC147/
    │   ├── annotation_FSC147_384.json
    │   ├── ImageClasses_FSC147.txt
    │   ├── Train_Test_Val_FSC_147.json
    │   └── images_384_VarV2/
    │       └── *.jpg
    ├── keypoints/
    │   └── crowdpose/
    │       ├── crowdpose_train.json
    │       └── images/
    ├── sroie-datasetv2/
    │   └── versions/
    │       └── 4/
    │           └── SROIE2019/
    │               └── train/
    │                   ├── box/
    │                   │   └── *.txt
    │                   └── img/
    │                       └── *.jpg
    ├── OCR/
    │   ├── icdar2013/
    │   │   ├── Challenge2_Training_Task1_GT/
    │   │   │   └── gt_*.txt
    │   │   └── Challenge2_Training_Task12_Images/
    │   │       └── *.jpg
    │   └── icdar2015/
    │       ├── ch4_training_images/
    │       │   └── *.jpg
    │       └── ch4_training_localization_transcription_gt/
    │           └── gt_*.txt
    └── coco2017/
        ├── annotations/
        │   └── person_keypoints_train2017.json
        └── train2017/
            └── *.jpg
```

For these examples, the converted outputs are written to:

```text
jsonl_train/
├── bbox/APTv2/processed_bbox_train.jsonl
├── bbox/blood-cell-object-detection/processed_bbox_train.jsonl
├── ocr/icdar2013/processed_word_bbox_train.jsonl
├── ocr/icdar2015/processed_word_bbox_train.jsonl
├── ocr/icdar2015/processed_word_poly_train.jsonl
├── ocr/SROIE/processed_ocr_train.jsonl
├── point/FSC147/processed_point_train.jsonl
├── visual_prompt/blood-cell-object-detection/processed_visual_train.jsonl
├── keypoints/coco2017/processed_keypoints_train.jsonl
└── keypoints/crowdpose/processed_keypoints_train.jsonl
```


## Run

For configured repository datasets, run the task wrapper from the repository
root. It executes any required source-to-intermediate stage, writes the final
training JSONL, and validates the first record and image.

```bash
LIMIT=1 bash tools/data_prepare/structured_visual_understanding/prepare.sh aptv2
```

For individual converter development, run the scripts from this directory.
Each converter already has a default
raw-data location under `data/` and a default output location under
`jsonl_train/`, so the simplest usage is just:

```bash
python converters/bbox_to_jsonl.py
python converters/bbox_to_jsonl.py --dataset aptv2
python converters/bbox_source_to_jsonl.py --dataset cdla-layout
python converters/bbox_to_jsonl.py --dataset cdla-layout
python converters/bbox_source_to_jsonl.py --dataset humanref-bbox
python converters/bbox_to_jsonl.py --dataset humanref-bbox
python converters/ocr_source_to_jsonl.py --dataset sroie-text-bbox
python converters/ocr_to_jsonl.py --dataset sroie-text-bbox
python converters/ocr_source_to_jsonl.py --dataset icdar2013-word-bbox
python converters/ocr_to_jsonl.py --dataset icdar2013-word-bbox
python converters/point_to_jsonl.py
python converters/visual_prompt_to_jsonl.py
python converters/visual_prompt_to_jsonl.py --dataset lvis-fruit-vegetable-visual
python converters/keypoint_to_jsonl.py
python converters/keypoint_source_to_jsonl.py --dataset ap-10k-keypoint
python converters/keypoint_intermediate_to_jsonl.py --dataset ap-10k-keypoint
```

The shared usage pattern is:

```bash
python converters/<script_name>.py \
  --input data/<dataset_root> \
  --output jsonl_train/<task>/<dataset_name>/<output_name>.jsonl \
  --limit 100
```

In practice:

- `bbox_to_jsonl.py`
  Uses the `blood-cell` preset by default. With `--dataset aptv2`, reads
  `data/APTv2` and writes `jsonl_train/bbox/APTv2/processed_bbox_train.jsonl`.
  Dataset layouts and category aliases are declared in
  `configs/bbox_datasets.json`; use `--config` to select another config file.
- `ocr_to_jsonl.py`
  Reads the common OCR JSONL selected in `configs/ocr_datasets.json` and
  writes the final bbox or polygon OCR training JSONL.
- `ocr_source_to_jsonl.py`
  Reads the raw OCR dataset selected in `configs/ocr_source_datasets.json` and
  writes the common OCR JSONL consumed by `ocr_to_jsonl.py`.
- `point_to_jsonl.py`
  Uses `fsc147` by default. HumanRef and RexVerse presets consume a common
  point JSONL configured in `configs/point_datasets.json`.
- `visual_prompt_to_jsonl.py`
  Reads COCO or the common bbox JSONL selected in
  `configs/visual_prompt_datasets.json` and writes visual-prompt training JSONL.
- `keypoint_to_jsonl.py`
  Reads `data/coco2017` by default. COCO-style datasets such as CrowdPose can
  override `--annotation`, `--image-dir`, and `--image-prefix`.

Most of the time you only need to override arguments in three cases:

- `--input`
  Your raw dataset is stored in a different location or uses a different root name.
- `--output`
  You want to save the converted file somewhere else.
- `--limit`
  You want a small debug subset before running the full conversion.
