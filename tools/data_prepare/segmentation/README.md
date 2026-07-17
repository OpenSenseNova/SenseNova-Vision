# Segmentation Data Preparation

This directory contains both benchmark-label preparation and training JSONL
conversion. The two flows stay together because they share mask decoding,
prompt definitions, and referring-segmentation dataset APIs.

```text
segmentation/
├── prepare_binary.py
├── prepare_semantic.py
├── check_reusable_data.sh
├── common/
│   ├── prompts.py
│   ├── refer.py
│   └── grefer.py
└── converters/
    ├── coco_to_binary.py
    ├── masks_to_coco.py
    └── structured_to_coco.py
```

Run the commands below from the repository root. Dataset downloads and the
expected `datas/` layout are documented in
[`docs/data_prepare.md`](../../../docs/data_prepare.md) and
[`docs/train_data_prepare.md`](../../../docs/train_data_prepare.md).

## Primary Entrypoints

`prepare_binary.py` handles RefCOCO/+/g, RefCLEF, gRefCOCO, ReasonSeg, and
COCO-Interactive. Training splits can write masks and training JSONL together;
benchmark splits can prepare masks without replacing released benchmark JSONL.

```bash
python tools/data_prepare/segmentation/prepare_binary.py --help
```

`prepare_semantic.py` handles COCO panoptic-to-semantic conversion and ADE20K
label normalization.

```bash
python tools/data_prepare/segmentation/prepare_semantic.py --help
```

`check_reusable_data.sh` checks whether an extracted directory or a valid local
archive can satisfy a required data path before another download is attempted.

## Additional Source Formats

The scripts under `converters/` cover representative datasets that do not use
the primary benchmark-derived entrypoints. Their pipeline is:

```text
original annotation or masks
  -> optional COCO annotation JSON
  -> binary masks and training JSONL
```

### Existing COCO Annotations

Use `coco_to_binary.py` when a dataset already provides image-level COCO
annotations. VizWiz is the configured representative case.

```bash
python tools/data_prepare/segmentation/converters/coco_to_binary.py \
  --dataset vizwiz \
  --data-root /path/to/datasets/vizwiz \
  --ann base_annotations.json \
  --dst-dir vizwiz_masks \
  --output-dir ./jsonl_generate_ref \
  --num-workers 4
```

### Structured Annotations

Use `structured_to_coco.py` to normalize structured video or track annotations
before the COCO-to-binary stage. VOS2022 is the configured representative case.

```bash
python tools/data_prepare/segmentation/converters/structured_to_coco.py \
  --dataset VOS2022 \
  --data-root /path/to/datasets/VOS2022 \
  --source train/instances.json \
  --output ./coco_annotations/VOS2022/coco_annotation.json

python tools/data_prepare/segmentation/converters/coco_to_binary.py \
  --dataset VOS2022 \
  --data-root /path/to/datasets/VOS2022 \
  --ann ./coco_annotations/VOS2022/coco_annotation.json \
  --dst-dir VOS2022_masks \
  --output-dir ./jsonl_generate_ref \
  --num-workers 4
```

### Image and Mask Directories

Use `masks_to_coco.py` when a dataset provides images and masks but no COCO
annotation file. DOORS is the configured representative case.

```bash
python tools/data_prepare/segmentation/converters/masks_to_coco.py \
  --dataset DOORS \
  --data-root /path/to/datasets/DOORS \
  --output ./coco_annotations/DOORS/coco_annotation.json

python tools/data_prepare/segmentation/converters/coco_to_binary.py \
  --dataset DOORS \
  --data-root /path/to/datasets/DOORS \
  --ann ./coco_annotations/DOORS/coco_annotation.json \
  --dst-dir DOORS_mask \
  --output-dir ./jsonl_generate_ref \
  --num-workers 4
```

The representative converters do not imply support for every dataset with a
similar annotation type. Add dataset-specific parsing and path rules before
documenting another dataset as supported.
