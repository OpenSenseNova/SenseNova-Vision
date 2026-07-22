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

`prepare_binary.py` handles RefCOCO/+/g, RefCLEF, gRefCOCO, ReasonSeg,
COCO-Interactive, DOORS, and VIS2022. The DOORS and VIS2022 subcommands fix
their source split, filtering, and output layout in code; users only select the
dataset root. Training splits can write masks and training JSONL together;
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

## Dataset-Specific Binary Training Data

After downloading and arranging the original data as described in
[`docs/train_data_prepare.md`](../../../docs/train_data_prepare.md), run:

```bash
python tools/data_prepare/segmentation/prepare_binary.py doors
python tools/data_prepare/segmentation/prepare_binary.py vis2022 --num-workers 8
```

The DOORS command reuses the original DS1 training masks and writes
`seg_binary_doors.jsonl`. The VIS2022 command creates category-level binary
masks under `train/BINARYMasks/` and writes `seg_binary_vis2022.jsonl`. Both
JSONL files are placed under `jsonl_generate/train_jsonls/segmentation/`.

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

`masks_to_coco.py` and `structured_to_coco.py` are lower-level normalization
helpers for contributor use. They are not alternate public preparation paths
for DOORS or VIS2022. The high-level commands above own the validated split,
mask, prompt, and JSONL contracts. A similar source format does not imply that
another dataset is supported without dataset-specific parsing and validation.
