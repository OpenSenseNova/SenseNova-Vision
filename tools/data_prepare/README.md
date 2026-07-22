# Data Preparation Tools

This directory groups data preparation tools by training task. Run documented
commands from the repository root unless a task guide says otherwise.

```text
tools/data_prepare/
├── general_understanding/
│   ├── download_ocr_vqa.py
│   └── prepare_llava_v1_5.py
├── general_editing/
│   ├── prepare_sharegpt_4o.py
│   └── prepare_gpt_image_edit.py
├── segmentation/
│   ├── prepare_binary.py
│   ├── prepare_semantic.py
│   ├── check_reusable_data.sh
│   ├── common/
│   └── converters/
├── dense_geometric_prediction/
│   ├── prepare_depth.py
│   └── prepare_normal.py
├── multi_view_visual_geometry/
│   ├── prepare_camera_pose.py
│   └── prepare_recon3d.py
└── structured_visual_understanding/
    ├── prepare.sh
    ├── converters/
    └── configs/
```

Each task owns its complete preparation flow. A flow may contain up to three
stages:

1. Read the original public annotations or generated assets.
2. Normalize dataset-specific annotations into a shared intermediate format
   when several final tasks reuse the same source.
3. Write training JSONL files and any derived local assets.

An intermediate format is an implementation detail inside a task directory;
it does not define a separate top-level tool family.

## Task Guides

| Task | Guide | Scope |
| --- | --- | --- |
| General understanding | [Training data preparation](../../docs/train_data_prepare.md#general-understanding-generation-and-editing) | LLaVA-v1.5 OCR-VQA image download and JSON-to-JSONL conversion with media validation; official FineVision and MAmmoTH-VL sources. |
| General generation | [Training data preparation](../../docs/train_data_prepare.md#general-understanding-generation-and-editing) | Official BLIP3o and ShareGPT-4o source downloads, with unresolved subset differences documented. |
| General editing | [Training data preparation](../../docs/train_data_prepare.md#general-understanding-generation-and-editing) | Official source downloads plus deterministic ShareGPT-4o and GPT-Image-Edit JSONL conversion. |
| Segmentation | [segmentation/README.md](segmentation/README.md) | Benchmark label preparation, training mask/JSONL generation, and representative source-to-COCO converters. |
| Dense geometric prediction | [dense_geometric_prediction/README.md](dense_geometric_prediction/README.md) | RGB-to-depth and RGB-to-surface-normal conversion. |
| Multi-view visual geometry | [multi_view_visual_geometry/README.md](multi_view_visual_geometry/README.md) | Multi-frame sampling and relative camera-pose JSONL generation. |
| Structured visual understanding | [structured_visual_understanding/README.md](structured_visual_understanding/README.md) | Bbox, point, visual-prompt, keypoint, OCR, layout, and GUI data conversion. |

## Entrypoints

General understanding currently provides separate OCR-VQA download and
LLaVA-v1.5 JSONL preparation entrypoints:

```bash
python tools/data_prepare/general_understanding/download_ocr_vqa.py --help
python tools/data_prepare/general_understanding/prepare_llava_v1_5.py --help
```

General editing provides direct dataset preparation entrypoints. Dataset
selection is an option on the GPT-Image-Edit tool, not a separate validation
subcommand; first-case image decoding is part of every conversion:

```bash
python tools/data_prepare/general_editing/prepare_sharegpt_4o.py --help
python tools/data_prepare/general_editing/prepare_gpt_image_edit.py --help
```

Segmentation keeps benchmark and training conversion together because both
flows share mask decoding and dataset APIs:

```bash
python tools/data_prepare/segmentation/prepare_binary.py --help
python tools/data_prepare/segmentation/prepare_semantic.py --help
```

Dense geometric and multi-view tools expose one entrypoint per target type:

```bash
python tools/data_prepare/dense_geometric_prediction/prepare_depth.py --help
python tools/data_prepare/dense_geometric_prediction/prepare_normal.py --help
python tools/data_prepare/multi_view_visual_geometry/prepare_camera_pose.py --help
python tools/data_prepare/multi_view_visual_geometry/prepare_recon3d.py --help
```

Structured visual understanding uses a task wrapper to run configured
source-to-intermediate and intermediate-to-training stages together:

```bash
LIMIT=1 bash tools/data_prepare/structured_visual_understanding/prepare.sh aptv2
```

Always follow the original dataset licenses and terms of use. Public download
locations and repository data layouts are documented in
[`docs/train_data_prepare.md`](../../docs/train_data_prepare.md).
