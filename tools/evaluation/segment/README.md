# Segmentation Evaluation

This directory contains the panoptic, grounded captioning segmentation, referring
segmentation, and reasoning segmentation metrics used by
`tools/evaluation/eval_all.sh`.

## Entrypoint

```bash
bash eval_segmentation.sh <tmp_dir> <metric_label>
```

`eval_all.sh` calls this script after detection, depth, and normal evaluation.

## Prediction Layout

Place segmentation predictions under:

```text
<tmp_dir>/segmentation/
```

Expected directory structure:

```text
<tmp_dir>/segmentation/pan/val/
<tmp_dir>/segmentation/ade/val/
<tmp_dir>/segmentation/gcg/val/
<tmp_dir>/segmentation/gcg/test/
<tmp_dir>/segmentation/ref/refcoco_val/
<tmp_dir>/segmentation/ref/refcocop_val/
<tmp_dir>/segmentation/ref/refcocog_val/
<tmp_dir>/segmentation/rea/val/
<tmp_dir>/segmentation/rea/test/
```

Missing directories are skipped with a warning, so partial segmentation
evaluation is supported.

## Expected Files

Panoptic and ADE-style tasks expect either a merged prediction file:

```text
predictions.json
```

or split files named by range:

```text
predictions_00000000_00001000.json
predictions_00001000_00002000.json
```

GCG tasks use the same prediction naming convention:

```text
predictions.json
predictions_00000000_00001000.json
```

RefSeg and ReaSeg tasks expect split metric files and confusion matrices:

```text
metrics_00000000_00001000.csv
metrics_00001000_00002000.csv
conf_matrix_00000000_00001000.npy
conf_matrix_00001000_00002000.npy
```

## Metric Scripts

`eval_segmentation.sh` dispatches to:

```text
evaluate_panoptic_segmentation.py
evaluate_grounded_caption_segmentation.py
evaluate_region_segmentation.py
```

Shared repository utilities such as RLE decoding and confusion-matrix
construction are imported from the top-level `utils/` package. Segmentation
evaluator-specific helpers, such as COCO wrappers, PQ calculation, caption
metrics, and summary-table export, live in `segment_utils/`.

Ground-truth resources default to the repository data layout:

```text
<repo>/datas/
```

Set `SEG_EVAL_DATA_ROOT=/path/to/datas` only when the prepared data lives
outside the repository. The script resolves ground-truth JSON and mask folders
from that data root directly; no tool-local `datas` symlink is required.

RefSeg/ReaSeg metrics are computed directly from `predictions_*.json` by
`evaluate_region_segmentation.py`. Each JSON item must contain `gt_name` and a
COCO RLE `pred_mask`. Saved PNG masks under `pred_masks/` are optional visual
artifacts and are not required for evaluation.

## Outputs

Metrics are written under:

```text
<tmp_dir>/segmentation/metrics/
```

The normalized summary JSONL is written as:

```text
<tmp_dir>/segmentation/metrics/<metric_label>_Segmentation.jsonl
```
