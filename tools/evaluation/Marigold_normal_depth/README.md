# Depth and Surface Normal Evaluation

This directory contains the depth and surface-normal metric runners used by
`tools/evaluation/eval_all.sh`.

## Entrypoints

```bash
bash eval_depth.sh <tmp_dir> <metric_label>
bash eval_normals.sh <tmp_dir> <metric_label>
```

`eval_all.sh` calls depth after detection, then calls normal evaluation.

## Depth Prediction Layout

Depth predictions are read from:

```text
<tmp_dir>/depth/
```

Each dataset may be stored with either directory layout:

```text
<tmp_dir>/depth/nyudepth_v2_depth/
<tmp_dir>/depth/kitti_depth/
<tmp_dir>/depth/eth3d_depth/
<tmp_dir>/depth/scannet_depth/
<tmp_dir>/depth/diode_depth/
```

or dotted-prefix layout:

```text
<tmp_dir>/depth.nyudepth_v2_depth/
<tmp_dir>/depth.kitti_depth/
<tmp_dir>/depth.eth3d_depth/
<tmp_dir>/depth.scannet_depth/
<tmp_dir>/depth.diode_depth/
```

The evaluator resolves both layouts automatically.

## Surface Normal Prediction Layout

Normal predictions are read from:

```text
<tmp_dir>/normal/
```

Each dataset may be stored with either directory layout:

```text
<tmp_dir>/normal/nyu_normal/
<tmp_dir>/normal/scannet_normal/
<tmp_dir>/normal/ibims_normal/
```

or dotted-prefix layout:

```text
<tmp_dir>/normal.nyu_normal/
<tmp_dir>/normal.scannet_normal/
<tmp_dir>/normal.ibims_normal/
```

## Outputs

Depth metrics are written under:

```text
<tmp_dir>/depth/metrics/
```

Normal metrics are written under:

```text
<tmp_dir>/normal/metrics/
```

The normalized summary files are:

```text
<tmp_dir>/depth/metrics/<metric_label>_Depth.jsonl
<tmp_dir>/normal/metrics/<metric_label>_Normal.jsonl
```

## Notes

Set `EVAL_PYTHON` to choose a Python executable. Depth evaluation dataset roots
are configured with `DEPTH_EVAL_DATA_ROOT` or the dataset-specific variables
`DEPTH_NYU_ROOT`, `DEPTH_KITTI_ROOT`, `DEPTH_ETH3D_ROOT`,
`DEPTH_SCANNET_ROOT`, and `DEPTH_DIODE_ROOT`. Surface normal evaluation uses
`NORMAL_EVAL_DATA_ROOT`.
