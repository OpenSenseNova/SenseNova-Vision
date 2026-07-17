# Dense Geometric Prediction JSONL Generation

This directory contains two scripts for converting dense geometric prediction
datasets into JSONL records:

- `prepare_depth.py`: RGB image + depth target
- `prepare_normal.py`: RGB image + surface normal target

Both scripts can inspect samples, save converted target images, and optionally
write JSONL files.

Run the commands below from the repository root.

## Depth Script

```bash
python tools/data_prepare/dense_geometric_prediction/prepare_depth.py DATASET \
  --root-dir /path/to/dataset \
  --out-depth-dir ./depth_images \
  --out-jsonl ./depth.jsonl \
  --limit -1
```

Supported `DATASET` keys:

```text
hypersim, irs, scenenet, tartanair, vkitti
```

### Depth Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `DATASET` | Yes | Dataset reader key. |
| `--root-dir` | No | Dataset root directory. If omitted, the script uses its dataset-specific environment variable or default path. |
| `--limit` | No | Maximum number of successful outputs. Use `-1` for all samples. Default is `5`. |
| `--print-every` | No | Print progress every N visited samples. Use `0` to disable progress lines. |
| `--check-read` | No | Read RGB and depth arrays while inspecting samples. |
| `--check-depth-image` | No | Convert depth to an 8-bit image while inspecting samples. |
| `--out-depth-dir` | No | Directory to save converted 8-bit depth images. |
| `--out-jsonl` | No | Output JSONL path. Requires `--out-depth-dir`. |
| `--append` | No | Append to `--out-jsonl` instead of overwriting. |
| `--start-id` | No | Starting JSONL record id. Defaults to `0`, or existing line count with `--append`. |
| `--seed` | No | Random seed for prompt selection. Use `-1` for non-deterministic prompts. |

## Normal Script

```bash
python tools/data_prepare/dense_geometric_prediction/prepare_normal.py DATASET \
  --root-dir /path/to/dataset \
  --out-normal-dir ./normal_images \
  --out-jsonl ./normal.jsonl \
  --limit -1
```

Supported `DATASET` keys:

```text
hypersim, interiorverse, irs, scenenet, tartanair
```

### Normal Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `DATASET` | Yes | Dataset reader key. |
| `--root-dir` | No | Dataset root directory. If omitted, the script uses its dataset-specific environment variable or default path. |
| `--limit` | No | Maximum number of successful outputs. Use `-1` for all samples. Default is `5`. |
| `--print-every` | No | Print progress every N visited samples. Use `0` to disable progress lines. |
| `--check-read` | No | Read RGB and normal arrays while inspecting samples. |
| `--check-normal-image` | No | Convert or read normal image arrays while inspecting samples. |
| `--out-normal-dir` | No | Directory to save converted normal images. |
| `--out-jsonl` | No | Output JSONL path. Requires `--out-normal-dir`. |
| `--append` | No | Append to `--out-jsonl` instead of overwriting. |
| `--start-id` | No | Starting JSONL record id. Defaults to `0`, or existing line count with `--append`. |
| `--seed` | No | Random seed for prompt selection. Use `-1` for non-deterministic prompts. |

## Examples

### Configured Public Conversions

The following converted entries are wired in `data/dataset_info.py`.
Generated JSONLs go under
`jsonl_generate/train_jsonls/dense_geometric_prediction/`. Generated target
images go under `datas/dense_geometric_prediction/`.

| Entry | Reader key | Input root | Target root | JSONL |
| --- | --- | --- | --- | --- |
| `hypersim_depth` | `hypersim` | `datas/train_data/Hypersim/` | `datas/dense_geometric_prediction/hypersim_depth/` | `jsonl_generate/train_jsonls/dense_geometric_prediction/hypersim_depth.jsonl` |
| `hypersim_normal` | `hypersim` | `datas/train_data/Hypersim/` | `datas/dense_geometric_prediction/hypersim_normal/` | `jsonl_generate/train_jsonls/dense_geometric_prediction/hypersim_normal.jsonl` |

Generate depth JSONL:

```bash
python tools/data_prepare/dense_geometric_prediction/prepare_depth.py hypersim \
  --root-dir datas/train_data/Hypersim \
  --out-depth-dir datas/dense_geometric_prediction/hypersim_depth \
  --out-jsonl jsonl_generate/train_jsonls/dense_geometric_prediction/hypersim_depth.jsonl \
  --limit -1 \
  --print-every 1000
```

Generate normal JSONL:

```bash
python tools/data_prepare/dense_geometric_prediction/prepare_normal.py hypersim \
  --root-dir datas/train_data/Hypersim \
  --out-normal-dir datas/dense_geometric_prediction/hypersim_normal \
  --out-jsonl jsonl_generate/train_jsonls/dense_geometric_prediction/hypersim_normal.jsonl \
  --limit -1 \
  --print-every 1000
```
