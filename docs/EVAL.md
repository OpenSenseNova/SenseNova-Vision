# SenseNova-Vision Benchmark Guide

This document describes the benchmark and evaluation process.

For benchmark dataset preparation, see [docs/data_prepare.md](data_prepare.md).

Before running the benchmark, set up your runtime environment as described in [docs/INFERENCE.md](INFERENCE.md).

## 1. Benchmark

`benchmark` runs `scripts/inference_benchmark.sh`, which launches local
benchmark inference jobs across GPUs:

```bash
bash scripts/run_sensenova_vision.sh benchmark [tasks] [sub_tasks] [benchmark args]
```

Before running benchmark commands, prepare `datas/` and `jsonl_generate/` by
following [docs/data_prepare.md](data_prepare.md). The benchmark entrypoint
checks the selected data and JSONL files before launching jobs.

**Recommended:** at least one 8 x 80GB GPU machine for the full benchmark.

Set `MODEL_PATH` to a completed local model directory before full benchmark
runs. This avoids repeated Hugging Face access checks in parallel workers.

Positional arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `tasks` | `all` | `all`, `seg`, `detection`, `depth`, `normal`, `recon3d`, `camera_pose`, or comma-separated values. |
| `sub_tasks` | `all` | Segmentation dataset/sub-task selector. Use `all` or comma-separated segmentation sub-task names. |

Common benchmark arguments:

| Argument | Required | Default | Description |
| --- | ---: | --- | --- |
| `--model_path` | Yes, unless `MODEL_PATH` is set | `MODEL_PATH` | Use a local model directory for the current command. |
| `--output_dir` | No | `output/benchmark` | Output directory for benchmark predictions. Runtime logs are written to `<output_dir>/local_logs/`. |
| `--tasks` | No | positional `tasks` value | Same as positional `tasks`. |
| `--sub_tasks` | No | positional `sub_tasks` value | Same as positional `sub_tasks`. |
| `--data_root` | No | `datas` | Benchmark data directory. Pass the `datas/` directory itself, for example `/path/to/datas`; keep the directory name `datas` for the provided segmentation JSONL files. |
| `--jsonl_root` | No | `jsonl_generate` | Benchmark JSONL root. Use only when JSONL files live outside the repository `jsonl_generate/` directory. |
| `--num_gpus` | No | `8` | Number of local GPUs. |
| `--tasks_per_gpu` | No | `1` | Concurrent jobs per GPU. NVIDIA H100 80GB GPUs can usually run with `--tasks_per_gpu 2`. |
| `--total_test_length` | No | full dataset | Sample limit per dataset for quick validation runs. |
| `--save_pred_masks` | No | off | Save segmentation prediction masks for visual inspection. |

Benchmark tasks:

| Task | Batch script | Dataset root |
| --- | --- | --- |
| `seg` | `inference/benchmark/batch_panoptic_segment.py`, `batch_gcg_segment.py`, `batch_binary_segment.py` | `datas` |
| `detection` | `inference/benchmark/batch_detect.py` | `datas/detection_data` |
| `depth` | `inference/benchmark/batch_dense_geometry.py --test_mode Depth` | `datas/geometry_data` |
| `normal` | `inference/benchmark/batch_dense_geometry.py --test_mode Normal` | `datas/geometry_data` |
| `recon3d` | `inference/benchmark/batch_recon3d.py` | `datas/multiview3d_data` |
| `camera_pose` | `inference/benchmark/batch_camera_pose.py` | `datas/multiview3d_data` |

Segmentation sub-tasks:

```text
pan_coco_val, ade20k_pan_val, gcg_val, gcg_test,
refcoco_val, refcocop_val, refcocog_val, reason_val, reason_test
```

Examples:

```bash
bash scripts/run_sensenova_vision.sh benchmark seg pan_coco_val \
  --total_test_length 16 \
  --num_gpus 1 \
  --tasks_per_gpu 1 \
  --save_pred_masks
```

```bash
bash scripts/run_sensenova_vision.sh benchmark detection all \
  --total_test_length 16 \
  --num_gpus 1 \
  --tasks_per_gpu 1
```

```bash
bash scripts/run_sensenova_vision.sh benchmark depth all \
  --data_root /path/to/datas \
  --total_test_length 16
```

`scripts/inference_benchmark.sh` checks required JSONL files and selected data
directories before launching jobs.

Runtime logs are always written to `<output_dir>/local_logs/`. Prediction
directories such as `<output_dir>/depth/scannet_depth/`,
`<output_dir>/normal/scannet_normal/`, and segmentation or detection result
directories should contain only prediction artifacts. Do not place `.log` files
inside prediction directories. Some evaluators infer prediction file extensions
from the files in the prediction directory; an extra log file can make a valid
prediction directory evaluate to empty or zero metrics.

## 2. Evaluation

After benchmark inference finishes, compute metrics through:

```bash
bash scripts/run_sensenova_vision.sh evaluate <output_dir> [tasks] [eval args]
```

This command routes to `tools/evaluation/eval_all.sh`.
The evaluation scripts are included under `tools/evaluation/`.

Common evaluation arguments:

| Argument | Required | Default | Description |
| --- | ---: | --- | --- |
| `output_dir` | Yes | none | Benchmark output directory produced by `benchmark`. |
| `--tasks` | No | `all` | `all`, `detection`, `depth`, `normal`, `segmentation`, `recon3d`, `camera_pose`, or comma-separated values. |
| `--parallel` | No | off | Run selected top-level metric tasks in parallel. This is task-level parallelism over detection/depth/normal/segmentation, not dataset- or split-level parallelism inside an evaluator. |

Evaluation tools use the active Python environment by default. Override it with
`EVAL_PYTHON=/path/to/python` when the metric dependencies are installed in a
separate environment.

Examples:

```bash
bash scripts/run_sensenova_vision.sh evaluate output/benchmark all
```

```bash
bash scripts/run_sensenova_vision.sh evaluate \
  output/benchmark \
  segmentation
```

Referring and reasoning segmentation predictions include enough information to
compute binary-mask metrics directly. Each `predictions_*.json` file written by
`inference/benchmark/batch_binary_segment.py` stores the ground-truth mask path
in `gt_name` and the predicted binary mask in COCO RLE format under
`pred_mask`. Ref/Rea metrics are computed from these JSON predictions by
`tools/evaluation/segment/evaluate_region_segmentation.py`. The optional
`--save_pred_masks` flag saves PNG masks for visual inspection only; these PNG
files are not required for evaluation.

```bash
python tools/evaluation/segment/evaluate_region_segmentation.py \
  --segmentation_root output/benchmark/segmentation \
  --repo_root . \
  --write_auto_tables
```

Evaluation metrics are saved under `output/benchmark`:

| Task | Location |
| --- | --- |
| Detection | detection/metrics/summary.md |
| Segmentation | segmentation/metrics/summary.md |
| Referring and reasoning segmentation | segmentation/metrics/ref_rea_metrics.md |
| Depth | depth/metrics/sensenova-vision_Depth.md |
| Normal | normal/metrics/sensenova-vision_Normal.md |
| Reconstruction | recon3d/\<metric\>.csv |
| Camera pose | camera_pose/\<metric\>.csv |

## Benchmark Preflight Checks

Run these checks before launching expensive benchmark jobs:

```bash
test -n "${MODEL_PATH:-}" && test -d "${MODEL_PATH}"
test -f "${MODEL_PATH}/llm_config.json"
bash scripts/run_sensenova_vision.sh --help
bash scripts/inference_benchmark.sh --help
bash tools/evaluation/eval_all.sh --help
LOCAL_INFER_DRY_RUN=1 bash scripts/run_sensenova_vision.sh benchmark detection all --num_gpus 1 --tasks_per_gpu 1
```

The dry-run command expands the benchmark jobs and validates the selected data
and JSONL paths without starting model inference.
