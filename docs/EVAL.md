# SenseNova-Vision Inference And Benchmark Guide

This document describes the public runtime entrypoints. Dataset preparation is
documented in [docs/data_prepare.md](data_prepare.md).

## Environment

Enter the repository root before running any command below:

```bash
cd /path/to/sensenova-vision
```

Prepare and activate the Python environment before launching inference:

```bash
bash setup.sh sensenova-vision
conda activate sensenova-vision
```

After activation, verify the runtime before starting inference or benchmark
jobs. The dependency set has been validated with PyTorch `2.5.1+cu124` and CUDA
toolkit `12.4`. Other CUDA `12.x` toolkits may work, but run the smoke check
below and pay particular attention to `flash-attn` compilation and import.
`setup.sh` installs `flash-attn==2.6.3` for the reference environment. If you
install dependencies manually or run on a different base image, choose a
`flash-attn` wheel/build that matches your Python, PyTorch, CUDA, and glibc
versions. A mismatched build often fails as `ImportError`, `undefined symbol`,
or `GLIBC_x.xx not found`.

```bash
which python
python -c 'import decord, fastevaluate, flash_attn, panopticapi, torch; print("decord=%s torch=%s cuda=%s flash_attn=%s panopticapi=ok fastevaluate=ok" % (decord.__version__, torch.__version__, torch.version.cuda, flash_attn.__version__))'
ldd --version | head -n 1
```

Use a local model directory for inference and benchmark runs:

```bash
export MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT/
```

`MODEL_PATH` must point to a fully downloaded model directory or Hugging Face
snapshot directory, not to the parent cache directory. For Hugging Face caches,
the expected form is:

```bash
export MODEL_PATH=/path/to/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/<revision>/
```

When downloading or refreshing the model from Hugging Face, configure the cache
and token in the same shell before the download command. For benchmark runs,
prefer `MODEL_PATH` pointing to the completed local model directory instead of
loading from a Hugging Face repository ID at runtime.

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_TOKEN=<your-huggingface-token>
```

Use `--model_path` in a command only when that command needs a different
model directory. It affects that command only and does not change `MODEL_PATH`.

The repository root is added to `PYTHONPATH` by
`scripts/run_sensenova_vision.sh`. If you run Python modules directly, set it
explicitly:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Quick Start

Run the built-in examples:

```bash
bash scripts/run_sensenova_vision.sh example
```

Run one custom inference:

```bash
bash scripts/run_sensenova_vision.sh inference \
  binary_seg \
  "person" \
  examples/images/2.jpg
```

Launch the web demo:

```bash
MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT \
  bash scripts/run_sensenova_vision.sh demo
```

## Unified Runtime Entrypoint

All public runtime commands are routed through:

```bash
bash scripts/run_sensenova_vision.sh <command> [options]
```

Supported commands:

| Command | Target | Use case |
| --- | --- | --- |
| `example` | `inference/example_visualize.py` | One-command visual example run. |
| `inference` | `inference/inference_demo.py` | Single custom task, query, image path, and model path. |
| `interactive` | `inference/inference_demo.py --interactive` | Keep the model loaded for iterative prompts. |
| `demo` | `inference/app.py` | Launch the Gradio web demo. |
| `benchmark` | `scripts/inference_benchmark.sh` | Run benchmark inference batches. |
| `evaluate` | `tools/evaluation/eval_all.sh` | Compute metrics from benchmark outputs. |

Common environment variables:

| Variable | Required | Description |
| --- | ---: | --- |
| `MODEL_PATH` | Yes for model-backed commands | Default local model directory. `--model_path` can select another path for the current command only. |

## 1. Example

`example` runs the built-in visualization script:

```bash
bash scripts/run_sensenova_vision.sh example
```

It loads `SenseNovaVisionModel`, runs fixed examples, and writes raw outputs and
visualization panels.

Default output directory:

```text
examples/output/example_visualize/
```

Covered examples:

| Example | Task | Output |
| --- | --- | --- |
| 1 | General understanding | Text printed to stdout. |
| 2 | Binary segmentation | Raw mask and overlay visualization. |
| 3 | Depth estimation | Depth image. |
| 4 | Normal estimation | Normal image. |
| 5 | GCG segmentation | Raw mask and overlay visualization. |
| 6 | Object detection | Detection overlay visualization. |
| 7 | Multi-view 3D reconstruction | Point maps and `glb` scene. |
| 8 | Panoptic segmentation | Raw mask and overlay visualization. |
| 9 | Interactive segmentation | Raw mask and prompt/prediction visualization. |

## 2. Inference

`inference` runs one `inference/inference_demo.py` task. Positional arguments map
to task, query, and image path:

```bash
bash scripts/run_sensenova_vision.sh inference [task] [query] [image_path] [extra args]
```

Defaults:

| Positional argument | Default |
| --- | --- |
| `task` | `raw_query` |
| `query` | `What are the main objects in this scene and their relationships?` |
| `image_path` | `examples/images/1.jpg` |

Examples:

```bash
bash scripts/run_sensenova_vision.sh inference \
  depth \
  "" \
  examples/images/3.jpg
```

```bash
bash scripts/run_sensenova_vision.sh inference \
  bbox_detection \
  "bird, boat, person, cell phone, backpack, handbag" \
  examples/images/5.jpg
```

Options after `[task] [query] [image_path]` are passed through to
`inference/inference_demo.py`.

Set a custom output directory:

```bash
bash scripts/run_sensenova_vision.sh inference \
  bbox_detection \
  "bird, boat, person, cell phone, backpack, handbag" \
  examples/images/5.jpg \
  --output_dir examples/output/demo_custom/
```

Use a different model path for one run:

```bash
bash scripts/run_sensenova_vision.sh inference \
  depth \
  "" \
  examples/images/3.jpg \
  --model_path /path/to/SenseNova-Vision-7B-MoT/
```

For `raw_query`, choose the backend mode:

```bash
bash scripts/run_sensenova_vision.sh inference \
  raw_query \
  "What are the main objects in this scene?" \
  examples/images/1.jpg \
  --mode understanding
```

Supported tasks:

| Task | Backend mode | Query |
| --- | --- | --- |
| `raw_query` | selected by `--mode` | Raw prompt. |
| `depth` | `dense_perception` | No need for further query. |
| `normal` | `dense_perception` | No need for further query. |
| `binary_seg` | `dense_perception` | Reference target. |
| `pan_seg` | `caption_generate` | Category list. If omitted, COCO categories are used. |
| `gcg_seg` | `caption_generate` | No need for further query. |
| `bbox_detection` | `dense_detection` | Category list. |
| `point_detection` | `dense_detection` | Category list. |
| `keypoint` | `dense_detection` | Category name, such as `person` or `cat`. |
| `ocr` | `dense_OCR` | No need for further query. |
| `recon3d` | `recon3d` | Comma-separated multi-view images. No need for further query. |
| `camera_pose` | `understanding` | Comma-separated multi-view images. No need for further query. |

Outputs are saved in a task-specific subdirectory under the selected output
directory:

```text
examples/output/demo/<task>/
```

Common files:

| File | Meaning |
| --- | --- |
| `*_prompt.txt` | Exact prompt sent to the model. |
| `*.txt` | Raw text output, when the model returns text. |
| `*.png` | Raw image output, when the model returns an image. |
| `vis/*.png` | Prompt, input, and prediction visualization. |
| `*_pts3d.npy` | Raw 3D point maps for `recon3d`. |
| `*.glb` | Postprocessed 3D scene for `recon3d`. |
| `*_pose.json` | Parsed camera poses for `camera_pose`. |

## 3. Interactive

`interactive` starts `inference/inference_demo.py --interactive` with the
general-understanding default case from the built-in examples:

```bash
bash scripts/run_sensenova_vision.sh interactive \
  --model_path /path/to/SenseNova-Vision-7B-MoT/
```

Change the initial case with forwarded arguments:

```bash
bash scripts/run_sensenova_vision.sh interactive \
  --image_path examples/images/2.jpg \
  --task binary_seg \
  --query "person furthest to the right" \
  --output_dir examples/output/demo_custom/
```

Interactive commands:

```text
/task TASK         Switch task and backend mode together
/mode MODE         Switch to raw_query with MODE
/image PATH        Switch image path; use commas for multi-view inputs
/status            Show current task, mode, and image path
/tasks             Show available tasks
/modes             Show raw_query backend modes
/help              Show help
q                  Exit
```

## 4. Web Demo

`demo` starts the Gradio app and prints the local URL before launching:

```bash
MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT \
  bash scripts/run_sensenova_vision.sh demo
```

**Recommended:** 1 x 80GB GPU for the full web demo.

Forward `inference/app.py` options only when needed:

```bash
bash scripts/run_sensenova_vision.sh demo --host 0.0.0.0 --port 9001
```

## 5. Benchmark

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
| `tasks` | `all` | `all`, `seg`, `detection`, `depth`, `normal`, or comma-separated values. |
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
directories before launching jobs. Missing JSONL errors point to
[docs/data_prepare.md](data_prepare.md).

Runtime logs are always written to `<output_dir>/local_logs/`. Prediction
directories such as `<output_dir>/depth/scannet_depth/`,
`<output_dir>/normal/scannet_normal/`, and segmentation or detection result
directories should contain only prediction artifacts. Do not place `.log` files
inside prediction directories. Some evaluators infer prediction file extensions
from the files in the prediction directory; an extra log file can make a valid
prediction directory evaluate to empty or zero metrics. For old outputs, move
stray logs into `<output_dir>/local_logs/` before computing metrics.

## 6. Evaluation

After benchmark inference finishes, compute metrics through:

```bash
bash scripts/run_sensenova_vision.sh evaluate <output_dir> [tasks] [eval args]
```

This command routes to `tools/evaluation/eval_all.sh`, which runs the metric
scripts for detection, depth, normal, and segmentation. The evaluation scripts
are included under `tools/evaluation/`.

Common evaluation arguments:

| Argument | Required | Default | Description |
| --- | ---: | --- | --- |
| `output_dir` | Yes | none | Benchmark output directory produced by `benchmark`. |
| `tasks` | No | `all` | Positional shortcut for `all`, `detection`, `depth`, `normal`, `segmentation`, or comma-separated values. |
| `--tasks` | No | `all` | `all`, `detection`, `depth`, `normal`, `segmentation`, or comma-separated values. |
| `--parallel` | No | off | Run selected top-level metric tasks in parallel. This is task-level parallelism over detection/depth/normal/segmentation, not dataset- or split-level parallelism inside an evaluator. |

Evaluation tools use the active Python environment by default. Override it with
`EVAL_PYTHON=/path/to/python` when the metric dependencies are installed in a
separate environment. By default, `evaluate` resolves benchmark data from the
standard repository layout under `datas/`. Use these environment variables only
when the prepared data lives elsewhere:

| Variable | Used by | Description |
| --- | --- | --- |
| `DETECTION_EVAL_DATA_ROOT` | detection | Root containing detection evaluation annotations. |
| `DETECTION_COCO_JSON` | detection | COCO annotation JSON. Overrides `DETECTION_EVAL_DATA_ROOT`. |
| `DETECTION_LVIS_JSON` | detection | LVIS annotation JSON. Overrides `DETECTION_EVAL_DATA_ROOT`. |
| `DEPTH_EVAL_DATA_ROOT` | depth | Root containing depth evaluation datasets. |
| `DEPTH_NYU_ROOT`, `DEPTH_KITTI_ROOT`, `DEPTH_ETH3D_ROOT`, `DEPTH_SCANNET_ROOT`, `DEPTH_DIODE_ROOT` | depth | Dataset-specific depth roots. These override `DEPTH_EVAL_DATA_ROOT`. |
| `NORMAL_EVAL_DATA_ROOT` | normal | Root containing normal evaluation datasets. |
| `SEG_EVAL_DATA_ROOT` | segmentation | The `datas/` directory used by segmentation evaluation. |

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
