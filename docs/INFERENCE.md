# SenseNova-Vision Inference Guide

This document describes the public runtime entrypoints.

## Environment

Prepare and activate the Python environment before launching inference:

```bash
bash setup.sh sensenova-vision
conda activate sensenova-vision
```

The dependency set has been validated with PyTorch `2.5.1+cu124` and CUDA
toolkit `12.4`. Other CUDA `12.x` toolkits may work, but run the smoke check
below and pay particular attention to `flash-attn` compilation and import.
`setup.sh` installs `flash-attn==2.6.3` for the reference environment.
If you use a different environment, choose a `flash-attn` version that matches
your Python, PyTorch and CUDA.

```bash
which python
python -c 'import decord, fastevaluate, flash_attn, panopticapi, torch; print("decord=%s torch=%s cuda=%s flash_attn=%s panopticapi=ok fastevaluate=ok" % (decord.__version__, torch.__version__, torch.version.cuda, flash_attn.__version__))'
ldd --version | head -n 1
```

Use a local model directory for inference and benchmark runs:

```bash
export MODEL_PATH=/path/to/SenseNova-Vision-7B-MoT/
```

Use `--model_path` in a command only when that command needs a different
model directory. It affects that command only and does not change `MODEL_PATH`.

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
bash scripts/run_sensenova_vision.sh demo --host 127.0.0.1 --port 9001
```
