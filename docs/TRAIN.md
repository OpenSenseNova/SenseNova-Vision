# Training

This guide describes the open-source SenseNova-Vision training entry and its
default multimodal dataset configuration.

Before launching training, prepare the required datasets by following
[docs/train_data_prepare.md](train_data_prepare.md).

## Environment

Use `setup.sh` to create the base Conda environment, install the Python
dependencies from `requirements.txt`, and build flash-attn:

```bash
bash setup.sh sensenova-vision
conda activate sensenova-vision
```

Set the pretrained model directory before launching training.
`SENSENOVA_REPO_DIR` and `SENSENOVA_ENV_PATH` are optional:

```bash
export SENSENOVA_MODEL_DIR=/path/to/pretrained/sensenova-vision
export SENSENOVA_REPO_DIR=/path/to/sensenova-vision
export SENSENOVA_ENV_PATH=/path/to/conda/env
```

## Data

The default training configuration uses:

```text
data/configs/cv_unify/cv_unify_baseline_v9.yaml
data/dataset_info.py
```

`data/dataset_info.py` contains the dataset registry and paths referenced by the
default configuration. Update the paths there to match your local storage
before training.

## Launch Training

Training requires at least 2 machines with 8 x 80GB GPUs each; 32 or more such
machines are recommended.

Submit the following command through a multi-node training system. The launcher
reads `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` from the distributed
runtime. The number of GPUs per node is detected with `nvidia-smi` and can be
overridden with `SENSENOVA_GPU_PER_NODE` when necessary.

```bash
bash scripts/train_cv_unify.sh data/configs/cv_unify/cv_unify_baseline_v9.yaml
```

## Dataset Smoke Test

Use the lightweight dataset test entry to check that config parsing, dataset
loading, and packed-batch construction work before starting a full run.

```bash
bash scripts/test_dataset.sh data/configs/cv_unify/cv_unify_baseline_v9.yaml
```
