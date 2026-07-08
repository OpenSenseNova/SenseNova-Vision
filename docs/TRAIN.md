# Training

This repository keeps the open-source training entry for SenseNova-Vision focused on
the CV-Unify recipe.

## Environment

Use `setup.sh` to create the base Conda environment, install the Python
dependencies from `requirements.txt`, and build flash-attn:

```bash
bash setup.sh sensenova-vision
conda activate sensenova-vision
```

Set the pretrained model directory before launching training:

```bash
export SENSENOVA_MODEL_DIR=/path/to/pretrained/sensenova-vision
```

If the runtime needs an explicit repository or conda environment path, set:

```bash
export SENSENOVA_REPO_DIR=/path/to/sensenova-vision
export SENSENOVA_ENV_PATH=/path/to/conda/env
```

## Data

The training recipe uses:

```text
data/configs/cv_unify/cv_unify_baseline_v9.yaml
data/dataset_info.py
```

`data/dataset_info.py` contains the dataset registry and dataset paths used by
the CV-Unify config. Update the paths there to match your local storage before
training.

## Launch Training

```bash
bash scripts/train_cv_unify.sh data/configs/cv_unify/cv_unify_baseline_v9.yaml
```

The script prints the resolved config, repository path, process count, shard
count, replica count, and gradient accumulation setting before launching
`train/pretrain_unified_navit.py` with the CV-Unify defaults.

## Dataset Smoke Test

Use the lightweight dataset test entry to check that config parsing, dataset
loading, and packed-batch construction work before starting a full run.

```bash
bash scripts/test_dataset.sh data/configs/cv_unify/cv_unify_baseline_v9.yaml
```
