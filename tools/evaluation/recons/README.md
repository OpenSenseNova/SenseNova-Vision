# Multi-View 3D Visual Geometry Evaluation

The multi-view 3D evaluation pipeline in **SenseNova-Vision** is adapted from the [evaluation code of G2VLM](https://github.com/InternRobotics/G2VLM/tree/main/eval_code/recons). The original license is included [here](LICENSE). Modifications made in this repository are licensed under the root license of **OpenSenseNova/SenseNova-Vision**.

## Install Additional Dependencies

Install the additional dependencies required for multi-view 3D evaluation:

```shell
pip install -r requirements.txt
```

## Prepare Benchmark Datasets

Before running the evaluation, please review each preprocessing script and ensure that you have obtained the required licenses and permissions from the original dataset providers.

```shell
bash datasets/preprocess/prepare_7scenes.sh   # Reconstruction
bash datasets/preprocess/prepare_eth3d.sh     # Reconstruction
bash datasets/preprocess/prepare_re10k.sh     # Camera pose estimation
bash datasets/preprocess/prepare_co3dv2.sh    # Camera pose estimation
```

## Multi-View Reconstruction (Point Map Estimation)

```shell
python mv_recon/eval.py varint.predict_root=/your/ouptut/benchmark
```

## Camera Pose Estimation (Relative Angular Error)

```shell
python relpose/eval_angle.py pose_type=c2w varint.predict_root=/your/ouptut/benchmark
```
