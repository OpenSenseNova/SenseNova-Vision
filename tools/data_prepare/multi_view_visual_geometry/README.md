# Multi-View Visual Geometry JSONL Generation

This document provides guidelines for converting multi-view visual geometry
datasets into JSONL records for camera pose estimation and 3D reconstruction tasks.

All camera pose annotations are expressed using the OpenCV's coordinate system
and stored as camera-to-world transformations.

## Camera Pose Estimation

`prepare_camera_pose.py` samples ordered frames from a scene and writes the
relative camera pose of each subsequent frame with respect to the first frame.
It contains readers for:

```text
co3dv2, demon, dl3dv, hypersim, irs, megasynth, mvs_synth,
objaverse, omniobject3d, scenenet, scannetpp, scannetv2,
tartanair, wildrgbd
```

Run the script from the repository root:

```bash
python tools/data_prepare/multi_view_visual_geometry/prepare_camera_pose.py \
  dl3dv \
  --root-dir /path/to/DL3DV-10K/ALL-960P \
  --out-jsonl /path/to/dl3dv_camera_pose.jsonl \
  --limit -1
```

Use `--help` for dataset-specific and output arguments. Original dataset
downloads and the repository data layout are documented in
[`docs/train_data_prepare.md`](../../../docs/train_data_prepare.md).

## Multi-View Reconstruction

Download the datasets listed below:

| Dataset | Download Source |
| --- | --- |
| Hypersim | <https://github.com/apple/ml-hypersim#downloading-the-hypersim-dataset> |
| IRS | <https://github.com/HKBU-HPML/IRS#download> |
| TartanAir | <https://github.com/castacks/tartanair_tools#download-the-training-data> |
| SceneNet RGB-D | <https://robotvault.bitbucket.io/scenenet-rgbd.html> |
| Aria Synthetic Environments | <https://www.projectaria.com/datasets/ase/> |
| BlendedMVG | <https://github.com/yoyo000/blendedmvs#download> |
| MegaSynth | <https://huggingface.co/datasets/hwjiang/MegaSynth> |
| MVSSynth | <https://phuang17.github.io/DeepMVS/mvs-synth.html> |
| OmniObject3D | <https://github.com/omniobject3d/OmniObject3D#download-the-dataset> |

After downloading the datasets, parse the data using the formats and guidelines
provided by each dataset, and convert them into JSONL records as described in
*Appendix A.4 (Multi-view Visual Geometry)* of our paper
[Vision as Unified Multimodal Generation](https://arxiv.org/abs/2607.06560).

The processed dataset should adopt the same layout as the recon3d datasets
provided in [data/dataset_info.py](/data/dataset_info.py).
For details on the target dataset structure, camera trajectory format, and
depth format, please refer to our published corpus
[SN-VC-50M](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M).

We provide an example for preparing the TartanAir dataset:

```shell
python tools/data_prepare/multi_view_visual_geometry/prepare_recon3d.py \
  --dataset tartanair \
  --root-dir /path/to/downloaded/tartanair \
  --out-jsonl jsonl_generate/train_jsonls/tartanair.jsonl \
  --out-anno datas/anno/tartanair/recon
```

After preparing TartanAir, you can add a new record, `recon3d_tartanair`, to
[`data/dataset_info.py`](/data/dataset_info.py) before configuring the dataset
for training:

```python
DATASET_INFO = {
    # ...
    "recon3d_jsonl_sft": {
        # ...
        "recon3d_tartanair": {
            "jsonl_path": "jsonl_generate/train_jsonls/tartanair.jsonl",
            "data_dir": {
                "image_dir": "/path/to/downloaded/tartanair",
                "depth_dir": "/path/to/downloaded/tartanair",
                "camera_dir": "datas/anno/tartanair/recon",
                "depth_scale": 1.0,
            },
            "num_total_samples": 369,
        },
}
```
