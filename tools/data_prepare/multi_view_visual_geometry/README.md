# Multi-View Visual Geometry JSONL Generation

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
