# Benchmark Data Preparation

This document describes the public benchmark data layout expected by
`scripts/inference_benchmark.sh`.

## Directory Layout

Run all commands in this document from the repository root.

Default layout:

```text
.
  datas/
    detection_data/
    geometry_data/
    gen_seg_data/
    ov_seg_data/
    gcg_seg_data/
    inter_seg_data/
    ref_seg_data/
    rea_seg_data/
    multiview3d_data/
    ...
  jsonl_generate/
    detection/
    ...
```

If data is prepared under `datas/` and JSONL files are prepared under
`jsonl_generate/` at the repository root, no path override is needed. Use
overrides only when the prepared directories live elsewhere:

```bash
DATA_ROOT=/absolute/path/datas
JSONL_ROOT=/absolute/path/jsonl_generate

bash scripts/run_sensenova_vision.sh benchmark all all -- \
  --data_root "$DATA_ROOT" \
  --jsonl_root "$JSONL_ROOT"
```

## JSONL Generate

Download the JSONL archive from
[SenseNova-Vision-Benchmark](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Benchmark)
and extract it under `jsonl_generate/`:

```bash
mkdir -p jsonl_generate
wget -c -O jsonl_generate/SenseNova-Vision_benchmark_jsonl.tar.gz \
  https://huggingface.co/datasets/sensenova/SenseNova-Vision-Benchmark/resolve/main/SenseNova-Vision_benchmark_jsonl.tar.gz
tar -xzf jsonl_generate/SenseNova-Vision_benchmark_jsonl.tar.gz -C jsonl_generate
rm jsonl_generate/SenseNova-Vision_benchmark_jsonl.tar.gz
```

Expected directory:

```text
jsonl_generate/
```

Detection JSONL files are expected under:

```text
jsonl_generate/detection/
```

## Segmentation Data

Segmentation data preparation follows the organization used by
[X-SAM](https://github.com/wanghao9610/X-SAM). We thank the X-SAM project for
the dataset collection and preparation notes in
[X-SAM datasets.md](https://github.com/wanghao9610/X-SAM/blob/main/docs/mds/datasets.md).

### Segmentation Directory Layout

Prepare the following layout for segmentation inference and metrics:

```text
datas/
  gen_seg_data/
    coco2014/
      train2014/
    coco2017/
      val2017/
      annotations/
        instances_val2017.json
        panoptic_val2017.json
      panoptic_val2017/
      panoptic_semseg_val2017/
  ov_seg_data/
    ade20k/
      images/
        validation/
      ade20k_panoptic_val/
      annotations_detectron2/
        validation/
      ade20k_panoptic_val.json
      ade20k_instance_val.json
  gcg_seg_data/
    annotations/
      val_test/
        val_gcg_coco_mask_gt.json
        val_gcg_coco_caption_gt.json
        test_gcg_coco_mask_gt.json
        test_gcg_coco_caption_gt.json
    images/
      GranDf_HA_images/
        val_test/
  ref_seg_data/
    images/
      coco2014/  # symlink or copy of datas/gen_seg_data/coco2014
    refcoco/
    refcoco+/
    refcocog/
    ref_seg/
      binary_masks/
        refcoco_val/
        refcoco+_val/
        refcocog_val/
  rea_seg_data/
    val/
    test/
    rea_seg/
      binary_masks/
        val/
        test/
  inter_seg_data/
    annotations/
      coco_interactive_train_psalm.json
      coco_interactive_val_psalm.json
    coco2017/  # symlink or copy of datas/gen_seg_data/coco2017
    inter_seg/
      binary_masks/
        coco_interactive_psalm/
          val/
            *.png
            box_visual_prompt_mask/
            mask_visual_prompt_mask/
            point_visual_prompt_mask/
            scribble_visual_prompt_mask/
```

### 1. Generic Segmentation Dataset

Download and extract COCO 2017 images and annotations:

```bash
mkdir -p datas/gen_seg_data/coco2017
COCO17_DIR=datas/gen_seg_data/coco2017

wget http://images.cocodataset.org/zips/val2017.zip -O "$COCO17_DIR/val2017.zip"
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$COCO17_DIR/annotations_trainval2017.zip"
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -O "$COCO17_DIR/panoptic_annotations_trainval2017.zip"

unzip "$COCO17_DIR/val2017.zip" -d "$COCO17_DIR"
unzip "$COCO17_DIR/annotations_trainval2017.zip" -d "$COCO17_DIR"
unzip "$COCO17_DIR/panoptic_annotations_trainval2017.zip" -d "$COCO17_DIR"
unzip "$COCO17_DIR/annotations/panoptic_val2017.zip" -d "$COCO17_DIR"
rm "$COCO17_DIR"/*.zip "$COCO17_DIR/annotations/panoptic_train2017.zip" "$COCO17_DIR/annotations/panoptic_val2017.zip"
unset COCO17_DIR
```

Convert the COCO panoptic masks to contiguous semantic category IDs for the
GenSeg semantic metric:

```bash
python tools/data_prepare/segmentation/prepare_semantic.py coco-panoptic \
  --data-root datas/gen_seg_data/coco2017 \
  --split val
```

Download and extract COCO 2014 images:

```bash
mkdir -p datas/gen_seg_data/coco2014
COCO14_DIR=datas/gen_seg_data/coco2014

wget http://images.cocodataset.org/zips/train2014.zip -O "$COCO14_DIR/train2014.zip"
unzip "$COCO14_DIR/train2014.zip" -d "$COCO14_DIR"
rm "$COCO14_DIR/train2014.zip"
unset COCO14_DIR
```

### 2. Open-Vocabulary Segmentation Dataset

Download ADE20K and create the converted panoptic, semantic, and instance GT.
The conversion commands follow the ADE20K preparation used by
[MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html):

```bash
mkdir -p datas/ov_seg_data tools/external

git clone --depth 1 https://github.com/open-mmlab/mmdetection.git tools/external/mmdetection
python tools/external/mmdetection/tools/misc/download_dataset.py \
  --dataset-name ade20k_2016 \
  --save-dir datas/ov_seg_data \
  --unzip

mv datas/ov_seg_data/ADEChallengeData2016 datas/ov_seg_data/ade20k
mv datas/ov_seg_data/annotations_instance datas/ov_seg_data/ade20k/
mv datas/ov_seg_data/categoryMapping.txt datas/ov_seg_data/ade20k/
mv datas/ov_seg_data/imgCatIds.json datas/ov_seg_data/ade20k/

python tools/external/mmdetection/tools/dataset_converters/ade20k2coco.py \
  datas/ov_seg_data/ade20k \
  --task panoptic
python tools/external/mmdetection/tools/dataset_converters/ade20k2coco.py \
  datas/ov_seg_data/ade20k \
  --task instance
python tools/data_prepare/segmentation/prepare_semantic.py ade20k \
  --data-root datas/ov_seg_data/ade20k \
  --split validation
```

### 3. Referring Segmentation Dataset

Download and extract RefCOCO, RefCOCO+, and RefCOCOg annotations:

```bash
mkdir -p datas/ref_seg_data/images
REF_DIR=datas/ref_seg_data

wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -O "$REF_DIR/refcoco.zip"
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -O "$REF_DIR/refcoco+.zip"
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -O "$REF_DIR/refcocog.zip"

unzip "$REF_DIR/refcoco.zip" -d "$REF_DIR"
unzip "$REF_DIR/refcoco+.zip" -d "$REF_DIR"
unzip "$REF_DIR/refcocog.zip" -d "$REF_DIR"
rm "$REF_DIR/refcoco.zip" "$REF_DIR/refcoco+.zip" "$REF_DIR/refcocog.zip"
[ -e "$REF_DIR/images/coco2014" ] || ln -s ../../gen_seg_data/coco2014 "$REF_DIR/images/coco2014"
unset REF_DIR

python tools/data_prepare/segmentation/prepare_binary.py refcoco \
  --data-root datas/ref_seg_data \
  --datasets refcoco refcoco+ refcocog \
  --split val
```

### 4. Reasoning Segmentation Dataset

Download ReasonSeg from the
[LISA release Google Drive](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy).

Place `val.zip` and `test.zip` under `datas/rea_seg_data/`, then extract:

```bash
mkdir -p datas/rea_seg_data
REA_DIR=datas/rea_seg_data

unzip "$REA_DIR/val.zip" -d "$REA_DIR"
unzip "$REA_DIR/test.zip" -d "$REA_DIR"
rm "$REA_DIR/val.zip" "$REA_DIR/test.zip"
unset REA_DIR

python tools/data_prepare/segmentation/prepare_binary.py reasonseg \
  --data-root datas/rea_seg_data --split val
python tools/data_prepare/segmentation/prepare_binary.py reasonseg \
  --data-root datas/rea_seg_data --split test
```

### 5. GCG Segmentation Dataset

Download GranD-f annotations from
[GranD-f on Hugging Face](https://huggingface.co/datasets/MBZUAI/GranD-f).
Use the `hfd` helper as documented in
[X-SAM datasets.md](https://github.com/wanghao9610/X-SAM/blob/main/docs/mds/datasets.md):

```bash
mkdir -p datas/gcg_seg_data/images
GCG_DIR=datas/gcg_seg_data

hfd MBZUAI/GranD-f --tool aria2c -x 8 --include val_test \
  --save_dir "$GCG_DIR" --dataset
mv "$GCG_DIR/GranD-f" "$GCG_DIR/annotations"
```

Download and extract
[GranDf_HA_images.zip](https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0):

```bash
GCG_DIR=datas/gcg_seg_data
wget -O "$GCG_DIR/GranDf_HA_images.zip" \
  "https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0"
unzip "$GCG_DIR/GranDf_HA_images.zip" -d "$GCG_DIR/images"
rm "$GCG_DIR/GranDf_HA_images.zip"
unset GCG_DIR
```

### 6. Image Interactive Segmentation Datasets

Download and extract
[PSALM_data.zip](https://drive.usercontent.google.com/download?id=1X4N5EJr3C63uAcnS1eC-yE2QoHU-aOKY&export=download):

```bash
mkdir -p datas/inter_seg_data/annotations
INTER_DIR=datas/inter_seg_data

wget -O "$INTER_DIR/PSALM_data.zip" \
  "https://drive.usercontent.google.com/download?id=1X4N5EJr3C63uAcnS1eC-yE2QoHU-aOKY&export=download"
unzip "$INTER_DIR/PSALM_data.zip" -d "$INTER_DIR"
mv "$INTER_DIR/PSALM_data/coco_interactive_train_psalm.json" "$INTER_DIR/annotations/"
mv "$INTER_DIR/PSALM_data/coco_interactive_val_psalm.json" "$INTER_DIR/annotations/"
[ -e "$INTER_DIR/coco2017" ] || ln -s ../gen_seg_data/coco2017 "$INTER_DIR/coco2017"
rm -rf "$INTER_DIR/PSALM_data" "$INTER_DIR/PSALM_data.zip"
unset INTER_DIR

python tools/data_prepare/segmentation/prepare_binary.py coco-interactive \
  --data-root datas/inter_seg_data \
  --dataset coco_interactive_psalm \
  --split val
```

## Detection Data

If you want to run detection evaluation, also prepare the Rex-Omni evaluation
data used by `tools/evaluation/detect/`.

The detection evaluation data layout here follows Rex-Omni. We use the
pre-organized Rex-Omni evaluation bundle directly instead of re-collecting and
re-formatting each original detection dataset in this repository.

Source:

- `https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData`

After downloading, place the files under:

```text
datas/detection_data/
```

Expected directory layout:

```text
datas/detection_data/
  *.tar.gz               # per-dataset image archives, such as coco.tar.gz
  _annotations/          # JSONL annotations for Rex-Omni evaluation
  _rex_omni_eval_results # evaluation outputs
```

Unpack the image archives before evaluation:

```bash
cd datas/detection_data
for f in *.tar.gz; do
  echo "Extracting $f" && tar -xzf "$f"
done
```

For LVIS evaluation, also place the downloaded missing annotation file lvis_v1_val_with_filename2.json at:

```text
datas/detection_data/coco/lvis_v1_val_with_filename2.json
```

After preparing this directory, run detection evaluation commands from the
repository root:

- `bash scripts/run_sensenova_vision.sh evaluate output/benchmark detection`

## Dense Geometry Data

Download the depth and normal benchmark data archive and extract it as
`datas/geometry_data/`:

```bash
mkdir -p datas/geometry_data
bash ./tools/evaluation/geometry/download_geometry_data.sh
```

Expected directory:

```text
datas/geometry_data/
```

Depth and normal benchmark jobs pass `--data_root datas/geometry_data` to
`inference/benchmark/batch_dense_geometry.py`.

## Multi-View 3D Benchmark Datasets

Dataset preparation scripts are provided under `tools/evaluation/recons/datasets/preprocess/`.

Read each of the scripts before running the following commands to download and prepare each dataset:

```bash
cd tools/evaluation/recons

bash datasets/preprocess/prepare_7scenes.sh   # reconstruction
bash datasets/preprocess/prepare_eth3d.sh     # reconstruction
bash datasets/preprocess/prepare_re10k.sh     # camera pose estimation
bash datasets/preprocess/prepare_co3dv2.sh    # camera pose estimation; manual download required
```

> **Note:** `prepare_co3dv2.sh` assumes the CO3D v2 dataset has already been downloaded manually. See the instructions in the script.

By default, the datasets are prepared under:

```text
tools/evaluation/recons/datas/
```

You should move the directory to your data root and leave a link at the original location:

```bash
DATA_ROOT=datas

mkdir -p "$DATA_ROOT"
mv tools/evaluation/recons/datas "$DATA_ROOT/multiview3d_data"
ln -sfn "$(realpath "$DATA_ROOT/multiview3d_data")" tools/evaluation/recons/datas
```

The expected directory layout is:

```text
$DATA_ROOT/
└── multiview3d_data/
    ├── 7scenes/
    ├── eth3d/
    ├── re10k/
    └── co3dv2/
        ├── annotations/
        └── data/

tools/evaluation/recons/datas -> $DATA_ROOT/multiview3d_data
```

## Dataset Settings

The following tables summarize benchmark sub-tasks, datasets, and public data
pages. JSONL files in this repository are task-specific converted annotations;
the links point to the original datasets or benchmark pages used to construct
those evaluation files.

### Segmentation

| Sub-task | Dataset / split | Public links |
| --- | --- | --- |
| `pan_coco_val` | COCO panoptic validation | [COCO](https://cocodataset.org/) |
| `ade20k_pan_val` | ADE20K panoptic validation | [ADE20K](https://github.com/CSAILVision/ADE20K) |
| `gcg_val` | GCG / GranD-f validation | [GranD-f](https://huggingface.co/datasets/MBZUAI/GranD-f) |
| `gcg_test` | GCG / GranD-f test | [GranD-f](https://huggingface.co/datasets/MBZUAI/GranD-f) |
| `refcoco_val` | RefCOCO validation | [RefCOCO / RefCOCO+ / RefCOCOg](https://tensorflow.google.cn/datasets/catalog/ref_coco) |
| `refcocop_val` | RefCOCO+ validation | [RefCOCO / RefCOCO+ / RefCOCOg](https://tensorflow.google.cn/datasets/catalog/ref_coco) |
| `refcocog_val` | RefCOCOg validation | [RefCOCO / RefCOCO+ / RefCOCOg](https://tensorflow.google.cn/datasets/catalog/ref_coco) |
| `reason_val` | ReasonSeg validation | [ReasonSeg / LISA](https://github.com/JIA-Lab-research/LISA) |
| `reason_test` | ReasonSeg test | [ReasonSeg / LISA](https://github.com/JIA-Lab-research/LISA) |

### Object Detection, OCR, Referring, Pointing, Visual Prompting

| Sub-task / mode | Datasets | Public links |
| --- | --- | --- |
| Common object detection | SROIE, LVIS, Total-Text, VisDrone, Dense200, ICDAR 2015, COCO | [SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE), [LVIS](https://www.lvisdataset.org/), [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [Dense200 / Rex-Omni EvalData](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData), [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4), [COCO](https://cocodataset.org/) |
| Dense OCR | HierText | [HierText](https://github.com/google-research-datasets/hiertext) |
| Dense detection | VisDrone, Dense200 | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [Dense200 / Rex-Omni EvalData](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData) |
| Referring object detection | HumanRef, RefCOCOg val, RefCOCOg test | [HumanRef](https://huggingface.co/datasets/IDEA-Research/HumanRef), [RefCOCO / RefCOCO+ / RefCOCOg](https://tensorflow.google.cn/datasets/catalog/ref_coco) |
| Object pointing | LVIS, VisDrone, Dense200, COCO | [LVIS](https://www.lvisdataset.org/), [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [Dense200 / Rex-Omni EvalData](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData), [COCO](https://cocodataset.org/) |
| Referring pointing | HumanRef, RefCOCOg val, RefCOCOg test | [HumanRef](https://huggingface.co/datasets/IDEA-Research/HumanRef), [RefCOCO / RefCOCO+ / RefCOCOg](https://tensorflow.google.cn/datasets/catalog/ref_coco) |
| Visual prompt detection | LVIS, Dense200, COCO, FSCD/FSC-147 | [LVIS](https://www.lvisdataset.org/), [Dense200 / Rex-Omni EvalData](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData), [COCO](https://cocodataset.org/), [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) |

### Layout Detection, GUI Grounding, Keypoint Detection

| Sub-task | Datasets | Public links |
| --- | --- | --- |
| Document layout detection | DocLayNet | [DocLayNet](https://github.com/DS4SD/DocLayNet) |
| GUI grounding | ScreenSpot-v2 desktop/mobile/web icon/text splits, ScreenSpot-Pro CAD/creative/dev/office/OS/science icon/text splits | [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2), [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) |
| Keypoint detection | AP-10K, COCO keypoints | [AP-10K](https://github.com/AlexTheBad/AP-10K), [COCO](https://cocodataset.org/) |

### Depth Estimation

| Sub-task | Datasets | Public links |
| --- | --- | --- |
| Monocular depth estimation | NYU Depth V2, KITTI Depth, ETH3D, ScanNet, DIODE | [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [KITTI Depth](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php), [ETH3D](https://www.eth3d.net/), [ScanNet](https://www.scan-net.org/), [DIODE](https://diode-dataset.org/diode-dataset.github.io) |

### Surface Normal Estimation

| Sub-task | Datasets | Public links |
| --- | --- | --- |
| Surface normal estimation | NYU Depth V2 / NYU, ScanNet, iBims-1 | [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [ScanNet](https://www.scan-net.org/), [iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/) |

### Multi-View 3D
| Sub-task | Datasets | Public links |
| --- | --- | --- |
| Reconstruction | 7Scenes, ETH3D | [7Scenes], [ETH3D] |
| Camera pose Estimation | Re10K, Co3Dv2 | [RealEstate10K], [Co3Dv2] |

[7Scenes]: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
[ETH3D]: https://www.eth3d.net/datasets
[RealEstate10K]: https://google.github.io/realestate10k/
[Co3Dv2]: https://github.com/facebookresearch/co3d/
