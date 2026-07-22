# Training Data Preparation

SenseNova-Vision training combines original images from public datasets with
training JSONL annotations and derived assets released in
[SenseNova-Vision-Corpus-50M](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M).
This document describes the repository layout expected by
`data/dataset_info.py` and organizes data preparation by training task.

Run all commands from the repository root. Review and follow the license,
terms of use, and citation requirements of every original dataset.

## Expected Roots

The training configuration uses the following repository-local roots:

```text
jsonl_generate/train_jsonls/          # training JSONL annotations
datas/                                # original public data and converted assets
datas/train_data/                     # source images for released training annotations
datas/SenseNova-Vision-Corpus-50M/    # released derived assets
```

Large datasets may be stored outside the repository. Symbolic links are
recommended as long as these paths are present from the repository root.

## Download SenseNova-Vision-Corpus-50M

After preparing and activating the project environment with `setup.sh`,
download the dataset repository with its pinned `huggingface_hub` package:

```bash
CORPUS_RELEASE_DIR=/absolute/path/SenseNova-Vision-Corpus-50M-release
python -m huggingface_hub.commands.huggingface_cli download \
  sensenova/SenseNova-Vision-Corpus-50M \
  --repo-type dataset \
  --local-dir "$CORPUS_RELEASE_DIR"
```

Copy only the four training annotation directories into
`jsonl_generate/train_jsonls/`. This preserves any benchmark annotations
already prepared under `jsonl_generate/`:

```bash
mkdir -p jsonl_generate/train_jsonls
cp -r "$CORPUS_RELEASE_DIR/dense_geometric_prediction" \
  jsonl_generate/train_jsonls/
cp -r "$CORPUS_RELEASE_DIR/multiview_visual_geometry" \
  jsonl_generate/train_jsonls/
cp -r "$CORPUS_RELEASE_DIR/segmentation" \
  jsonl_generate/train_jsonls/
cp -r "$CORPUS_RELEASE_DIR/structure_view_understanding" \
  jsonl_generate/train_jsonls/
```

The corpus assets are distributed as ordered
`SenseNova-Vision-Corpus-50M.tar.gzNNN` shards. Join and extract them:

```bash
CORPUS_ASSET_PARENT=/absolute/path/sensenova-corpus-assets
mkdir -p "$CORPUS_ASSET_PARENT"

cat "$CORPUS_RELEASE_DIR"/SenseNova-Vision-Corpus-50M.tar.gz* | \
  tar -xz -C "$CORPUS_ASSET_PARENT"

mkdir -p datas
ln -s "$CORPUS_ASSET_PARENT/SenseNova-Vision-Corpus-50M" \
  datas/SenseNova-Vision-Corpus-50M
```

The resulting release layout is:

```text
jsonl_generate/train_jsonls/
├── dense_geometric_prediction/
├── multiview_visual_geometry/
├── segmentation/
└── structure_view_understanding/

datas/SenseNova-Vision-Corpus-50M/
├── coco2017/
├── object365/
├── sa_1b/
├── scannetpp/
├── DL3DV/
├── wild_rgbd/
├── Cityscapes/
├── coconut/
└── ...
```

The release does not duplicate original source images. Its source-image links
are listed in the corpus
[download references](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M/blob/main/Dataset_download_description.md).
The full SN-VC source inventory is defined by the report sections linked below.

## Segmentation

Corpus details: [Section 7.3, Tables 12–13](https://arxiv.org/html/2607.06560v1#S7.SS3).

### Source Images for Released Segmentation Annotations

The released segmentation JSONL files load original images from
`datas/train_data/` and `datas/gcg_seg_data/`, while their target assets are
loaded from `datas/SenseNova-Vision-Corpus-50M/`. The benchmark-compatible
COCO 2014 and COCO 2017 layouts remain under `datas/gen_seg_data/`.

| Source dataset | Download | Target path |
| --- | --- | --- |
| COCONut-XL | [COCONut](https://github.com/bytedance/coconut_cvpr2024) | COCO images under `datas/train_data/coco2017/` |
| Objects365 | [Objects365](https://huggingface.co/datasets/jxu124/objects365) | `datas/train_data/object365/` |
| Cityscapes | [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) | `datas/train_data/Cityscapes/` |
| Hypersim | [Hypersim](https://github.com/apple/ml-hypersim) | `datas/train_data/Hypersim/` |
| EntityV2 | [EntitySeg](https://github.com/adobe-research/EntitySeg-Dataset) | `datas/train_data/Entityv2/images/` |
| Trashcan | [TrashCan download](https://github.com/dataset-ninja/trash-can/blob/main/DOWNLOAD.md) | `datas/train_data/TrashCan/dataset/instance_version/` |
| Pidray | [PIDRay](https://github.com/lutao2021/PIDray) | `datas/train_data/PIDRay/` |
| ZeroWaste-f | [ZeroWaste](https://github.com/dbash/zerowaste) | `datas/train_data/ZeroWaste-f/` |
| LVIS | [LVIS](https://www.lvisdataset.org/dataset) | COCO images under `datas/train_data/coco2017/` |
| IDD-1/2 | [IDD](https://github.com/IshanKuchroo/IDD-Indian-Driving-Dataset) | `datas/train_data/IDD/IDD_Segmentation/` and `datas/train_data/IDD/idd20kII/` |
| IDDAv3 | [IDDA](https://idda-dataset.github.io/home/download/) | `datas/train_data/IDDA/IDDAv3/` |
| Mapillary Vistas | [Mapillary Vistas](https://huggingface.co/datasets/valentinamihalescu/mapillary-vistas-dataset/tree/main) | `datas/train_data/MapillaryVistas/` |
| NuScenes | [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) | `datas/train_data/nuImages/samples/` |
| 51WORLD | [DataOne sample](https://huggingface.co/datasets/51WORLD/DataOne-synthetic-v1.0-sample) | `datas/train_data/51WORLD/train/` |
| StreetHazards | [StreetHazards](https://github.com/hendrycks/anomaly-seg) | `datas/train_data/StreetHazards/train/images/` |
| KITTI | [KITTI](https://www.cvlibs.net/datasets/kitti/user_login.php) | `datas/train_data/KITTI/` |
| TAS500 | [TAS500 download](https://github.com/dataset-ninja/tas500/blob/main/DOWNLOAD.md) | `datas/train_data/TAS500/` |
| UDD5/6 | [UDD](https://github.com/MarcWong/UDD) | `datas/train_data/UDD/UDD5/` and `datas/train_data/UDD/UDD6/` |
| TTPLA | [TTPLA](https://github.com/r3ab/ttpla_dataset) | `datas/train_data/TTPLA/` |
| LoveDA | [LoveDA preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda) | `datas/train_data/LoveDA/` |
| VIPSeg | [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) | `datas/train_data/VIPSeg/imgs/` |
| GranDf | [GroundingLMM data](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/datasets.md) | `datas/gcg_seg_data/images/GranDf_HA_images/` |
| RefCOCOg | [REFER](https://github.com/lichengunc/refer) | `datas/gcg_seg_data/images/coco2014/` |
| PSG | [OpenPSG](https://github.com/Jingkang50/OpenPSG) | `datas/gcg_seg_data/images/coco2017/` |
| Flickr30k | [Flickr30k Entities](https://hockenmaier.cs.illinois.edu/DenotationGraph/) | `datas/gcg_seg_data/images/flickr30k/` |

The segmentation layout follows the public preparation conventions documented
by [X-SAM](https://github.com/wanghao9610/X-SAM/blob/main/docs/mds/datasets.md).
The benchmark preparation downloads COCO 2017 validation images but not the
training images required here. Download and extract `train2017` into the same
COCO directory:

```bash
mkdir -p datas/gen_seg_data/coco2017
COCO17_DIR=datas/gen_seg_data/coco2017

wget -c http://images.cocodataset.org/zips/train2017.zip \
  -O "$COCO17_DIR/train2017.zip"
unzip "$COCO17_DIR/train2017.zip" -d "$COCO17_DIR"
rm "$COCO17_DIR/train2017.zip"
unset COCO17_DIR
```

Only the original training images are needed; the derived training targets are
provided by SenseNova-Vision-Corpus-50M. Reuse common image collections with
symbolic links where necessary:

```bash
mkdir -p datas/train_data datas/gen_seg_data datas/gcg_seg_data/images

ln -s ../gen_seg_data/coco2017 datas/train_data/coco2017
ln -s ../../gen_seg_data/coco2014 datas/gcg_seg_data/images/coco2014
ln -s ../../gen_seg_data/coco2017 datas/gcg_seg_data/images/coco2017
```

### Binary Segmentation from Public Annotations

Prepare the public binary-mask datasets listed in
[Section 7.3, Table 12](https://arxiv.org/html/2607.06560v1#S7.T12) as follows.

#### Referring, Reasoning, and Interactive Segmentation

Generate the following JSONL files under
`jsonl_generate/train_jsonls/segmentation/`.

| Dataset entry | JSONL file | Required media directories |
| --- | --- | --- |
| `refcoco_train` | `seg_refcoco_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcoco_train/` |
| `refcoc+_train` | `seg_refcoco+_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcoco+_train/` |
| `refcocog_train` | `seg_refcocog_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcocog_train/` |
| `refclef_train` | `seg_refclef_train_binary.jsonl` | `datas/ref_seg_data/images/saiapr_tc-12/`, `datas/ref_seg_data/ref_seg/binary_masks/refclef_train/` |
| `grefcoco_train` | `seg_grefcoco_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/grefcoco_train/` |
| `rea_train` | `seg_reason_train_repeat100.jsonl` | `datas/rea_seg_data/train/`, `datas/rea_seg_data/rea_seg/binary_masks/train/` |
| `coco_interactive_psalm` | `seg_coco_interactive_psalm.jsonl` | `datas/gen_seg_data/coco2017/train2017/`, `datas/inter_seg_data/inter_seg/binary_masks/coco_interactive_psalm/train/` |

Reuse the COCO 2014 images, RefCOCO-family annotations, and COCO-Interactive
annotations prepared in [`docs/data_prepare.md`](data_prepare.md). Check them
before downloading:

```bash
bash tools/data_prepare/segmentation/check_reusable_data.sh \
  datas/gen_seg_data/coco2014/train2014 \
  datas/gen_seg_data/coco2014/train2014.zip \
  'datas/ref_seg_data/refcoco/refs(unc).p' \
  datas/ref_seg_data/refcoco.zip \
  'datas/ref_seg_data/refcoco+/refs(unc).p' \
  datas/ref_seg_data/refcoco+.zip \
  'datas/ref_seg_data/refcocog/refs(umd).p' \
  datas/ref_seg_data/refcocog.zip \
  datas/inter_seg_data/annotations/coco_interactive_train_psalm.json \
  datas/inter_seg_data/PSALM_data.zip
```

`[READY]` means the extracted data can be reused. `[ARCHIVE READY]` means a
valid ZIP is available for extraction. For missing COCO or RefCOCO-family data,
follow the corresponding instructions in
[`docs/data_prepare.md`](data_prepare.md).

Download the remaining train-only sources following the public
[X-SAM preparation](https://github.com/wanghao9610/X-SAM/blob/main/docs/mds/datasets.md).
RefCLEF requires both the REFER annotations and the ReferItGame image subset:

```bash
REF_DIR=datas/ref_seg_data
mkdir -p "$REF_DIR/images"

if [ ! -f "$REF_DIR/refclef/refs(unc).p" ]; then
  wget -c \
    https://web.archive.org/web/20220413011631/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip \
    -O "$REF_DIR/refclef.zip"
  unzip "$REF_DIR/refclef.zip" -d "$REF_DIR"
fi

if [ ! -d "$REF_DIR/images/saiapr_tc-12" ]; then
  wget -c \
    https://web.archive.org/web/20220413011744/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip \
    -O "$REF_DIR/saiapr_tc-12.zip"
  unzip "$REF_DIR/saiapr_tc-12.zip" -d "$REF_DIR/images"
fi
```

Download the official
[gRefCOCO annotations](https://huggingface.co/datasets/FudanCVL/gRefCOCO)
directly into the directory used by the converter:

```bash
if [ ! -f datas/ref_seg_data/grefcoco/instances.json ] || \
   [ ! -f 'datas/ref_seg_data/grefcoco/grefs(unc).json' ]; then
  python -m huggingface_hub.commands.huggingface_cli download \
    FudanCVL/gRefCOCO \
    --repo-type dataset \
    --local-dir datas/ref_seg_data/grefcoco
fi
```

Download `train.zip` and `train.json` from the
[LISA/ReasonSeg release](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy)
to `datas/rea_seg_data/`. Reuse `val.zip` and `test.zip` from benchmark
preparation; only the train split is added here:

```bash
REA_DIR=datas/rea_seg_data
mkdir -p "$REA_DIR/explanatory"

[ -d "$REA_DIR/train" ] || unzip "$REA_DIR/train.zip" -d "$REA_DIR"
[ -f "$REA_DIR/explanatory/train.json" ] || \
  mv "$REA_DIR/train.json" "$REA_DIR/explanatory/train.json"
```

The benchmark download of `PSALM_data.zip` already contains both train and val
annotations. If the archive exists but the train annotation has not been
extracted, recover only that JSON file:

```bash
INTER_DIR=datas/inter_seg_data
mkdir -p "$INTER_DIR/annotations"

if [ ! -f "$INTER_DIR/annotations/coco_interactive_train_psalm.json" ]; then
  unzip -j "$INTER_DIR/PSALM_data.zip" \
    '*/coco_interactive_train_psalm.json' \
    -d "$INTER_DIR/annotations"
fi
```

Create shared image links without duplicating COCO images:

```bash
mkdir -p datas/ref_seg_data/images datas/inter_seg_data
[ -e datas/ref_seg_data/images/coco2014 ] || \
  ln -s ../../gen_seg_data/coco2014 datas/ref_seg_data/images/coco2014
[ -e datas/inter_seg_data/coco2017 ] || \
  ln -s ../gen_seg_data/coco2017 datas/inter_seg_data/coco2017
```

The resulting layout should include:

```text
datas/
├── gen_seg_data/
│   ├── coco2014/train2014/
│   └── coco2017/train2017/
├── ref_seg_data/
│   ├── grefcoco/
│   ├── images/
│   │   ├── coco2014/
│   │   └── saiapr_tc-12/
│   ├── refclef/
│   ├── refcoco/
│   ├── refcoco+/
│   ├── refcocog/
│   └── ref_seg/binary_masks/
├── rea_seg_data/
│   ├── explanatory/
│   ├── train/
│   └── rea_seg/binary_masks/train/
└── inter_seg_data/
    ├── annotations/
    ├── coco2017/
    └── inter_seg/binary_masks/coco_interactive_psalm/train/
```

Run the converters after preparing the source data:

```bash
python tools/data_prepare/segmentation/prepare_binary.py refcoco
python tools/data_prepare/segmentation/prepare_binary.py reasonseg
python tools/data_prepare/segmentation/prepare_binary.py coco-interactive
```

The commands write training JSONL files to
`jsonl_generate/train_jsonls/segmentation/`. Benchmark and test JSONL files
remain under `jsonl_generate/`.

#### Long-Tail Binary Segmentation Sources

Download each dataset to the target directory listed below.

| Source dataset | Download | Target path |
| --- | --- | --- |
| DOORS | [Zenodo](https://zenodo.org/records/7107409) | `datas/ref_seg_data/DOORS/` |
| NDISPark | [Zenodo](https://zenodo.org/records/6560823) | `datas/ref_seg_data/NDISPark/` |
| MinneApple | [Project repository](https://github.com/nicolaihaeni/MinneApple) | `datas/ref_seg_data/MinneApple/` |
| EYTH | [EgoYouTubeHands project](https://aurooj.github.io/Hand-Segmentation-in-the-Wild/) | `datas/ref_seg_data/EYTH/` |
| PST900 | [Project repository](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) | `datas/ref_seg_data/PST900/` |
| PSTRGB | [Project repository](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) | `datas/ref_seg_data/PSTRGB/` |
| SUIM | [UMN IRVLab](https://irvlab.cs.umn.edu/image-segmentation/suim) | `datas/ref_seg_data/SUIM/` |
| MyFood | [AIcrowd Food Recognition Challenge](https://www.aicrowd.com/challenges/food-recognition-challenge) | `datas/ref_seg_data/MyFood/` |
| CO-SKEL | [Project repository](https://github.com/jkoteswarrao/Object-Co-skeletonization-with-Co-segmentation) | `datas/ref_seg_data/CO-SKEL/` |
| YouTube VOS 2022 (`VIS2022`) | [YouTube-VIS data page](https://youtube-vos.org/dataset/vis/) | `datas/ref_seg_data/VIS2022/` |
| MVTec D2S (`MVTecD2S`) | [MVTec dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-d2s) | `datas/ref_seg_data/MVTecD2S/` |
| VizWiz-FewShot | [VizWiz download page](https://vizwiz.org/tasks-and-datasets/object-localization/) | `datas/ref_seg_data/VizWiz-FewShot/` |
| Trans10K | [Trans10K-v1 project](https://xieenze.github.io/projects/TransLAB/TransLAB.html) | `datas/ref_seg_data/Trans10K/` |
| CIHP | [LIP Challenge](https://competitions.codalab.org/competitions/23431) | `datas/ref_seg_data/CIHP/` |
| ATR | [HumanParsing-Dataset repository](https://github.com/lemondan/HumanParsing-Dataset) | `datas/ref_seg_data/ATR/` |
| LIP | [SYSU-HCP dataset page](https://sysu-hcp.net/resources/datasets/index.html) | `datas/ref_seg_data/LIP/` |
| FAT-single / FAT-mixed | [NVIDIA Falling Things](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation) | `datas/ref_seg_data/FAT/` |
| Fashionpedia | [Official download page](https://fashionpedia.github.io/home/Fashionpedia_download.html) | `datas/ref_seg_data/Fashionpedia/` |
| PartImageNet / PartImageNet-Whole | [Project repository](https://github.com/TACJu/PartImageNet) | `datas/ref_seg_data/PartImageNet/` |
| WaterOVS | [Hugging Face dataset](https://huggingface.co/datasets/kkk2026/WaterOVS) | `datas/ref_seg_data/WaterOVS/` |
| RaidaR-rainy / RaidaR-sunny | [Official download page](https://raidar-dataset.com/download) | `datas/ref_seg_data/RaidaR/` |
| FSS-1000 | [Project repository](https://github.com/HKUSTCV/FSS-1000) | `datas/ref_seg_data/FSS-1000/` |
| DAVIS 2017 (`DAVIS`) | [Official download page](https://davischallenge.org/davis2017/code.html) | `datas/ref_seg_data/DAVIS/` |
| OCID-VLG (`OCID`) | [Project repository](https://github.com/gtziafas/OCID-VLG) | `datas/ref_seg_data/OCID-VLG/` |
| PIC | [IEEE Person in Context Challenge](https://signalprocessingsociety.org/publications-resources/data-challenges/person-context-challenge) | `datas/ref_seg_data/PIC/` |
| LaPa | [Official repository](https://github.com/jd-opensource/lapa-dataset) | `datas/ref_seg_data/LaPa/` |
| DeepFashion2 | [Official repository](https://github.com/switchablenorms/DeepFashion2) | `datas/ref_seg_data/DeepFashion2/` |
| MattingHumanHalf | [Matting Human Datasets](https://github.com/aisegmentcn/matting_human_datasets) | `datas/ref_seg_data/MattingHumanHalf/` |

Follow each source's access and license requirements.

Download DOORS v1.0 from [Zenodo](https://zenodo.org/records/7107409). The
archive is CC BY 4.0.

```bash
mkdir -p datas/ref_seg_data

if [ ! -f datas/ref_seg_data/DOORS.zip ]; then
  wget -c 'https://zenodo.org/records/7107409/files/DOORS.zip?download=1' \
    -O datas/ref_seg_data/DOORS.zip
fi

if [ ! -d datas/ref_seg_data/DOORS/Segmentation/DS1/DS ]; then
  unzip datas/ref_seg_data/DOORS.zip -d datas/ref_seg_data
fi

python tools/data_prepare/segmentation/prepare_binary.py doors
```

For YouTube-VIS 2022, accept the official
[terms of use](https://youtube-vos.org/dataset/term/), then download the 2022
training images and annotations from the
[YouTube-VIS data page](https://youtube-vos.org/dataset/vis/). The annotations
are CC BY 4.0; the dataset is limited to non-commercial research use.

Use this layout:

```text
datas/ref_seg_data/VIS2022/
└── train/
    ├── instances.json
    └── JPEGImages/
        └── <video_id>/
            └── <frame>.jpg
```

Run:

```bash
python tools/data_prepare/segmentation/prepare_binary.py vis2022 --num-workers 8
```

## Dense Geometric Prediction

Corpus details: [Section 7.2, Table 10](https://arxiv.org/html/2607.06560v1#S7.T10).

### Dataset Downloads and Directory Layout

Reuse Hypersim, COCO 2017, and Objects365 from
[Segmentation](#segmentation). Download the remaining sources.

| Source dataset | Download | Target path |
| --- | --- | --- |
| Hypersim | [Hypersim](https://github.com/apple/ml-hypersim) | `datas/train_data/Hypersim/` |
| Virtual KITTI | [Virtual KITTI 1.3.1](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) | `datas/train_data/vkitti_depth/` |
| InteriorVerse | [InteriorVerse](https://interiorverse.github.io/) | `datas/train_data/InteriorVerse_85/` |
| IRS | [IRS dataset](https://github.com/ccj5351/IRS-dataset) | `datas/train_data/IRS/` |
| TartanAir | [TartanAir](https://tartanair.org/) | `datas/train_data/tartanair/` |
| SceneNet RGB-D | [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html) | `datas/train_data/ScenenetRGBD/` |
| Taskonomy | [Taskonomy](http://taskonomy.stanford.edu/) | `datas/train_data/taskonomy/` |
| ScanNet++ | [ScanNet++](https://github.com/scannetpp/scannetpp) | `datas/train_data/scannetpp/` |
| COCO 2017 | [COCO downloads](https://cocodataset.org/#download) | `datas/train_data/coco2017/` |
| SA-1B | [Segment Anything](https://ai.meta.com/datasets/segment-anything/) | `datas/train_data/sa_1b/` |
| Objects365 | [Objects365](https://huggingface.co/datasets/jxu124/objects365) | `datas/train_data/object365/` |

Use the iPhone-captured ScanNet++ data.

### Prepare Training JSONLs

#### Use Released Training JSONLs

The corpus download steps above install the released depth and normal JSONLs
under `jsonl_generate/train_jsonls/dense_geometric_prediction/`. Download the
corresponding source images listed above; the generated targets are loaded from
`datas/SenseNova-Vision-Corpus-50M/`.

| Source dataset | Dataset entries |
| --- | --- |
| COCO 2017 | `coco_depth`, `coco_normal` |
| Objects365 | `object365_depth`, `object365_normal` |
| ScanNet++ | `scannetpp_depth`, `scannetpp_normal` |
| SA-1B | `sa_1b_depth`, `SA_1B_normal` |
| Taskonomy | `taskonomy_depth`, `taskonomy_normal` |

#### Convert Public Raw Data

For public raw-data conversion, see the
[Dense Geometric Prediction conversion guide](../tools/data_prepare/dense_geometric_prediction/README.md).

## Multi-View Visual Geometry

Corpus details: [Section 7.4, Table 16](https://arxiv.org/html/2607.06560v1#S7.T16).

### Dataset Downloads and Directory Layout

Reuse Hypersim, IRS, TartanAir, SceneNet RGB-D, and ScanNet++ from
[Dense Geometric Prediction](#dense-geometric-prediction). Download the
remaining sources.

| Source dataset | Download | Target path |
| --- | --- | --- |
| Hypersim | [Hypersim](https://github.com/apple/ml-hypersim) | `datas/train_data/Hypersim/` |
| IRS | [IRS dataset](https://github.com/ccj5351/IRS-dataset) | `datas/train_data/IRS/` |
| TartanAir | [TartanAir](https://tartanair.org/) | `datas/train_data/tartanair/` |
| SceneNet RGB-D | [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html) | `datas/train_data/ScenenetRGBD/` |
| AriaSyntheticENV | [Aria Synthetic Environments](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset) | `datas/train_data/AriaSyntheticEnvironment/` |
| BlendedMVG | [BlendedMVS and BlendedMVG](https://github.com/YoYo000/BlendedMVS) | `datas/train_data/BlendedMVG/` |
| MegaSynth | [MegaSynth](https://huggingface.co/datasets/hwjiang/MegaSynth) | `datas/train_data/MegaSynth/` |
| MvsSynth | [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html) | `datas/train_data/MVS-Synth/` |
| OmniObject3D | [OmniObject3D](https://omniobject3d.github.io/) | `datas/train_data/OmniObject3D/` |
| Objaverse | [Objaverse](https://objaverse.allenai.org/docs/intro/) | `datas/train_data/objaverse_v1/` |
| CO3Dv2 | [CO3Dv2](https://github.com/facebookresearch/co3d) | `datas/train_data/CO3Dv2/` |
| DeMoN-MVE | [DeMoN datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/DeMoN.en.html) | `datas/train_data/demon-mve/` |
| ScanNetV2 | [OpenDataLab](https://openxlab.org.cn/datasets/OpenDataLab/ScanNet_v2) | `datas/train_data/scannetv2/` |
| ScanNet++ | [ScanNet++](https://github.com/scannetpp/scannetpp) | `datas/train_data/scannetpp/` |
| DL3DV | [DL3DV-10K](https://github.com/DL3DV-10K/Dataset#dataset-download) | `datas/train_data/DL3DV/ALL-960P/` |
| WildRGB-D | [WildRGB-D](https://github.com/wildrgbd/wildrgbd) | `datas/train_data/wild_rgbd/` |

The released reconstruction JSONLs use DL3DV, ScanNet++, ScanNetV2, and
WildRGB-D. Use the 960P resolution version for DL3DV, and extract ScanNetV2
scenes from their `.sens` files with the [ScanNet tools](https://github.com/ScanNet/ScanNet).
Camera-pose JSONLs for the other sources can be generated with the
[multi-view conversion guide](../tools/data_prepare/multi_view_visual_geometry/README.md).

## Structured Visual Understanding

Corpus details: [Section 7.1, Tables 6–7](https://arxiv.org/html/2607.06560v1#S7.SS1).

### Dataset Downloads and Directory Layout

Reuse SA-1B, Objects365, and COCO 2017 from
[Dense Geometric Prediction](#dense-geometric-prediction), and reuse NuImages
and the RefCOCO family from [Segmentation](#segmentation). Download the
remaining sources.

| Source dataset | Download | Target path |
| --- | --- | --- |
| APTv2 | [APTv2](https://huggingface.co/datasets/DenisKochetov/APTv2) | `datas/train_data/APTv2/` |
| BDD100K | [BDD100K](http://bdd-data.berkeley.edu/download.html) | `datas/train_data/BDD100K/` |
| Blood Cell | [Roboflow](https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu) | `datas/train_data/Blood Cell Detection/` |
| CARPK | [CARPK project](https://lafi.github.io/LPN/) | `datas/train_data/CARPK/` |
| CrowdHuman | [Official download page](https://www.crowdhuman.org/download.html) | `datas/train_data/CrowdHuman/` |
| DOTAv2 | [DOTA](https://captain-whu.github.io/DOTA/dataset.html) | `datas/train_data/DOTAv2/` |
| DeepFashion | [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) | `datas/train_data/DeepFashion/` |
| EgoObjects | [EgoObjects](https://ai.meta.com/datasets/egoobjects-downloads/) | `datas/train_data/EgoObjects/` |
| FAIR1M | [FAIR1M 2.0](https://www.gaofen-challenge.com/benchmark) | `datas/train_data/FAIR1M/` |
| FSC147 | [Learning to Count Everything](https://github.com/cvlab-stonybrook/LearningToCountEverything) | `datas/train_data/FSC147/` |
| FiftyOne | [Hugging Face](https://huggingface.co/datasets/shubh303/dense_object_detection_FiftyOne) | `datas/train_data/dense_object_detection_FiftyOne/` |
| Fish | [Roboflow](https://public.roboflow.com/object-detection/fish) | `datas/train_data/fish-detection-dataset/` |
| Football | [Roboflow](https://universe.roboflow.com/augmented-startups/football-player-detection-kucab) | `datas/train_data/football-object-detection/` |
| GroceryStore | [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset/tree/master/dataset) | `datas/train_data/GroceryStore/` |
| HomeObjects-3k | [Ultralytics](https://platform.ultralytics.com/jayce-gaddis/datasets/homeobjects-3k) | `datas/train_data/homeobjects-3K/` |
| HumanParts | [Human-Parts](https://github.com/xiaojie1017/Human-Parts) | `datas/train_data/HumanParts/` |
| ImageNetPart | [PartImageNet](https://huggingface.co/datasets/turkeyju/PartImageNet) | `datas/train_data/ImageNetPart/` |
| Industrial Site Safety | [Hugging Face](https://huggingface.co/datasets/Chappieut/Industrial-Site-Safety-Detection-v1-DATASET) | `datas/train_data/Industrial-Site-Safety-Detection-v1-DATASET/` |
| LVIS Fruits & Vegetables | [Hugging Face](https://huggingface.co/datasets/henningheyen/LVIS_Fruits_And_Vegetables) | `datas/train_data/LVIS_Fruits_And_Vegetables/` |
| Locount | [Dataset repository](https://github.com/SakiRinn/mmdetection-locount) | `datas/train_data/Locount/` |
| METU-ALET | [Dataset repository](https://github.com/metu-kovan/METU-ALET) | `datas/train_data/METU-ALET/` |
| NuImages | [nuImages](https://www.nuscenes.org/nuimages#download) | `datas/train_data/nuImages/` |
| OWOD | [Open World Dense Object Detection](https://huggingface.co/datasets/shubh303/open-world-dense-object-detection) | `datas/train_data/owdod/` |
| Objects365 | [Objects365](https://huggingface.co/datasets/jxu124/objects365) | `datas/train_data/object365/` |
| PACO-LVIS | [PACO annotations](https://github.com/facebookresearch/paco) and [COCO images](https://cocodataset.org/#download) | `datas/train_data/PACO/` |
| PixMo-Points | [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | `datas/train_data/pixmo/` |
| S2TLD | [Dataset repository](https://github.com/Thinklab-SJTU/S2TLD) | `datas/train_data/S2TLD/` |
| SA-1B | [Segment Anything](https://ai.meta.com/datasets/segment-anything/) | `datas/train_data/sa_1b/` |
| SKU110K | [Official repository](https://github.com/eg4000/SKU110K_CVPR19) | `datas/train_data/SKU110k/` |
| Shoes | [Kaggle](https://www.kaggle.com/datasets/nishthakukreti/shoedataset) | `datas/train_data/Shoes_data/` |
| TinyPerson | [TinyBenchmark](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark/dataset) | `datas/train_data/TinyPerson/` |
| V3Det-OVD | [V3Det](https://github.com/V3Det/V3Det) | `datas/train_data/V3Det___V3Det/raw/` |
| VisDrone | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | `datas/train_data/VisDrone/` |
| WiderPerson | [Dataset index](https://fmi-data-index.github.io/wider_person.html) | `datas/train_data/WiderPerson/` |
| Pill | [Medical Pills](https://huggingface.co/datasets/Ultralytics/Medical-pills) | `datas/train_data/Medical-pills/` |
| Sheep | [Aerial Sheep](https://universe.roboflow.com/riis/aerial-sheep) | `datas/train_data/aerial-sheep-object-detection/` |
| HumanRef | [Hugging Face](https://huggingface.co/datasets/IDEA-Research/HumanRef) | `datas/train_data/humanref_cot_45k_converted/` |
| OpenImages | [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | `datas/train_data/openimages/` |
| RefCOCO/+/g | [REFER](https://github.com/lichengunc/refer) and [COCO 2014](https://cocodataset.org/#download) | `datas/ref_seg_data/` |
| RexVerse | [RexVerse-2M](https://huggingface.co/datasets/IDEA-Research/Rexverse-2M) | `datas/train_data/RexVerse-2M/` |
| BLIP3-OCR-200M | [Hugging Face](https://huggingface.co/datasets/Salesforce/blip3-ocr-200m) | `datas/train_data/OCR/blip3-ocr-200m/` |
| HierText | [Dataset repository](https://github.com/google-research-datasets/hiertext) | `datas/train_data/OCR/Hiertext/` |
| ICDAR2013 | [ICDAR 2013 Robust Reading](https://rrc.cvc.uab.es/?ch=2&com=downloads) | `datas/train_data/OCR/icdar2013/` |
| ICDAR2015 | [ICDAR 2015 Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4&com=downloads) | `datas/train_data/OCR/icdar2015/` |
| ICDAR2019 | [ICDAR 2019 ArT](https://rrc.cvc.uab.es/?ch=14&com=downloads) | `datas/train_data/OCR/icdar2019/` |
| LSVT2019 | [ICDAR 2019 LSVT](https://rrc.cvc.uab.es/?ch=16&com=downloads) | `datas/train_data/OCR/LSVT2019/` |
| MTWI | [Tianchi](https://tianchi.aliyun.com/dataset/137084) | `datas/train_data/OCR/mtwi/` |
| RCTW | [RCTW-17](https://rctw.vlrlab.net/) | `datas/train_data/OCR/RCTW/` |
| ReCTS | [ICDAR 2019 ReCTS](https://rrc.cvc.uab.es/?ch=12&com=downloads) | `datas/train_data/OCR/ReCTS/` |
| SROIE | [SROIE 2019](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data) | `datas/train_data/sroie-datasetv2/` |
| SynthText | [SynthText in the Wild](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c) | `datas/train_data/OCR/SynthText/` |
| TextOCR | [Official dataset page](https://textvqa.org/textocr/dataset/) | `datas/train_data/OCR/TextOCR/` |
| WildReceipt | [OpenMMLab archive](https://download.openmmlab.com/mmocr/data/wildreceipt.tar) | `datas/train_data/OCR/wildreceipt/` |
| AP-10K | [AP-10K](https://github.com/AlexTheBad/AP-10K) | `datas/train_data/keypoints/ap-10k/` |
| APT36K | [APT-36K](https://github.com/pandorgan/APT-36K) | `datas/train_data/keypoints/APT36k/` |
| COCO2017 | [COCO](https://cocodataset.org/#download) | `datas/train_data/coco2017/` |
| CrowdPose | [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) | `datas/train_data/keypoints/crowdpose/` |
| Human-Art | [Human-Art](https://idea-research.github.io/HumanArt/) | `datas/train_data/keypoints/Human-Art/` |
| MPII | [MPII Human Pose](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download) | `datas/train_data/keypoints/mpii/` |
| MacaquePose V1 | [MacaquePose](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) | `datas/train_data/keypoints/macaquepose_v1/` |
| OCHuman | [OCHuman](https://github.com/liruilong940607/OCHumanApi) | `datas/train_data/keypoints/ochuman/` |
| CDLA | [CDLA](https://github.com/buptlihang/CDLA) | `datas/train_data/Layout/CDLA_DATASET/` |
| DocLayNet Core | [DocLayNet](https://github.com/DS4SD/DocLayNet) | `datas/train_data/Layout/DocLayNet_core/` |
| PubLayNet | [Official repository](https://github.com/ibm-aur-nlp/PubLayNet) | `datas/train_data/Layout/publaynet/` |
| TabRecSet | [TabRecSet](https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788) | `datas/train_data/Layout/TabRecSet/` |
| TableBank | [TableBank](https://github.com/doc-analysis/TableBank) | `datas/train_data/Layout/TableBank/` |
| OS-Atlas | [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) | `datas/train_data/GUI/OS-Atlas-data/` |
| ShowUI Desktop | [ShowUI-desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop) | `datas/train_data/GUI/ShowUI-desktop/` |

Apply these dataset-specific adjustments after downloading the source data.

Objects365: keep the annotation Parquet shards under
`datas/train_data/object365/data/` and the training images under
`datas/train_data/object365/patch*/`.

PACO-LVIS: place `paco_lvis_v1_train.json` in `datas/train_data/PACO/`. From
that directory, link the COCO 2017 images:

```bash
ln -s ../coco2017/train2017 train2017
```

EgoObjects: use `EgoObjectsV1_images.zip` and
`EgoObjectsV1_unified_train.json`.

APTv2: extract both archives so that `datas/train_data/APTv2/` contains
`annotations/train_annotations.json`, `data/easy/`, and `data/hard/`.

OCR datasets: keep the released folder names from the original downloads.

- SROIE:
  extract to `datas/train_data/sroie-datasetv2/versions/4/SROIE2019/train/`
  so both `box/` and `img/` are present.
- ICDAR2013:
  place the training images in
  `datas/train_data/OCR/icdar2013/Challenge2_Training_Task12_Images/` and the
  word annotations in
  `datas/train_data/OCR/icdar2013/Challenge2_Training_Task1_GT/`.
- ICDAR2015:
  place the training images in
  `datas/train_data/OCR/icdar2015/ch4_training_images/` and the word
  annotations in
  `datas/train_data/OCR/icdar2015/ch4_training_localization_transcription_gt/`.

### Prepare Training JSONLs

Generated JSONLs are written to
`jsonl_generate/train_jsonls/structure_view_understanding/`.

#### Use Released Training JSONLs

Copy the published
[`structure_view_understanding`](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M/tree/main/structure_view_understanding)
JSONLs to `jsonl_generate/train_jsonls/structure_view_understanding/`. Download
the source images listed above; do not run a converter for these entries.

| Source dataset | Dataset entries |
| --- | --- |
| SA-1B | `grounding_SA1B`, `SA1B_pointing`, `SA_1B_visual` |
| Objects365 | `Objects365_pointing`, `object365_refbbox_merge`, `object365_refpoint_merge` |
| OpenImages | `openimages_refbbox_merge`, `openimages_refpoint_merge` |
| APTv2 | `APT_pointing` |
| DeepFashion | `DeepFashion_pointing` |
| EgoObjects | `EgoObjects_pointing` |
| HumanParts | `HumanParts_pointing` |
| ImageNetPart | `ImageNetPart_pointing` |
| PACO-LVIS | `PACO_LVIS_pointing` |
| V3Det-OVD | `V3Det_ovd_pointing` |
| BDD100K, DOTAv2, FAIR1M, NuImages, VisDrone | `BDD100K_pointing`, `DOTAv2_pointing`, `FAIR1M_pointing`, `NuImages_pointing`, `VisDrone_pointing` |
| PixMo-Points | `pixmo_detect` |
| GroceryStore | `GroceryStore_detect`, `GroceryStore_visual` |
| FSC147 | `FSC147_detect`, `FSC147_visual` |
| BLIP3-OCR-200M | `blip3_ocr_200m_text_bbox`, `blip3_ocr_200m_text_poly_OCR`, `blip3_ocr_200m_word_bbox_OCR`, `blip3_ocr_200m_word_poly_OCR` |

#### Convert Raw Annotations Directly

These datasets are converted directly from the downloaded annotations without
an intermediate JSONL.

| Source dataset | Dataset entries | Workflow cases |
| --- | --- | --- |
| APTv2 | `APT_detect` | `aptv2` |
| Blood Cell | `blood_cell_detect`, `blood_cell_visual` | `blood-cell-bbox`, `blood-cell-visual` |
| FSC147 | `FSC147_pointing` | `fsc147` |
| COCO 2017 | `coco2017_keypoint` | `coco2017-keypoint` |
| CrowdPose | `crowdpose_keypoint` | `crowdpose-keypoint` |

The commands below cover every workflow case in this table. Run only the cases
for the datasets you prepared.

```bash
bash tools/data_prepare/structured_visual_understanding/prepare.sh aptv2
bash tools/data_prepare/structured_visual_understanding/prepare.sh blood-cell-bbox blood-cell-visual
bash tools/data_prepare/structured_visual_understanding/prepare.sh fsc147
bash tools/data_prepare/structured_visual_understanding/prepare.sh coco2017-keypoint
bash tools/data_prepare/structured_visual_understanding/prepare.sh crowdpose-keypoint
```

#### Convert Through a Unified Intermediate JSONL

The workflow first converts the raw annotations to a common bbox, keypoint, or
OCR JSONL, then creates the final training JSONL.

| Source dataset | Dataset entries | Workflow cases |
| --- | --- | --- |
| Objects365 | `Objects365_detect`, `Objects365_visual` | `objects365-bbox`, `objects365-visual` |
| PACO-LVIS | `PACO_LVIS_detect` | `paco-lvis-bbox` |
| SKU110K | `SKU110k_detect`, `SKU110k_visual` | `sku110k-bbox`, `sku110k-visual` |
| EgoObjects | `EgoObjects_detect` | `egoobjects-bbox` |
| V3Det-OVD | `V3Det_ovd_detect` | `v3det-ovd-bbox` |
| OWOD | `owdod_detect`, `owdod_visual` | `owdod-bbox`, `owdod-visual` |
| LVIS Fruits & Vegetables | `LVIS_Fruits_And_Vegetables_detect`, `LVIS_Fruits_And_Vegetables_visual` | `lvis-fruit-vegetable-bbox`, `lvis-fruit-vegetable-visual` |
| Sheep | `sheep_detect`, `sheep_visual` | `sheep-bbox`, `sheep-visual` |
| Football | `football_detect`, `football_visual` | `football-bbox`, `football-visual` |
| Industrial Site Safety | `Industrial_Site_Safety_detect`, `Industrial_Site_Safety_visual` | `industrial-safety-bbox`, `industrial-safety-visual` |
| AP-10K | `ap-10k_keypoint` | `ap-10k-keypoint` |
| APT36K | `APT36k_keypoint` | `apt36k-keypoint` |
| Human-Art | `Human-Art_keypoint` | `human-art-keypoint` |
| MacaquePose V1 | `macaquepose_v1_keypoint` | `macaquepose-keypoint` |
| MPII | `mpii_keypoint` | `mpii-keypoint` |
| OCHuman | `ochuman_keypoint` | `ochuman-keypoint` |
| SROIE | `SROIE_text_bbox_OCR` | `sroie` |
| ICDAR2013 | `icdar2013_word_bbox_OCR` | `icdar2013-word-bbox` |
| ICDAR2015 | `icdar2015_word_bbox_OCR`, `icdar2015_word_poly_OCR` | `icdar2015-word-bbox`, `icdar2015-word-poly` |
| CDLA | `CDLA_Layout` | `cdla-layout` |
| DocLayNet Core | `DocLayNet_core_Layout` | `doclaynet-core-layout` |
| TableBank | `TableBank_Layout` | `tablebank-layout` |
| TabRecSet | `TabRecSet_Layout` | `tabrecset-layout` |
| OS-Atlas | `OS-Atlas-data_desktop_domain_GUI`, `OS-Atlas-data_mobile_domain_GUI`, `OS-Atlas-data_rico_GUI`, `OS-Atlas-data_web_domain_GUI` | `os-atlas-desktop-gui`, `os-atlas-mobile-gui`, `os-atlas-rico-gui`, `os-atlas-web-gui` |
| ShowUI Desktop | `ShowUI-desktop_GUI` | `showui-desktop-gui` |

The commands below are representative bbox/visual, keypoint, layout, and GUI
examples. For other datasets, use the workflow case listed in the table.

```bash
bash tools/data_prepare/structured_visual_understanding/prepare.sh objects365-bbox objects365-visual
bash tools/data_prepare/structured_visual_understanding/prepare.sh owdod-bbox owdod-visual
bash tools/data_prepare/structured_visual_understanding/prepare.sh ap-10k-keypoint
bash tools/data_prepare/structured_visual_understanding/prepare.sh sroie icdar2013-word-bbox icdar2015-word-bbox icdar2015-word-poly
bash tools/data_prepare/structured_visual_understanding/prepare.sh cdla-layout
bash tools/data_prepare/structured_visual_understanding/prepare.sh os-atlas-rico-gui
```

## General Understanding, Generation, and Editing

These auxiliary multimodal sources belong to the training mixture described in
[Section 4](https://arxiv.org/html/2607.06560v1#S4); they are not part of the
SN-VC source tables in Appendix Section 7.

### Public Sources

| Task | Dataset entries | Official source | Local source path | Preparation |
| --- | --- | --- | --- | --- |
| Understanding | `llava_v1_5` | [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [LLaVA data guide](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning) | `datas/train_data/llava_images` | Supported converter |
| Understanding | `finevision_image`, `finevision_multi_image`, `finevision_text` | [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | `datas/train_data/finevision_source` | Official source only |
| Understanding | `mammoth_image`, `mammoth_text` | [MAmmoTH-VL-Instruct-12M](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M) | `datas/train_data/mammoth_vl_source` | Official source only |
| Generation | `BLIP3o-Pretrain-Long-Caption`, `BLIP3o-Pretrain-Short-Caption`, `BLIP3o-Long-part2` | [BLIP3o datasets](https://huggingface.co/BLIP3o/datasets) | `datas/train_data/blip3o` | Official source only |
| Generation/editing | ShareGPT-4o text-to-image, `ShareGPT_4o_edit` | [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) | `datas/train_data/sharegpt_4o` | Editing converter only |
| Editing | `Nano-consistent-150k` | [Nano-consistent-150k](https://huggingface.co/datasets/Yejy53/Nano-consistent-150k) | `datas/train_data/nano_consistent_150k` | Official source only |
| Editing | `multi_edit` | [MultiEdit](https://huggingface.co/datasets/inclusionAI/MultiEdit) | `datas/train_data/multiedit` | Official source only; gated |
| Editing | `GPT_Image_Edit_OmniEdit`, `GPT_Image_Edit_HQEdit`, `GPT_Image_Edit_UltraEdit` | [GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M), [training JSON](https://huggingface.co/UCSC-VLAA/gpt-image-edit-training/tree/main/training_json) | `datas/train_data/gpt_image_edit/gpt-edit` | Supported converter |

### Download and Layout

Prepare LLaVA annotations and images. Reuse COCO 2017 from the segmentation
preparation with a relative symbolic link:

```bash
LLAVA_ANNOTATION_DIR=/absolute/path/LLaVA-Instruct-150K
LLAVA_IMAGES_DIR=datas/train_data/llava_images

mkdir -p \
  "$LLAVA_ANNOTATION_DIR" \
  "$LLAVA_IMAGES_DIR/coco" \
  "$LLAVA_IMAGES_DIR/gqa" \
  "$LLAVA_IMAGES_DIR/ocr_vqa/images" \
  "$LLAVA_IMAGES_DIR/textvqa" \
  "$LLAVA_IMAGES_DIR/vg"

wget -c \
  https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json \
  -O "$LLAVA_ANNOTATION_DIR/llava_v1_5_mix665k.json"

ln -s ../../coco2017/train2017 "$LLAVA_IMAGES_DIR/coco/train2017"

wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip \
  -O "$LLAVA_IMAGES_DIR/gqa/images.zip"
unzip "$LLAVA_IMAGES_DIR/gqa/images.zip" -d "$LLAVA_IMAGES_DIR/gqa"

wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip \
  -O "$LLAVA_IMAGES_DIR/textvqa/train_val_images.zip"
unzip "$LLAVA_IMAGES_DIR/textvqa/train_val_images.zip" \
  -d "$LLAVA_IMAGES_DIR/textvqa"

wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip \
  -O "$LLAVA_IMAGES_DIR/vg/images.zip"
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip \
  -O "$LLAVA_IMAGES_DIR/vg/images2.zip"
unzip "$LLAVA_IMAGES_DIR/vg/images.zip" -d "$LLAVA_IMAGES_DIR/vg"
unzip "$LLAVA_IMAGES_DIR/vg/images2.zip" -d "$LLAVA_IMAGES_DIR/vg"

wget -c \
  'https://drive.usercontent.google.com/download?id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q&export=download&confirm=t' \
  -O "$LLAVA_IMAGES_DIR/ocr_vqa/dataset.json"
python tools/data_prepare/general_understanding/download_ocr_vqa.py
```

Download the other public releases with the project-pinned Hugging Face CLI:

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  HuggingFaceM4/FineVision --repo-type dataset \
  --local-dir datas/train_data/finevision_source

python -m huggingface_hub.commands.huggingface_cli download \
  MAmmoTH-VL/MAmmoTH-VL-Instruct-12M --repo-type dataset \
  --local-dir datas/train_data/mammoth_vl_source

python -m huggingface_hub.commands.huggingface_cli download \
  BLIP3o/BLIP3o-Pretrain-Long-Caption --repo-type dataset \
  --local-dir datas/train_data/blip3o/long_caption
python -m huggingface_hub.commands.huggingface_cli download \
  BLIP3o/BLIP3o-Pretrain-Short-Caption --repo-type dataset \
  --local-dir datas/train_data/blip3o/short_caption

python -m huggingface_hub.commands.huggingface_cli download \
  Yejy53/Nano-consistent-150k --repo-type dataset \
  --local-dir datas/train_data/nano_consistent_150k

# MultiEdit requires Hugging Face login and acceptance of its access terms.
python -m huggingface_hub.commands.huggingface_cli download \
  inclusionAI/MultiEdit --repo-type dataset \
  --local-dir datas/train_data/multiedit

python -m huggingface_hub.commands.huggingface_cli download \
  FreedomIntelligence/ShareGPT-4o-Image --repo-type dataset \
  --include '*.json' '*.tar' \
  --local-dir datas/train_data/sharegpt_4o

for archive in datas/train_data/sharegpt_4o/*.tar; do
  tar -xf "$archive" -C datas/train_data/sharegpt_4o
done
```

GPT-Image-Edit provides a parallel downloader for its 4.53 TB image release.
GNU `parallel` is required:

```bash
GPT_EDIT_DIR=datas/train_data/gpt_image_edit
GPT_EDIT_IMAGES="$GPT_EDIT_DIR/gpt-edit"
GPT_EDIT_ANNOTATIONS="$GPT_EDIT_DIR/annotations"

python -m huggingface_hub.commands.huggingface_cli download \
  UCSC-VLAA/GPT-Image-Edit-1.5M --repo-type dataset \
  --include download.sh --local-dir "$GPT_EDIT_DIR/release"

for family in hqedit omniedit ultraedit; do
  bash "$GPT_EDIT_DIR/release/download.sh" \
    -d "$family" -o "$GPT_EDIT_IMAGES" -p 8
  cat "$GPT_EDIT_IMAGES/$family/$family.tar.gz.part"* | \
    tar -xz -C "$GPT_EDIT_IMAGES/$family"
done

python -m huggingface_hub.commands.huggingface_cli download \
  UCSC-VLAA/gpt-image-edit-training \
  --include 'training_json/*.json' \
  --local-dir "$GPT_EDIT_ANNOTATIONS"
```

### Conversion and Configuration

Create the task JSONL directories and run the three deterministic converters:

```bash
mkdir -p \
  jsonl_generate/train_jsonls/understanding \
  jsonl_generate/train_jsonls/editing

python tools/data_prepare/general_understanding/prepare_llava_v1_5.py \
  --input-json "$LLAVA_ANNOTATION_DIR/llava_v1_5_mix665k.json"

python tools/data_prepare/general_editing/prepare_sharegpt_4o.py

python tools/data_prepare/general_editing/prepare_gpt_image_edit.py
```

Each converter checks its expected count and decodes every image in the first
case for each source component. There is no separate validation subcommand.
