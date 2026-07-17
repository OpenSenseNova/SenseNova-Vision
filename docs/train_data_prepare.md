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

The release does not duplicate original source images. Download those images
for each task below. The complete source list is maintained in the corpus
[dataset download references](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M/blob/main/Dataset_download_description.md).

## Segmentation

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
| TrashCan | [TrashCan download](https://github.com/dataset-ninja/trash-can/blob/main/DOWNLOAD.md) | `datas/train_data/TrashCan/dataset/instance_version/` |
| PIDRay | [PIDRay](https://github.com/lutao2021/PIDray) | `datas/train_data/PIDRay/` |
| ZeroWaste | [ZeroWaste](https://github.com/dbash/zerowaste) | `datas/train_data/ZeroWaste-f/` |
| LVIS | [LVIS](https://www.lvisdataset.org/dataset) | COCO images under `datas/train_data/coco2017/` |
| IDD-1/2 | [IDD](https://github.com/IshanKuchroo/IDD-Indian-Driving-Dataset) | `datas/train_data/IDD/IDD_Segmentation/` and `datas/train_data/IDD/idd20kII/` |
| IDDA v3 | [IDDA](https://idda-dataset.github.io/home/download/) | `datas/train_data/IDDA/IDDAv3/` |
| Mapillary Vistas | [Mapillary Vistas](https://huggingface.co/datasets/valentinamihalescu/mapillary-vistas-dataset/tree/main) | `datas/train_data/MapillaryVistas/` |
| nuImages | [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) | `datas/train_data/nuImages/samples/` |
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

### Referring, Reasoning, and Interactive Segmentation

The training configuration also includes seven datasets converted from their
original public annotations. Their generated JSONL files are separate from the
edited annotations released in SenseNova-Vision-Corpus-50M and should be placed
under `jsonl_generate/train_jsonls/segmentation/`.

| Dataset entry | JSONL file | Required media directories |
| --- | --- | --- |
| `refcoco_train` | `seg_refcoco_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcoco_train/` |
| `refcoc+_train` | `seg_refcoco+_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcoco+_train/` |
| `refcocog_train` | `seg_refcocog_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/refcocog_train/` |
| `refclef_train` | `seg_refclef_train_binary.jsonl` | `datas/ref_seg_data/images/saiapr_tc-12/`, `datas/ref_seg_data/ref_seg/binary_masks/refclef_train/` |
| `grefcoco_train` | `seg_grefcoco_train_binary.jsonl` | `datas/ref_seg_data/images/coco2014/train2014/`, `datas/ref_seg_data/ref_seg/binary_masks/grefcoco_train/` |
| `rea_train` | `seg_reason_train_repeat100.jsonl` | `datas/rea_seg_data/train/`, `datas/rea_seg_data/rea_seg/binary_masks/train/` |
| `coco_interactive_psalm` | `seg_coco_interactive_psalm.jsonl` | `datas/gen_seg_data/coco2017/train2017/`, `datas/inter_seg_data/inter_seg/binary_masks/coco_interactive_psalm/train/` |

These are the train splits of the same public datasets used by the segmentation
benchmarks. If you have already followed [`docs/data_prepare.md`](data_prepare.md),
reuse its COCO 2014 images, RefCOCO-family annotations, and COCO-Interactive
annotations. Check the extracted files before downloading anything again:

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

`[READY]` means the extracted benchmark data can be reused without touching its
archive. `[ARCHIVE READY]` means the target is missing but a valid ZIP is
available for extraction. For a missing COCO or
RefCOCO-family path, run the corresponding COCO 2014 or referring-segmentation
download commands in [`docs/data_prepare.md`](data_prepare.md). Do not create a
second copy under the training JSONL directory.

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

If an existing benchmark workspace already contains verified training JSONLs
at the top level of `jsonl_generate/`, copy them into the configured training
root without replacing files that are already there:

```bash
mkdir -p jsonl_generate/train_jsonls/segmentation

for name in \
  seg_reason_train_repeat100.jsonl \
  seg_coco_interactive_psalm.jsonl; do
  source_path="jsonl_generate/$name"
  target_path="jsonl_generate/train_jsonls/segmentation/$name"
  if [ -s "$source_path" ] && [ ! -e "$target_path" ]; then
    cp -p "$source_path" "$target_path"
  fi
done
```

Run the training converters after the source data is in place. Each command
creates both the derived masks and the training JSONL in one pass:

```bash
python tools/data_prepare/segmentation/prepare_binary.py refcoco
python tools/data_prepare/segmentation/prepare_binary.py reasonseg
python tools/data_prepare/segmentation/prepare_binary.py coco-interactive
```

The converters reuse the same mask decoding functions as the benchmark GT
preparation and process source records in a deterministic order with a fixed
random seed.

Benchmark/test JSONL files remain under `jsonl_generate/`. The commands above
write train JSONL files to `jsonl_generate/train_jsonls/segmentation/` by
default.

These JSONL records contain repository-relative paths spanning multiple
`datas/` subdirectories, so their dataset entries resolve media from the
repository root.

## Dense Geometric Prediction

### Dataset Downloads and Directory Layout

| Source dataset | Download | Target path |
| --- | --- | --- |
| COCO 2017 | [COCO downloads](https://cocodataset.org/#download) | `datas/train_data/coco2017/` |
| Objects365 | [Objects365](https://huggingface.co/datasets/jxu124/objects365) | `datas/train_data/object365/` |
| ScanNet++ | [scannetpp](https://github.com/scannetpp/scannetpp) | `datas/train_data/scannetpp/` |
| ScanNet v2 | [OpenDataLab](https://openxlab.org.cn/datasets/OpenDataLab/ScanNet_v2) | `datas/train_data/scannetv2/` |
| SA-1B | [Segment Anything](https://ai.meta.com/datasets/segment-anything/) | `datas/train_data/sa_1b/` |
| Taskonomy | [Taskonomy](http://taskonomy.stanford.edu/) | `datas/train_data/taskonomy/` |

COCO 2017 and Objects365 are already prepared under `datas/train_data/` by the
segmentation section.

Use the iPhone-captured ScanNet++ data. ScanNet v2 scenes must be extracted
from their `.sens` files following the
[ScanNet tools](https://github.com/ScanNet/ScanNet).

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

### Source Images for Released Reconstruction Annotations

| Source dataset | Download | Target path |
| --- | --- | --- |
| DL3DV | [DL3DV-10K](https://github.com/DL3DV-10K/Dataset#dataset-download) | `datas/train_data/DL3DV/` |
| WildRGB-D | [WildRGB-D](https://github.com/wildrgbd/wildrgbd) | `datas/train_data/wild_rgbd/` |

The released multi-view JSONL files reuse `datas/train_data/scannetpp/` and
`datas/train_data/scannetv2/` prepared above. Use the 960P DL3DV data.

## Structured Visual Understanding

### Dataset Downloads and Directory Layout

| Source dataset | Download | Target path |
| --- | --- | --- |
| APTv2 | [APTv2](https://huggingface.co/datasets/DenisKochetov/APTv2) | `datas/train_data/APTv2/` |
| BDD100K | [BDD100K](http://bdd-data.berkeley.edu/download.html) | `datas/train_data/BDD100K/` |
| DOTA v2 | [DOTA](https://captain-whu.github.io/DOTA/dataset.html) | `datas/train_data/DOTAv2/` |
| DeepFashion | [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) | `datas/train_data/DeepFashion/` |
| EgoObjects | [EgoObjects](https://ai.meta.com/datasets/egoobjects-downloads/) | `datas/train_data/EgoObjects/` |
| FAIR1M 2.0 | [FAIR1M](https://www.gaofen-challenge.com/benchmark) | `datas/train_data/FAIR1M/` |
| FSC147 | [Learning to Count Everything](https://github.com/cvlab-stonybrook/LearningToCountEverything) | `datas/train_data/FSC147/` |
| GroceryStore | [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset/tree/master/dataset) | `datas/train_data/GroceryStore/` |
| HumanParts | [Human-Parts](https://github.com/xiaojie1017/Human-Parts) | `datas/train_data/HumanParts/` |
| ImageNetPart | [PartImageNet](https://huggingface.co/datasets/turkeyju/PartImageNet) | `datas/train_data/ImageNetPart/` |
| nuImages | [nuImages](https://www.nuscenes.org/nuimages#download) | `datas/train_data/nuImages/` |
| PACO-LVIS | [PACO annotations](https://github.com/facebookresearch/paco) and [COCO images](https://cocodataset.org/#download) | `datas/train_data/PACO/` |
| PixMo-Points | [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | `datas/train_data/pixmo/` |
| V3Det-OVD | [V3Det](https://github.com/V3Det/V3Det) | `datas/train_data/V3Det___V3Det/raw/` |
| VisDrone | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | `datas/train_data/VisDrone/` |
| OpenImages V7 | [OpenImages](https://storage.googleapis.com/openimages/web/download_v7.html) | `datas/train_data/openimages/` |
| BLIP3-OCR | [BLIP3-OCR-200M](https://huggingface.co/datasets/Salesforce/blip3-ocr-200m) | `datas/train_data/OCR/blip3-ocr-200m/` |
| Blood Cell Detection | [Blood Cell Detection](https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu) | `datas/train_data/Blood Cell Detection/` |
| CrowdPose | [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) | `datas/train_data/keypoints/crowdpose/` |
| ICDAR 2013 Word OCR | [ICDAR 2013 Robust Reading](https://rrc.cvc.uab.es/?ch=2&com=downloads) | `datas/train_data/OCR/icdar2013/` |
| ICDAR 2015 Word OCR | [ICDAR 2015 Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4&com=downloads) | `datas/train_data/OCR/icdar2015/` |
| SKU110K + Gooreal | [YOLO-format release](https://huggingface.co/datasets/kukientinhky/SKU110k/tree/130022cae0cb4e559db0043d5789be340ff2df42) | `datas/train_data/SKU110k/` |
| SROIE 2019 | [SROIE 2019](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data) | `datas/train_data/sroie-datasetv2/` |
| AP-10K | [AP-10K](https://github.com/AlexTheBad/AP-10K) | `datas/train_data/keypoints/ap-10k/` |
| APT-36K | [APT-36K](https://github.com/pandorgan/APT-36K) | `datas/train_data/keypoints/APT36k/` |
| Human-Art | [Human-Art](https://idea-research.github.io/HumanArt/) | `datas/train_data/keypoints/Human-Art/` |
| MacaquePose | [MacaquePose](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) | `datas/train_data/keypoints/macaquepose_v1/` |
| MPII Human Pose | [MPII Human Pose](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download) | `datas/train_data/keypoints/mpii/` |
| OCHuman | [OCHuman](https://github.com/liruilong940607/OCHumanApi) | `datas/train_data/keypoints/ochuman/` |
| Aerial Sheep | [Aerial Sheep](https://universe.roboflow.com/riis/aerial-sheep) | `datas/train_data/aerial-sheep-object-detection/` |
| Football Player Detection | [Football Player Detection](https://universe.roboflow.com/augmented-startups/football-player-detection-kucab) | `datas/train_data/football-object-detection/` |
| Industrial Site Safety | [Industrial Site Safety](https://huggingface.co/datasets/Chappieut/Industrial-Site-Safety-Detection-v1-DATASET) | `datas/train_data/Industrial-Site-Safety-Detection-v1-DATASET/` |
| LVIS Fruits and Vegetables | [LVIS Fruits and Vegetables](https://huggingface.co/datasets/henningheyen/LVIS_Fruits_And_Vegetables) | `datas/train_data/LVIS_Fruits_And_Vegetables/` |
| Open World Dense Object Detection | [OWOD](https://huggingface.co/datasets/shubh303/open-world-dense-object-detection) | `datas/train_data/owdod/` |
| CDLA | [CDLA](https://github.com/buptlihang/CDLA) | `datas/train_data/Layout/CDLA_DATASET/` |
| DocLayNet core | [DocLayNet](https://github.com/DS4SD/DocLayNet) | `datas/train_data/Layout/DocLayNet_core/` |
| TableBank | [TableBank](https://github.com/doc-analysis/TableBank) | `datas/train_data/Layout/TableBank/` |
| TabRecSet | [TabRecSet](https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788) | `datas/train_data/Layout/TabRecSet/` |
| OS-Atlas | [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) | `datas/train_data/GUI/OS-Atlas-data/` |
| ShowUI desktop | [ShowUI-desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop) | `datas/train_data/GUI/ShowUI-desktop/` |

Reuse SA-1B, Objects365, and COCO 2017 from the dense geometric prediction
section, and nuImages from the segmentation section. Apply these
dataset-specific adjustments after downloading the source data.

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

- SROIE 2019:
  extract to `datas/train_data/sroie-datasetv2/versions/4/SROIE2019/train/`
  so both `box/` and `img/` are present.
- ICDAR 2013:
  place the training images in
  `datas/train_data/OCR/icdar2013/Challenge2_Training_Task12_Images/` and the
  word annotations in
  `datas/train_data/OCR/icdar2013/Challenge2_Training_Task1_GT/`.
- ICDAR 2015:
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
| BDD100K, DOTA v2, FAIR1M, nuImages, VisDrone | `BDD100K_pointing`, `DOTAv2_pointing`, `FAIR1M_pointing`, `NuImages_pointing`, `VisDrone_pointing` |
| PixMo-Points | `pixmo_detect` |
| GroceryStore | `GroceryStore_detect`, `GroceryStore_visual` |
| FSC147 | `FSC147_detect`, `FSC147_visual` |
| BLIP3-OCR | `blip3_ocr_200m_text_bbox`, `blip3_ocr_200m_text_poly_OCR`, `blip3_ocr_200m_word_bbox_OCR`, `blip3_ocr_200m_word_poly_OCR` |

#### Convert Raw Annotations Directly

These datasets are converted directly from the downloaded annotations without
an intermediate JSONL.

| Source dataset | Dataset entries | Workflow cases |
| --- | --- | --- |
| APTv2 | `APT_detect` | `aptv2` |
| Blood Cell Detection | `blood_cell_detect`, `blood_cell_visual` | `blood-cell-bbox`, `blood-cell-visual` |
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
| SKU110K + Gooreal | `SKU110k_detect`, `SKU110k_visual` | `sku110k-bbox`, `sku110k-visual` |
| EgoObjects | `EgoObjects_detect` | `egoobjects-bbox` |
| V3Det-OVD | `V3Det_ovd_detect` | `v3det-ovd-bbox` |
| Open World Dense Object Detection | `owdod_detect`, `owdod_visual` | `owdod-bbox`, `owdod-visual` |
| LVIS Fruits and Vegetables | `LVIS_Fruits_And_Vegetables_detect`, `LVIS_Fruits_And_Vegetables_visual` | `lvis-fruit-vegetable-bbox`, `lvis-fruit-vegetable-visual` |
| Aerial Sheep | `sheep_detect`, `sheep_visual` | `sheep-bbox`, `sheep-visual` |
| Football Player Detection | `football_detect`, `football_visual` | `football-bbox`, `football-visual` |
| Industrial Site Safety | `Industrial_Site_Safety_detect`, `Industrial_Site_Safety_visual` | `industrial-safety-bbox`, `industrial-safety-visual` |
| AP-10K | `ap-10k_keypoint` | `ap-10k-keypoint` |
| APT-36K | `APT36k_keypoint` | `apt36k-keypoint` |
| Human-Art | `Human-Art_keypoint` | `human-art-keypoint` |
| MacaquePose | `macaquepose_v1_keypoint` | `macaquepose-keypoint` |
| MPII Human Pose | `mpii_keypoint` | `mpii-keypoint` |
| OCHuman | `ochuman_keypoint` | `ochuman-keypoint` |
| SROIE 2019 | `SROIE_text_bbox_OCR` | `sroie` |
| ICDAR 2013 Word OCR | `icdar2013_word_bbox_OCR` | `icdar2013-word-bbox` |
| ICDAR 2015 Word OCR | `icdar2015_word_bbox_OCR`, `icdar2015_word_poly_OCR` | `icdar2015-word-bbox`, `icdar2015-word-poly` |
| CDLA | `CDLA_Layout` | `cdla-layout` |
| DocLayNet core | `DocLayNet_core_Layout` | `doclaynet-core-layout` |
| TableBank | `TableBank_Layout` | `tablebank-layout` |
| TabRecSet | `TabRecSet_Layout` | `tabrecset-layout` |
| OS-Atlas | `OS-Atlas-data_desktop_domain_GUI`, `OS-Atlas-data_mobile_domain_GUI`, `OS-Atlas-data_rico_GUI`, `OS-Atlas-data_web_domain_GUI` | `os-atlas-desktop-gui`, `os-atlas-mobile-gui`, `os-atlas-rico-gui`, `os-atlas-web-gui` |
| ShowUI desktop | `ShowUI-desktop_GUI` | `showui-desktop-gui` |

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
