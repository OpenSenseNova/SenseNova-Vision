# Reference: https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/README.md
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

# firstly, Download the Co3Dv2 dataset from [the official repository](https://github.com/facebookresearch/co3d)
# download it to `datas/co3dv2/data`
if [ ! -f datas/co3dv2/data/apple/198_21290_41362/images/frame000069.jpg ]; then
  >&2 echo "Error: Before running $0,"
  >&2 echo "  you should download the Co3Dv2 dataset from the official repository (https://github.com/facebookresearch/co3d) to"
  >&2 echo "  $PWD/datas/co3dv2/data."
  exit 1
fi

# generate annotations
mkdir -p datas/co3dv2/annotations
python preprocess_co3d.py --category all --co3d_v2_dir datas/co3dv2/data --output_dir datas/co3dv2/annotations
