import argparse
import json
import os
import tarfile
import tempfile

from PIL import Image
from io import BytesIO

from data.prompts import (
    DEPTH_TESTSET_PROMPT_TEMPLATES,
    NORMAL_TESTSET_PROMPT_TEMPLATES,
    ensure_image_placeholders,
)
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from utils import ensure_dir
from utils.timing import timer_context

DEPTH_DATASETS = {
    'nyudepth_v2':{
        'image_path':'evaluation_depth_dataset/nyuv2/nyu_labeled_extracted.tar',
        'image_text':'tools/evaluation/geometry/evaluation_depth_dataset_text/nyu_depth/filename_list_test.txt'
    },
    'kitti':{
        'image_path':'evaluation_depth_dataset/kitti/kitti_eigen_split_test.tar',
        'image_text':'tools/evaluation/geometry/evaluation_depth_dataset_text/kitti_depth/eigen_test_files_with_gt.txt'
    },
    'eth3d':{
        'image_path':'evaluation_depth_dataset/eth3d/eth3d.tar',
        'image_text':'tools/evaluation/geometry/evaluation_depth_dataset_text/eth3d_depth/eth3d_filename_list.txt'
    },
    'scannet':{
        'image_path':'evaluation_depth_dataset/scannet/scannet_val_sampled_800_1.tar',
        'image_text':'tools/evaluation/geometry/evaluation_depth_dataset_text/scannet_depth/scannet_val_sampled_list_800_1.txt'
    },
    'diode':{
        'image_path':'evaluation_depth_dataset/diode/diode_val.tar',
        'image_text':'tools/evaluation/geometry/evaluation_depth_dataset_text/diode_depth/diode_val_all_filename_list.txt'
    },
}


NORMAL_DATASETS = {
    "scannet": {
        "image_path": "evaluation_normal_dataset/scannet",
        "image_text": "tools/evaluation/geometry/evaluation_normal_dataset_text/scannet_normals/scannet_test.txt",
    },
    "ibims": {
        "image_path": "evaluation_normal_dataset/ibims/ibims",
        "image_text": "tools/evaluation/geometry/evaluation_normal_dataset_text/ibims_normals/ibims_test.txt",
    },
    "nyu": {
        "image_path": "evaluation_normal_dataset/nyuv2/test",
        "image_text": "tools/evaluation/geometry/evaluation_normal_dataset_text/nyu_normals/nyuv2_test.txt",
    },
}


def normalize_tar_member_name(path: str) -> str:
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def open_image_from_tar(tar: tarfile.TarFile, relative_path: str) -> Image.Image:
    relative_path = normalize_tar_member_name(relative_path)
    candidates = [relative_path, f"./{relative_path}"]
    member = None
    for candidate in candidates:
        try:
            member = tar.getmember(candidate)
            break
        except KeyError:
            continue
    if member is None:
        raise FileNotFoundError(f"Image not found in tar: {relative_path}")
    file_obj = tar.extractfile(member)
    if file_obj is None:
        raise FileNotFoundError(f"Cannot extract image from tar: {relative_path}")
    return Image.open(BytesIO(file_obj.read())).convert("RGB")


def read_first_column_list(text_path: str) -> list[str]:
    with open(text_path, "r", encoding="utf-8") as f:
        return [
            line.strip().split()[0]
            for line in f
            if line.strip()
        ]


def build_output_dir(args, test_dataset: str) -> str:
    is_depth = "Depth" in args.test_mode
    base_folder = "depth" if is_depth else "normal"
    suffix = "depth" if is_depth else "normal"

    if args.output_dir:
        leaf = f"{test_dataset}_{suffix}"
        return os.path.abspath(os.path.join(args.output_dir, base_folder, leaf))

    model_name = os.path.basename(os.path.normpath(args.model_path)) or "model"
    leaf = f"{model_name}_ema.{test_dataset}_{suffix}"
    return os.path.abspath(os.path.join(".", base_folder, leaf))


def select_datasets(test_mode: str, dataset: str | None) -> dict:
    dataset_map = DEPTH_DATASETS if test_mode == "Depth" else NORMAL_DATASETS
    if dataset is None:
        return dataset_map
    if dataset not in dataset_map:
        supported = ", ".join(sorted(dataset_map))
        raise ValueError(
            f"Unsupported {test_mode} dataset: {dataset}. Supported: {supported}"
        )
    return {dataset: dataset_map[dataset]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-Vision dense geometry benchmark inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sensenova/SenseNova-Vision-7B-MoT",
        help="Path to SenseNova-Vision model directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for model")
    parser.add_argument(
        "--override_json",
        type=str,
        help="JSON string or path to JSON file to override hyper‑params",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. If omitted, run all datasets for --test_mode.",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=["Depth", "Normal"],
        default="Depth",
        help="Dense geometry task.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datas/geometry_data",
        help="Root directory for depth/normal datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional. Root directory to save results. If set, override the legacy ./result_depth|normal naming.",
    )
    parser.add_argument(
        "--total_test_length",
        type=int,
        default=None,
        help="Optional. Limit the number of samples per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Per-sample seed is seed + original sample index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with timer_context("Init model"):
        model = SenseNovaVisionModel(
            model_path=args.model_path,
            device=args.device,
        )
    total_depth_choices = len(DEPTH_TESTSET_PROMPT_TEMPLATES)
    total_normal_choices = len(NORMAL_TESTSET_PROMPT_TEMPLATES)

    params = dict(BASE_PARAMS["dense_perception"])
    if args.override_json:
        try:
            override_dict = json.loads(args.override_json)
        except json.JSONDecodeError:
            # maybe it's a file path
            if os.path.isfile(args.override_json):
                with open(args.override_json, "r", encoding="utf-8") as f:
                    override_dict = json.load(f)
            else:
                raise
        params.update(override_dict)
    image_dir_list = select_datasets(args.test_mode, args.dataset)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))

    def data_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(data_root, path))

    sample_offset = 0
    for test_dataset in image_dir_list:
        dataset_info = image_dir_list[test_dataset]
        print(f"{len(image_dir_list)} datasets total: now processing  {test_dataset}")
        is_depth_task = "Depth" in args.test_mode
        # if "image_text" in dataset_info:
        image_dir = data_path(dataset_info["image_path"])

        image_list = read_first_column_list(dataset_info["image_text"])
   
        if args.total_test_length is not None:
            image_list = image_list[: args.total_test_length]

        out_image_dir = build_output_dir(args, test_dataset)
        ensure_dir(out_image_dir)
        print(f"[OUTPUT_DIR] {out_image_dir}")
        tar_handle = tarfile.open(image_dir, "r") if is_depth_task and image_dir.endswith(".tar") else None

        try:
            for local_idx, image_name in enumerate(image_list):
                sample_seed = args.seed + sample_offset + local_idx
                if "scannet" == args.dataset and "Normal" in args.test_mode:
                    out_image_name = (
                        image_name.split("/")[-2]
                        + "-"
                        + image_name.split("/")[-1].split(".")[0]
                        + ".png"
                    )
                elif "diode" == args.dataset and "Normal" in args.test_mode:
                    out_image_name = (
                        image_name.split("/")[-3]
                        + "-"
                        + image_name.split("/")[-1].split(".")[0]
                        + ".png"
                    )
                elif "Normal" in args.test_mode:
                    out_image_name = image_name.split("/")[-1]
                else:
                    out_image_name = normalize_tar_member_name(image_name)[:-4] + ".png"
                infer_image_path = None
                temp_image_path = None
                print(f"process {image_name} ...")
                if tar_handle is not None:
                    img = open_image_from_tar(tar_handle, image_name)
                else:
                    image_path = image_name if os.path.isabs(image_name) else os.path.join(image_dir, image_name)
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    img = Image.open(image_path).convert("RGB")
                if test_dataset == "kitti":
                    width, height = img.size
                    KB_CROP_HEIGHT = 352
                    KB_CROP_WIDTH = 1216

                    top_margin = int(height - KB_CROP_HEIGHT)
                    left_margin = int((width - KB_CROP_WIDTH) / 2)
                    img = img.crop(
                        (
                            left_margin,
                            top_margin,
                            left_margin + KB_CROP_WIDTH,
                            top_margin + KB_CROP_HEIGHT,
                        )
                    )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_image_path = tmp.name
                img.save(temp_image_path)
                infer_image_path = temp_image_path

                output_path = os.path.join(out_image_dir, out_image_name)
                output_parent = os.path.dirname(output_path)
                if output_parent:
                    os.makedirs(output_parent, exist_ok=True)
                if os.path.exists(output_path):
                    if temp_image_path is not None:
                        os.remove(temp_image_path)
                    continue
                if "Depth" in args.test_mode:
                    depth_idx = sample_seed % total_depth_choices
                    new_prompt = DEPTH_TESTSET_PROMPT_TEMPLATES[depth_idx]
                elif "Normal" in args.test_mode:
                    normal_idx = sample_seed % total_normal_choices
                    new_prompt = NORMAL_TESTSET_PROMPT_TEMPLATES[normal_idx]
                with timer_context("inference"):
                    print(new_prompt)
                    output = model.generate(
                        question=ensure_image_placeholders(new_prompt, 1),
                        images=[infer_image_path],
                        mode="dense_perception",
                        noise_seed=sample_seed,
                        return_intermediate_outputs=True,
                        **params,
                    )
                if temp_image_path is not None:
                    os.remove(temp_image_path)
                if output.get("text") is not None:
                    print("=== Text Output ===")
                    print(output["text"])
                if output.get("image") is not None:
                    output["image"].save(output_path)
                    print(f"Image saved to {output_path}")
        finally:
            if tar_handle is not None:
                tar_handle.close()
        sample_offset += len(image_list)


if __name__ == "__main__":
    main()
