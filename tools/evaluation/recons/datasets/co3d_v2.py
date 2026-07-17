"""
Reference:
[VGGT](https://github.com/facebookresearch/vggt/blob/3d0e21d4043a37f0bfd1fd9e28dc31f76011cd98/evaluation/test_co3d.py)
[PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/datasets/co3d_v2.py)
[RayDiffusion](https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/dataset/co3d_v2.py)
"""

import gzip
import json
import os.path as osp
import numpy as np
import torch

from typing import Optional, Union, Iterable
from PIL import Image, ImageFile
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

def opencv_from_cameras_projection_RT(R, T):
    R = R[None, :, :]
    T = T[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    return R[0], tvec[0]

def opencv_from_cameras_projection_intr(focal, p0, image_size):
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    focal_pytorch3d = focal
    p0_pytorch3d = p0

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros((1, 3, 3))
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return camera_matrix[0]

def convert_pt3d_RT_to_opencv(Rot, Trans):
    # Convert pt3d extrinsic to opencv
    rot_pt3d   = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2]  *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d         = rot_pt3d.transpose(1, 0)

    # extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    extri_opencv         = np.eye(4)
    extri_opencv[:3, :3] = rot_pt3d
    extri_opencv[:3, 3]  = trans_pt3d
    return extri_opencv

class Co3dDataset(Dataset):
    def __init__(
        self,
        CO3D_DIR,
        CO3D_ANNOTATION_DIR,
        categories: Union[str, list, None] = None,
        split_name: str = "test",
        min_num_images: int = 50,
        sort_by_filename: bool = False,
    ):
        """
        Args:
            categories (iterable): List of categories to use. if not specified, will use TEST_CATEGORIES.
            num_images (int): Default number of images in each batch.
        """
        categories = TEST_CATEGORIES if categories is None else categories
        if isinstance(categories, str):
            if categories == "test":
                categories = TEST_CATEGORIES
            elif categories == "debug":
                categories = DEBUG_CATEGORIES
            elif categories == "train":
                categories = TRAINING_CATEGORIES
            elif categories == "all":
                categories = TRAINING_CATEGORIES + TEST_CATEGORIES
            else:
                raise ValueError(f"Unknown str category: {categories}")
        elif isinstance(categories, list):
            categories = categories
        else:
            raise ValueError(f"Unknown categories: {categories}")
        self.split_name = split_name
        self.categories = categories

        self.low_quality_translations = []
        self.metadata = {}
        self.category_map = {}

        print(f"[CO3DV2] CO3D_DIR is {CO3D_DIR}")
        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR
        self.min_num_images = min_num_images

        for c in categories:
            annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            if not osp.exists(annotation_file):
                continue
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < min_num_images:
                    print(f"[CO3DV2] sequence {seq_name} in category {c} has only {len(seq_data)} images, filter it.")
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            # "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        }
                    )

                if not bad_seq:
                    self.metadata[seq_name] = filtered_data

            # print(f"Found {counter} sequences in {annotation_file}")

        self.sequence_list: list = list(self.metadata.keys())
        self.sort_by_filename    = sort_by_filename

        print(f"[CO3DV2] Low quality translation sequences, not used: {self.low_quality_translations}")
        print(f"[CO3DV2] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name])

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[Iterable, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        category = self.category_map[sequence_name]

        if ids is None:
            ids = np.arange(len(metadata))
        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        image_paths: list      = [""] * len(annos)
        focal_lengths: list    = [0] * len(annos)
        principal_points: list = [0] * len(annos)

        extrinsics: torch.Tensor = torch.eye(4, 4)[None].repeat(len(annos), 1, 1)
        # intrinsics: torch.Tensor = torch.eye(3, 3)[None].repeat(len(annos), 1, 1)

        for idx, anno in enumerate(annos):
            filepath = anno['filepath']
            impath = osp.join(self.CO3D_DIR, filepath)

            # focal_length = np.array(anno['focal_length'])
            # principal_point = np.array(anno['principal_point'])

            extri_opencv = convert_pt3d_RT_to_opencv(anno["R"], anno["T"])

            image_paths[idx]      = impath
            extrinsics[idx]       = torch.tensor(extri_opencv)
            focal_lengths[idx]    = torch.tensor(anno['focal_length'])
            principal_points[idx] = torch.tensor(anno['principal_point'])

        batch = {"seq_id": sequence_name, "category": category, "n": len(metadata), "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths

        batch["extrs"] = extrinsics
        batch["fl"] = torch.stack(focal_lengths)
        batch["pp"] = torch.stack(principal_points)

        return batch

    def get_data_ori(self, index=None, sequence_name=None, ids=(0, 1), return_path = False):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        category = self.category_map[sequence_name]

        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        image_paths: list      = [""] * len(annos)
        # images: list           = [0] * len(annos)
        # depth_maps: list       = [0] * len(annos)
        focal_lengths: list    = [0] * len(annos)
        principal_points: list = [0] * len(annos)

        extrinsics: torch.Tensor = torch.eye(4, 4)[None].repeat(len(annos), 1, 1)

        for idx, anno in enumerate(annos):
            filepath = anno['filepath']
            impath = osp.join(self.CO3D_DIR, filepath)

            # load camera params
            R = np.array(anno['R'])
            T = np.array(anno['T'])

            focal_length = np.array(anno['focal_length'])
            principal_point = np.array(anno['principal_point'])

            R, tvec = opencv_from_cameras_projection_RT(R, T)

            image_paths[idx]        = impath
            extrinsics[idx, :3, :3] = torch.tensor(R)
            extrinsics[idx, :3, 3]  = torch.tensor(tvec)
            focal_lengths[idx]      = torch.tensor(focal_length)
            principal_points[idx]   = torch.tensor(principal_point)

        batch = {"seq_id": sequence_name, "category": category, "n": len(metadata), "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths

        batch["extrs"] = extrinsics
        batch["fl"] = torch.stack(focal_lengths)
        batch["pp"] = torch.stack(principal_points)

        return batch


TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = ["ball", "book", "couch", "frisbee", "hotdog", "kite", "remote", "sandwich", "skateboard", "suitcase"]

DEBUG_CATEGORIES = ["baseballbat", "hotdog"]
