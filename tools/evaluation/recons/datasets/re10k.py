"""
Reference:
[PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/datasets/re10k.py)
"""

import os.path as osp
import os
import numpy as np
import torch
import json

from typing import Optional, Union, Iterable
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Re10KDataset(Dataset):
    def __init__(
        self,
        Re10K_DIR,
        split="test",
        min_num_images=50,
        sort_by_filename=False,
        seq_file=None, # "datasets/sequences/re10k_test_1719.txt"
    ):
        self.Re10K_DIR = Re10K_DIR
        print(f"[Re10K-{split}] Re10K_DIR is {Re10K_DIR}")

        self.split = split

        if split == "test":
            if seq_file is not None:
                with open(seq_file, "r") as f:
                    seq_list = f.readlines()
                self.sequence_list = [x.strip() for x in seq_list]
            else:
                self.sequence_list = os.listdir(Re10K_DIR)

            self.metadata = {}
            for seq in tqdm(self.sequence_list, desc=f"[Re10K-{split}] Creating metadata..."):
                anno_path = osp.join(Re10K_DIR, seq, "annotations.json")
                if not osp.exists(anno_path):
                    continue
                with open(anno_path, "r") as f:
                    annos = json.load(f)

                seq_info = []
                for anno in annos:
                    seq_info.append({
                        "idx": anno["idx"],
                        "filepath": anno["filepath"],
                        "intrinsics": torch.tensor(anno["intrinsics"]),
                        "extrinsics": torch.tensor(anno["extrinsics"]),
                    })

                self.metadata[seq] = seq_info

        elif split == "train":
            raise ValueError("We don't want to train on Re10K")
        else:
            raise ValueError("please specify correct set")

        self.min_num_images = min_num_images
        self.sort_by_filename = sort_by_filename

        print(f"[Re10K-{split}] Data size: {len(self)}")

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

        if ids is None:
            ids = np.arange(len(metadata))
        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        image_paths: list      = [""] * len(annos)

        extrinsics: torch.Tensor = torch.eye(4, 4)[None].repeat(len(annos), 1, 1)
        intrinsics: torch.Tensor = torch.eye(3, 3)[None].repeat(len(annos), 1, 1)

        for idx, anno in enumerate(annos):
            filepath = anno['filepath']
            impath = osp.join(self.Re10K_DIR, filepath)

            image_paths[idx]      = impath
            extrinsics[idx]       = anno["extrinsics"]
            intrinsics[idx]       = anno["intrinsics"]

        batch = {"seq_id": sequence_name, "n": len(metadata), "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths
        batch["extrs"] = extrinsics
        batch["intrs"] = intrinsics

        return batch
