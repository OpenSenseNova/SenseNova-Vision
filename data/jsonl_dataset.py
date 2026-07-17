# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import traceback

from .data_utils import load_image, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


class JSONLIterableDataset(DistributedIterableDataset):

    INPUT_ROLE = "input"
    OUTPUT_ROLE = "output"

    def __init__(
        self,
        dataset_name,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, data_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_roots = self._normalize_data_dir(data_dir)
            data_paths.extend((json_data, data_roots) for json_data in raw_data)
        return data_paths

    def get_resume_row(self, worker_id):
        if self.data_status is not None and worker_id in self.data_status:
            return self.data_status[worker_id] + 1
        return 0

    def log_resume(self, worker_id, row_start_id, num_worker_paths):
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id} out of {num_worker_paths}"
        )

    def log_bad_sample(self, worker_id, row_idx, exc, error_type="Loading Image Error"):
        print(
            f"{error_type}: rank-{self.local_rank} worker-{worker_id} "
            f"dataset-{self.dataset_name}: skip bad sample at row#{row_idx}, error={exc}"
        )
        traceback.print_exc()

    def new_sample(self):
        return {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }

    @staticmethod
    def _normalize_data_dir(data_dir):
        """Normalize legacy and role-aware media root configurations."""
        if isinstance(data_dir, (str, os.PathLike)):
            root = os.fspath(data_dir)
            return {
                "input_dir": root,
                "output_dir": root,
            }

        if not isinstance(data_dir, dict):
            raise TypeError(
                "data_dir must be a path string or a dict containing "
                "input_dir/output_dir"
            )

        roots = {
            "input_dir": data_dir.get("input_dir"),
            "output_dir": data_dir.get("output_dir"),
        }
        if roots["input_dir"] is None and roots["output_dir"] is None:
            raise ValueError(
                "data_dir dict must define at least one of input_dir/output_dir"
            )
        return roots

    @staticmethod
    def _is_explicit_path(path):
        path = os.fspath(path)
        return os.path.isabs(path) or "://" in path

    def resolve_media_path(self, data_roots, media_path, role):
        data_roots = self._normalize_data_dir(data_roots)
        media_path = os.fspath(media_path)
        if self._is_explicit_path(media_path):
            return media_path

        if role not in (self.INPUT_ROLE, self.OUTPUT_ROLE):
            raise ValueError(f"unsupported media role: {role}")

        root_key = f"{role}_dir"
        root = data_roots.get(root_key)
        if root in (None, ""):
            raise ValueError(
                f"relative {role} path {media_path!r} requires {root_key}"
            )
        return os.path.join(os.fspath(root), media_path)

    def load_image(self, data_roots, image_path, role=INPUT_ROLE):
        image_path = self.resolve_media_path(data_roots, image_path, role)
        return pil_img2rgb(load_image(image_path))

    def load_image_list(self, data_roots, image_paths, role=INPUT_ROLE):
        if image_paths is None:
            return []
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        return [
            self.load_image(data_roots, image_path, role=role)
            for image_path in image_paths
        ]
