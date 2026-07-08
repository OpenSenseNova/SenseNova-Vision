# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import traceback

from .data_utils import load_image, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


class JSONLIterableDataset(DistributedIterableDataset):

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
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend((json_data, image_dir) for json_data in raw_data)
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

    def load_image(self, image_dir, image_path):
        return pil_img2rgb(load_image(os.path.join(image_dir, image_path)))

    def load_image_list(self, image_dir, image_paths):
        if image_paths is None:
            return []
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        return [self.load_image(image_dir, image_path) for image_path in image_paths]
