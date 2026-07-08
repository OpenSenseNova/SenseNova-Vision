# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json

from PIL import Image

from .jsonl_dataset import JSONLIterableDataset

Image.MAX_IMAGE_PIXELS = 200_000_000


class T2IJSONLIterableDataset(JSONLIterableDataset):

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
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
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        self.transform = transform
        self.tokenizer = tokenizer
        super().__init__(
            dataset_name,
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            local_rank,
            world_size,
            num_workers,
            data_status,
            shuffle_lines,
            shuffle_seed,
        )

    def _get_caption_token(self, data_item):
        if 'conversations' not in data_item:
            return self.tokenizer.encode(' ')

        caption_list = data_item['conversations']
        if not caption_list or caption_list[0].get('from') != 'human':
            return self.tokenizer.encode(' ')

        caption = caption_list[0].get('value', ' ')
        return self.tokenizer.encode(caption)

    def _build_sample(self, image_tensor, caption_token, row_idx, worker_id, num_tokens):
        return {
            'image_tensor_list': [image_tensor],
            'text_ids_list': [caption_token],
            'num_tokens': num_tokens,
            'sequence_plan': [
                {
                    'type': 'text',
                    'enable_cfg': 1,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
                {
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            ],
            'data_indexes': {
                "data_indexes": row_idx,
                "worker_id": worker_id,
                "dataset_name": self.dataset_name,
            },
        }

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.get_resume_row(worker_id)
        transform_stride = self.transform.stride

        self.log_resume(worker_id, row_start_id, len(data_paths_per_worker))

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                try:
                    data_item = json.loads(data)
                    image = self.load_image(image_dir, data_item['image'])
                except Exception as e:
                    self.log_bad_sample(worker_id, row_idx, e)
                    continue

                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                num_tokens += width * height // transform_stride**2

                try:
                    caption_token = self._get_caption_token(data_item)
                except Exception as e:
                    self.log_bad_sample(worker_id, row_idx, e, error_type="Caption Error")
                    continue

                if len(caption_token) == 0:
                    print(f'no caption in {data} in {self.dataset_name}')
                    caption_token = self.tokenizer.encode(' ')
                num_tokens += len(caption_token)
                yield self._build_sample(
                    image_tensor, caption_token, row_idx, worker_id, num_tokens
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
