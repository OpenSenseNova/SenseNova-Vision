# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json

from PIL import Image

from .jsonl_dataset import JSONLIterableDataset
from .prompts import strip_task_prompt

Image.MAX_IMAGE_PIXELS = 200000000


class EditJSONLIterableDataset(JSONLIterableDataset):

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
        task_prompt="raw_prompt",
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.task_prompt = task_prompt
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

    def _get_instruction(self, data_item):
        if "instruction" in data_item:
            return strip_task_prompt(
                data_item["instruction"], self.task_prompt
            ).rstrip()

        if "conversations" in data_item:
            conversations = data_item["conversations"]
            if len(conversations) == 2 and conversations[0]["from"] == "human":
                instruction = conversations[0]["value"].replace("<image>", "")
                return strip_task_prompt(instruction, self.task_prompt).rstrip()

        raise ValueError(
            "sample must contain instruction or a two-turn human conversation"
        )

    def _add_text(self, sample, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        sample["num_tokens"] += len(text_ids)
        sample["text_ids_list"].append(text_ids)
        sample["sequence_plan"].append(
            {
                "type": "text",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return sample

    def _add_image(self, sample, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            sample["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            sample["num_tokens"] += width * height // self.transform.stride**2
            sample["image_tensor_list"].append(image_tensor)

        if need_vae:
            sample["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            sample["num_tokens"] += width * height // self.transform.stride**2
            sample["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            sample["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            sample["num_tokens"] += width * height // self.vit_transform.stride**2
            sample["image_tensor_list"].append(vit_image_tensor)

        return sample

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.get_resume_row(worker_id)

        self.log_resume(worker_id, row_start_id, len(data_paths_per_worker))

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, data_roots) in enumerate(
                data_paths_per_worker_, start=row_start_id
            ):
                sample = self.new_sample()
                try:
                    data_item = json.loads(data)

                    sample = self._add_image(
                        sample,
                        self.load_image(
                            data_roots,
                            data_item["image"][0],
                            role=self.INPUT_ROLE,
                        ),
                        need_loss=False,
                        need_vae=True,
                        need_vit=True,
                    )
                    sample = self._add_text(
                        sample, self._get_instruction(data_item), need_loss=False
                    )
                    sample = self._add_image(
                        sample,
                        self.load_image(
                            data_roots,
                            data_item["image"][1],
                            role=self.OUTPUT_ROLE,
                        ),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
                except Exception as exc:
                    self.log_bad_sample(worker_id, row_idx, exc)
                    continue
                sample["data_indexes"] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield sample
            row_start_id = 0
            print(
                f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}"
            )
