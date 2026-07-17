# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json

from PIL import Image, ImageFile, PngImagePlugin

from .jsonl_dataset import JSONLIterableDataset
from .prompts import strip_task_prompt

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDataset(JSONLIterableDataset):

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        frame_sampler,
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
        # Keep the argument for existing config compatibility. VLM media inputs
        # are intentionally limited to images (including multi-image samples).
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

    def _load_raw_images(self, data_item, data_roots):
        if "video" in data_item:
            raise ValueError("video inputs are not supported")
        return self.load_image_list(
            data_roots,
            data_item.get("image"),
            role=self.INPUT_ROLE,
        )

    def _build_elements(self, data, num_images):
        elements = []
        for conversation in data["conversations"]:
            if conversation["from"] == "human":
                value = conversation["value"]
                value = strip_task_prompt(value, self.task_prompt)

                if "<image>" not in value:
                    elements.append(
                        {
                            "type": "text",
                            "has_loss": 0,
                            "text": value,
                        }
                    )
                else:
                    text_list = value.split("<image>")
                    for idx, text in enumerate(text_list):
                        if text.strip() != "":
                            elements.append(
                                {
                                    "type": "text",
                                    "has_loss": 0,
                                    "text": text.strip(),
                                }
                            )
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append(
                                {
                                    "type": "image",
                                }
                            )
            elif conversation["from"] == "gpt":
                elements.append(
                    {
                        "type": "text",
                        "has_loss": 1,
                        "text": conversation["value"],
                    }
                )
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.get_resume_row(worker_id)
        transform_stride = self.transform.stride

        self.log_resume(worker_id, row_start_id, len(data_paths_per_worker))

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, data_roots) in enumerate(
                data_paths_per_worker_, start=row_start_id
            ):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    data_item = json.loads(data)
                    raw_images = self._load_raw_images(data_item, data_roots)
                except Exception as exc:
                    self.log_bad_sample(worker_id, row_idx, exc)
                    continue

                if raw_images:
                    for raw_image in raw_images:
                        image_tensor = self.transform(
                            raw_image, img_num=len(raw_images)
                        )
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride**2

                elements = self._build_elements(data_item, len(image_tensor_list))

                for item in elements:
                    if item["type"] == "text":
                        text_data = item["text"]
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                "type": "text",
                                "enable_cfg": 0,
                                "loss": item["has_loss"],
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                            sequence_plan.append(current_plan)
                    elif item["type"] == "image":
                        current_plan = {
                            "type": "vit_image",
                            "enable_cfg": 0,
                            "loss": 0,
                            "special_token_loss": 0,
                            "special_token_label": None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item["loss"] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f"No loss defined, skipped.")
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            row_start_id = 0
            print(
                f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}"
            )
