# Copyright (c) 2026 SenseTime Group Inc. and/or its affiliates.

import argparse
import json
import os
from pathlib import Path

import jsonlines
import tqdm

from data.prompts import ensure_image_placeholders
from inference.sensenova_vision import (
    BASE_PARAMS,
    CAMERA_POSE_PROMPT,
    SenseNovaVisionModel,
)
from inference.utils_3d import resolve_pose_string
from utils import ensure_dir


TASKS = {
    "co3dv2": {
        "dataset_dir": "co3dv2/data",
        "id_map": "tools/evaluation/recons/datasets/seq-id-maps/CO3Dv2_relpose_seq-id-map_seed42.json",
        "output": "co3dv2",
    },
    "re10k": {
        "dataset_dir": "re10k",
        "id_map": "tools/evaluation/recons/datasets/seq-id-maps/Re10K_relpose_seq-id-map_seed42.json",
        "output": "re10k",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-Vision camera pose benchmark"
    )
    parser.add_argument("--model_path", default="sensenova/SenseNova-Vision-7B-MoT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=TASKS.keys(), required=True)
    parser.add_argument(
        "--data_root",
        default="datas/multiview3d_data",
        help="Root directory for test datasets.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--total_test_length",
        type=int,
        help="Limit the number of samples per dataset (for debug only).",
    )
    return parser.parse_args()


def build_output_dir(args):
    leaf = TASKS[args.dataset]["output"]
    output_root = args.output_dir or ""
    return os.path.abspath(os.path.join(output_root, "camera_pose", leaf))


def load_params(args):
    params = dict(BASE_PARAMS["understanding"])
    return params


def main():
    args = parse_args()
    params = load_params(args)
    task = TASKS[args.dataset]
    dataset_dir = Path(args.data_root) / task["dataset_dir"]
    with open(task["id_map"], "r") as f:
        seq_map = json.load(f)
    if args.total_test_length:
        seq_map = {
            k: v
            for i, (k, v) in enumerate(seq_map.items())
            if i < args.total_test_length
        }

    model = SenseNovaVisionModel(model_path=args.model_path, device=args.device)

    output_dir = build_output_dir(args)
    ensure_dir(output_dir)
    output_jsonl = os.path.join(output_dir, "results.jsonl")
    output_jsonl_raw = os.path.join(output_dir, "results_raw.jsonl")

    for path in [output_jsonl, output_jsonl_raw]:
        if os.path.exists(path):
            os.remove(path)

    print(f"[OUTPUT_DIR] {output_dir}")

    for seq_name, seq_list in tqdm.tqdm(seq_map.items(), desc="sequence"):
        image_paths = []
        for item in seq_list:
            if isinstance(item, dict):
                rel_path = item["filepath"]
            else:
                rel_path = item
            image_path = dataset_dir / rel_path
            image_paths.append(image_path)

        raw_text = model.generate(
            question=ensure_image_placeholders(CAMERA_POSE_PROMPT, len(image_paths)),
            images=image_paths,
            mode="understanding",
            vit_transform=model.camera_vit_transform,
            **params,
        )

        result = resolve_pose_string(raw_text)

        if result is not None:
            with jsonlines.open(output_jsonl, mode="a") as writer:
                writer.write({seq_name: result})

        with jsonlines.open(output_jsonl_raw, mode="a") as writer:
            writer.write({seq_name: raw_text})


if __name__ == "__main__":
    main()
