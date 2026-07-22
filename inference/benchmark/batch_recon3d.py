# Copyright (c) 2026 SenseTime Group Inc. and/or its affiliates.

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tqdm

from inference.sensenova_vision import (
    BASE_PARAMS,
    RECON3D_PROMPT,
    SenseNovaVisionModel,
)
from utils import ensure_dir


TASKS = {
    "7scenes": {
        "dataset_dir": "7scenes",
        "id_map": "tools/evaluation/recons/datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf100.json",
        "output": "7scenes_kf100",
    },
    "eth3d": {
        "dataset_dir": "eth3d",
        "id_map": "tools/evaluation/recons/datasets/seq-id-maps/ETH3D_mv-recon_seq-id-map-nf10.json",
        "output": "eth3d_nf10",
    },
    "dtu": {
        "dataset_dir": "dtu",
        "id_map": "tools/evaluation/recons/datasets/seq-id-maps/DTU_mv-recon_seq-id-map-kf5.json",
        "output": "dtu_kf5",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-Vision recon3d benchmark inference."
    )
    parser.add_argument(
        "--model_path",
        default="sensenova/SenseNova-Vision-7B-MoT",
        help="Path to SenseNova-Vision model directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--dataset",
        choices=list(TASKS.keys()),
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_root",
        default="datas/multiview3d_data",
        help="Root directory for test datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root directory to save results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
    )
    parser.add_argument(
        "--total_test_length",
        type=int,
        default=None,
        help="Limit the number of samples per dataset (for debug only).",
    )
    return parser.parse_args()


def build_output_dir(args):
    leaf = TASKS[args.dataset]["output"]
    output_root = args.output_dir or ""
    return os.path.join(output_root, "recon3d", leaf)


def load_params(args):
    params = dict(BASE_PARAMS["recon3d"])
    params.update(
        noise_seed=args.seed,
        cfg_text_scale=4.0,
    )
    return params


def eval_7scenes(model, params, *, data_root, output_dir, total_test_length):
    dataset_dir = Path(data_root) / TASKS["7scenes"]["dataset_dir"]
    id_map_path = Path(TASKS["7scenes"]["id_map"])

    with id_map_path.open() as f:
        seq_id_map = json.load(f)
    if total_test_length:
        seq_id_map = {
            k: v for i, (k, v) in enumerate(seq_id_map.items()) if i < total_test_length
        }

    for seq_name, ids in tqdm.tqdm(seq_id_map.items()):
        images = []
        for idx in ids:
            img = dataset_dir / seq_name / f"frame-{idx:06d}.color.png"
            images.append(str(img))

        output = model.reconstruct_3d(
            images=images,
            prompt=RECON3D_PROMPT,
            postprocess_predictions=False,
            **params,
        )

        pointmaps = output["pts3d"]
        for i, pm in enumerate(pointmaps):
            out_name = seq_name.replace("/", "_") + f"_frame-{ids[i]:06d}.npy"
            np.save(os.path.join(output_dir, out_name), pm)


def eval_eth3d(model, params, *, data_root, output_dir, total_test_length):
    dataset_dir = Path(data_root) / TASKS["eth3d"]["dataset_dir"]
    id_map_path = Path(TASKS["eth3d"]["id_map"])

    with id_map_path.open() as f:
        seq_id_map = json.load(f)
    if total_test_length:
        seq_id_map = {
            k: v for i, (k, v) in enumerate(seq_id_map.items()) if i < total_test_length
        }

    metadata = {}
    for seq in sorted(d.name for d in dataset_dir.iterdir() if d.is_dir()):
        imgs = sorted(
            (dataset_dir / seq / "images" / "custom_undistorted").glob("*.JPG")
        )
        metadata[seq] = imgs

    for seq_name, ids in tqdm.tqdm(seq_id_map.items()):
        image_paths = [str(metadata[seq_name][i]) for i in ids]

        output = model.reconstruct_3d(
            images=image_paths,
            prompt=RECON3D_PROMPT,
            postprocess_predictions=False,
            **params,
        )

        for i, pm in enumerate(output["pts3d"]):
            np.save(
                os.path.join(output_dir, f"{seq_name}_frame-{ids[i]:06d}.npy"),
                pm,
            )


def eval_dtu(model, params, *data_root, output_dir, total_test_length):
    dataset_dir = Path(data_root) / TASKS["dtu"]["dataset_dir"]
    id_map_path = Path(TASKS["dtu"]["id_map"])

    with id_map_path.open() as f:
        seq_id_map = json.load(f)
    if total_test_length:
        seq_id_map = {
            k: v for i, (k, v) in enumerate(seq_id_map.items()) if i < total_test_length
        }

    for seq_name, ids in tqdm.tqdm(seq_id_map.items()):
        image_paths = [
            str(dataset_dir / seq_name / "images" / f"{i:08d}.jpg") for i in ids
        ]

        output = model.reconstruct_3d(
            images=image_paths,
            prompt=RECON3D_PROMPT,
            postprocess_predictions=False,
            **params,
        )

        for i, pm in enumerate(output["pts3d"]):
            np.save(
                os.path.join(output_dir, f"{seq_name}_frame-{ids[i]:06d}.npy"),
                pm,
            )


def main():
    args = parse_args()
    params = load_params(args)
    model = SenseNovaVisionModel(
        model_path=args.model_path,
        device=args.device,
    )

    output_dir = build_output_dir(args)
    ensure_dir(output_dir)
    print(f"[OUTPUT_DIR] {output_dir}")

    if args.dataset == "7scenes":
        eval_7scenes(
            model,
            params,
            data_root=args.data_root,
            output_dir=output_dir,
            total_test_length=args.total_test_length,
        )
    elif args.dataset == "eth3d":
        eval_eth3d(
            model,
            params,
            data_root=args.data_root,
            output_dir=output_dir,
            total_test_length=args.total_test_length,
        )
    elif args.dataset == "dtu":
        eval_dtu(
            model,
            params,
            data_root=args.data_root,
            output_dir=output_dir,
            total_test_length=args.total_test_length,
        )


if __name__ == "__main__":
    main()
