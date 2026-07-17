#!/usr/bin/env python3
"""Prepare semantic benchmark labels from source-specific annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def rgb2id(color: np.ndarray) -> np.ndarray:
    """Convert a COCO panoptic RGB map to integer segment IDs."""
    if color.ndim == 2:
        return color.astype(np.uint32)
    color = color.astype(np.uint32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def prepare_coco_panoptic_to_semantic(args: argparse.Namespace) -> None:
    """Convert COCO panoptic segment IDs to contiguous semantic category IDs."""
    data_root = Path(args.data_root)
    ann_path = data_root / "annotations" / f"panoptic_{args.split}2017.json"
    panoptic_dir = data_root / f"panoptic_{args.split}2017"
    if not panoptic_dir.is_dir():
        panoptic_dir = data_root / "annotations" / f"panoptic_{args.split}2017"
    output_dir = data_root / f"panoptic_semseg_{args.split}2017"
    output_dir.mkdir(parents=True, exist_ok=True)

    with ann_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    category_id_to_contiguous_id = {
        int(category["id"]): contiguous_id
        for contiguous_id, category in enumerate(annotations["categories"])
    }

    for annotation in tqdm(
        annotations["annotations"],
        desc=f"COCO panoptic to semantic {args.split}2017",
    ):
        panoptic = rgb2id(
            np.asarray(Image.open(panoptic_dir / annotation["file_name"]))
        )
        semantic = np.full(panoptic.shape, 255, dtype=np.uint8)
        for segment in annotation["segments_info"]:
            semantic[panoptic == int(segment["id"])] = (
                category_id_to_contiguous_id[int(segment["category_id"])]
            )
        Image.fromarray(semantic).save(output_dir / annotation["file_name"])

    print(f"Wrote {output_dir}")


def prepare_ade20k(args: argparse.Namespace) -> None:
    """Convert ADE20K labels from one-based to zero-based category IDs."""
    data_root = Path(args.data_root)
    input_dir = data_root / "annotations" / args.split
    output_dir = data_root / "annotations_detectron2" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(
        sorted(input_dir.glob("*.png")), desc=f"ADE20K semantic {args.split}"
    ):
        label = np.asarray(Image.open(input_path), dtype=np.uint8)
        Image.fromarray(label - np.uint8(1)).save(output_dir / input_path.name)

    print(f"Wrote {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)

    coco_parser = subparsers.add_parser(
        "coco-panoptic",
        help="convert COCO panoptic RGB-ID annotations to semantic labels",
    )
    coco_parser.add_argument("--data-root", default="datas/gen_seg_data/coco2017")
    coco_parser.add_argument("--split", default="val", choices=["train", "val"])
    coco_parser.set_defaults(func=prepare_coco_panoptic_to_semantic)

    ade20k_parser = subparsers.add_parser(
        "ade20k",
        help="convert ADE20K semantic labels to zero-based category IDs",
    )
    ade20k_parser.add_argument("--data-root", default="datas/ov_seg_data/ade20k")
    ade20k_parser.add_argument(
        "--split", default="validation", choices=["training", "validation"]
    )
    ade20k_parser.set_defaults(func=prepare_ade20k)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
