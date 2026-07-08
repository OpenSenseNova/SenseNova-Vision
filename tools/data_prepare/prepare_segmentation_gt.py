#!/usr/bin/env python3
"""Prepare segmentation benchmark GT files.

The commands in docs/data_prepare.md call this script. Keep this file focused on
deterministic GT conversion; it should not generate training prompts or JSONL.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


VISUAL_PROMPT_TYPES = (
    "point_visual_prompt_mask",
    "scribble_visual_prompt_mask",
    "box_visual_prompt_mask",
    "mask_visual_prompt_mask",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rgb2id(color: np.ndarray) -> np.ndarray:
    if color.ndim == 2:
        return color.astype(np.uint32)
    color = color.astype(np.uint32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def normalize_rle(rle: dict) -> dict:
    rle = dict(rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    return rle


def decode_coco_mask(segmentation, height: int, width: int) -> np.ndarray:
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = normalize_rle(segmentation)
        decoded = mask_utils.decode(rle)
    elif isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], dict):
        rles = [normalize_rle(seg) for seg in segmentation]
        decoded = mask_utils.decode(rles)
    elif isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return (decoded > 0).astype(np.uint8)


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def enhance_with_circles(mask: np.ndarray, radius: int) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.uint8)
    y, x = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for center_y, center_x in np.argwhere(mask > 0):
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        output[distance <= radius] = 1
    return output


def prepare_coco_panoptic_semseg(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    ann_path = data_root / "annotations" / f"panoptic_{args.split}2017.json"
    panoptic_dir = data_root / f"panoptic_{args.split}2017"
    if not panoptic_dir.is_dir():
        panoptic_dir = data_root / "annotations" / f"panoptic_{args.split}2017"
    out_dir = data_root / f"panoptic_semseg_{args.split}2017"

    ensure_dir(out_dir)
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    category_id_to_contiguous_id = {
        int(category["id"]): contiguous_id
        for contiguous_id, category in enumerate(data["categories"])
    }

    for ann in tqdm(data["annotations"], desc=f"COCO panoptic_semseg_{args.split}2017"):
        pan_path = panoptic_dir / ann["file_name"]
        panoptic = rgb2id(np.asarray(Image.open(pan_path)))
        semseg = np.full(panoptic.shape, 255, dtype=np.uint8)
        for seg in ann["segments_info"]:
            category_id = int(seg["category_id"])
            semseg[panoptic == int(seg["id"])] = category_id_to_contiguous_id[category_id]
        Image.fromarray(semseg).save(out_dir / ann["file_name"])

    print(f"Wrote {out_dir}")


def prepare_ade20k_semantic(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    in_dir = data_root / "annotations" / args.split
    out_dir = data_root / "annotations_detectron2" / args.split
    ensure_dir(out_dir)

    for in_path in tqdm(sorted(in_dir.glob("*.png")), desc=f"ADE20K semantic {args.split}"):
        label = np.asarray(Image.open(in_path), dtype=np.uint8)
        converted = label - np.uint8(1)
        Image.fromarray(converted).save(out_dir / in_path.name)

    print(f"Wrote {out_dir}")


def load_refcoco(data_root: Path, dataset: str):
    split_by = "umd" if dataset == "refcocog" else "unc"
    ds_root = data_root / dataset
    with (ds_root / f"refs({split_by}).p").open("rb") as f:
        refs = pickle.load(f)
    with (ds_root / "instances.json").open("r", encoding="utf-8") as f:
        instances = json.load(f)

    ann_by_id = {ann["id"]: ann for ann in instances["annotations"]}
    img_by_id = {img["id"]: img for img in instances["images"]}
    return refs, ann_by_id, img_by_id


def prepare_refcoco_binary(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    for dataset in args.datasets:
        refs, ann_by_id, img_by_id = load_refcoco(data_root, dataset)
        out_dir = data_root / "ref_seg" / "binary_masks" / f"{dataset}_{args.split}"
        ensure_dir(out_dir)

        selected = [ref for ref in refs if ref.get("split") == args.split]
        for ref in tqdm(selected, desc=f"{dataset}_{args.split}"):
            ann = ann_by_id[ref["ann_id"]]
            img = img_by_id[ann["image_id"]]
            mask = decode_coco_mask(ann["segmentation"], int(img["height"]), int(img["width"]))
            stem = img["file_name"].replace(".jpg", "").replace("/", "_")
            save_binary(mask, out_dir / f"{stem}_{ref['ref_id']}.png")

        print(f"Wrote {out_dir}")


def labelme_shapes_to_mask(width: int, height: int, shapes: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label == "flag":
            continue
        points = np.asarray([shape.get("points", [])], dtype=np.int32)
        if points.size:
            cv2.fillPoly(mask, points, 0 if "ignore" in label else 1)
    return mask


def prepare_reason_binary(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    for split in args.splits:
        image_dir = data_root / split
        out_dir = data_root / "rea_seg" / "binary_masks" / split
        ensure_dir(out_dir)

        for image_path in tqdm(sorted(image_dir.glob("*.jpg")), desc=f"ReasonSeg {split}"):
            json_path = image_path.with_suffix(".json")
            if not json_path.is_file():
                continue
            with json_path.open("r", encoding="utf-8") as f:
                item = json.load(f)

            shapes = [
                shape
                for shape in item.get("shapes", [])
                if str(shape.get("label", "")).lower() != "flag"
            ]
            if not shapes:
                continue

            image_bytes = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image: {image_path}")
            height, width = image.shape[:2]
            mask = labelme_shapes_to_mask(width, height, shapes)
            save_binary(mask, out_dir / f"{image_path.stem}.png")

        print(f"Wrote {out_dir}")


def prepare_interseg_binary(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    ann_path = data_root / "annotations" / f"coco_interactive_{args.split}_psalm.json"
    out_dir = data_root / "inter_seg" / "binary_masks" / args.dataset / args.split
    ensure_dir(out_dir)
    for prompt_type in VISUAL_PROMPT_TYPES:
        ensure_dir(out_dir / prompt_type)

    with ann_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    for item in tqdm(items, desc=f"{args.dataset}_{args.split}"):
        image_info = item["image_info"]
        height = int(image_info["height"])
        width = int(image_info["width"])
        stem = Path(image_info["file_name"]).stem
        anns = item.get("annotations", item.get("anns", []))

        for ann in anns:
            mask = decode_coco_mask(ann["segmentation"], height, width)
            if not mask.any():
                continue
            mask_name = f"{stem}_{ann['id']}_class{ann['category_id']}.png"
            save_binary(mask, out_dir / mask_name)

            for prompt_type in VISUAL_PROMPT_TYPES:
                prompt = ann.get(prompt_type)
                if prompt is None:
                    continue
                prompt_mask = decode_coco_mask(prompt, height, width)
                if prompt_type == "point_visual_prompt_mask":
                    prompt_mask = enhance_with_circles(prompt_mask, radius=10)
                elif prompt_type == "scribble_visual_prompt_mask":
                    prompt_mask = enhance_with_circles(prompt_mask, radius=5)
                if prompt_mask.any():
                    save_binary(prompt_mask, out_dir / prompt_type / mask_name)

    print(f"Wrote {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    coco = subparsers.add_parser("coco-panoptic-semseg")
    coco.add_argument("--data-root", default="datas/gen_seg_data/coco2017")
    coco.add_argument("--split", default="val", choices=["train", "val"])
    coco.set_defaults(func=prepare_coco_panoptic_semseg)

    ade = subparsers.add_parser("ade20k-semantic")
    ade.add_argument("--data-root", default="datas/ov_seg_data/ade20k")
    ade.add_argument("--split", default="validation", choices=["training", "validation"])
    ade.set_defaults(func=prepare_ade20k_semantic)

    ref = subparsers.add_parser("refcoco-binary")
    ref.add_argument("--data-root", default="datas/ref_seg_data")
    ref.add_argument("--datasets", nargs="+", default=["refcoco", "refcoco+", "refcocog"])
    ref.add_argument("--split", default="val")
    ref.set_defaults(func=prepare_refcoco_binary)

    rea = subparsers.add_parser("reason-binary")
    rea.add_argument("--data-root", default="datas/rea_seg_data")
    rea.add_argument("--splits", nargs="+", default=["val", "test"])
    rea.set_defaults(func=prepare_reason_binary)

    inter = subparsers.add_parser("interseg-binary")
    inter.add_argument("--data-root", default="datas/inter_seg_data")
    inter.add_argument("--dataset", default="coco_interactive_psalm")
    inter.add_argument("--split", default="val")
    inter.set_defaults(func=prepare_interseg_binary)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
