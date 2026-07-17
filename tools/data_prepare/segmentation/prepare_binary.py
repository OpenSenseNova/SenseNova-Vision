#!/usr/bin/env python3
"""Prepare RefCOCO-family, ReasonSeg, and COCO-Interactive binary data.

The three subcommands preserve the processing behavior of the corresponding
reference converters. Training splits write masks and JSONL together;
benchmark splits write masks only because benchmark JSONL files are released
separately. ``--jsonl-only`` rebuilds a JSONL for any split without writing
masks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

if __package__:
    from .common.grefer import G_REFER
    from .common.prompts import (
        DEFAULT_IMAGE_TOKEN,
        EXPLANATORY_QUESTIONS,
        LONG_QUESTIONS,
        MASK_ANSWER_LIST,
        MASK_QUESTION_LIST,
        SHORT_QUESTIONS,
        VISION_QUESTION_LIST,
        tag_categories,
    )
    from .common.refer import REFER
else:
    from common.grefer import G_REFER
    from common.prompts import (
        DEFAULT_IMAGE_TOKEN,
        EXPLANATORY_QUESTIONS,
        LONG_QUESTIONS,
        MASK_ANSWER_LIST,
        MASK_QUESTION_LIST,
        SHORT_QUESTIONS,
        VISION_QUESTION_LIST,
        tag_categories,
    )
    from common.refer import REFER


VISUAL_PROMPT_TYPES = (
    "point_visual_prompt_mask",
    "scribble_visual_prompt_mask",
    "box_visual_prompt_mask",
    "mask_visual_prompt_mask",
)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def media_path(repo_root: Path, path: str | Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def write_jsonl(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    os.replace(temporary_path, path)
    print(f"Wrote {len(lines)} samples to {path}")


# ---------------------------------------------------------------------------
# RefCOCO / RefCOCO+ / RefCOCOg / RefCLEF / G-RefCOCO
# ---------------------------------------------------------------------------


def refcoco_mask(refer_api, ref: dict, image_info: dict, dataset: str) -> np.ndarray:
    height, width = image_info["height"], image_info["width"]
    mask = np.zeros((height, width), dtype=np.uint8)

    if dataset == "grefcoco":
        anns = refer_api.refToAnn.get(ref["ref_id"], [])
        if anns and anns[0] is not None:
            for ann in anns:
                if ann is None or "segmentation" not in ann:
                    continue
                segmentation = ann["segmentation"]
                if not segmentation:
                    continue
                if isinstance(segmentation, list):
                    rle = mask_util.frPyObjects(segmentation, height, width)
                elif isinstance(segmentation, dict):
                    rle = segmentation
                    if not isinstance(rle["counts"], bytes):
                        if isinstance(rle["counts"], list):
                            rle = mask_util.frPyObjects(rle, height, width)
                        elif isinstance(rle["counts"], str):
                            rle["counts"] = rle["counts"].encode()
                        else:
                            raise TypeError(
                                f"Unexpected counts type: {type(rle['counts'])}"
                            )
                else:
                    continue
                decoded = mask_util.decode(rle)
                if decoded.ndim == 3:
                    decoded = np.sum(decoded, axis=2)
                mask += decoded
            mask[mask > 1] = 1
    else:
        ann = refer_api.refToAnn[ref["ref_id"]]
        if isinstance(ann["segmentation"][0], list):
            rle = mask_util.frPyObjects(ann["segmentation"], height, width)
        else:
            rle = ann["segmentation"]
            for item in rle:
                if not isinstance(item["counts"], bytes):
                    item["counts"] = item["counts"].encode()
        decoded = mask_util.decode(rle)
        if decoded.ndim == 3:
            decoded = np.sum(decoded, axis=2)
        mask = (decoded > 0).astype(np.uint8)

    return (mask * 255).astype(np.uint8)


def refcoco_image_path(dataset: str, file_name: str) -> str:
    if dataset == "refclef":
        return str(Path("datas/ref_seg_data/images/saiapr_tc-12") / file_name)
    return str(Path("datas/ref_seg_data/images/coco2014/train2014") / file_name)


def process_refcoco_ref(
    refer_api,
    ref: dict,
    image_info: dict,
    mask_dir: Path,
    repo_root: Path,
    dataset: str,
    split: str,
    write_masks: bool,
    emit_jsonl: bool,
) -> list[str]:
    mask = refcoco_mask(refer_api, ref, image_info, dataset)
    mask_name = (
        f"{image_info['file_name'].replace('.jpg', '').replace('/', '_')}"
        f"_{ref['ref_id']}.png"
    )
    mask_path = mask_dir / mask_name
    if write_masks:
        Image.fromarray(mask).save(mask_path)

    if not emit_jsonl:
        return []

    if split == "train":
        count = random.randint(1, len(ref["sentences"]))
        sentences = ref["sentences"] + random.sample(ref["sentences"], count)
    else:
        sentences = list(
            {sentence["sent"]: sentence for sentence in ref["sentences"]}.values()
        )

    lines = []
    for sentence_info in sentences:
        question = random.choice(MASK_QUESTION_LIST).format(
            categories=tag_categories([sentence_info["sent"]]),
            task_type="binary",
        )
        item = {
            "image": refcoco_image_path(dataset, image_info["file_name"]),
            "conversations": [
                {"from": "human", "value": DEFAULT_IMAGE_TOKEN + question},
                {"from": "gpt", "value": random.choice(MASK_ANSWER_LIST)},
            ],
            "seg": media_path(repo_root, mask_path),
            "sent_info": sentence_info,
        }
        lines.append(json.dumps(item))
    return lines


def prepare_refcoco(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    data_root = resolve_path(repo_root, args.data_root)
    mask_root = resolve_path(repo_root, args.mask_output_dir)
    jsonl_root = resolve_path(repo_root, args.jsonl_output_dir)
    emit_jsonl = args.split == "train" or args.jsonl_only

    for dataset in args.datasets:
        split_by = "umd" if dataset == "refcocog" else "unc"
        if dataset == "grefcoco":
            refer_api = G_REFER(str(data_root), dataset, split_by)
        else:
            refer_api = REFER(str(data_root), dataset, split_by)

        ref_ids = refer_api.getRefIds(split=args.split)
        image_ids = refer_api.getImgIds(ref_ids=ref_ids)
        images = refer_api.loadImgs(image_ids=image_ids)
        mask_dir = mask_root / f"{dataset}_{args.split}"
        if not args.jsonl_only:
            ensure_dir(mask_dir)

        all_lines = []
        random.seed(args.seed)
        for image_info in tqdm(
            images, desc=f"seg_{dataset}_{args.split}_binary.jsonl"
        ):
            refs = refer_api.imgToRefs[image_info["id"]]
            for ref in refs:
                try:
                    all_lines.extend(
                        process_refcoco_ref(
                            refer_api,
                            ref,
                            image_info,
                            mask_dir,
                            repo_root,
                            dataset,
                            args.split,
                            write_masks=not args.jsonl_only,
                            emit_jsonl=emit_jsonl,
                        )
                    )
                    if args.max_samples and len(all_lines) >= args.max_samples:
                        break
                except Exception as error:
                    print(
                        f"[ERROR] {image_info['file_name']} "
                        f"ref {ref['ref_id']}: {error}"
                    )
            if args.max_samples and len(all_lines) >= args.max_samples:
                break

        if emit_jsonl:
            output_path = jsonl_root / f"seg_{dataset}_{args.split}_binary.jsonl"
            write_jsonl(output_path, all_lines)
        else:
            print(f"Wrote benchmark masks to {mask_dir}")


# ---------------------------------------------------------------------------
# ReasonSeg
# ---------------------------------------------------------------------------


def create_reason_mask(height: int, width: int, shapes: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        label = shape["label"].strip().lower()
        if label == "flag":
            continue
        points = np.array([shape["points"]], dtype=np.int32)
        cv2.fillPoly(mask, points, 0 if "ignore" in label else 1)
    return mask


def load_reason_image(image_path: Path) -> np.ndarray:
    with image_path.open("rb") as handle:
        image_bytes = handle.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return image


def process_reason_image(
    image_path: Path,
    annotation_path: Path,
    mask_dir: Path,
    repo_root: Path,
    explanations: dict,
    write_masks: bool,
    emit_jsonl: bool,
    repeat: int,
) -> list[str]:
    if not annotation_path.is_file():
        return []
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    image = load_reason_image(image_path)
    height, width = image.shape[:2]
    shapes = [
        shape
        for shape in annotation.get("shapes", [])
        if shape["label"].lower() != "flag"
    ]
    if not shapes:
        return []

    mask_path = mask_dir / f"{image_path.stem}.png"
    mask = create_reason_mask(height, width, shapes)
    if write_masks:
        if not cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8)):
            raise ValueError(f"Failed to save mask: {mask_path}")
        saved_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if saved_mask is None or saved_mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                "Mask/image size mismatch after save: "
                f"image={image.shape[:2]} "
                f"mask={None if saved_mask is None else saved_mask.shape[:2]} "
                f"image_path={image_path} mask_path={mask_path}"
            )

    if not emit_jsonl:
        return []

    explanation = explanations.get(image_path.name)
    is_sentence = annotation.get("is_sentence", True)
    lines = []
    for text in annotation.get("text", []):
        questions = LONG_QUESTIONS if is_sentence else SHORT_QUESTIONS
        question = random.choice(questions)
        answer = random.choice(MASK_ANSWER_LIST)
        if explanation and random.random() < 0.5:
            question = random.choice(EXPLANATORY_QUESTIONS)
            answer = f"<explain>{explanation} {random.choice(MASK_ANSWER_LIST)}"
        item = {
            "image": media_path(repo_root, image_path),
            "conversations": [
                {
                    "from": "human",
                    "value": DEFAULT_IMAGE_TOKEN
                    + question.format(sent=tag_categories([text.strip()])),
                },
                {"from": "gpt", "value": answer},
            ],
            "seg": media_path(repo_root, mask_path),
        }
        line = json.dumps(item)
        lines.extend([line] * repeat)
    return lines


def prepare_reasonseg(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    data_root = resolve_path(repo_root, args.data_root)
    mask_root = resolve_path(repo_root, args.mask_output_dir)
    jsonl_root = resolve_path(repo_root, args.jsonl_output_dir)
    emit_jsonl = args.split == "train" or args.jsonl_only
    repeat = args.repeat if args.split == "train" else 1

    mask_dir = mask_root / args.split
    if not args.jsonl_only:
        ensure_dir(mask_dir)

    explanations = {}
    explanation_path = data_root / "explanatory" / f"{args.split}.json"
    if explanation_path.is_file():
        with explanation_path.open("r", encoding="utf-8") as handle:
            explanations = {
                item["image"]: item["outputs"] for item in json.load(handle)
            }

    all_lines = []
    image_dir = data_root / args.split
    image_names = [name for name in os.listdir(image_dir) if name.endswith(".jpg")]
    for image_name in tqdm(image_names, desc=f"Processing {args.split} split"):
        image_path = image_dir / image_name
        annotation_path = image_path.with_suffix(".json")
        try:
            all_lines.extend(
                process_reason_image(
                    image_path,
                    annotation_path,
                    mask_dir,
                    repo_root,
                    explanations,
                    write_masks=not args.jsonl_only,
                    emit_jsonl=emit_jsonl,
                    repeat=repeat,
                )
            )
            if args.max_samples and len(all_lines) >= args.max_samples:
                break
        except Exception as error:
            print(f"[ERROR] {image_name}: {error}")

    if emit_jsonl:
        suffix = "" if repeat == 1 else f"_repeat{repeat}"
        output_path = jsonl_root / f"seg_reason_{args.split}{suffix}.jsonl"
        write_jsonl(output_path, all_lines)
    else:
        print(f"Wrote benchmark masks to {mask_dir}")


# ---------------------------------------------------------------------------
# COCO-Interactive
# ---------------------------------------------------------------------------


def decode_interactive_mask(segmentation, height: int, width: int) -> np.ndarray:
    binary_mask = np.zeros((height, width), dtype=np.int8)
    if isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            segmentation = mask_util.frPyObjects(segmentation, *segmentation["size"])
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
        mask = mask_util.decode(segmentation).astype(np.int8)
        binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], dict):
        for segment in segmentation:
            mask = mask_util.decode(segment).astype(np.int8)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], list):
        for segment in segmentation:
            rles = mask_util.frPyObjects([segment], height, width)
            rle = mask_util.merge(rles)
            mask = mask_util.decode(rle)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    else:
        raise ValueError(f"Invalid segmentation type: {type(segmentation)}")
    return binary_mask


def draw_circle(mask: np.ndarray, center: tuple[int, int], radius: int) -> None:
    y, x = np.ogrid[: mask.shape[0], : mask.shape[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[distance <= radius] = 1


def enhance_with_circles(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    binary_mask = np.asarray(binary_mask).astype(np.uint8)
    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for point in np.argwhere(binary_mask == 1):
        draw_circle(output_mask, (point[0], point[1]), radius)
    return output_mask


def process_interactive_annotation(
    ann: dict,
    image_info: dict,
    mask_dir: Path,
    repo_root: Path,
    split: str,
    write_masks: bool,
    emit_jsonl: bool,
) -> list[str]:
    height, width = image_info["height"], image_info["width"]
    segmentation_mask = decode_interactive_mask(
        ann["segmentation"], height, width
    )
    if segmentation_mask.sum() == 0:
        return []
    base_name = Path(image_info["file_name"]).stem
    mask_name = f"{base_name}_{ann['id']}_class{ann['category_id']}.png"
    mask_path = mask_dir / mask_name
    if write_masks:
        segmentation_mask = (segmentation_mask > 0).astype(np.uint8) * 255
        Image.fromarray(segmentation_mask).save(mask_path)

    lines = []
    for prompt_type in VISUAL_PROMPT_TYPES:
        if prompt_type not in ann:
            continue
        prompt_mask = decode_interactive_mask(ann[prompt_type], height, width)
        if write_masks:
            if prompt_type == "point_visual_prompt_mask":
                prompt_mask = enhance_with_circles(prompt_mask, radius=10)
            elif prompt_type == "scribble_visual_prompt_mask":
                prompt_mask = enhance_with_circles(prompt_mask, radius=5)
        if prompt_mask.sum() == 0:
            continue
        prompt_path = mask_dir / prompt_type / mask_name
        if write_masks:
            prompt_mask = (prompt_mask > 0).astype(np.uint8) * 255
            Image.fromarray(prompt_mask).save(prompt_path)
        if not emit_jsonl:
            continue

        prompt_key = prompt_type.split("_")[0]
        category = f"<{prompt_key}>{DEFAULT_IMAGE_TOKEN}"
        question = random.choice(VISION_QUESTION_LIST).format(regions=category)
        image_root = (
            "datas/gen_seg_data/coco2017"
            if split == "train"
            else "datas/inter_seg_data/coco2017"
        )
        image_path = str(
            Path(image_root) / f"{split}2017" / image_info["file_name"]
        )
        item = {
            "image": [image_path, media_path(repo_root, prompt_path)],
            "visual_prompt_type": prompt_type,
            "conversations": [
                {"from": "human", "value": DEFAULT_IMAGE_TOKEN + question},
                {"from": "gpt", "value": random.choice(MASK_ANSWER_LIST)},
            ],
            "seg": media_path(repo_root, mask_path),
        }
        lines.append(json.dumps(item, ensure_ascii=False))
    return lines


def process_interactive_chunk(task) -> list[str]:
    source_items, mask_dir, repo_root, split, write_masks, emit_jsonl = task
    mask_dir = Path(mask_dir)
    repo_root = Path(repo_root)
    lines = []
    for source_item in source_items:
        image_info = source_item["image_info"]
        annotations = source_item.get("annotations", source_item.get("anns", []))
        for ann in annotations:
            lines.extend(
                process_interactive_annotation(
                    ann,
                    image_info,
                    mask_dir,
                    repo_root,
                    split,
                    write_masks,
                    emit_jsonl,
                )
            )
    return lines


def prepare_coco_interactive(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    data_root = resolve_path(repo_root, args.data_root)
    mask_root = resolve_path(repo_root, args.mask_output_dir)
    jsonl_root = resolve_path(repo_root, args.jsonl_output_dir)
    emit_jsonl = args.split == "train" or args.jsonl_only

    annotation_path = (
        data_root / "annotations" / f"coco_interactive_{args.split}_psalm.json"
    )
    with annotation_path.open("r", encoding="utf-8") as handle:
        source_items = json.load(handle)

    mask_dir = mask_root / args.dataset / args.split
    if not args.jsonl_only:
        ensure_dir(mask_dir)
        for prompt_type in VISUAL_PROMPT_TYPES:
            ensure_dir(mask_dir / prompt_type)

    worker_count = 1 if args.num_workers <= 0 else min(args.num_workers, mp.cpu_count())
    if worker_count == 1:
        all_lines = process_interactive_chunk(
            (
                source_items,
                str(mask_dir),
                str(repo_root),
                args.split,
                not args.jsonl_only,
                emit_jsonl,
            )
        )
    else:
        chunk_size = int(np.ceil(len(source_items) / worker_count))
        tasks = [
            (
                source_items[index : index + chunk_size],
                str(mask_dir),
                str(repo_root),
                args.split,
                not args.jsonl_only,
                emit_jsonl,
            )
            for index in range(0, len(source_items), chunk_size)
        ]
        context = mp.get_context("spawn")
        with context.Pool(processes=worker_count) as pool:
            all_lines = []
            futures = [pool.apply_async(process_interactive_chunk, (task,)) for task in tasks]
            pool.close()
            for future in tqdm(futures, desc="Collecting results"):
                all_lines.extend(future.get())

    if args.max_samples:
        all_lines = all_lines[: args.max_samples]
    if emit_jsonl:
        output_name = (
            f"seg_{args.dataset}.jsonl"
            if args.split == "train"
            else f"seg_coco_interactive_{args.split}_psalm.jsonl"
        )
        output_path = jsonl_root / output_name
        write_jsonl(output_path, all_lines)
    else:
        print(f"Wrote benchmark masks to {mask_dir}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--jsonl-output-dir",
        default="jsonl_generate/train_jsonls/segmentation",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Generate JSONL without writing or checking masks.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    refcoco = subparsers.add_parser("refcoco")
    add_common_arguments(refcoco)
    refcoco.add_argument("--data-root", default="datas/ref_seg_data")
    refcoco.add_argument(
        "--mask-output-dir",
        default="datas/ref_seg_data/ref_seg/binary_masks",
    )
    refcoco.add_argument(
        "--datasets",
        nargs="+",
        choices=("refcoco", "refcoco+", "refcocog", "refclef", "grefcoco"),
        default=("refcoco", "refcoco+", "refcocog", "refclef", "grefcoco"),
    )
    refcoco.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test", "testA", "testB"),
    )
    refcoco.add_argument("--seed", type=int, default=1453)
    refcoco.set_defaults(func=prepare_refcoco)

    reasonseg = subparsers.add_parser("reasonseg")
    add_common_arguments(reasonseg)
    reasonseg.add_argument("--data-root", default="datas/rea_seg_data")
    reasonseg.add_argument(
        "--mask-output-dir",
        default="datas/rea_seg_data/rea_seg/binary_masks",
    )
    reasonseg.add_argument("--split", default="train", choices=("train", "val", "test"))
    reasonseg.add_argument("--repeat", type=int, default=100)
    reasonseg.set_defaults(func=prepare_reasonseg)

    interactive = subparsers.add_parser("coco-interactive")
    add_common_arguments(interactive)
    interactive.add_argument("--data-root", default="datas/inter_seg_data")
    interactive.add_argument(
        "--mask-output-dir",
        default="datas/inter_seg_data/inter_seg/binary_masks",
    )
    interactive.add_argument("--dataset", default="coco_interactive_psalm")
    interactive.add_argument("--split", default="train", choices=("train", "val"))
    interactive.add_argument("--num-workers", type=int, default=0)
    interactive.set_defaults(func=prepare_coco_interactive)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
