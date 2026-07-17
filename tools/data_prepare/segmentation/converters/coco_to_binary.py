#!/usr/bin/env python3
"""Build binary segmentation JSONL records from local COCO-style annotations.

The script is intentionally local-only for open-source use: it does not depend
on private storage SDKs, remote path conventions, or repository-local helper
modules. Pass one dataset key with ``--dataset``, a local annotation JSON with
``--ann``, and a binary-mask directory with ``--dst-dir``.
"""

import argparse
import io
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

DEFAULT_IMAGE_TOKEN = "<image>"

MASK_QUESTION_LIST = [
    "Please segment all {categories} in the image and output a {task_type} mask.",
    "Find the {categories} region in the image and provide the {task_type} segmentation mask.",
    "Generate a {task_type} mask for {categories} in this image.",
    "Identify and segment {categories}; return the result as a {task_type} mask.",
]

MASK_ANSWER_LIST = [
    "The requested segmentation mask is provided.",
    "Here is the segmentation mask.",
    "Done.",
]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def tag_categories(categories):
    return ", ".join(f"<p>{name}</p>" for name in categories)


COCO_CATEGORIES = [
    {"id": 1, "name": "person", "isthing": 1},
    {"id": 2, "name": "bicycle", "isthing": 1},
    {"id": 3, "name": "car", "isthing": 1},
    {"id": 4, "name": "motorcycle", "isthing": 1},
    {"id": 5, "name": "airplane", "isthing": 1},
    {"id": 6, "name": "bus", "isthing": 1},
    {"id": 7, "name": "train", "isthing": 1},
    {"id": 8, "name": "truck", "isthing": 1},
    {"id": 9, "name": "boat", "isthing": 1},
    {"id": 10, "name": "traffic light", "isthing": 1},
    {"id": 11, "name": "fire hydrant", "isthing": 1},
    {"id": 13, "name": "stop sign", "isthing": 1},
    {"id": 14, "name": "parking meter", "isthing": 1},
    {"id": 15, "name": "bench", "isthing": 1},
    {"id": 16, "name": "bird", "isthing": 1},
    {"id": 17, "name": "cat", "isthing": 1},
    {"id": 18, "name": "dog", "isthing": 1},
    {"id": 19, "name": "horse", "isthing": 1},
    {"id": 20, "name": "sheep", "isthing": 1},
    {"id": 21, "name": "cow", "isthing": 1},
    {"id": 22, "name": "elephant", "isthing": 1},
    {"id": 23, "name": "bear", "isthing": 1},
    {"id": 24, "name": "zebra", "isthing": 1},
    {"id": 25, "name": "giraffe", "isthing": 1},
    {"id": 27, "name": "backpack", "isthing": 1},
    {"id": 28, "name": "umbrella", "isthing": 1},
    {"id": 31, "name": "handbag", "isthing": 1},
    {"id": 32, "name": "tie", "isthing": 1},
    {"id": 33, "name": "suitcase", "isthing": 1},
    {"id": 34, "name": "frisbee", "isthing": 1},
    {"id": 35, "name": "skis", "isthing": 1},
    {"id": 36, "name": "snowboard", "isthing": 1},
    {"id": 37, "name": "sports ball", "isthing": 1},
    {"id": 38, "name": "kite", "isthing": 1},
    {"id": 39, "name": "baseball bat", "isthing": 1},
    {"id": 40, "name": "baseball glove", "isthing": 1},
    {"id": 41, "name": "skateboard", "isthing": 1},
    {"id": 42, "name": "surfboard", "isthing": 1},
    {"id": 43, "name": "tennis racket", "isthing": 1},
    {"id": 44, "name": "bottle", "isthing": 1},
    {"id": 46, "name": "wine glass", "isthing": 1},
    {"id": 47, "name": "cup", "isthing": 1},
    {"id": 48, "name": "fork", "isthing": 1},
    {"id": 49, "name": "knife", "isthing": 1},
    {"id": 50, "name": "spoon", "isthing": 1},
    {"id": 51, "name": "bowl", "isthing": 1},
    {"id": 52, "name": "banana", "isthing": 1},
    {"id": 53, "name": "apple", "isthing": 1},
    {"id": 54, "name": "sandwich", "isthing": 1},
    {"id": 55, "name": "orange", "isthing": 1},
    {"id": 56, "name": "broccoli", "isthing": 1},
    {"id": 57, "name": "carrot", "isthing": 1},
    {"id": 58, "name": "hot dog", "isthing": 1},
    {"id": 59, "name": "pizza", "isthing": 1},
    {"id": 60, "name": "donut", "isthing": 1},
    {"id": 61, "name": "cake", "isthing": 1},
    {"id": 62, "name": "chair", "isthing": 1},
    {"id": 63, "name": "couch", "isthing": 1},
    {"id": 64, "name": "potted plant", "isthing": 1},
    {"id": 65, "name": "bed", "isthing": 1},
    {"id": 67, "name": "dining table", "isthing": 1},
    {"id": 70, "name": "toilet", "isthing": 1},
    {"id": 72, "name": "tv", "isthing": 1},
    {"id": 73, "name": "laptop", "isthing": 1},
    {"id": 74, "name": "mouse", "isthing": 1},
    {"id": 75, "name": "remote", "isthing": 1},
    {"id": 76, "name": "keyboard", "isthing": 1},
    {"id": 77, "name": "cell phone", "isthing": 1},
    {"id": 78, "name": "microwave", "isthing": 1},
    {"id": 79, "name": "oven", "isthing": 1},
    {"id": 80, "name": "toaster", "isthing": 1},
    {"id": 81, "name": "sink", "isthing": 1},
    {"id": 82, "name": "refrigerator", "isthing": 1},
    {"id": 84, "name": "book", "isthing": 1},
    {"id": 85, "name": "clock", "isthing": 1},
    {"id": 86, "name": "vase", "isthing": 1},
    {"id": 87, "name": "scissors", "isthing": 1},
    {"id": 88, "name": "teddy bear", "isthing": 1},
    {"id": 89, "name": "hair drier", "isthing": 1},
    {"id": 90, "name": "toothbrush", "isthing": 1},
]

DATASET_CONFIG = {
    "vizwiz": dict(
        img="images",
    ),
    "DOORS": dict(
        img="./",
    ),
    "VOS2022": dict(
        img="train/JPEGImages",
    ),
}


def safe_category_name(cat_name):
    return cat_name.replace(" ", "-").replace("/", "-")


def log(msg):
    print(msg, flush=True)


def is_remote_path(path):
    return isinstance(path, str) and "://" in path


def read_image_from_path(path, mode="RGB"):
    if is_remote_path(path):
        raise ValueError(
            f"Remote paths are not supported by this open-source local script: {path}"
        )

    if not os.path.isfile(path):
        return None
    with Image.open(path) as img:
        return img.convert(mode).copy() if mode is not None else img.copy()


def read_mask_from_path(path):
    mask_img = read_image_from_path(path, mode="L")
    if mask_img is None:
        return None
    return np.array(mask_img)


def decode_rle_mask(rle_obj):
    if isinstance(rle_obj.get("counts"), list):
        rle_obj = mask_utils.frPyObjects(rle_obj, rle_obj["size"][0], rle_obj["size"][1])
    mask = mask_utils.decode(rle_obj)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8) * 255


def decode_segmentation_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        valid_polygons = [
            poly for poly in segmentation
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0
        ]
        if not valid_polygons:
            return None

        rles = mask_utils.frPyObjects(valid_polygons, height, width)
        rle = mask_utils.merge(rles)
        return decode_rle_mask(rle)

    if isinstance(segmentation, dict):
        return decode_rle_mask(segmentation)

    return None


def merge_masks(mask_list):
    if not mask_list:
        return None
    merged = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        merged = np.maximum(merged, (mask > 0).astype(np.uint8) * 255)
    return merged


def extract_masks_from_instance_annotations(anns, cat_id2name, is_thing_map, height, width):
    masks_by_name = defaultdict(list)
    positive_names = set()

    for ann in anns:
        cat_id = ann.get("category_id")
        cat_name = cat_id2name.get(cat_id)
        if cat_name is None or not is_thing_map.get(cat_id, 1):
            continue

        positive_names.add(cat_name)
        mask = None
        if ann.get("sam_mask") is not None:
            mask = decode_rle_mask(ann["sam_mask"])
        elif ann.get("segmentation") is not None:
            mask = decode_segmentation_mask(ann["segmentation"], height, width)

        if mask is not None:
            masks_by_name[cat_name].append(mask)

    merged_masks = {name: merge_masks(mask_list) for name, mask_list in masks_by_name.items()}
    return positive_names, merged_masks


def extract_masks_from_image_annotation(ann, cat_id2name, is_thing_map):
    masks_by_name = defaultdict(list)
    positive_names = set()

    for cat_name, grounding in ann.get("groundings", {}).items():
        if grounding.get("rle_masks"):
            positive_names.add(cat_name)
            for rle in grounding["rle_masks"]:
                masks_by_name[cat_name].append(decode_rle_mask(rle))

    for seg in ann.get("segments_info", []):
        cat_id = seg.get("category_id")
        cat_name = cat_id2name.get(cat_id)
        if cat_name is not None and is_thing_map.get(cat_id, 1):
            positive_names.add(cat_name)

    merged_masks = {name: merge_masks(mask_list) for name, mask_list in masks_by_name.items()}
    return positive_names, merged_masks


def load_or_create_mask(mask_src_path, mask_dst_path, generated_mask, allow_write):
    if not allow_write:
        mask = read_mask_from_path(mask_src_path)
        if mask is None:
            return None, None
        return mask, mask_src_path

    if generated_mask is not None:
        if mask_dst_path is None:
            return generated_mask, mask_src_path
        ensure_dir(os.path.dirname(mask_dst_path))
        Image.fromarray(generated_mask).save(mask_dst_path)
        return generated_mask, mask_dst_path

    mask = read_mask_from_path(mask_src_path)
    if mask is None:
        return None, None

    if mask_dst_path is None or mask_dst_path == mask_src_path:
        return mask, mask_src_path

    ensure_dir(os.path.dirname(mask_dst_path))
    if not os.path.isfile(mask_dst_path):
        Image.fromarray(mask).save(mask_dst_path)
    return mask, mask_dst_path


def build_mask_path_candidates(binary_map_folder, image_file, cat_name, data_type=None):
    image_stem, _ = os.path.splitext(image_file)
    image_basename = os.path.splitext(os.path.basename(image_file))[0]
    safe_name = safe_category_name(cat_name)

    return [
        os.path.join(binary_map_folder, f"{image_stem}_{safe_name}.png"),
        os.path.join(binary_map_folder, f"{image_basename}_{safe_name}.png"),
        os.path.join(binary_map_folder, f"{image_stem}.png"),
        os.path.join(binary_map_folder, f"{image_basename}.png"),
    ]


def resolve_mask_path(binary_map_folder, image_file, cat_name, data_type=None):
    candidates = build_mask_path_candidates(binary_map_folder, image_file, cat_name, data_type=data_type)
    for path in candidates:
        if is_remote_path(path):
            raise ValueError(f"Remote paths are not supported: {path}")
        if os.path.isfile(path):
            return path
    return candidates[0]


def to_output_path(path, data_root):
    if path is None:
        return None
    if data_root is None:
        return path

    if is_remote_path(path) or is_remote_path(data_root):
        raise ValueError("Remote paths are not supported by this local script.")
    return os.path.relpath(path, data_root)


def prepare_output_image_path(image_path, data_type):
    return image_path


def find_top_level_array_offset(json_path, key, chunk_size=64 * 1024 * 1024, show_progress=False):
    needle = f'"{key}"'.encode("utf-8")
    lookahead = len(needle) + 32

    progress_bar = None
    if show_progress:
        total_bytes = os.path.getsize(json_path)
        progress_bar = tqdm(
            total=total_bytes,
            desc=f"scan_{key}",
            unit="B",
            unit_scale=True,
            leave=False,
        )

    try:
        with open(json_path, "rb") as f:
            offset = 0
            prev = b""

            while True:
                data = f.read(chunk_size)
                if not data:
                    return None
                if progress_bar is not None:
                    progress_bar.update(len(data))

                buf = prev + data
                base_offset = offset - len(prev)
                search_pos = 0

                while True:
                    idx = buf.find(needle, search_pos)
                    if idx == -1:
                        break

                    cursor = idx + len(needle)
                    while cursor < len(buf) and buf[cursor] in b" \t\r\n":
                        cursor += 1

                    if cursor >= len(buf):
                        break
                    if buf[cursor:cursor + 1] != b":":
                        search_pos = idx + 1
                        continue

                    cursor += 1
                    while cursor < len(buf) and buf[cursor] in b" \t\r\n":
                        cursor += 1

                    if cursor >= len(buf):
                        break
                    if buf[cursor:cursor + 1] == b"[":
                        return base_offset + cursor

                    search_pos = idx + 1

                prev = buf[-lookahead:]
                offset += len(data)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def iter_json_array_from_offset(json_path, array_offset, chunk_size=8 * 1024 * 1024):
    decoder = json.JSONDecoder()

    with open(json_path, "rb") as raw_f:
        raw_f.seek(array_offset)
        with io.TextIOWrapper(raw_f, encoding="utf-8") as text_f:
            opener = text_f.read(1)
            if opener != "[":
                raise ValueError(f"Expected '[' at offset {array_offset} in {json_path}, got {opener!r}")

            buffer = ""
            eof = False

            while True:
                chunk = text_f.read(chunk_size)
                if not chunk:
                    eof = True
                buffer += chunk

                while True:
                    buffer = buffer.lstrip()
                    if not buffer:
                        break

                    if buffer[0] == "]":
                        return
                    if buffer[0] == ",":
                        buffer = buffer[1:]
                        continue

                    try:
                        obj, consumed = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        break

                    yield obj
                    buffer = buffer[consumed:]

                if eof:
                    if buffer.strip() in {"", "]"}:
                        return
                    raise ValueError(f"Unexpected trailing content while parsing {json_path}")


def iter_top_level_array(json_path, key, chunk_size=8 * 1024 * 1024, show_scan_progress=False):
    array_offset = find_top_level_array_offset(json_path, key, show_progress=show_scan_progress)
    if array_offset is None:
        return
    yield from iter_json_array_from_offset(json_path, array_offset, chunk_size=chunk_size)


def load_categories_streaming(json_path, show_scan_progress=False):
    categories = list(iter_top_level_array(json_path, "categories", show_scan_progress=show_scan_progress))
    if categories:
        cat_id2name = {cat["id"]: cat["name"] for cat in categories}
        is_thing_map = {cat["id"]: cat.get("isthing", cat.get("is_thing", 1)) for cat in categories}
    else:
        cat_id2name = {cat["id"]: cat["name"] for cat in COCO_CATEGORIES}
        is_thing_map = {cat["id"]: cat.get("isthing", 1) for cat in COCO_CATEGORIES}

    for cat in COCO_CATEGORIES:
        is_thing_map.setdefault(cat["id"], cat.get("isthing", 1))

    return cat_id2name, is_thing_map


def iter_streaming_instance_items(json_path, show_scan_progress=False):
    image_iter = iter_top_level_array(json_path, "images", show_scan_progress=show_scan_progress)
    ann_iter = iter_top_level_array(json_path, "annotations", show_scan_progress=show_scan_progress)
    current_ann = next(ann_iter, None)

    for img in image_iter:
        image_id = img["id"]
        anns = []

        while current_ann is not None and current_ann.get("image_id") == image_id:
            anns.append(current_ann)
            current_ann = next(ann_iter, None)

        if current_ann is not None and current_ann.get("image_id") < image_id:
            raise RuntimeError(
                "Streaming parser requires annotations to be grouped by image_id in the same order as images."
            )

        if not anns:
            continue

        yield {
            "mode": "instance",
            "image_id": image_id,
            "file_name": img["file_name"],
            "height": img.get("height"),
            "width": img.get("width"),
            "anns": anns,
        }

    if current_ann is not None:
        raise RuntimeError("Annotations remain after images are exhausted; image/annotation ordering is inconsistent.")


def is_annotation_image_id_monotonic(json_path, show_progress=False):
    """Check whether annotations are non-decreasing by image_id (required by strict streaming parser)."""
    prev_image_id = None
    ann_iter = iter_top_level_array(json_path, "annotations", show_scan_progress=show_progress)
    if show_progress:
        ann_iter = tqdm(ann_iter, desc="check_ann_order", unit="ann")

    for ann in ann_iter:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        if prev_image_id is not None and image_id < prev_image_id:
            return False
        prev_image_id = image_id
    return True


def load_json_file(data_path):
    if is_remote_path(data_path):
        raise ValueError(f"Remote JSON paths are not supported: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_items_non_streaming(data_path):
    coco_ann = load_json_file(data_path)
    images = coco_ann.get("images", [])
    videos = coco_ann.get("videos", [])
    categories = coco_ann.get("categories", [])
    annotations = coco_ann.get("annotations", [])

    if categories:
        cat_id2name = {cat["id"]: cat["name"] for cat in categories}
        is_thing_map = {cat["id"]: cat.get("isthing", cat.get("is_thing", 1)) for cat in categories}
    else:
        cat_id2name = {cat["id"]: cat["name"] for cat in COCO_CATEGORIES}
        is_thing_map = {cat["id"]: cat.get("isthing", 1) for cat in COCO_CATEGORIES}

    for cat in COCO_CATEGORIES:
        is_thing_map.setdefault(cat["id"], cat.get("isthing", 1))

    items = []
    sample_ann = annotations[0] if annotations else {}
    is_instance_dataset = bool(
        images and annotations and (
            "segmentation" in sample_ann or
            "sam_mask" in sample_ann
        )
    )
    is_video_instance_dataset = bool(
        videos and annotations and ("segmentations" in sample_ann)
    )

    if is_video_instance_dataset:
        vid_id2meta = {video["id"]: video for video in videos}
        anns_by_video_frame = defaultdict(list)

        for ann in annotations:
            video_id = ann.get("video_id")
            if video_id is None:
                continue
            segmentations = ann.get("segmentations") or []
            for frame_idx, segmentation in enumerate(segmentations):
                if segmentation is None:
                    continue
                anns_by_video_frame[(video_id, frame_idx)].append({
                    "category_id": ann.get("category_id"),
                    "segmentation": segmentation,
                })

        for video_id, video in vid_id2meta.items():
            file_names = video.get("file_names", [])
            for frame_idx, file_name in enumerate(file_names):
                frame_anns = anns_by_video_frame.get((video_id, frame_idx), [])
                if not frame_anns:
                    continue
                items.append({
                    "mode": "instance",
                    "video_id": video_id,
                    "file_name": file_name,
                    "height": video.get("height"),
                    "width": video.get("width"),
                    "anns": frame_anns,
                })
    elif is_instance_dataset:
        img_id2meta = {img["id"]: img for img in images}
        anns_by_image = defaultdict(list)
        for ann in annotations:
            anns_by_image[ann["image_id"]].append(ann)

        for image_id, img in img_id2meta.items():
            items.append({
                "mode": "instance",
                "image_id": image_id,
                "file_name": img["file_name"],
                "height": img.get("height"),
                "width": img.get("width"),
                "anns": anns_by_image.get(image_id, []),
            })
    else:
        img_id2file = {img["id"]: img["file_name"] for img in images} if images else {}
        for ann in annotations:
            file_name = ann.get("file_name") or img_id2file.get(ann["image_id"])
            if file_name is None:
                raise KeyError(f"Cannot resolve file_name for annotation: {ann}")
            items.append({
                "mode": "image",
                "image_id": ann["image_id"],
                "file_name": file_name,
                "height": ann.get("height"),
                "width": ann.get("width"),
                "ann": ann,
            })

    return cat_id2name, is_thing_map, items


def write_results(result, out_files, line_counts):
    for key, lines in result.items():
        out_f = out_files[key]
        for line in lines:
            out_f.write(line + "\n")
            line_counts[key] += 1


def process_annotation(item, cat_id2name, is_thing_map, image_folder, binary_mask_folder,
                       data_root, data_type, min_ratio=0.01, max_ratio=0.5):
    """Process one image, reusing or creating binary masks in dst_dir."""
    image_file = item["file_name"]
    image_path = os.path.join(image_folder, image_file) if image_folder else image_file
    results = {
        "binary": [],
    }

    if not is_remote_path(image_path) and not os.path.isfile(image_path):
        log(f"[WARN] Image not found, skip: {image_path}")
        return results

    output_image_path = prepare_output_image_path(image_path, data_type)

    H = item.get("height")
    W = item.get("width")
    if H is None or W is None:
        img = read_image_from_path(image_path, mode="RGB")
        if img is None:
            log(f"[WARN] Cannot load image to infer size, skip: {image_path}")
            return results
        W, H = img.size

    if item["mode"] == "instance":
        positive_names, generated_masks = extract_masks_from_instance_annotations(
            item["anns"], cat_id2name, is_thing_map, H, W
        )
    else:
        positive_names, generated_masks = extract_masks_from_image_annotation(
            item["ann"], cat_id2name, is_thing_map
        )

    has_positive_source = bool(generated_masks)
    if not has_positive_source:
        for cat_name in positive_names:
            src_mask_path = resolve_mask_path(binary_mask_folder, image_file, cat_name, data_type=data_type)
            if read_mask_from_path(src_mask_path) is not None:
                has_positive_source = True
                break

    if positive_names and not has_positive_source:
        raise ValueError(f"No existing binary or decodable mask found for: {image_file}")

    # ---------------- Binary JSONL ----------------
    for cat_id, cat_name in cat_id2name.items():
        if not is_thing_map.get(cat_id, 1):
            continue

        generated_mask = generated_masks.get(cat_name)
        if generated_mask is None and cat_name not in positive_names:
            continue

        src_mask_path = resolve_mask_path(binary_mask_folder, image_file, cat_name, data_type=data_type)
        dst_mask_path = resolve_mask_path(binary_mask_folder, image_file, cat_name, data_type=data_type)

        mask, output_mask_path = load_or_create_mask(
            src_mask_path,
            dst_mask_path,
            generated_mask,
            allow_write=True,
        )
        if mask is None or output_mask_path is None:
            continue
        mask = (mask > 0).astype(np.uint8) * 255

        # Filter masks by their image-area ratio.
        ratio = mask.sum() / (H * W * 255)
        if not (min_ratio <= ratio <= max_ratio):
            continue

        question = random.choice(MASK_QUESTION_LIST).format(categories=tag_categories([cat_name]), task_type="binary")
        answer = random.choice(MASK_ANSWER_LIST)
        binary_item = {
            "image": to_output_path(output_image_path, data_root),
            "conversations": [
                {"from": "human", "value": DEFAULT_IMAGE_TOKEN + question},
                {"from": "gpt", "value": answer}
            ],
            "seg": to_output_path(output_mask_path, data_root)
        }
        results["binary"].append(json.dumps(binary_item))

    return results


def process_annotation_star(args):
    return process_annotation(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List supported dataset keys and exit.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset key to convert: vizwiz, DOORS, or VOS2022.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Local dataset root. Relative ann/img/dst-dir paths are resolved from this directory.",
    )
    parser.add_argument(
        "--ann",
        type=str,
        help="Local COCO-style annotation JSON, absolute or relative to data root.",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        help="Binary mask directory, absolute or relative to data root. Existing masks are reused; generated masks are written here.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./jsonl_generate_ref",
        help="Output directory for generated JSONL files.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--min-ratio", type=float, default=0.01, help="Minimum mask area ratio.")
    parser.add_argument("--max-ratio", type=float, default=1.0, help="Maximum mask area ratio.")
    args = parser.parse_args()

    if args.list_datasets:
        for name in sorted(DATASET_CONFIG):
            print(name)
        return

    if not args.dataset:
        parser.error("--dataset is required unless --list-datasets is used.")
    if not args.ann:
        parser.error("--ann is required unless --list-datasets is used.")
    if not args.dst_dir:
        parser.error("--dst-dir is required unless --list-datasets is used.")

    dataset = args.dataset
    if dataset not in DATASET_CONFIG:
        parser.error(f"unknown dataset {dataset!r}. Use --list-datasets to see supported keys.")

    dataset_cfg = DATASET_CONFIG[dataset]
    data_root = args.data_root

    def resolve_path(path):
        if path is None:
            return None
        if is_remote_path(path):
            raise ValueError(f"Remote paths are not supported: {path}")
        if os.path.isabs(path):
            return path
        return os.path.join(data_root, path)

    data_path = resolve_path(args.ann)
    image_folder = resolve_path(dataset_cfg["img"])
    binary_mask_folder = resolve_path(args.dst_dir)

    ensure_dir(binary_mask_folder)
    ensure_dir(args.output_dir)
    random.seed(1453)

    log(
        f"[INFO] dataset={dataset}, data_path={data_path}, "
        f"num_workers={args.num_workers}"
    )
    log(
        f"[INFO] image_folder={image_folder}, binary_mask_folder={binary_mask_folder}"
    )

    out_paths = {
        "binary": os.path.join(args.output_dir, f"seg_binary_{dataset}.jsonl"),
    }

    log("[INFO] Using non-streaming parser (will load full annotation JSON into memory).")
    cat_id2name, is_thing_map, items = build_items_non_streaming(data_path)
    item_iter = iter(items)
    total_items = len(items)

    task_iter = (
        (
            item, cat_id2name, is_thing_map, image_folder, binary_mask_folder,
            data_root, dataset, args.min_ratio, args.max_ratio
        )
        for item in item_iter
    )
    line_counts = {k: 0 for k in out_paths}
    log(f"[INFO] Start processing. Output: {out_paths['binary']}")

    with ExitStack() as stack:
        out_files = {
            key: stack.enter_context(open(path, "w"))
            for key, path in out_paths.items()
        }

        if args.num_workers == 1:
            for task in tqdm(task_iter, total=total_items, desc="process", unit="item"):
                write_results(process_annotation_star(task), out_files, line_counts)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                for res in tqdm(
                    executor.map(process_annotation_star, task_iter),
                    total=total_items,
                    desc="process",
                    unit="item",
                ):
                    write_results(res, out_files, line_counts)

    for k, path in out_paths.items():
        log(f"{k} -> {line_counts[k]} lines -> {path}")


if __name__ == '__main__':
    main()
