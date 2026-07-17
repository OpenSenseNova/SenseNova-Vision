#!/usr/bin/env python3
"""Generate COCO annotations from local binary mask datasets."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
class BaseDataset:
    name: str
    image_dirs: Sequence[str]
    mask_dirs: Sequence[str]
    mask_tokens: Sequence[str]

    def is_image_candidate(self, path: str) -> bool:
        raise NotImplementedError

    def is_mask_candidate(self, path: str) -> bool:
        raise NotImplementedError

    def decode_regions(self, array: np.ndarray) -> Iterator[Tuple[str, np.ndarray, int]]:
        raise NotImplementedError


class DoorsDataset(BaseDataset):
    name = "DOORS"
    image_dirs = ("Segmentation/DS1/DS/img",)
    mask_dirs = ("Segmentation/DS1/DS/mask",)
    mask_tokens = ("mask", "masks", "label", "labels", "annotation", "annotations")
    category = "boulder"

    def is_image_candidate(self, path: str) -> bool:
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            return False
        stem = Path(path).stem.lower()
        return not any(
            re.search(rf"(?:^|[._-]){re.escape(token)}(?:$|[._-])", stem)
            for token in self.mask_tokens
        )

    def is_mask_candidate(self, path: str) -> bool:
        return Path(path).suffix.lower() in MASK_EXTENSIONS

    def decode_regions(self, array: np.ndarray) -> Iterator[Tuple[str, np.ndarray, int]]:
        binary = np.any(array[..., :3] != 0, axis=2) if array.ndim == 3 else array != 0
        if binary.any():
            yield self.category, binary, 1


DATASETS: Dict[str, BaseDataset] = {
    "DOORS": DoorsDataset(),
}


def relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def iter_files(root: str) -> Iterator[str]:
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            yield str(Path(directory) / filename)


def choose_root(data_root: str, explicit: Optional[str], candidates: Sequence[str], label: str) -> str:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path(data_root) / path
        if not path.is_dir():
            raise NotADirectoryError(f"{label} does not exist or is not a directory: {path}")
        return str(path)

    for candidate in candidates:
        path = Path(data_root) / candidate
        if path.is_dir():
            return str(path)

    raise FileNotFoundError(
        f"Cannot find {label} under {data_root}; tried {list(candidates)}. "
        f"Pass --{label.replace('_', '-')} explicitly."
    )


def normalized_stem(path: str, tokens: Sequence[str]) -> str:
    stem = Path(path).stem.lower()
    for token in tokens:
        stem = re.sub(rf"(?:[._-]?{re.escape(token)})$", "", stem, flags=re.IGNORECASE)
    return stem


def relative_key(path: str, root: str, tokens: Sequence[str]) -> str:
    rel = relative_path(path, root)
    return posixpath.join(posixpath.dirname(rel).lower(), normalized_stem(rel, tokens))


def build_image_index(
    image_root: str,
    dataset: BaseDataset,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    exact: Dict[str, str] = {}
    by_stem: Dict[str, List[str]] = {}
    for path in iter_files(image_root):
        if not dataset.is_image_candidate(path):
            continue
        exact[relative_key(path, image_root, ())] = path
        by_stem.setdefault(Path(path).stem.lower(), []).append(path)
    return exact, by_stem


def find_image_for_mask(
    mask_path: str,
    mask_root: str,
    image_root: str,
    dataset: BaseDataset,
    exact: Mapping[str, str],
    by_stem: Mapping[str, List[str]],
) -> str:
    key = relative_key(mask_path, mask_root, dataset.mask_tokens)
    if key in exact:
        return exact[key]

    candidates = by_stem.get(normalized_stem(mask_path, dataset.mask_tokens), [])
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(f"No unique image match for mask {mask_path} (key={key!r})")


def rle_geometry(binary: np.ndarray) -> Tuple[Dict[str, Any], float, List[float]]:
    encoded = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded, float(mask_utils.area(encoded)), [float(value) for value in mask_utils.toBbox(encoded).tolist()]


def convert(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = DATASETS[args.dataset]
    data_root = str(Path(args.data_root).expanduser().resolve())
    if not Path(data_root).is_dir():
        raise NotADirectoryError(f"--data-root does not exist or is not a directory: {data_root}")

    image_root = choose_root(data_root, args.image_root, dataset.image_dirs, "image_root")
    mask_root = choose_root(data_root, args.mask_root, dataset.mask_dirs, "mask_root")
    exact, by_stem = build_image_index(image_root, dataset)
    print(f"indexed {len(exact)} {dataset.name} images under {image_root}")

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = []
    image_ids: Dict[str, int] = {}
    category_ids: Dict[str, int] = {}
    visited = skipped = empty_masks = 0

    for mask_path in iter_files(mask_root):
        if not dataset.is_mask_candidate(mask_path):
            continue
        if args.limit >= 0 and visited >= args.limit:
            break
        visited += 1

        try:
            image_path = find_image_for_mask(mask_path, mask_root, image_root, dataset, exact, by_stem)
            with Image.open(mask_path) as mask_image:
                array = np.asarray(mask_image)
                width, height = mask_image.size
            if args.read_image_size:
                with Image.open(image_path) as image:
                    width, height = image.size

            image_id = image_ids.get(image_path)
            if image_id is None:
                image_id = len(images) + 1
                image_ids[image_path] = image_id
                images.append({
                    "id": image_id,
                    "file_name": relative_path(image_path, data_root),
                    "width": width,
                    "height": height,
                })

            produced = 0
            for name, binary, raw_value in dataset.decode_regions(array):
                if int(binary.sum()) < args.min_area:
                    continue

                category_id = category_ids.get(name)
                if category_id is None:
                    category_id = len(categories) + 1
                    category_ids[name] = category_id
                    categories.append({"id": category_id, "name": name, "supercategory": "object"})

                segmentation, area, bbox = rle_geometry(binary)
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "mask_file": relative_path(mask_path, data_root),
                    "source_label": raw_value,
                })
                produced += 1

            if not produced:
                empty_masks += 1
        except Exception as exc:
            skipped += 1
            if not args.skip_bad:
                raise
            print(f"[{dataset.name}] skip {mask_path}: {exc}")

        if args.print_every and visited % args.print_every == 0:
            print(f"[{dataset.name}] masks={visited} annotations={len(annotations)} skipped={skipped}")

    return {
        "info": {
            "description": f"{dataset.name} converted from original local masks",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "data_root": data_root,
            "image_root": image_root,
            "mask_root": mask_root,
            "mask_mode": "binary",
            "visited_masks": visited,
            "skipped_masks": skipped,
            "empty_masks": empty_masks,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="Dataset converter to use.")
    parser.add_argument("--data-root", required=True, help="Local root directory of the downloaded dataset.")
    parser.add_argument("--image-root", help="Local image directory, absolute or relative to data root.")
    parser.add_argument("--mask-root", help="Local mask directory, absolute or relative to data root.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--read-image-size", action="store_true")
    parser.add_argument("--skip-bad", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = convert(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, ensure_ascii=False, indent=args.indent), encoding="utf-8")
    print(
        f"wrote {args.output}: images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])} categories={len(coco['categories'])} "
        f"empty_masks={coco['info']['empty_masks']} skipped={coco['info']['skipped_masks']}"
    )


if __name__ == "__main__":
    main()
