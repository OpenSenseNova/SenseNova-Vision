#!/usr/bin/env python3
"""Normalize locally downloaded VIS2022 structured annotations to COCO."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from pycocotools import mask as mask_utils


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_explicit(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def resolve_source(root: Path, value: str) -> Path:
    path = resolve_explicit(root, value)
    if not path.is_file():
        raise FileNotFoundError(f"--source does not exist or is not a file: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"--source must be a JSON file: {path}")
    return path


def normalize_categories(
    categories: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    for category in categories:
        item = dict(category)
        item["id"] = int(item["id"])
        item["name"] = str(item.get("name", f"category_{item['id']}"))
        item.setdefault("supercategory", "object")
        result.append(item)
    return sorted(result, key=lambda item: item["id"])


def ensure_coco(coco: Mapping[str, Any], description: str) -> Dict[str, Any]:
    result = dict(coco)
    result.setdefault("info", {})
    result["info"] = dict(result["info"])
    result["info"].setdefault("description", description)
    result["info"].setdefault("date_created", datetime.now(timezone.utc).isoformat())
    result.setdefault("licenses", [])
    result["images"] = [dict(image) for image in result.get("images", [])]
    result["annotations"] = [
        dict(annotation) for annotation in result.get("annotations", [])
    ]
    result["categories"] = normalize_categories(result.get("categories", []))
    return result


def segmentation_geometry(
    segmentation: Any, height: int, width: int
) -> Tuple[Any, float, List[float]]:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        area = float(mask_utils.area(rle))
        bbox = [float(value) for value in mask_utils.toBbox(rle).tolist()]
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        return rle, area, bbox

    polygons = segmentation if isinstance(segmentation, list) else []
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    rles = mask_utils.frPyObjects(polygons, height, width)
    merged = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return (
        polygons,
        float(mask_utils.area(merged)),
        [float(value) for value in mask_utils.toBbox(merged).tolist()],
    )


def finalize_coco(coco: Dict[str, Any]) -> Dict[str, Any]:
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    for annotation_id, annotation in enumerate(coco["annotations"], 1):
        annotation["id"] = annotation_id
        annotation["image_id"] = int(annotation["image_id"])
        annotation["category_id"] = int(annotation["category_id"])
        annotation.setdefault("iscrowd", 0)
        if "segmentation" not in annotation:
            continue
        if annotation.get("area") is not None and annotation.get("bbox") is not None:
            continue

        image = image_by_id.get(annotation["image_id"], {})
        height, width = int(image.get("height", 0)), int(image.get("width", 0))
        if not height or not width:
            continue
        segmentation, area, bbox = segmentation_geometry(
            annotation["segmentation"], height, width
        )
        annotation["segmentation"] = segmentation
        if annotation.get("area") is None:
            annotation["area"] = area
        if annotation.get("bbox") is None:
            annotation["bbox"] = bbox
    return coco


def apply_annotation_limit(coco: Dict[str, Any], limit: int) -> Dict[str, Any]:
    if limit < 0:
        return coco

    coco["annotations"] = coco["annotations"][:limit]
    image_ids = {int(item["image_id"]) for item in coco["annotations"]}
    category_ids = {int(item["category_id"]) for item in coco["annotations"]}
    coco["images"] = [item for item in coco["images"] if int(item["id"]) in image_ids]
    coco["categories"] = [
        item for item in coco["categories"] if int(item["id"]) in category_ids
    ]
    return coco


class BaseDataset:
    name: str

    def convert(self, source: Path) -> Dict[str, Any]:
        raise NotImplementedError


class VIS2022Dataset(BaseDataset):
    name = "VIS2022"

    def convert(self, source: Path) -> Dict[str, Any]:
        value = read_json(source)
        if "videos" not in value:
            return ensure_coco(value, self.name)

        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        frame_ids: Dict[Tuple[int, int], int] = {}

        for video in value["videos"]:
            video_id = int(video["id"])
            for frame_index, file_name in enumerate(video.get("file_names", [])):
                image_id = len(images) + 1
                frame_ids[(video_id, frame_index)] = image_id
                images.append(
                    {
                        "id": image_id,
                        "file_name": str(file_name),
                        "width": int(video.get("width", 0)),
                        "height": int(video.get("height", 0)),
                        "video_id": video_id,
                        "frame_id": frame_index,
                    }
                )

        for track in value.get("annotations", []):
            video_id = int(track["video_id"])
            segmentations = track.get("segmentations", [])
            bboxes = track.get("bboxes", [])
            areas = track.get("areas", [])
            for frame_index, segmentation in enumerate(segmentations):
                if not segmentation:
                    continue
                item: Dict[str, Any] = {
                    "image_id": frame_ids[(video_id, frame_index)],
                    "category_id": int(track["category_id"]),
                    "segmentation": segmentation,
                    "iscrowd": int(track.get("iscrowd", 0)),
                    "track_id": int(track["id"]),
                }
                if frame_index < len(bboxes) and bboxes[frame_index] is not None:
                    item["bbox"] = bboxes[frame_index]
                if frame_index < len(areas) and areas[frame_index] is not None:
                    item["area"] = areas[frame_index]
                annotations.append(item)

        return ensure_coco(
            {
                "images": images,
                "annotations": annotations,
                "categories": value.get("categories", []),
            },
            self.name,
        )


DATASETS: Dict[str, BaseDataset] = {
    "VIS2022": VIS2022Dataset(),
}


def convert(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(
            f"--data-root does not exist or is not a directory: {data_root}"
        )

    dataset = DATASETS[args.dataset]
    source = resolve_source(data_root, args.source)
    coco = dataset.convert(source)
    return finalize_coco(apply_annotation_limit(coco, args.limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--data-root",
        required=True,
        help="Local root of the downloaded VIS2022 dataset.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local VIS2022 annotation JSON, absolute or relative to data root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--indent", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = convert(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(coco, ensure_ascii=False, indent=args.indent), encoding="utf-8"
    )
    print(
        f"wrote {args.output}: images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])} categories={len(coco['categories'])}"
    )


if __name__ == "__main__":
    main()
