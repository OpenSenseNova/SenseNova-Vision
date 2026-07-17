import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "keypoint_datasets.json"


def load_presets(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError("keypoint config requires default_dataset and datasets")
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else TOOL_ROOT / value)


def prefixed_path(prefix: str, value: str) -> str:
    value = value.replace("\\", "/")
    if prefix:
        return f"{prefix.rstrip('/')}/{value.lstrip('/')}"
    return value.lstrip("/")


def normalize_image_name(raw_name: str, preset: dict) -> str:
    name = raw_name.replace("\\", "/")
    if ":" in name:
        name = name.split(":", 1)[1]
    name = name.lstrip("/")
    strip_prefix = preset.get("strip_image_prefix", "")
    if strip_prefix and name.startswith(strip_prefix):
        name = name[len(strip_prefix) :].lstrip("/")
    tail_components = preset.get("image_tail_components")
    if tail_components:
        name = "/".join(name.split("/")[-int(tail_components) :])
    return prefixed_path(preset.get("image_prefix", ""), name)


def flatten_keypoints(values: list, count: int | None = None) -> list:
    if count is None:
        count = len(values) // 3
    result = []
    for index in range(count):
        result.extend([values[index * 3], values[index * 3 + 1]])
    return result


def make_record(image_name: str, width: int, height: int, boxes: list, keypoints: list):
    annotation = {"keypoints": keypoints}
    if boxes:
        annotation["boxes"] = boxes
    return {
        "image_name": image_name,
        "image_info": {"width": width, "height": height},
        "annotation": annotation,
    }


def iter_coco_keypoints(root: str, preset: dict):
    images = {}
    categories = {}
    annotations = []
    seen_annotation_ids = set()
    for relative_path in preset["annotation_files"]:
        with open(os.path.join(root, relative_path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        images.update({item["id"]: item for item in data["images"]})
        categories.update({item["id"]: item["name"] for item in data["categories"]})
        for annotation in data["annotations"]:
            annotation_id = annotation.get("id")
            if preset.get("deduplicate_annotation_ids") and annotation_id in seen_annotation_ids:
                continue
            seen_annotation_ids.add(annotation_id)
            annotations.append(annotation)

    by_image = defaultdict(list)
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)

    for image_id, image in images.items():
        image_annotations = by_image.get(image_id, [])
        if not image_annotations:
            continue
        image_name = normalize_image_name(image["file_name"], preset)
        if preset.get("require_images") and not os.path.isfile(os.path.join(root, image_name)):
            continue

        boxes = []
        keypoints = []
        for annotation in image_annotations:
            values = annotation.get(preset.get("keypoints_field", "keypoints")) or []
            count = preset.get("keypoint_count")
            if len(values) < 3 * (count or 1):
                continue
            phrase = preset.get("fixed_phrase") or categories.get(
                annotation.get("category_id"), ""
            )
            if not phrase:
                continue
            flat = flatten_keypoints(values, count)
            keypoints.append({"keypoint": flat, "phrase": phrase})
            if preset.get("include_boxes", True):
                x, y, width, height = annotation.get("bbox", [0, 0, 0, 0])
                if width > 0 and height > 0:
                    boxes.append(
                        {"bbox": [x, y, x + width, y + height], "phrase": phrase}
                    )
                else:
                    boxes.append({"bbox": [0, 0, 0, 0], "phrase": phrase})
        if keypoints and (not boxes or len(boxes) == len(keypoints)):
            yield make_record(
                image_name, image["width"], image["height"], boxes, keypoints
            )


def iter_macaque_csv(root: str, preset: dict):
    csv_path = os.path.join(root, preset["annotation_file"])
    image_root = os.path.join(root, preset["image_root"])
    expected_count = len(preset["keypoint_names"])
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            image_leaf = row[0].strip()
            image_path = os.path.join(image_root, image_leaf)
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except OSError:
                continue
            try:
                instances = json.loads(row[1])
            except json.JSONDecodeError:
                continue

            keypoints = []
            for instance in instances:
                points = [item.get("position") for item in instance]
                if len(points) != expected_count or any(
                    not point or len(point) != 2 for point in points
                ):
                    continue
                flat = [coordinate for point in points for coordinate in point]
                keypoints.append(
                    {"keypoint": flat, "phrase": preset.get("fixed_phrase", "macaque")}
                )
            if keypoints:
                yield make_record(
                    prefixed_path(preset.get("image_prefix", ""), image_leaf),
                    width,
                    height,
                    [],
                    keypoints,
                )


def iter_mpii(root: str, preset: dict):
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("the MPII adapter requires scipy") from error

    mat = loadmat(
        os.path.join(root, preset["annotation_file"]), struct_as_record=False
    )
    release = mat["RELEASE"][0, 0]
    image_root = os.path.join(root, preset["image_root"])
    point_count = len(preset["keypoint_names"])
    for index in range(release.annolist.shape[1]):
        if release.img_train[0, index] == 0:
            continue
        item = release.annolist[0, index]
        rectangles = item.annorect
        if rectangles.size == 0 or rectangles.shape[1] == 0:
            continue
        image_leaf = item.image[0, 0].name[0]
        try:
            with Image.open(os.path.join(image_root, image_leaf)) as image:
                width, height = image.size
        except OSError:
            continue

        keypoints = []
        for rectangle_index in range(rectangles.shape[1]):
            rectangle = rectangles[0, rectangle_index]
            if not hasattr(rectangle, "annopoints") or rectangle.annopoints.size == 0:
                continue
            points = rectangle.annopoints[0, 0].point
            point_map = {}
            for point_index in range(points.shape[1]):
                point = points[0, point_index]
                point_map[int(point.id[0, 0])] = [
                    float(point.x[0, 0]),
                    float(point.y[0, 0]),
                ]
            flat = []
            for point_index in range(point_count):
                flat.extend(point_map.get(point_index, [0, 0]))
            if any(flat):
                keypoints.append(
                    {"keypoint": flat, "phrase": preset.get("fixed_phrase", "person")}
                )
        if keypoints:
            yield make_record(
                prefixed_path(preset.get("image_prefix", ""), image_leaf),
                width,
                height,
                [],
                keypoints,
            )


def iter_ochuman(root: str, preset: dict):
    with open(os.path.join(root, preset["annotation_file"]), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    phrase = preset.get("fixed_phrase", "person")
    for image in data["images"]:
        boxes = []
        keypoints = []
        valid = True
        for annotation in image.get("annotations", []):
            values = annotation.get("keypoints")
            if values is None:
                valid = False
                break
            x, y, width, height = annotation["bbox"]
            boxes.append(
                {"bbox": [x, y, x + width, y + height], "phrase": phrase}
            )
            keypoints.append(
                {"keypoint": flatten_keypoints(values), "phrase": phrase}
            )
        if valid and keypoints:
            yield make_record(
                prefixed_path(preset.get("image_prefix", ""), image["file_name"]),
                image["width"],
                image["height"],
                boxes,
                keypoints,
            )


ADAPTERS = {
    "coco_keypoints": iter_coco_keypoints,
    "macaque_csv": iter_macaque_csv,
    "mpii_mat": iter_mpii,
    "ochuman": iter_ochuman,
}


def write_records(records, output_path: str, limit: int | None) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            if limit is not None and count >= limit:
                break
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw keypoint annotations to common keypoint JSONL."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    default_dataset, datasets = load_presets(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in datasets:
        parser.error(f"unknown dataset {dataset_name!r}; choose one of {sorted(datasets)}")
    preset = datasets[dataset_name]
    adapter = preset.get("adapter")
    if adapter not in ADAPTERS:
        parser.error(f"unsupported adapter {adapter!r}")
    root = args.input or resolve_tool_path(preset["default_input"])
    output = args.output or resolve_tool_path(preset["default_intermediate"])
    count = write_records(ADAPTERS[adapter](root, preset), output, args.limit)
    print(f"Wrote {count} common keypoint records to {output}")


if __name__ == "__main__":
    main()
