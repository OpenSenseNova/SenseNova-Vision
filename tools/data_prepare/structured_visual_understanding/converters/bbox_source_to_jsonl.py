import argparse
import glob
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "bbox_source_datasets.json"


def load_config(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError(
            "bbox source config must contain string 'default_dataset' and object "
            "'datasets'"
        )
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    for name, preset in datasets.items():
        missing = {"adapter", "default_input", "default_output"}.difference(preset)
        if missing:
            raise ValueError(f"dataset {name!r} is missing fields: {sorted(missing)}")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def prefixed_path(prefix: str, name: str) -> str:
    if not prefix:
        return name.replace(os.sep, "/")
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}".replace(os.sep, "/")


def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def iter_json_array(path: str, chunk_size: int = 1024 * 1024):
    """Stream objects from a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False

    with open(path, "r", encoding="utf-8") as handle:
        while True:
            if position >= len(buffer):
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                buffer = chunk
                position = 0

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array: {path}")
                position += 1
                started = True

            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[position:] + chunk
                    position = 0
                    continue
                yield item
                position = end

                if position > chunk_size:
                    buffer = buffer[position:]
                    position = 0
                if position >= len(buffer):
                    break


def iter_json_items(path: str, input_format: str):
    if input_format == "json_array":
        yield from iter_json_array(path)
        return
    if input_format == "jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    raise ValueError(f"unsupported JSON input format: {input_format!r}")


def make_record(
    image_name: str,
    width: int,
    height: int,
    boxes: list[dict],
    image_field: str = "image_name",
) -> dict:
    return {
        image_field: image_name,
        "annotation": {"boxes": boxes},
        "image_info": {"width": width, "height": height},
    }


def clean_phrase(phrase: str, preset: dict) -> str:
    phrase = phrase.strip()
    if preset.get("strip_trailing_digits"):
        phrase = re.sub(r"\d+$", "", phrase)
    return preset.get("category_aliases", {}).get(phrase, phrase)


def iter_coco(dataset_root: str, preset: dict):
    excluded = set(preset.get("excluded_categories", []))
    for annotation_file in preset["annotation_files"]:
        annotation_path = os.path.join(dataset_root, annotation_file)
        with open(annotation_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = {
            item["id"]: clean_phrase(str(item["name"]), preset)
            for item in coco.get("categories", [])
        }
        image_file_field = preset.get("image_file_field", "file_name")
        annotation_phrase_field = preset.get("annotation_phrase_field")
        bbox_format = preset.get("bbox_format", "xywh")
        if bbox_format not in {"xywh", "xyxy"}:
            raise ValueError(f"unsupported COCO bbox format: {bbox_format!r}")
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for image in coco["images"]:
            image_leaf = image[image_file_field]
            if preset.get("require_images", False):
                source_image = os.path.join(
                    dataset_root,
                    preset.get("image_root", ""),
                    image_leaf,
                )
                if not os.path.isfile(source_image):
                    continue
            boxes = []
            for ann in anns_by_image.get(image["id"], []):
                if annotation_phrase_field:
                    phrase = clean_phrase(
                        str(ann.get(annotation_phrase_field, "")), preset
                    )
                else:
                    phrase = categories.get(ann.get("category_id"), "")
                if not phrase or phrase in excluded:
                    continue
                x, y, width, height = ann["bbox"]
                if bbox_format == "xyxy":
                    x1, y1 = width, height
                else:
                    x1, y1 = x + width, y + height
                if x1 <= x or y1 <= y:
                    continue
                boxes.append({"bbox": [x, y, x1, y1], "phrase": phrase})
            if not boxes:
                continue
            yield make_record(
                prefixed_path(preset.get("image_prefix", ""), image_leaf),
                image["width"],
                image["height"],
                boxes,
            )


def iter_unified_bbox(dataset_root: str, preset: dict):
    """Normalize JSON/JSONL records containing ``annotation.boxes``.

    The source may already use absolute ``xyxy`` or absolute ``xywh`` boxes.
    Image dimensions are copied when present and otherwise read from the
    original image. This adapter deliberately stops at the common bbox JSONL
    boundary; prompt construction remains in ``bbox_to_jsonl.py``.
    """
    input_format = preset.get("input_format", "jsonl")
    bbox_format = preset.get("bbox_format", "xyxy")
    if bbox_format not in {"xyxy", "xywh"}:
        raise ValueError(f"unsupported bbox format: {bbox_format!r}")

    for annotation_file in preset["annotation_files"]:
        annotation_path = os.path.join(dataset_root, annotation_file)
        for item in iter_json_items(annotation_path, input_format):
            image_name = item.get("image_name", item.get("image_path"))
            if not image_name:
                continue
            source_image = os.path.join(
                dataset_root, preset.get("image_root", ""), image_name
            )
            if preset.get("require_images", False) and not os.path.isfile(source_image):
                continue

            image_info = item.get("image_info") or {}
            width, height = image_info.get("width"), image_info.get("height")
            if not width or not height:
                try:
                    width, height = image_size(source_image)
                except (OSError, ValueError):
                    continue

            boxes = []
            for box in item.get("annotation", {}).get("boxes", []):
                phrase = clean_phrase(str(box.get("phrase", "")), preset)
                coordinates = box.get("bbox")
                if not phrase or not isinstance(coordinates, list) or len(coordinates) != 4:
                    continue
                x0, y0, x1, y1 = coordinates
                if bbox_format == "xywh":
                    x1, y1 = x0 + x1, y0 + y1
                if x1 <= x0 or y1 <= y0:
                    continue
                boxes.append({"bbox": [x0, y0, x1, y1], "phrase": phrase})
            if boxes:
                record = make_record(
                    prefixed_path(preset.get("image_prefix", ""), image_name),
                    width,
                    height,
                    boxes,
                    image_field=preset.get("output_image_field", "image_name"),
                )
                source_annotation = item.get("annotation", {})
                for field in preset.get("preserve_annotation_fields", []):
                    if field in source_annotation:
                        record["annotation"][field] = source_annotation[field]
                yield record


def iter_yolo(dataset_root: str, preset: dict):
    """Convert normalized YOLO ``class cx cy width height`` labels."""
    decimals = preset.get("coordinate_decimals", 3)
    coordinate_offset = preset.get("coordinate_offset", 0)
    extensions = preset.get("image_extensions", [".jpg", ".jpeg", ".png"])
    class_names = preset.get("class_names", {})
    class_names_file = preset.get("class_names_file")
    if class_names_file:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "class_names_file requires PyYAML; install pyyaml or set class_names"
            ) from error
        with open(
            os.path.join(dataset_root, class_names_file), "r", encoding="utf-8"
        ) as handle:
            names = yaml.safe_load(handle).get("names", {})
        if isinstance(names, list):
            class_names = {str(index): name for index, name in enumerate(names)}
        elif isinstance(names, dict):
            class_names = {str(index): name for index, name in names.items()}
        else:
            raise ValueError(f"invalid YOLO names in {class_names_file}")

    for source in preset["sources"]:
        label_root = os.path.join(dataset_root, source["label_root"])
        image_root = os.path.join(dataset_root, source["image_root"])
        coordinate_width_scale = source.get("coordinate_width_scale", 1)
        coordinate_height_scale = source.get("coordinate_height_scale", 1)
        for label_path in sorted(glob.glob(os.path.join(label_root, "**", "*.txt"), recursive=True)):
            relative_stem = os.path.splitext(os.path.relpath(label_path, label_root))[0]
            image_path = None
            image_leaf = None
            for extension in extensions:
                candidate = os.path.join(image_root, relative_stem + extension)
                if os.path.isfile(candidate):
                    image_path = candidate
                    image_leaf = relative_stem + extension
                    break
            if image_path is None or image_leaf is None:
                continue

            try:
                width, height = image_size(image_path)
            except (OSError, ValueError):
                continue

            boxes = []
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    class_id, cx, cy, box_width, box_height = fields
                    phrase = source.get("phrase") or class_names.get(class_id, "")
                    phrase = clean_phrase(phrase, preset)
                    if not phrase:
                        continue
                    cx, cy, box_width, box_height = map(
                        float, (cx, cy, box_width, box_height)
                    )
                    coordinate_width = width * coordinate_width_scale
                    coordinate_height = height * coordinate_height_scale
                    x0 = round(
                        (cx - box_width / 2) * coordinate_width
                        + coordinate_offset,
                        decimals,
                    )
                    y0 = round(
                        (cy - box_height / 2) * coordinate_height
                        + coordinate_offset,
                        decimals,
                    )
                    x1 = round(
                        (cx + box_width / 2) * coordinate_width
                        + coordinate_offset,
                        decimals,
                    )
                    y1 = round(
                        (cy + box_height / 2) * coordinate_height
                        + coordinate_offset,
                        decimals,
                    )
                    if x1 <= x0 or y1 <= y0:
                        continue
                    boxes.append({"bbox": [x0, y0, x1, y1], "phrase": phrase})
            if boxes:
                yield make_record(
                    prefixed_path(source.get("image_prefix", ""), image_leaf),
                    width,
                    height,
                    boxes,
                    image_field=preset.get("output_image_field", "image_name"),
                )


def build_image_index(dataset_root: str, patterns: list[str]) -> dict[str, str]:
    index = {}
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(dataset_root, pattern), recursive=True)):
            if os.path.isfile(path):
                index.setdefault(os.path.basename(path), path)
    return index


def iter_labelme(dataset_root: str, preset: dict):
    image_index = build_image_index(dataset_root, preset.get("image_globs", []))
    annotation_paths = []
    for pattern in preset["annotation_globs"]:
        annotation_paths.extend(
            glob.glob(os.path.join(dataset_root, pattern), recursive=True)
        )

    for annotation_path in sorted(set(annotation_paths)):
        with open(annotation_path, "r", encoding="utf-8") as handle:
            item = json.load(handle)

        image_basename = os.path.basename(
            item.get("imagePath") or str(Path(annotation_path).with_suffix(".jpg"))
        )
        source_image = image_index.get(image_basename)
        if source_image:
            image_name = os.path.relpath(source_image, dataset_root).replace(os.sep, "/")
        else:
            if preset.get("require_images", False):
                continue
            image_name = prefixed_path(
                preset.get("image_prefix", ""), image_basename
            )

        width = item.get("imageWidth")
        height = item.get("imageHeight")
        if not width or not height:
            if not source_image:
                continue
            width, height = image_size(source_image)

        boxes = []
        for shape in item.get("shapes", []):
            phrase = clean_phrase(str(shape.get("label", "")), preset)
            points = shape.get("points") or []
            if not phrase or len(points) < 2:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append({"bbox": [x0, y0, x1, y1], "phrase": phrase})
        if boxes:
            yield make_record(image_name, width, height, boxes)


def iter_json_records(dataset_root: str, preset: dict):
    size_cache = {}
    for source in preset["sources"]:
        annotation_path = os.path.join(dataset_root, source["annotation"])

        image_field = source.get("image_field", "img_filename")
        boxes_field = source.get("boxes_field")
        phrase_field = source.get("phrase_field", "instruction")
        bbox_field = source.get("bbox_field", "bbox")
        image_root = os.path.join(dataset_root, source["image_root"])
        image_prefix = source.get("image_prefix", source["image_root"])
        normalized = source.get("bbox_normalized", False)
        emit_per_box = source.get("emit_per_box", False)

        for item in iter_json_array(annotation_path):
            image_leaf = item[image_field]
            image_path = os.path.join(image_root, image_leaf)
            if image_path not in size_cache:
                try:
                    size_cache[image_path] = image_size(image_path)
                except (OSError, ValueError):
                    size_cache[image_path] = None
            if size_cache[image_path] is None:
                continue
            width, height = size_cache[image_path]
            elements = item.get(boxes_field, []) if boxes_field else [item]
            boxes = []
            for element in elements:
                phrase = clean_phrase(str(element.get(phrase_field, "")), preset)
                coordinates = element.get(bbox_field)
                if not phrase or not isinstance(coordinates, list) or len(coordinates) != 4:
                    continue
                x0, y0, x1, y1 = coordinates
                if normalized:
                    x0, x1 = x0 * width, x1 * width
                    y0, y1 = y0 * height, y1 * height
                box = {"bbox": [x0, y0, x1, y1], "phrase": phrase}
                if emit_per_box:
                    yield make_record(
                        prefixed_path(image_prefix, image_leaf), width, height, [box]
                    )
                else:
                    boxes.append(box)
            if boxes and not emit_per_box:
                yield make_record(
                    prefixed_path(image_prefix, image_leaf), width, height, boxes
                )


def iter_parquet(dataset_root: str, preset: dict):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("the parquet adapter requires pyarrow") from error

    for pattern in preset["parquet_globs"]:
        paths = sorted(glob.glob(os.path.join(dataset_root, pattern)))
        for path in paths:
            table = parquet.read_table(path)
            for row in table.to_pylist():
                image_value = row[preset.get("image_field", "image_url")]
                strip_image_prefix = preset.get("strip_image_prefix", "")
                if strip_image_prefix and image_value.startswith(strip_image_prefix):
                    image_value = image_value[len(strip_image_prefix) :].lstrip("/")
                phrase = clean_phrase(
                    str(row[preset.get("phrase_field", "instruction")]), preset
                )
                coordinates = list(row[preset.get("bbox_field", "bbox")])
                embedded = row.get(preset.get("embedded_image_field", "image"))
                image_bytes = embedded.get("bytes") if isinstance(embedded, dict) else None
                if image_bytes:
                    with Image.open(BytesIO(image_bytes)) as image:
                        width, height = image.size
                else:
                    image_path = os.path.join(dataset_root, image_value)
                    try:
                        width, height = image_size(image_path)
                    except (OSError, ValueError):
                        continue
                if preset.get("bbox_normalized", False):
                    coordinates = [
                        coordinates[0] * width,
                        coordinates[1] * height,
                        coordinates[2] * width,
                        coordinates[3] * height,
                    ]
                yield make_record(
                    prefixed_path(preset.get("image_prefix", ""), image_value),
                    width,
                    height,
                    [{"bbox": coordinates, "phrase": phrase}],
                )


def iter_parquet_objects(dataset_root: str, preset: dict):
    """Convert Hugging Face image/object parquet rows to common bbox JSONL."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("the parquet adapter requires pyarrow") from error

    image_field = preset.get("image_field", "image")
    image_name_field = preset.get("image_name_field", "file_name")
    objects_field = preset.get("objects_field", "objects")
    bbox_field = preset.get("bbox_field", "bbox")
    phrase_field = preset.get("phrase_field", "category")
    bbox_format = preset.get("bbox_format", "xywh")
    if bbox_format not in {"xywh", "xyxy"}:
        raise ValueError(f"unsupported parquet bbox format: {bbox_format!r}")

    extract_root = preset.get("extract_image_root")
    for pattern in preset["parquet_globs"]:
        for path in sorted(glob.glob(os.path.join(dataset_root, pattern))):
            table = parquet.read_table(path)
            for row in table.to_pylist():
                image_name = row.get(image_name_field)
                embedded = row.get(image_field)
                if not image_name and isinstance(embedded, dict):
                    image_name = embedded.get("path")
                if not image_name:
                    continue

                width = row.get("width")
                height = row.get("height")
                image_bytes = embedded.get("bytes") if isinstance(embedded, dict) else None
                if (not width or not height) and image_bytes:
                    with Image.open(BytesIO(image_bytes)) as image:
                        width, height = image.size
                if not width or not height:
                    continue

                if extract_root and image_bytes:
                    image_path = os.path.join(dataset_root, extract_root, image_name)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    if not os.path.isfile(image_path):
                        with open(image_path, "wb") as handle:
                            handle.write(image_bytes)

                objects = row.get(objects_field) or {}
                coordinates_list = objects.get(bbox_field) or []
                phrases = objects.get(phrase_field) or []
                boxes = []
                for coordinates, phrase_value in zip(coordinates_list, phrases):
                    phrase = clean_phrase(str(phrase_value), preset)
                    if not phrase or len(coordinates) != 4:
                        continue
                    x0, y0, x1, y1 = coordinates
                    if bbox_format == "xywh":
                        x1, y1 = x0 + x1, y0 + y1
                    if x1 <= x0 or y1 <= y0:
                        continue
                    boxes.append({"bbox": [x0, y0, x1, y1], "phrase": phrase})
                if boxes:
                    yield make_record(
                        prefixed_path(preset.get("image_prefix", ""), image_name),
                        width,
                        height,
                        boxes,
                    )


def iter_parquet_annotations(dataset_root: str, preset: dict):
    """Convert Parquet rows containing image metadata and annotation records.

    Objects365 publishes one row per image with ``image_info`` and an
    ``anns_info`` list.  Read the shards in batches so the full annotation
    table is never materialized in memory.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("the parquet adapter requires pyarrow") from error

    image_name_field = preset.get("image_name_field", "image_path")
    image_info_field = preset.get("image_info_field", "image_info")
    annotations_field = preset.get("annotations_field", "annotations")
    bbox_field = preset.get("bbox_field", "bbox")
    phrase_field = preset.get("phrase_field", "category")
    bbox_format_field = preset.get("bbox_format_field")
    default_bbox_format = preset.get("bbox_format", "xyxy")
    strip_image_prefix = preset.get("strip_image_prefix", "")
    excluded = set(preset.get("excluded_categories", []))
    batch_size = preset.get("batch_size", 1024)

    for pattern in preset["parquet_globs"]:
        for path in sorted(glob.glob(os.path.join(dataset_root, pattern))):
            parquet_file = parquet.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    image_name = row.get(image_name_field)
                    image_info = row.get(image_info_field) or {}
                    width = image_info.get("width")
                    height = image_info.get("height")
                    if not image_name or not width or not height:
                        continue
                    if strip_image_prefix and image_name.startswith(
                        strip_image_prefix
                    ):
                        image_name = image_name[len(strip_image_prefix) :].lstrip(
                            "/"
                        )

                    bbox_format = (
                        row.get(bbox_format_field, default_bbox_format)
                        if bbox_format_field
                        else default_bbox_format
                    )
                    if bbox_format not in {"xywh", "xyxy"}:
                        raise ValueError(
                            f"unsupported parquet bbox format: {bbox_format!r}"
                        )

                    boxes = []
                    for annotation in row.get(annotations_field) or []:
                        phrase = clean_phrase(
                            str(annotation.get(phrase_field, "")), preset
                        )
                        coordinates = annotation.get(bbox_field)
                        if (
                            not phrase
                            or phrase in excluded
                            or not isinstance(coordinates, list)
                            or len(coordinates) != 4
                        ):
                            continue
                        x0, y0, x1, y1 = coordinates
                        if bbox_format == "xywh":
                            x1, y1 = x0 + x1, y0 + y1
                        if x1 <= x0 or y1 <= y0:
                            continue
                        boxes.append(
                            {"bbox": [x0, y0, x1, y1], "phrase": phrase}
                        )

                    if boxes:
                        yield make_record(
                            prefixed_path(
                                preset.get("image_prefix", ""), image_name
                            ),
                            width,
                            height,
                            boxes,
                        )


ADAPTERS = {
    "coco": iter_coco,
    "json_records": iter_json_records,
    "labelme": iter_labelme,
    "parquet": iter_parquet,
    "parquet_annotations": iter_parquet_annotations,
    "parquet_objects": iter_parquet_objects,
    "unified_bbox": iter_unified_bbox,
    "yolo": iter_yolo,
}


def iter_records(dataset_root: str, preset: dict, limit: int | None = None):
    adapter_name = preset["adapter"]
    if adapter_name not in ADAPTERS:
        raise ValueError(
            f"unsupported adapter {adapter_name!r}; choose one of {sorted(ADAPTERS)}"
        )
    for index, record in enumerate(ADAPTERS[adapter_name](dataset_root, preset)):
        if limit is not None and index >= limit:
            return
        yield record


def write_jsonl(records, output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a unified bbox JSONL from raw public annotations."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None, help="Root containing raw datasets.")
    parser.add_argument("--output", default=None, help="Unified bbox JSONL path.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    default_dataset, datasets = load_config(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in datasets:
        parser.error(f"unknown dataset {dataset_name!r}; choose one of {sorted(datasets)}")
    preset = datasets[dataset_name]
    dataset_root = args.input or resolve_tool_path(preset["default_input"])
    output_path = args.output or resolve_tool_path(preset["default_output"])
    count = write_jsonl(
        iter_records(dataset_root, preset, limit=args.limit), output_path
    )
    print(f"Wrote {count} unified bbox records to {output_path}")


if __name__ == "__main__":
    main()
