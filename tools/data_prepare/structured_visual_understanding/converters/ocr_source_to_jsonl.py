import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "ocr_source_datasets.json"


def load_presets(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError("OCR source config requires default_dataset and datasets")
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(TOOL_ROOT / candidate)


def prefixed_path(prefix: str, value: str) -> str:
    value = value.replace("\\", "/").lstrip("/")
    if prefix:
        return f"{prefix.rstrip('/')}/{value}"
    return value


def clean_text(value: str, preset: dict) -> str:
    text = " ".join(str(value or "").lstrip("\ufeff").split())
    if not text or text in set(preset.get("ignore_texts", [])):
        return ""
    return text


def rect_to_polygon(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def polygon_to_bbox(values: list[float]) -> list[float] | None:
    if len(values) < 6 or len(values) % 2 != 0:
        return None
    xs = values[0::2]
    ys = values[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def make_record(image_name: str, width: int, height: int, regions: list[dict]) -> dict:
    return {
        "image_name": image_name,
        "image_info": {"width": width, "height": height},
        "annotation": {"ocr": regions},
    }


def resolve_image_path(image_root: str, stem: str, preset: dict) -> tuple[str, str]:
    stem_strip_prefix = preset.get("annotation_stem_strip_prefix", "")
    if stem_strip_prefix and stem.startswith(stem_strip_prefix):
        stem = stem[len(stem_strip_prefix) :]
    image_name = preset.get("image_name_template", "{stem}").format(stem=stem)
    image_path = os.path.join(image_root, image_name)
    return image_name, image_path


def iter_text_files(root: str, preset: dict):
    annotation_root = os.path.join(root, preset["annotation_root"])
    image_root = os.path.join(root, preset["image_root"])
    delimiter = preset.get("delimiter", ",")
    coord_count = int(preset["coord_count"])
    granularity = preset["granularity"]
    for annotation_path in sorted(
        glob.glob(os.path.join(annotation_root, preset["annotation_glob"]))
    ):
        stem = Path(annotation_path).stem
        image_name, image_path = resolve_image_path(image_root, stem, preset)
        with Image.open(image_path) as image:
            width, height = image.size
        regions = []
        with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(delimiter, coord_count)
                if len(parts) != coord_count + 1:
                    continue
                try:
                    coords = [float(part.strip()) for part in parts[:coord_count]]
                except ValueError:
                    continue
                text = clean_text(parts[coord_count], preset)
                if not text:
                    continue
                if coord_count == 4:
                    bbox = [coords[0], coords[1], coords[2], coords[3]]
                    polygon = rect_to_polygon(bbox)
                else:
                    polygon = coords
                    bbox = polygon_to_bbox(coords)
                if bbox is None:
                    continue
                regions.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "polygon": polygon,
                        "granularity": granularity,
                    }
                )
        if regions:
            yield make_record(
                prefixed_path(preset.get("image_prefix", ""), image_name),
                width,
                height,
                regions,
            )


ADAPTERS = {
    "text_files": iter_text_files,
}


def iter_records(dataset_root: str, preset: dict, limit: int | None = None):
    adapter_name = preset["adapter"]
    if adapter_name not in ADAPTERS:
        raise ValueError(
            f"unsupported OCR adapter {adapter_name!r}; choose one of {sorted(ADAPTERS)}"
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
        description="Prepare a unified OCR JSONL from raw public annotations."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None, help="Root containing raw datasets.")
    parser.add_argument("--output", default=None, help="Unified OCR JSONL path.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    default_dataset, datasets = load_presets(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in datasets:
        parser.error(
            f"unknown OCR source dataset {dataset_name!r}; choose one of {sorted(datasets)}"
        )
    preset = datasets[dataset_name]
    dataset_root = args.input or resolve_tool_path(preset["default_input"])
    output_path = args.output or resolve_tool_path(preset["default_output"])
    count = write_jsonl(iter_records(dataset_root, preset, args.limit), output_path)
    print(f"Wrote {count} OCR source records to {output_path}")


if __name__ == "__main__":
    main()
