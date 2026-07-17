import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "ocr_datasets.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.prompts import NATUR_PROMPT_TEMPLATE


def load_presets(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError("OCR config requires default_dataset and datasets")
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(TOOL_ROOT / candidate)


def truncate_norm(value: float) -> float:
    return min(max(value, 0.0), 0.999)


def format_bbox(box: list[float], width: int, height: int) -> str:
    x0, y0, x1, y1 = box
    return (
        f"<bbox>[{truncate_norm(x0 / width):.3f}, {truncate_norm(y0 / height):.3f}, "
        f"{truncate_norm(x1 / width):.3f}, {truncate_norm(y1 / height):.3f}]</bbox>"
    )


def format_polygon(values: list[float], width: int, height: int) -> str:
    normalized = []
    for index, value in enumerate(values):
        size = width if index % 2 == 0 else height
        normalized.append(f"{truncate_norm(value / size):.3f}")
    return f"<polygon>[{', '.join(normalized)}]</polygon>"


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


def bbox_to_polygon(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def clean_text(value: str) -> str:
    return " ".join(str(value or "").lstrip("\ufeff").split())


def prompt_pool(granularity: str, output_kind: str) -> list[str]:
    if granularity == "word" and output_kind == "bbox":
        return NATUR_PROMPT_TEMPLATE["OCR_word_promt"]
    if granularity == "word" and output_kind == "polygon":
        return NATUR_PROMPT_TEMPLATE["OCR_word_polygon_promt"]
    if granularity == "text_line" and output_kind == "bbox":
        return NATUR_PROMPT_TEMPLATE["OCR_textline_promt"]
    if granularity == "text_line" and output_kind == "polygon":
        return NATUR_PROMPT_TEMPLATE["OCR_textline_polygon_promt"]
    raise ValueError(
        f"unsupported OCR prompt combination: granularity={granularity!r}, "
        f"output_kind={output_kind!r}"
    )


def iter_records(input_path: str, preset: dict, seed: int, limit: int | None):
    rng = random.Random(seed)
    granularity = preset["granularity"]
    output_kind = preset["output_kind"]
    prompts = prompt_pool(granularity, output_kind)
    emitted = 0
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_name = item.get("image_name", item.get("image_path"))
            image_info = item.get("image_info") or {}
            width = image_info.get("width")
            height = image_info.get("height")
            regions = (item.get("annotation") or {}).get("ocr", [])
            if not image_name or not width or not height or not regions:
                continue

            grouped = OrderedDict()
            for region in regions:
                if region.get("granularity") != granularity:
                    continue
                text = clean_text(region.get("text"))
                if not text:
                    continue
                if output_kind == "bbox":
                    bbox = region.get("bbox")
                    if bbox is None and region.get("polygon"):
                        bbox = polygon_to_bbox(region["polygon"])
                    if bbox is None:
                        continue
                    payload = format_bbox(bbox, width, height)
                else:
                    polygon = region.get("polygon")
                    if polygon is None and region.get("bbox"):
                        polygon = bbox_to_polygon(region["bbox"])
                    if polygon is None:
                        continue
                    payload = format_polygon(polygon, width, height)
                grouped.setdefault(text, []).append(payload)
            if not grouped:
                continue

            answer = ", ".join(
                f"<p>{text}</p>{''.join(payloads)}"
                for text, payloads in grouped.items()
            ) + "."
            yield {
                "id": emitted,
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": f"<image>{rng.choice(prompts)}"},
                    {"from": "gpt", "value": answer},
                ],
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert common OCR JSONL to training JSONL."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    default_dataset, datasets = load_presets(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in datasets:
        parser.error(
            f"unknown OCR dataset {dataset_name!r}; choose one of {sorted(datasets)}"
        )
    preset = datasets[dataset_name]
    input_path = args.input or resolve_tool_path(preset["default_input"])
    output_path = args.output or resolve_tool_path(preset["default_output"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in iter_records(input_path, preset, args.seed, args.limit):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} OCR training records to {output_path}")


if __name__ == "__main__":
    main()
