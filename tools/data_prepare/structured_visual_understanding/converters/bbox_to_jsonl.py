import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "bbox_datasets.json"

PROMPTS = [
    "Detect all instances of PHRASE in the image. Output the results as a structured text list with each detection including category and bounding box coordinates in <bbox> format.",
    "Locate and identify PHRASE within the scene. Output detection results as text entries, each containing the object class and pixel coordinates defining the object bounding box.",
    "Find all objects belonging to the specified categories: PHRASE. Return a text-based list where each detection includes the category name and bounding box coordinates in pixel values.",
    "Detect all instances of PHRASE in the image. The output should be a structured text response with each object class and precise location coordinates—unlike image outputs from other vision tasks.",
    "Identify objects from the following categories: PHRASE. Output detection results as text, with each entry specifying category and bounding box coordinates. This text format differs from the image outputs of depth or segmentation tasks.",
    "Detect all PHRASE present in the image. Unlike tasks that output modified images, provide results as structured text with class labels and bounding box coordinates for each detected object.",
    "Find and classify objects from the categories: PHRASE. Return detection results as structured text, with each object represented by its class and bounding box coordinates in a <bbox> format suitable for further processing.",
    "Locate and list PHRASE visible in the image. Provide the output as structured text data, with each detection specifying the object category and its bounding box coordinates in pixel values.",
    "Detect objects from these categories: PHRASE. Unlike tasks that produce visual outputs like depth maps or segmentation masks, return a text-based list of detections with category labels and bounding box coordinates.",
    "Perform object detection to identify PHRASE. Return detections as text, with each object represented by its category and a bounding box defined by four coordinate values, <bbox> format.",
]


def truncate_norm(value: float) -> float:
    return min(max(value, 0.0), 0.999)


def format_box(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> str:
    nx0 = truncate_norm(x0 / width)
    ny0 = truncate_norm(y0 / height)
    nx1 = truncate_norm(x1 / width)
    ny1 = truncate_norm(y1 / height)
    return f"<bbox>[{nx0:.3f}, {ny0:.3f}, {nx1:.3f}, {ny1:.3f}]</bbox>"


def normalize_category(name: str) -> str:
    # We recommend maintaining dataset-specific category normalization rules.
    # In practice, different sources may mix casing or near-synonyms such as
    # "car(automobile)", "car", "automobile", or plural forms like "cars".
    # Keeping a small normalization layer per dataset makes supervision more
    # consistent and easier to merge across public datasets.
    return (
        name.replace("-merged", "")
        .replace("-other", "")
        .replace("-stuff", "")
        .replace("-negative", "")
        .replace("-", " ")
        .lower()
        .strip()
    )


def normalize_bbox_phrase(name: str) -> str:
    """Match the phrase cleanup used by the unified bbox JSONL pipeline."""
    if "/" in name:
        name = name.split("/", 1)[0]
    return name.replace("_", " ").strip()


def load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_config(path: str) -> tuple[str, dict]:
    config = load_coco(path)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError(
            "bbox config must contain string 'default_dataset' and object 'datasets'"
        )
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")

    required = {
        "default_input",
        "default_output",
        "annotation",
        "image_prefix",
        "excluded_categories",
        "category_aliases",
    }
    for name, preset in datasets.items():
        missing = required.difference(preset)
        if missing:
            raise ValueError(f"dataset {name!r} is missing fields: {sorted(missing)}")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def build_records(
    dataset_root: str,
    annotation_file: str,
    image_prefix: str,
    excluded_categories: set[str],
    category_aliases: dict[str, str],
    seed: int,
    limit: int | None = None,
):
    random.seed(seed)
    ann_path = os.path.join(dataset_root, annotation_file)
    coco = load_coco(ann_path)

    categories = {}
    for item in coco["categories"]:
        category = normalize_category(item["name"])
        categories[item["id"]] = category_aliases.get(category, category)
    images = {item["id"]: item for item in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    records = []
    for image_id in sorted(images):
        image_info = images[image_id]
        image_anns = anns_by_image.get(image_id, [])
        if not image_anns:
            continue

        width = image_info["width"]
        height = image_info["height"]
        phrase_to_boxes = defaultdict(list)
        phrase_order = []

        for ann in image_anns:
            phrase = categories.get(ann["category_id"], "").strip()
            if not phrase or phrase in excluded_categories:
                continue
            if phrase not in phrase_to_boxes:
                phrase_order.append(phrase)
            x, y, w, h = ann["bbox"]
            phrase_to_boxes[phrase].append((x, y, x + w, y + h))

        if not phrase_order:
            continue

        phrase_text = ", ".join(f"<p>{phrase}</p>" for phrase in phrase_order)
        prompt = random.choice(PROMPTS).replace("PHRASE", phrase_text)

        answer_parts = []
        for phrase in phrase_order:
            boxes = "".join(
                format_box(x0, y0, x1, y1, width, height)
                for x0, y0, x1, y1 in phrase_to_boxes[phrase]
            )
            answer_parts.append(f"<p>{phrase}</p>{boxes}")

        image_name = image_info["file_name"]
        if image_prefix:
            image_name = f"{image_prefix.rstrip('/')}/{image_name.lstrip('/')}"
        records.append(
            {
                "id": len(records),
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": f"<image>{prompt}"},
                    {"from": "gpt", "value": ", ".join(answer_parts) + "."},
                ],
            }
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def iter_bbox_jsonl_records(
    dataset_root: str,
    annotation_file: str,
    image_prefix: str,
    excluded_categories: set[str],
    category_aliases: dict[str, str],
    task_prefix: str,
    verify_images: bool,
    seed: int,
    limit: int | None = None,
):
    """Convert unified ``image_name + annotation.boxes`` JSONL records.

    Layout and GUI source readers use this common intermediate representation.
    Boxes are absolute ``xyxy`` coordinates and ``image_info`` supplies the
    source dimensions. ``verify_images`` reproduces datasets whose prepared
    annotations contain rows for images that are no longer present.
    """
    rng = random.Random(seed)
    ann_path = os.path.join(dataset_root, annotation_file)
    emitted = 0

    with open(ann_path, "r", encoding="utf-8") as handle:
        for source_id, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            image_name = item.get("image_name", item.get("image_path"))
            if not image_name:
                continue
            if image_prefix:
                image_name = f"{image_prefix.rstrip('/')}/{image_name.lstrip('/')}"

            image_info = item.get("image_info") or {}
            width = image_info.get("width")
            height = image_info.get("height")
            image_path = (
                image_name
                if os.path.isabs(image_name)
                else os.path.join(dataset_root, image_name)
            )
            if verify_images and not os.path.exists(image_path):
                continue
            if not width or not height:
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except (OSError, ValueError):
                    continue
            if not width or not height:
                continue

            phrase_to_boxes = defaultdict(list)
            phrase_order = []
            for box in item.get("annotation", {}).get("boxes", []):
                phrase = normalize_bbox_phrase(str(box.get("phrase", "")))
                phrase = category_aliases.get(phrase, phrase)
                if not phrase or phrase in excluded_categories:
                    continue
                coordinates = box.get("bbox")
                if not isinstance(coordinates, list) or len(coordinates) != 4:
                    continue
                if phrase not in phrase_to_boxes:
                    phrase_order.append(phrase)
                phrase_to_boxes[phrase].append(coordinates)

            if not phrase_order:
                continue

            phrase_text = ", ".join(f"<p>{phrase}</p>" for phrase in phrase_order)
            prompt = rng.choice(PROMPTS).replace("PHRASE", phrase_text)
            answer_parts = []
            for phrase in phrase_order:
                boxes = "".join(
                    format_box(*coordinates, width, height)
                    for coordinates in phrase_to_boxes[phrase]
                )
                answer_parts.append(f"<p>{phrase}</p>{boxes}")

            yield {
                "id": source_id,
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": f"<image>{task_prefix}{prompt}"},
                    {"from": "gpt", "value": ", ".join(answer_parts) + "."},
                ],
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def write_jsonl(records, output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a supported COCO-style bbox dataset to training jsonl."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Dataset preset config. Relative preset paths use this tool directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset layout preset.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Dataset root. Defaults to the selected dataset preset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. Defaults to the selected dataset preset.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Optional annotation path relative to the dataset root.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for prompt sampling."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional record limit for debugging."
    )
    args = parser.parse_args()

    default_dataset, presets = load_dataset_config(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in presets:
        parser.error(
            f"unknown dataset {dataset_name!r}; choose one of {sorted(presets)}"
        )
    preset = presets[dataset_name]
    input_path = args.input or resolve_tool_path(preset["default_input"])
    output_path = args.output or resolve_tool_path(preset["default_output"])
    annotation_file = args.annotation or preset["annotation"]
    input_format = preset.get("input_format", "coco")
    if input_format == "coco":
        records = build_records(
            input_path,
            annotation_file=annotation_file,
            image_prefix=preset["image_prefix"],
            excluded_categories=set(preset["excluded_categories"]),
            category_aliases=preset["category_aliases"],
            seed=args.seed,
            limit=args.limit,
        )
    elif input_format == "bbox_jsonl":
        records = iter_bbox_jsonl_records(
            input_path,
            annotation_file=annotation_file,
            image_prefix=preset["image_prefix"],
            excluded_categories=set(preset["excluded_categories"]),
            category_aliases=preset["category_aliases"],
            task_prefix=preset.get("task_prefix", ""),
            verify_images=preset.get("verify_images", False),
            seed=args.seed,
            limit=args.limit,
        )
    else:
        parser.error(
            f"unsupported input_format {input_format!r} for dataset {dataset_name!r}"
        )
    count = write_jsonl(records, output_path)
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
