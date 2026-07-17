import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "visual_prompt_datasets.json"

PROMPTS = [
    "Given the visually indicated objects <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox> as references, detect all matching instances in the image. Output results as structured text with category and bounding box coordinates in [x0, y0, x1, y1] format.",
    "Using <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox> as visual exemplars, locate all same-category objects in the scene. Return detections as text entries, each containing the object class and bounding box in [x0, y0, x1, y1] pixel coordinates.",
    "Detect all instances that match the categories visually specified by <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>. Provide a structured text list with each entry including category name and bounding box in [x0, y0, x1, y1] format.",
    "Based on the reference objects <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>, find all semantically similar instances in the image. Output as structured text with class labels and [x0, y0, x1, y1] bounding boxes, unlike image-based outputs from segmentation or depth tasks.",
    "Identify all objects belonging to the same classes as the visually provided <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>. Return results in text form, with each detection specifying category and bounding box coordinates in [x0, y0, x1, y1].",
    "Given visual prompts for <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>, detect all corresponding instances in the image. Output must be structured text, not an image, with each object represented by its category and [x0, y0, x1, y1] bounding box.",
    "Locate all objects that are visually consistent with the reference categories <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>. Return a text-based list where each entry includes the class name and bounding box in [x0, y0, x1, y1] format.",
    "Using the example regions associated with <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>, find all matching objects in the image. Output detections as structured text with category and precise [x0, y0, x1, y1] coordinates.",
    "Detect every instance that matches the visually indicated categories <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>. Unlike tasks that produce masks or depth maps, return results as textual detections with [x0, y0, x1, y1] bounding boxes.",
    "Perform object detection based on visual references for <p>object1</p><bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>. Output a structured text response listing each detected object category and its bounding box in [x0, y0, x1, y1] format."
]


def truncate_norm(value: float) -> float:
    return min(max(value, 0.0), 0.999)


def load_presets(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError("visual prompt config requires default_dataset and datasets")
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else REPO_ROOT / value)


def normalize_category(name: str, preset: dict) -> str:
    category = name.replace("_", " ").split("/")[0].strip().lower()
    aliases = preset.get("category_aliases", {})
    return aliases.get(category, category)


def format_bbox(box) -> str:
    return f"<bbox>[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]</bbox>"


def normalize_xyxy(x0: float, y0: float, x1: float, y1: float, width: int, height: int):
    return [
        truncate_norm(x0 / width),
        truncate_norm(y0 / height),
        truncate_norm(x1 / width),
        truncate_norm(y1 / height),
    ]


def sample_reference_boxes(boxes: list[list[float]], rng: random.Random) -> list[list[float]]:
    count = len(boxes)
    if count == 1:
        return [boxes[0]]
    if count == 2:
        return [boxes[0] if rng.random() < 0.5 else boxes[1]]

    if rng.random() < 0.6:
        return [rng.choice(boxes)]

    # Keep the prompt compact for dense categories: we want enough visual
    # references to disambiguate the class, but not so many that the prompt is
    # dominated by exemplar boxes. Using at most about one third of all answer
    # boxes is the heuristic used by the released visual-prompt data.
    max_k = max(count // 3, 1)
    k = rng.randint(1, max_k)
    return rng.sample(boxes, min(k, count))


def bucket_name(box_count: int) -> str:
    # The single / double / multi split is defined by the number of boxes in the
    # GPT answer, not by how many reference boxes appear in the human prompt.
    # In this converter the answer contains all boxes for the category, so that
    # count is len(boxes): 1 -> single, 2 -> double, >=3 -> multi.
    if box_count == 1:
        return "single"
    if box_count == 2:
        return "double"
    return "multi"


def load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_coco_images(dataset_root: str, preset: dict):
    coco = load_coco(os.path.join(dataset_root, preset["annotation"]))
    categories = {
        item["id"]: normalize_category(item["name"], preset)
        for item in coco["categories"]
    }
    images = {item["id"]: item for item in coco["images"]}
    anns_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)

    image_prefix = preset.get("image_prefix", "")
    for image_id in sorted(images):
        image_info = images[image_id]
        boxes = []
        for annotation in anns_by_image.get(image_id, []):
            phrase = categories.get(annotation["category_id"], "")
            if not phrase:
                continue
            x, y, width, height = annotation["bbox"]
            boxes.append(
                {
                    "phrase": phrase,
                    "bbox": [x, y, x + width, y + height],
                }
            )
        image_name = image_info["file_name"]
        if image_prefix:
            image_name = f"{image_prefix.rstrip('/')}/{image_name.lstrip('/')}"
        yield image_name, image_info["width"], image_info["height"], boxes


def iter_bbox_jsonl(input_path: str, preset: dict):
    image_prefix = preset.get("image_prefix", "")
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_name = item.get("image_name", item.get("image_path"))
            image_info = item.get("image_info", {})
            boxes = item.get("annotation", {}).get("boxes", [])
            if not image_name or not image_info.get("width") or not image_info.get("height"):
                continue
            if image_prefix:
                image_name = f"{image_prefix.rstrip('/')}/{image_name.lstrip('/')}"
            yield image_name, image_info["width"], image_info["height"], boxes


def build_final_subset(
    bucketed_records: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    # Reproduce the observed visual-prompt mixing rule used by the existing
    # released jsonl files:
    #
    #   cap = int(len(multi) * 0.2)
    #   final_subset = (
    #       random_sample(single, min(len(single), cap), seed=42)
    #       + random_sample(double, min(len(double), cap), seed=42)
    #       + all_multi
    #   )
    #
    # The final subset is then shuffled and ids are reassigned.
    multi_records = bucketed_records["multi"]
    cap = int(len(multi_records) * 0.2)

    final_records = []
    final_records.extend(
        rng.sample(bucketed_records["single"], min(len(bucketed_records["single"]), cap))
    )
    final_records.extend(
        rng.sample(bucketed_records["double"], min(len(bucketed_records["double"]), cap))
    )
    final_records.extend(multi_records)
    rng.shuffle(final_records)
    return final_records


def build_records(input_path: str, preset: dict, seed: int, limit: int | None = None):
    rng = random.Random(seed)
    merge_rng = random.Random(seed)
    input_format = preset.get("input_format", "coco")
    if input_format == "coco":
        images = iter_coco_images(input_path, preset)
    elif input_format == "bbox_jsonl":
        images = iter_bbox_jsonl(input_path, preset)
    else:
        raise ValueError(f"unsupported visual prompt input format: {input_format}")

    excluded_categories = {
        normalize_category(category, preset)
        for category in preset.get("excluded_categories", [])
    }

    bucketed_records = {"single": [], "double": [], "multi": []}
    for image_name, width, height, image_boxes in images:
        if not image_boxes:
            continue

        phrase_to_boxes = defaultdict(list)
        phrase_order = []
        for box in image_boxes:
            phrase = normalize_category(str(box.get("phrase", "")), preset)
            coordinates = box.get("bbox", [])
            if not phrase or phrase in excluded_categories or len(coordinates) != 4:
                continue
            if phrase not in phrase_to_boxes:
                phrase_order.append(phrase)
            phrase_to_boxes[phrase].append(normalize_xyxy(*coordinates, width, height))

        for phrase in phrase_order:
            boxes = sorted(phrase_to_boxes[phrase], key=lambda item: (item[0], item[1]))
            refs = sample_reference_boxes(boxes, rng)
            ref_text = "".join(format_bbox(box) for box in refs)
            prompt = rng.choice(PROMPTS).replace(
                "<bbox>[0.xxx, 0.xxx, 0.xxx, 0.xxx]</bbox>",
                ref_text,
            )
            answer = "".join(format_bbox(box) for box in boxes)
            record = {
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": f"<image>{prompt}"},
                    {"from": "gpt", "value": f"<p>object1</p>{answer}."},
                ],
                "category": phrase,
            }
            bucketed_records[bucket_name(len(boxes))].append(record)

    # Use a fresh RNG for the final single/double sampling step so the merged
    # subset is reproducible independently of how many random choices were
    # consumed while building prompts and visual references.
    if limit is None:
        final_records = build_final_subset(bucketed_records, merge_rng)
    else:
        final_records = (
            bucketed_records["single"]
            + bucketed_records["double"]
            + bucketed_records["multi"]
        )[:limit]

    records = []
    for idx, record in enumerate(final_records):
        records.append(
            {
                "id": idx,
                "image": record["image"],
                "conversations": record["conversations"],
                "category": record["category"],
            }
        )
    return records


def write_jsonl(records, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO or common bbox annotations to visual-prompt JSONL."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None, help="Dataset root or common bbox JSONL.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument(
        "--image-prefix",
        default=None,
        help="Optional image-prefix override.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt/reference sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for debugging.")
    args = parser.parse_args()

    default_dataset, datasets = load_presets(args.config)
    dataset_name = args.dataset or default_dataset
    if dataset_name not in datasets:
        parser.error(f"unknown dataset {dataset_name!r}; choose one of {sorted(datasets)}")
    preset = dict(datasets[dataset_name])
    if args.image_prefix is not None:
        preset["image_prefix"] = args.image_prefix
    input_path = args.input or resolve_tool_path(preset["default_input"])
    output_path = args.output or resolve_tool_path(preset["default_output"])

    records = build_records(input_path, preset, seed=args.seed, limit=args.limit)
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
