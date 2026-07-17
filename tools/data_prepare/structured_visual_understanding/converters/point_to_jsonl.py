import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "point_datasets.json"

PROMPTS = [
    "Detect all instances of PHRASE in the image. Output the results as a structured text list with each detection including category and point coordinates in <point> format.",
    "Locate and identify PHRASE within the scene. Output detection results as text entries, each containing the object class and pixel coordinates defining the object point location.",
    "Find all objects belonging to the specified categories: PHRASE. Return a text-based list where each detection includes the category name and point coordinates in pixel values.",
    "Detect all instances of PHRASE in the image. The output should be a structured text response with each object class and precise point location coordinates—unlike image outputs from other vision tasks.",
    "Identify objects from following categories: PHRASE. Output detection results as text, with each entry specifying category and point coordinates. This text format differs from the image outputs of depth or segmentation tasks.",
    "Detect all PHRASE present in the image. Unlike tasks that output modified images, provide results as structured text with class labels and point coordinates for each detected object.",
    "Find and classify objects from the categories: PHRASE. Return detection results as structured text, with each object represented by its class and point coordinates in a <point> format suitable for further processing.",
    "Locate and list PHRASE visible in the image. Provide the output as structured text data, with each detection specifying the object category and its point coordinates in pixel values.",
    "Detect objects from these categories: PHRASE. Unlike tasks that produce visual outputs like depth maps or segmentation masks, return a text-based list of detections with category labels and point coordinates.",
    "Perform object detection to identify PHRASE. Return detections as text, with each object represented by its category and a point defined by two coordinate values, <point> format.",
]


def truncate_norm(value: float) -> float:
    return min(max(value, 0.0), 0.999)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_config(path: str) -> tuple[str, dict]:
    config = load_json(path)
    default_dataset = config.get("default_dataset")
    datasets = config.get("datasets")
    if not isinstance(default_dataset, str) or not isinstance(datasets, dict):
        raise ValueError(
            "point config must contain string 'default_dataset' and object 'datasets'"
        )
    if default_dataset not in datasets:
        raise ValueError(f"default dataset {default_dataset!r} is not configured")
    for name, preset in datasets.items():
        missing = {"default_input", "default_output", "input_format"}.difference(preset)
        if missing:
            raise ValueError(f"dataset {name!r} is missing fields: {sorted(missing)}")
    return default_dataset, datasets


def resolve_tool_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def load_class_map(path: str) -> dict[str, str]:
    class_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name, phrase = line.split("\t", 1)
            class_map[image_name] = phrase.strip()
    return class_map


def build_records(dataset_root: str, output_split: str, seed: int, limit: int | None = None):
    random.seed(seed)
    annotations = load_json(os.path.join(dataset_root, "annotation_FSC147_384.json"))
    split_info = load_json(os.path.join(dataset_root, "Train_Test_Val_FSC_147.json"))
    class_map = load_class_map(os.path.join(dataset_root, "ImageClasses_FSC147.txt"))

    image_names = split_info[output_split]
    records = []
    for image_name in image_names:
        ann = annotations.get(image_name)
        phrase = class_map.get(image_name)
        if ann is None or not phrase:
            continue

        image_path = os.path.join(dataset_root, "images_384_VarV2", image_name)
        with Image.open(image_path) as image:
            width, height = image.size
        points = ann.get("points", [])
        if not points:
            continue

        prompt = random.choice(PROMPTS).replace("PHRASE", f"<p>{phrase}</p>")
        # We map point coordinates to 0.000-0.999 using the actual image size.
        point_text = "".join(
            f"<point>[{truncate_norm(x / width):.3f}, {truncate_norm(y / height):.3f}]</point>"
            for x, y in points
        )
        records.append(
            {
                "id": len(records),
                "image": f"FSC147/images_384_VarV2/{image_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>{prompt}"},
                    {"from": "gpt", "value": f"<p>{phrase}</p>{point_text}."},
                ],
            }
        )
        if limit is not None and len(records) >= limit:
            break

    return records


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
        description="Convert FSC147 points to training jsonl."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None, help="Dataset and image root.")
    parser.add_argument("--output", default=None, help="Output jsonl path.")
    parser.add_argument("--split", default=None, help="FSC147 split override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for debugging.")
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
    if preset["input_format"] == "fsc147":
        records = build_records(
            input_path,
            output_split=args.split or preset.get("split", "train"),
            seed=args.seed,
            limit=args.limit,
        )
    else:
        parser.error(
            f"unsupported input_format {preset['input_format']!r} for "
            f"dataset {dataset_name!r}"
        )
    count = write_jsonl(records, output_path)
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
