import argparse
import json
import os
import random
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "keypoint_datasets.json"

BOX_PROMPTS = [
    "Detect all instances of PHRASE in the image. For each instance, output a bounding box in <bbox> format and the coordinates of its KEYPOINTS as [x, y] pairs.",
    "Locate and identify PHRASE within the scene. For each detected object, provide its bounding box and KEYPOINTS coordinates.",
]
POINT_PROMPTS = [
    "Predict the KEYPOINTS for all instances of PHRASE in the image. Return each category and its keypoint coordinates.",
    "Locate the KEYPOINTS of PHRASE within the scene. Return structured text without bounding boxes.",
]


def load_presets(path: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config["default_dataset"], config["datasets"]


def resolve_tool_path(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else TOOL_ROOT / value)


def norm(value: float, size: float) -> float:
    return min(max(value / size, 0.0), 0.999)


def derived_box(points: list[float]) -> list[float]:
    pairs = list(zip(points[0::2], points[1::2]))
    valid = [(x, y) for x, y in pairs if x > 0 or y > 0]
    if not valid:
        return [0, 0, 0, 0]
    xs, ys = zip(*valid)
    return [min(xs), min(ys), max(xs), max(ys)]


def iter_records(input_path: str, preset: dict, seed: int, limit: int | None):
    rng = random.Random(seed)
    keypoint_names = [name.replace("_", " ").strip() for name in preset["keypoint_names"]]
    emitted = 0
    with open(input_path, "r", encoding="utf-8") as handle:
        for source_id, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            image_name = item.get("image_name", item.get("image_path"))
            image_info = item.get("image_info") or item.get("annotation", {}).get("image_info", {})
            width, height = image_info.get("width"), image_info.get("height")
            keypoint_items = item.get("annotation", {}).get("keypoints", [])
            original_boxes = item.get("annotation", {}).get("boxes", [])
            if not image_name or not width or not height or not keypoint_items:
                continue
            if original_boxes and len(original_boxes) != len(keypoint_items):
                continue

            boxes = original_boxes or [
                {"bbox": derived_box(entry["keypoint"]), "phrase": entry["phrase"]}
                for entry in keypoint_items
            ]
            phrases = list(dict.fromkeys(entry["phrase"].lower() for entry in keypoint_items))
            prompt_template = rng.choice(BOX_PROMPTS if original_boxes else POINT_PROMPTS)
            prompt = prompt_template.replace(
                "PHRASE", ", ".join(f"<p>{phrase}</p>" for phrase in phrases)
            ).replace("KEYPOINTS", ", ".join(keypoint_names))

            answers = []
            for box, keypoint_item in zip(boxes, keypoint_items):
                points = keypoint_item["keypoint"]
                if len(points) != 2 * len(keypoint_names):
                    answers = []
                    break
                keypoint_parts = []
                for index, name in enumerate(keypoint_names):
                    x, y = points[index * 2 : index * 2 + 2]
                    value = (
                        "unvisible"
                        if x <= 0 and y <= 0
                        else f"[{norm(x, width):.3f}, {norm(y, height):.3f}]"
                    )
                    keypoint_parts.append(f"{name}<kpt>{value}</kpt>")
                bbox = box["bbox"]
                bbox_text = ""
                if original_boxes and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    bbox_text = (
                        f"<bbox>[{norm(bbox[0], width):.3f}, {norm(bbox[1], height):.3f}, "
                        f"{norm(bbox[2], width):.3f}, {norm(bbox[3], height):.3f}]</bbox>"
                    )
                answers.append(
                    f"<p>{keypoint_item['phrase'].lower()}</p><ins>{bbox_text}{''.join(keypoint_parts)}</ins>"
                )
            if not answers:
                continue
            yield {
                "id": source_id,
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": f"<image><task: keypoint detection>: {prompt}"},
                    {"from": "gpt", "value": "".join(answers) + "."},
                ],
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert common keypoint JSONL to training JSONL."
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
        parser.error(f"unknown dataset {dataset_name!r}; choose one of {sorted(datasets)}")
    preset = datasets[dataset_name]
    input_path = args.input or resolve_tool_path(preset["default_intermediate"])
    output_path = args.output or resolve_tool_path(preset["default_output"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in iter_records(input_path, preset, args.seed, args.limit):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} keypoint training records to {output_path}")


if __name__ == "__main__":
    main()
