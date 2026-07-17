import argparse
import json
import os
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "coco2017"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "jsonl_train"
    / "keypoints"
    / "coco2017"
    / "processed_keypoints_train.jsonl"
)

DEFAULT_ANNOTATION = "annotations/person_keypoints_train2017.json"

KEYPOINT_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]

PROMPTS = [
    "Detect all instances of PHRASE in the image. For each instance, output a bounding box in <bbox> format and the coordinates of its KEYPOINTS as [x, y] pairs. Return results as a structured list.",
    "Locate and identify PHRASE within the scene. For each detected object, provide its bounding box and KEYPOINTS coordinates. Output must be a text list—distinct from image-based outputs like segmentation masks.",
    "Find all objects belonging to the categories: PHRASE. For each, return a text entry containing category name, normalized bounding box coordinates, and KEYPOINTS as a list of normalized [x, y] values.",
    "Detect all instances of PHRASE in the image. Unlike depth or pose visualization tasks, output structured text with each object class, <bbox>, and KEYPOINTS coordinates for further processing.",
    "Identify objects from the following categories: PHRASE. For each, output category, bounding box, and KEYPOINTS. This text-based response differs from visual outputs used in other vision tasks.",
    "Detect all PHRASE present in the image. Provide results as structured text, where each detection includes class label, bounding box in <bbox> format, and KEYPOINTS as normalized [x, y] coordinates.",
    "Find and classify objects from the categories: PHRASE. Return detection and keypoint results with each object represented by its class, normalized bounding box, and a list of normalized KEYPOINTS in [x, y] format.",
    "Locate and list PHRASE visible in the image. For each instance, output category, normalized bounding box coordinates, and KEYPOINTS positions as [x, y] pairs. Format the entire response as valid text.",
    "Detect objects from these categories: PHRASE. Unlike tasks that produce images (e.g., segmentation), return a list where each entry contains class, <bbox>, and KEYPOINTS coordinates for precise localization.",
    "Perform joint object and keypoint detection for PHRASE. Output a response listing each object with its category, normalized bounding box, and KEYPOINTS as an array of normalized [x, y] coordinate pairs."
]


def truncate_norm(value: float) -> float:
    return min(max(value, 0.0), 0.999)


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


def normalize_keypoint_name(name: str) -> str:
    return name.replace("_", " ").strip()


def format_point(x: float, y: float, width: int, height: int) -> str:
    # We map keypoints to 0.000-0.999 using the original COCO image size.
    nx = truncate_norm(x / width)
    ny = truncate_norm(y / height)
    return f"[{nx:.3f}, {ny:.3f}]"


def format_bbox(bbox, width: int, height: int) -> str:
    x, y, w, h = bbox
    x0 = truncate_norm(x / width)
    y0 = truncate_norm(y / height)
    x1 = truncate_norm((x + w) / width)
    y1 = truncate_norm((y + h) / height)
    return f"<bbox>[{x0:.3f}, {y0:.3f}, {x1:.3f}, {y1:.3f}]</bbox>"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_records(
    dataset_root: str,
    annotation_file: str,
    image_dir: str,
    image_prefix: str,
    strip_image_prefix: str,
    keypoints_field: str,
    category_id: int,
    seed: int,
    limit: int | None = None,
):
    random.seed(seed)
    records = []
    missing_image_count = 0
    coco = load_json(os.path.join(dataset_root, annotation_file))
    categories = {
        item["id"]: normalize_category(item["name"])
        for item in coco["categories"]
    }
    keypoint_names = KEYPOINT_NAMES
    for item in coco["categories"]:
        if item["id"] == category_id and item.get("keypoints"):
            keypoint_names = [normalize_keypoint_name(name) for name in item["keypoints"]]
            break
    anns_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("category_id") != category_id or keypoints_field not in ann:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_info in coco["images"]:
        image_id = image_info["id"]
        image_anns = anns_by_image.get(image_id, [])
        if not image_anns:
            continue

        width = image_info["width"]
        height = image_info["height"]
        image_name = image_info["file_name"]
        if strip_image_prefix and image_name.startswith(strip_image_prefix):
            image_name = image_name[len(strip_image_prefix) :]
        image_path = os.path.join(dataset_root, image_dir, image_name)
        if not os.path.exists(image_path):
            missing_image_count += 1
            continue

        phrase_order = []
        for ann in image_anns:
            phrase = categories.get(ann["category_id"], "").strip()
            if phrase and phrase not in phrase_order:
                phrase_order.append(phrase)
        if not phrase_order:
            continue

        prompt = random.choice(PROMPTS)
        prompt = prompt.replace(
            "PHRASE",
            ", ".join(f"<p>{phrase}</p>" for phrase in phrase_order),
        ).replace("KEYPOINTS", ", ".join(keypoint_names))

        answer_parts = []
        for ann in image_anns:
            phrase = categories.get(ann["category_id"], "").strip()
            if not phrase:
                continue

            bbox_text = format_bbox(ann["bbox"], width, height)
            keypoints = ann[keypoints_field]
            keypoint_parts = []
            for idx, keypoint_name in enumerate(keypoint_names):
                x = keypoints[idx * 3]
                y = keypoints[idx * 3 + 1]
                v = keypoints[idx * 3 + 2]
                if v <= 0 or (x <= 0 and y <= 0):
                    value = "unvisible"
                else:
                    value = format_point(x, y, width, height)
                keypoint_parts.append(f"{keypoint_name}<kpt>{value}</kpt>")

            answer_parts.append(
                f"<p>{phrase}</p><ins>{bbox_text}{''.join(keypoint_parts)}</ins>"
            )

        if not answer_parts:
            continue

        records.append(
            {
                "id": len(records),
                "image": f"{image_prefix.rstrip('/')}/{image_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>{prompt}"},
                    {"from": "gpt", "value": "".join(answer_parts) + "."},
                ],
            }
        )
        if limit is not None and len(records) >= limit:
            if missing_image_count:
                print(f"Skipped {missing_image_count} annotated images because train2017 files were missing.")
            return records

    if missing_image_count:
        print(f"Skipped {missing_image_count} annotated images because train2017 files were missing.")
    return records


def write_jsonl(records, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO 2017 keypoints to training jsonl.")
    parser.add_argument("--input", default=str(DEFAULT_ROOT), help="COCO 2017 dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output jsonl path.")
    parser.add_argument(
        "--annotation",
        default=DEFAULT_ANNOTATION,
        help="Annotation file relative to dataset root.",
    )
    parser.add_argument(
        "--image-dir",
        default="train2017",
        help="Image directory relative to the dataset root.",
    )
    parser.add_argument(
        "--image-prefix",
        default="coco2017/train2017",
        help="Image prefix written to each JSONL record.",
    )
    parser.add_argument(
        "--strip-image-prefix",
        default="",
        help="Optional prefix removed from annotation file_name values.",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category id containing the target keypoint schema.",
    )
    parser.add_argument(
        "--keypoints-field",
        default="keypoints",
        help="Annotation field containing flattened x/y/visibility triplets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for debugging.")
    args = parser.parse_args()

    records = build_records(
        args.input,
        args.annotation,
        image_dir=args.image_dir,
        image_prefix=args.image_prefix,
        strip_image_prefix=args.strip_image_prefix,
        keypoints_field=args.keypoints_field,
        category_id=args.category_id,
        seed=args.seed,
        limit=args.limit,
    )
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
