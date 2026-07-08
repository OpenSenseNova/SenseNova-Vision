import argparse
import json
import os
from typing import Dict, List

from tqdm import tqdm


def load_coco_categories(coco_json_path: str) -> Dict:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    name_to_id = {
        cat["name"].lower().replace("_", " "): cat["id"]
        for cat in coco_data["categories"]
    }
    id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    return {
        "data": coco_data,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
    }


def build_image_path_to_id(coco_data: Dict) -> Dict[str, int]:
    return {
        os.path.basename(img["file_name"]): img["id"] for img in coco_data["images"]
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Directly convert our prediction JSONL (with extracted_predictions) to "
            "FastEval TSV using COCO/LVIS metadata."
        )
    )
    parser.add_argument(
        "--our_pred_jsonl",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/eval_results/box_eval/COCO/answer.jsonl",
        help="Path to our prediction jsonl (contains extracted_predictions)",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/coco/instances_val2017.json",
        help="Path to COCO/LVIS json with images/annotations/categories",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Optional image root used when input rows provide image_name instead of image_path",
    )
    parser.add_argument(
        "--out_tsv",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/eval_results/box_eval/COCO/fast_eval.tsv",
        help="Output TSV path for FastEval",
    )
    parser.add_argument(
        "--positive_only",
        action="store_true",
        default=True,
        help="Keep only GT-positive categories per image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Direct, single-step conversion: our_pred JSONL -> FastEval TSV
    info = load_coco_categories(args.coco_json)
    coco_data = info["data"]
    name_to_id = info["name_to_id"]
    id_to_name = info["id_to_name"]
    image_path_to_id = build_image_path_to_id(coco_data)

    # Build GT image_id -> cat_ids for positive filtering
    image_id_to_cat_ids: Dict[int, List[int]] = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        image_id_to_cat_ids.setdefault(image_id, []).append(cat_id)

    final_result: Dict[int, List[Dict]] = {}
    with open(args.our_pred_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting predictions to FastEval TSV"):
            data = json.loads(line)
            image_path = data.get("image_path") or os.path.join(
                args.image_root, data.get("image_name", "")
            )
            image_path = os.path.basename(image_path)
            if image_path not in image_path_to_id:
                raise ValueError(f"Image path {image_path} not found in coco json")
            image_id = image_path_to_id[image_path]

            extracted = data.get("extracted_predictions", {})
            entries: List[Dict] = []
            for cate_name, coords in extracted.items():
                cate_key = str(cate_name).lower().replace("_", " ")
                if cate_key not in name_to_id:
                    continue
                cat_id = name_to_id[cate_key]
                for coord in coords:
                    try:
                        x1, y1, x2, y2 = coord
                    except Exception:
                        continue
                    w = x2 - x1
                    h = y2 - y1
                    if args.positive_only and image_id in image_id_to_cat_ids:
                        if cat_id not in image_id_to_cat_ids[image_id]:
                            continue
                    entries.append(
                        {
                            "class": id_to_name[cat_id],
                            "conf": 0.2,
                            "rect": [x1, y1, w, h],
                        }
                    )
            final_result[image_id] = entries
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for image_id, results in final_result.items():
            f.write(f"{image_id}\t{json.dumps(results, ensure_ascii=False)}\n")

    print(f"Saved FastEval TSV to: {args.out_tsv}")
    return


if __name__ == "__main__":
    main()
