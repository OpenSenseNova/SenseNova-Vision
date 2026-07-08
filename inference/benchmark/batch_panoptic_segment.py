import argparse
import json
import os
from collections import defaultdict

import numpy as np
from panopticapi.utils import id2rgb
from PIL import Image
from tqdm import tqdm

from data.prompts import ensure_image_placeholders
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from utils import (
    ensure_dir,
    load_jsonl_split,
    normalize_category,
    resolve_path,
    safe_filename_part,
    safe_get_single_image_from_output,
)
from utils.parsing_output import class_mask_from_raw_output, parse_panoptic_phrase
from utils.timing import timer_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-Vision benchmark inference with JSONL input"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sensenova/SenseNova-Vision-7B-MoT",
        help="Path to SenseNova-Vision model directory",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to jsonl file (each line contains image and conversations; seg is ignored if present)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/", help="Path to save results"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for model")
    parser.add_argument(
        "--total_test_length",
        type=int,
        default=None,
        help="Total number of test samples. If not set, use the full jsonl.",
    )
    parser.add_argument(
        "--total_split", type=int, default=1, help="Total number of splits."
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=0,
        help="Current split index [0, total_split-1].",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        default="datas/gen_seg_data/coco2017/annotations/panoptic_val2017.json",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Base directory for relative paths in jsonl, not the jsonl directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Per-sample seed is seed + original JSONL index.",
    )
    parser.add_argument(
        "--save_pred_masks",
        action="store_true",
        help="Save panoptic prediction masks as PNG files.",
    )
    parser.add_argument(
        "--pred_mask_dir",
        type=str,
        default=None,
        help="Directory for --save_pred_masks. Defaults to OUTPUT_DIR/pred_masks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    pred_mask_dir = args.pred_mask_dir or os.path.join(args.output_dir, "pred_masks")
    if args.save_pred_masks:
        ensure_dir(pred_mask_dir)
    data_root = os.path.abspath(args.data_path) if args.data_path else None
    panoptic_eval_dir = os.path.join(args.output_dir, "panoptic_eval/")
    semantic_eval_dir = os.path.join(args.output_dir, "semantic_eval/")
    ensure_dir(panoptic_eval_dir)
    ensure_dir(semantic_eval_dir)

    with open(args.coco_json, "r") as f:
        prediction_data = json.load(f)

    category_name_to_id = dict()
    category_name_to_semantic_label_id = dict()
    for semantic_label_id, category in enumerate(prediction_data["categories"]):
        normalized_category_name = normalize_category(category["name"])
        category_name_to_id[normalized_category_name] = category["id"]
        category_name_to_semantic_label_id[normalized_category_name] = semantic_label_id
    file_name_to_image_id = {
        image_info["file_name"]: image_info["id"]
        for image_info in prediction_data["images"]
    }

    with timer_context("Init model"):
        model = SenseNovaVisionModel(
            model_path=args.model_path,
            device=args.device,
        )

    lines, start_idx, end_idx = load_jsonl_split(
        args.input_jsonl,
        total_test_length=args.total_test_length,
        total_split=args.total_split,
        split_num=args.split_num,
    )

    prediction_annotations = []
    for sample_idx, line in enumerate(tqdm(lines)):
        sample_seed = args.seed + start_idx + sample_idx
        data = json.loads(line.strip())

        json_image_path = data["image"]
        image_path = resolve_path(json_image_path, data_root)

        image_file_name = os.path.basename(json_image_path)
        image_stem = os.path.splitext(image_file_name)[0]
        mask_file_name = os.path.splitext(image_file_name)[0] + ".png"
        image_id = data.get("image_id", file_name_to_image_id.get(image_file_name))

        conversations = data["conversations"]
        prompt = conversations[0]["value"].strip()

        image = Image.open(image_path).convert("RGB")
        params = dict(BASE_PARAMS["caption_generate"])

        with timer_context(f"Inference {sample_idx}"):
            output = model.generate(
                question=ensure_image_placeholders(prompt, 1),
                images=[image_path],
                mode="caption_generate",
                noise_seed=sample_seed,
                return_intermediate_outputs=True,
                **params,
            )

        text_output = output.get("text", "")
        print(text_output)

        pred_mask_image = safe_get_single_image_from_output(output.get("image"))
        pred_mask_image = pred_mask_image.resize(image.size, resample=Image.NEAREST)
        parsed_mask = class_mask_from_raw_output(pred_mask_image, text_output)
        gcg_phrases = parsed_mask["gcg_phrases"]
        gcg_caption = parsed_mask["gcg_caption"]
        if not gcg_phrases:
            print(f"No color in {text_output}")
        global_sample_idx = start_idx + sample_idx

        pred_class_mask = parsed_mask["class_mask"]

        segments_info = []
        panoptic_mask = np.zeros_like(pred_class_mask)
        semantic_mask = np.ones_like(pred_class_mask) * 255
        occurrence_by_category_name = defaultdict(int)
        used_ids = set()
        for instance_id, phrase in enumerate(gcg_phrases):
            normalized_category, category_id, panoptic_id, _ = parse_panoptic_phrase(
                phrase, occurrence_by_category_name, used_ids, category_name_to_id
            )
            if category_id is None:
                print(f"Error: unhandled category {phrase}")
                continue
            if np.sum(pred_class_mask == instance_id) == 0:
                print(f"Error: No {phrase} in mask output")
                continue
            semantic_label_id = int(
                category_name_to_semantic_label_id[normalized_category]
            )
            panoptic_mask[pred_class_mask == instance_id] = panoptic_id
            semantic_mask[pred_class_mask == instance_id] = semantic_label_id
            segments_info.append(
                {
                    "id": panoptic_id,
                    "category_id": category_id,
                    "score": 1.0,
                }
            )
        panoptic_rgb = id2rgb(panoptic_mask.astype(np.int32))
        semantic_label_mask = semantic_mask.astype(np.uint8)
        Image.fromarray(panoptic_rgb).save(
            os.path.join(panoptic_eval_dir, mask_file_name), "PNG"
        )
        Image.fromarray(semantic_label_mask).save(
            os.path.join(semantic_eval_dir, mask_file_name), "PNG"
        )

        prediction_annotations.append(
            {
                "image_id": image_id,
                "idx": sample_idx,
                "global_idx": global_sample_idx,
                "file_name": mask_file_name,
                "segments_info": segments_info,
                "gcg_phrases": gcg_phrases,
                "gcg_caption": gcg_caption,
            }
        )

        if args.save_pred_masks:
            safe_image_id = safe_filename_part(image_id)
            safe_image_stem = safe_filename_part(image_stem)
            mask_output_path = os.path.join(
                pred_mask_dir,
                f"sample_{global_sample_idx:06d}_{safe_image_stem}_{safe_image_id}_pred.png",
            )
            pred_mask_image.save(mask_output_path)

    prediction_data["annotations"] = prediction_annotations
    prediction_json_path = os.path.join(
        args.output_dir, f"predictions_{start_idx:08d}_{end_idx:08d}.json"
    )
    with open(prediction_json_path, "w") as f:
        json.dump(prediction_data, f)
    print(f"Saved COCO-style predictions to {prediction_json_path}")


if __name__ == "__main__":
    main()
