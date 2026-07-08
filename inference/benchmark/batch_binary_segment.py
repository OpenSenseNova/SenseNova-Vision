import argparse
import json
import os
import re

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

from data.prompts import ensure_image_placeholders
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from utils import (
    ensure_dir,
    load_jsonl_split,
    resolve_path,
    safe_filename_part,
    safe_get_single_image_from_output,
)
from utils.mask import to_binary_mask
from utils.timing import timer_context


def extract_binary_categories_from_prompt(prompt):
    """
    Extract all category information from the prompt:
    - Semantic categories inside <p>...</p>
    - Special category tags that appear, such as <point>, <box>, <scribble>, and <mask>

    Return a flat list, for example:
    ['cat', 'dog', 'scribble']
    """
    # 1. Extract the content inside <p>...</p>
    categories = re.findall(r"<p>(.*?)</p>", prompt)

    # 2. Check for special category tags that appear in the prompt
    special_tags = ["point", "box", "scribble", "mask"]
    for tag in special_tags:
        if f"<{tag}>" in prompt:
            categories.append(tag)

    return categories


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
        help="Path to jsonl file (each line contains image, seg, conversations)",
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
        "--data_path",
        type=str,
        default=None,
        help="Base directory for relative paths in jsonl (data root), not the jsonl directory.",
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
        help="Save binary prediction masks as PNG files.",
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

    predictions = []
    for sample_idx, line in enumerate(tqdm(lines)):
        sample_seed = args.seed + start_idx + sample_idx
        data = json.loads(line.strip())

        if "visual_prompt_type" in data:
            image_paths = data["image"]
            json_image_path = image_paths[0]
            json_visual_prompt_path = image_paths[1]

            image_path = resolve_path(json_image_path, data_root)
            visual_prompt_path = resolve_path(json_visual_prompt_path, data_root)
        else:
            json_image_path = data["image"]
            image_path = resolve_path(json_image_path, data_root)

        json_mask_path = data["seg"]
        gt_mask_path = resolve_path(json_mask_path, data_root)

        conversations = data["conversations"]
        prompt = conversations[0]["value"].strip()

        categories = extract_binary_categories_from_prompt(prompt)
        foreground_category = categories[0].strip()

        image = Image.open(image_path).convert("RGB")

        params = dict(BASE_PARAMS["dense_perception"])

        with timer_context(f"Inference {sample_idx}"):
            if "visual_prompt_type" in data:
                image_input_list = [image_path, visual_prompt_path]
            else:
                image_input_list = [image_path]
            output = model.generate(
                question=ensure_image_placeholders(prompt, len(image_input_list)),
                images=image_input_list,
                mode="dense_perception",
                noise_seed=sample_seed,
                return_intermediate_outputs=True,
                **params,
            )

        if output.get("text") is not None:
            print(f"=== Text Output ({sample_idx}) ===")
            print(output["text"])

        pred_mask_image = safe_get_single_image_from_output(output.get("image"))
        pred_mask_image = pred_mask_image.resize(image.size, resample=Image.NEAREST)

        pred_binary_mask = to_binary_mask(pred_mask_image)
        global_sample_idx = start_idx + sample_idx

        prediction_entry = {
            "idx": sample_idx,
            "global_idx": global_sample_idx,
            "file_name": image_path,
            "gt_name": gt_mask_path,
            "categories": categories,
        }
        try:
            encoded_mask = mask_util.encode(
                np.asfortranarray(pred_binary_mask.astype(np.uint8))
            )
            prediction_entry["pred_mask"] = encoded_mask
            prediction_entry["pred_mask"] = {
                "size": encoded_mask["size"],
                "counts": encoded_mask["counts"].decode("utf-8"),
            }
        except Exception as e:
            prediction_entry["pred_mask"] = {
                "size": list(pred_binary_mask.shape),
                "counts": [],
            }
            print(f"[Warning] failed to encode mask for idx={sample_idx}: {e}")
        predictions.append(prediction_entry)

        if args.save_pred_masks:
            safe_foreground_category = safe_filename_part(foreground_category)
            safe_gt_mask_stem = safe_filename_part(
                os.path.splitext(os.path.basename(gt_mask_path))[0]
            )
            mask_output_path = os.path.join(
                pred_mask_dir,
                f"sample_{global_sample_idx:06d}_{safe_gt_mask_stem}_{safe_foreground_category}_pred.png",
            )
            Image.fromarray((pred_binary_mask * 255).astype(np.uint8)).save(
                mask_output_path
            )

    prediction_json_path = os.path.join(
        args.output_dir, f"predictions_{start_idx:08d}_{end_idx:08d}.json"
    )
    with open(prediction_json_path, "w") as f:
        json.dump(predictions, f)
    print(f"Saved split predictions to {prediction_json_path}")


if __name__ == "__main__":
    main()
