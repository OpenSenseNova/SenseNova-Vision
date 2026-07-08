import argparse
import json
import os

import numpy as np
from PIL import Image
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
from utils.mask import encode_mask
from utils.parsing_output import class_mask_from_raw_output
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
        help="Save GCG prediction masks as PNG files.",
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

    data_root = os.path.abspath(args.data_path) if args.data_path else None
    if data_root is None:
        print(
            "[WARN] --data_path is None; relative paths will be resolved "
            "against current working directory."
        )

    ensure_dir(args.output_dir)
    pred_mask_dir = args.pred_mask_dir or os.path.join(args.output_dir, "pred_masks")
    if args.save_pred_masks:
        ensure_dir(pred_mask_dir)

    predictions = []

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

    for sample_idx, line in enumerate(tqdm(lines)):
        sample_seed = args.seed + start_idx + sample_idx
        data = json.loads(line.strip())

        json_image_path = data["image"]
        image_path = resolve_path(json_image_path, data_root)

        image_file_name = os.path.basename(json_image_path)
        image_stem = os.path.splitext(image_file_name)[0]
        image_id = data.get("image_id", image_file_name.split(".")[0])

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
        # Convert predicted class indices to RLE masks in GCG phrase order.
        segmentation = []
        num_phrases = len(gcg_phrases)

        for phrase_idx in range(num_phrases):
            phrase_mask = (pred_class_mask == phrase_idx).astype(np.uint8)
            encoded_mask = encode_mask(phrase_mask)
            segmentation.append(encoded_mask)

        predictions.append(
            {
                "image_id": image_id,
                "idx": sample_idx,
                "global_idx": global_sample_idx,
                "file_name": image_file_name,
                "segmentation": segmentation,
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

    prediction_json_path = os.path.join(
        args.output_dir, f"predictions_{start_idx:08d}_{end_idx:08d}.json"
    )
    with open(prediction_json_path, "w") as f:
        json.dump(predictions, f)
    print(f"Saved COCO-style predictions to {prediction_json_path}")


if __name__ == "__main__":
    main()
