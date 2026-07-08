import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from data.prompts import ensure_image_placeholders, strip_task_prompt
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from utils import ensure_dir, normalize_category
from utils.parsing_output import (
    convert_detection_output_to_pixel,
    parse_detection_text_output,
)
from utils.timing import timer_context

OCR_CASE_SENSITIVE_DATASETS = {"SROIE", "HierText", "IC15", "TotalText"}


def make_summary_filename(input_jsonl: str, task_name: str, split_num: int) -> str:
    input_basename = os.path.basename(input_jsonl)
    input_name, input_ext = os.path.splitext(input_basename)
    prefix_map = {
        "pointing": "point.",
        "pointing_referring": "point.",
        "visual_prompt_detection": "visual.",
        "gui": "gui.",
        "keypoint": "keypoint.",
    }
    prefix = prefix_map.get(task_name, "")
    return f"{prefix}{input_name}_{split_num:03d}{input_ext}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-Vision detection benchmark inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sensenova/SenseNova-Vision-7B-MoT",
        help="Path to SenseNova-Vision model directory",
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_det/")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datas/detection_data",
        help="Root directory for detection benchmark images.",
    )
    parser.add_argument(
        "--mode", type=str, choices=list(BASE_PARAMS.keys()), default="understanding"
    )  # generate
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_test_length", type=int, default=None)
    parser.add_argument("--total_split", type=int, default=1)
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument(
        "--save_single_files",
        action="store_true",
        help="Whether to keep per-image JSON files (by default only the summary file is saved).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Per-sample seed is seed + original JSONL index.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=[
            "common_object_detection",
            "referring_object_detection",
            "pointing",
            "pointing_referring",
            "visual_prompt_detection",
            "gui",
            "keypoint",
        ],
        default="common_object_detection",
        help=(
            "Specify the task name, choose from common_object_detection or "
            "referring_object_detection"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # ===== Put optional per-image results in a task-specific subdirectory =====
    single_dir = os.path.join(args.output_dir, "single", args.task_name)

    if args.save_single_files:
        ensure_dir(single_dir)

    summary_filename = make_summary_filename(
        args.input_jsonl, args.task_name, args.split_num
    )
    summary_file_path = os.path.join(args.output_dir, summary_filename)

    with timer_context("Init model"):
        model = SenseNovaVisionModel(
            model_path=args.model_path,
            device=args.device,
        )

    with open(args.input_jsonl, "r") as f:
        lines = f.readlines()

    total_len = args.total_test_length or len(lines)
    lines = lines[:total_len]

    split_size = (total_len + args.total_split - 1) // args.total_split
    start_idx = args.split_num * split_size
    end_idx = min(start_idx + split_size, total_len)
    lines = lines[start_idx:end_idx]

    print(
        f"Processing split {args.split_num}/{args.total_split}, range [{start_idx}, {end_idx})"
    )
    print(f"Summary file will be saved to: {summary_file_path}")
    print(f"Current task type: {args.task_name}")

    # Open the summary file and prepare to write one line at a time
    with open(summary_file_path, "w", encoding="utf-8") as summary_file:
        for idx, line in enumerate(tqdm(lines)):
            sample_seed = args.seed + start_idx + idx
            data = json.loads(line.strip())
            image_path = data["image"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(args.data_root, image_path)

            # Extract GT and prompt
            gt_text = ""
            prompt = ""
            for conv in data["conversations"]:
                if conv["from"] == "gpt":
                    gt_text = conv["value"]
                elif conv["from"] == "human":
                    prompt = conv["value"].strip()
            prompt = strip_task_prompt(prompt, "no_prefix_promt")
            dataset_name = data.get("dataset_name") or os.path.splitext(
                os.path.basename(args.input_jsonl)
            )[0]
            preserve_ocr_case = (
                args.task_name == "common_object_detection"
                and dataset_name in OCR_CASE_SENSITIVE_DATASETS
            )

            gt_detections = parse_detection_text_output(
                gt_text, "bbox", normalize_labels=not preserve_ocr_case
            )

            # Load image
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            params = dict(BASE_PARAMS[args.mode])

            with timer_context(f"Inference {idx}"):
                output = model.generate(
                    question=ensure_image_placeholders(prompt, 1),
                    images=[image_path],
                    mode=args.mode,
                    noise_seed=sample_seed,
                    return_intermediate_outputs=True,
                    **params,
                )
            category = None
            text_output = output.get("text", "").strip()
            if not text_output:
                print(f"Warning: No text output for {image_path}")
                detections = {}
            else:
                # ======== New: keypoint task processing branch ========
                if args.task_name == "keypoint":
                    # Parse model output (normalized coordinates)
                    raw_predictions = parse_detection_text_output(
                        text_output, "keypoint"
                    )
                    # import pdb; pdb.set_trace()

                    # Convert to pixel coordinates and generate phrase
                    extracted_predictions = {}
                    pixel_predictions = convert_detection_output_to_pixel(
                        raw_predictions, orig_w, orig_h, "keypoint"
                    )
                    for cat, instances in pixel_predictions.items():
                        cat_instances = []
                        for idx, inst in enumerate(instances, 1):
                            # Generate phrase (for example, "antelope1")
                            phrase = f"{cat}{idx}"

                            # Build instance
                            instance_dict = {
                                "bbox": inst.get("bbox"),
                                "keypoints": inst["keypoints"],
                                "phrase": phrase,
                            }
                            cat_instances.append(instance_dict)

                        extracted_predictions[cat] = cat_instances

                    # Parse GT from conversations
                    gt_raw = (
                        parse_detection_text_output(gt_text, "keypoint")
                        if gt_text
                        else {}
                    )

                    # Convert GT to target format: pixel coordinates,
                    # normalized keypoint names, and filtered invisible values.
                    gt_pixel_instances = {}
                    for cat, instances in gt_raw.items():
                        cat_instances = []
                        for (
                            inst
                        ) in (
                            instances
                        ):  # Iterate over GT instances without adding indexes
                            # Convert bbox to pixel coordinates (integer)
                            pixel_bbox = None
                            if "bbox" in inst and inst["bbox"]:
                                x1, y1, x2, y2 = [float(x) for x in inst["bbox"]]
                                pixel_bbox = [
                                    round(x1 * orig_w),
                                    round(y1 * orig_h),
                                    round(x2 * orig_w),
                                    round(y2 * orig_h),
                                ]

                            # Convert keypoints to integer pixel coordinates.
                            pixel_keypoints = {}
                            for name, coords in inst.get("keypoints", {}).items():
                                # Filter unvisible points (coordinates are [-1,-1] or None)
                                if coords is None or (
                                    isinstance(coords, list) and coords[0] == -1
                                ):
                                    continue
                                # Keypoint names: replace spaces with underscores
                                name_underscore = name.replace(" ", "_")
                                # Convert to pixel coordinates and round to integer
                                x_norm, y_norm = coords
                                x_pixel = round(x_norm * orig_w)
                                y_pixel = round(y_norm * orig_h)
                                pixel_keypoints[name_underscore] = [x_pixel, y_pixel]

                            # Phrase has no index (required by GT format)
                            phrase = cat

                            instance_dict = {
                                "bbox": pixel_bbox,
                                "keypoints": pixel_keypoints,
                                "phrase": phrase,
                            }
                            cat_instances.append(instance_dict)
                        gt_pixel_instances[cat] = cat_instances
                    # import pdb; pdb.set_trace()

                    result_item = {
                        "image_path": data["image"],
                        "extracted_predictions": extracted_predictions,
                        "gt": gt_pixel_instances,
                        "question": prompt,
                        "dataset_name": dataset_name,
                        "raw_response": text_output,
                        "task_name": args.task_name,
                        # Dynamically extract all keypoint names that appear (for statistics)
                        "keypoint_names": list(
                            {
                                name
                                for instances in extracted_predictions.values()
                                for inst in instances
                                for name in inst.get("keypoints", {}).keys()
                            }
                        ),
                    }
                # ======== End of keypoint branch ========
                elif args.task_name == "visual_prompt_detection":
                    # Parse model output, but force category to data["category"]
                    raw_detections = parse_detection_text_output(text_output, "bbox")
                    # Map all <p>object1</p> entries to the real category
                    category = normalize_category(data.get("category", "unknown"))
                    detections = {category: []}
                    for bboxes in raw_detections.values():
                        detections[category].extend(bboxes)
                elif "point" in args.task_name:
                    detections = parse_detection_text_output(text_output, "point")
                else:
                    detections = parse_detection_text_output(
                        text_output, "bbox", normalize_labels=not preserve_ocr_case
                    )

            if args.task_name != "keypoint":
                # Convert normalized predictions to pixel coordinates.
                if "point" in args.task_name:
                    extracted_predictions = convert_detection_output_to_pixel(
                        detections, orig_w, orig_h, "point"
                    )
                else:
                    extracted_predictions = convert_detection_output_to_pixel(
                        detections, orig_w, orig_h, "bbox"
                    )

                if args.task_name == "gui":
                    # Convert normalized GT bbox to pixel coordinates (kept to two decimal places)
                    gt_pixels = convert_detection_output_to_pixel(
                        gt_detections, orig_w, orig_h, "bbox"
                    )
                    for cat, bboxes in gt_pixels.items():
                        gt_pixel = bboxes[0]
                elif args.task_name == "visual_prompt_detection":
                    gt_pixel = {}
                    # Handle GT similarly: GT also uses object1, but should map to the real category
                    gt_category = normalize_category(data.get("category", "unknown"))
                    if gt_detections:
                        all_gt_bboxes = []
                        for bboxes in gt_detections.values():
                            all_gt_bboxes.extend(bboxes)
                        gt_pixel = {
                            gt_category: convert_detection_output_to_pixel(
                                {gt_category: all_gt_bboxes},
                                orig_w,
                                orig_h,
                                "bbox",
                            )[gt_category]
                        }
                    else:
                        gt_pixel = {gt_category: []}
                else:
                    gt_pixel = convert_detection_output_to_pixel(
                        gt_detections, orig_w, orig_h, "bbox"
                    )

                # Build result item
                result_item = {
                    "image_path": data["image"],
                    "category": category,
                    "extracted_predictions": extracted_predictions,
                    "gt": data["gt_mask"] if "point" in args.task_name else gt_pixel,
                    "question": prompt,
                    "dataset_name": dataset_name,
                    "raw_response": text_output,
                    "task_name": args.task_name,
                }

            # Write to summary file (one JSON object per line)
            json.dump(result_item, summary_file, ensure_ascii=False)
            summary_file.write("\n")

            # Optional: save individual files (split_num can also be added if needed)
            if args.save_single_files:
                # Individual files can include a split marker when needed.
                img_basename = (
                    os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
                )
                out_name = f"{args.task_name}_{img_basename}.json"
                out_path = os.path.join(single_dir, out_name)
                with open(out_path, "w") as f:
                    json.dump(result_item, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Summary results saved to: {summary_file_path}")
    print(f"Task type: {args.task_name}")
    if args.save_single_files:
        print(
            f"Individual result files saved to: {args.output_dir} "
            f"(filenames include split{args.split_num:03d} marker)"
        )
    else:
        print(
            "All detection results have been summarized and saved to "
            f"{summary_file_path} (individual files were not saved)"
        )


if __name__ == "__main__":
    main()
