import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.colormap import build_palette  # noqa: E402
from utils import (  # noqa: E402
    load_json_or_jsonl,
    load_jsonl,
    resolve_path,
    safe_stem,
)
from utils.parsing_output import (  # noqa: E402
    semantic_category_maps_from_panoptic_json,
)
from utils.visualize import (  # noqa: E402
    VisualizationConfig,
    draw_visual_prompt,
    visualize_binary_segmentation,
    visualize_concat_col,
    visualize_gcg_prediction,
    visualize_gcg_segmentation,
    visualize_panoptic_prediction,
    visualize_semantic_segmentation,
)


INTERACTIVE_PROMPT_STYLES = {
    "point": "fill",
    "scribble": "fill",
    "box": "boundary",
    "mask": "boundary",
}


def parse_root_args(values):
    roots = []
    for value in values or []:
        for part in str(value).replace(",", " ").split():
            part = part.strip().strip("[]").strip("'\"")
            if part:
                roots.append(part)
    return roots


def prediction_jsons_from_root(root):
    if os.path.isfile(root):
        return [root]
    merged_path = os.path.join(root, "predictions.json")
    if os.path.exists(merged_path):
        return [merged_path]
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(root, "predictions_*.json"))))
    candidates.extend(sorted(glob.glob(os.path.join(root, "predictions_*.jsonl"))))
    return candidates


def load_prediction_jsons(root):
    predictions = []
    paths = prediction_jsons_from_root(root)
    for path in paths:
        data = load_json_or_jsonl(path)
        if isinstance(data, dict):
            if "annotations" in data:
                predictions.extend(data["annotations"])
            else:
                predictions.append(data)
        else:
            predictions.extend(data)
    return predictions, paths


def load_panoptic_predictions(root):
    paths = prediction_jsons_from_root(root)
    if not paths:
        raise FileNotFoundError(f"No predictions.json or predictions_*.json found under: {root}")

    merged = {
        "info": None,
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    seen_images = set()
    seen_categories = set()
    for path in paths:
        data = load_json_or_jsonl(path)
        if isinstance(data, list):
            merged["annotations"].extend(data)
            continue
        if not isinstance(data, dict):
            continue

        if merged["info"] is None and data.get("info") is not None:
            merged["info"] = data.get("info")
        if not merged["licenses"] and data.get("licenses"):
            merged["licenses"] = data.get("licenses")

        for image in data.get("images", []):
            key = image.get("id", image.get("file_name", str(image)))
            if key not in seen_images:
                merged["images"].append(image)
                seen_images.add(key)
        for category in data.get("categories", []):
            key = category.get("id", category.get("name", str(category)))
            if key not in seen_categories:
                merged["categories"].append(category)
                seen_categories.add(key)

        if "annotations" in data:
            merged["annotations"].extend(data["annotations"])
        elif data:
            merged["annotations"].append(data)

    if merged["info"] is None:
        merged.pop("info")
    return merged, paths


def load_record_index(input_jsonl, data_path):
    records = []
    index = {}
    if not input_jsonl:
        return records, index
    for row_idx, item in enumerate(load_jsonl(input_jsonl)):
        image_value = item.get("image") or item.get("image_path")
        raw_image = image_value[0] if isinstance(image_value, list) else image_value
        raw_prompt = image_value[1] if isinstance(image_value, list) and len(image_value) > 1 else None
        raw_seg = item.get("seg")
        file_name = os.path.basename(str(raw_image)) if raw_image else str(row_idx)
        image_id = str(item.get("image_id", os.path.splitext(file_name)[0]))
        record = {
            "row_idx": row_idx,
            "raw_image": raw_image,
            "raw_seg": raw_seg,
            "image_path": resolve_path(raw_image, data_path),
            "visual_prompt_path": resolve_path(raw_prompt, data_path) if raw_prompt else None,
            "visual_prompt_type": item.get("visual_prompt_type", ""),
            "seg_path": resolve_path(raw_seg, data_path) if raw_seg else None,
            "file_name": file_name,
            "image_id": image_id,
            "caption": item.get("caption", ""),
            "prompt": extract_prompt(item),
            "gt_rich_caption": extract_gpt_response(item),
        }
        records.append(record)
        for key in (image_id, file_name, os.path.splitext(file_name)[0], f"row:{row_idx}"):
            index[str(key)] = record
    return records, index


def extract_prompt(item):
    sent_info = item.get("sent_info") or {}
    for key in ("raw", "sent"):
        if sent_info.get(key):
            return str(sent_info[key])
    for conv in item.get("conversations", []):
        if conv.get("from") == "human":
            return str(conv.get("value", ""))
    return str(item.get("caption", ""))


def extract_gpt_response(item):
    for conv in item.get("conversations", []):
        if conv.get("from") == "gpt":
            return str(conv.get("value", ""))
    return ""


def find_record_for_ann(ann, index):
    keys = [
        ann.get("image_id"),
        ann.get("file_name"),
        os.path.splitext(str(ann.get("file_name", "")))[0],
        ann.get("idx"),
        ann.get("global_idx"),
    ]
    for key in keys:
        if key is None:
            continue
        key = str(key)
        if key in index:
            return index[key]
        row_key = f"row:{key}"
        if row_key in index:
            return index[row_key]
    return None


def normalize_prediction_path(path):
    if not path:
        return ""
    return os.path.normpath(str(path).lstrip("./"))


def find_record_for_pred(pred, index):
    image_id = str(pred.get("image_id", ""))
    file_name = str(pred.get("file_name", ""))
    stem = os.path.splitext(file_name)[0]
    return index.get(image_id) or index.get(file_name) or index.get(stem)


def interactive_record_index(records):
    seg_index = {}
    file_to_records = {}
    file_counts = {}
    for record in records:
        for value in (record.get("raw_seg"), record.get("seg_path")):
            if not value:
                continue
            seg_index[normalize_prediction_path(value)] = record
            seg_index[os.path.basename(value)] = record
        for value in (record.get("raw_image"), record.get("image_path"), record.get("file_name")):
            if not value:
                continue
            for key in (normalize_prediction_path(value), os.path.basename(value)):
                file_to_records[key] = record
                file_counts[key] = file_counts.get(key, 0) + 1
    unique_file_index = {key: record for key, record in file_to_records.items() if file_counts.get(key) == 1}
    return seg_index, unique_file_index


def find_record_for_interactive_pred(pred, records, index, record_indexes):
    seg_index, unique_file_index = record_indexes
    gt_name = pred.get("gt_name")
    if gt_name:
        record = seg_index.get(normalize_prediction_path(gt_name)) or seg_index.get(os.path.basename(gt_name))
        if record is not None:
            return record
    file_name = pred.get("file_name")
    if file_name:
        record = unique_file_index.get(normalize_prediction_path(file_name)) or unique_file_index.get(os.path.basename(file_name))
        if record is not None:
            return record
    for key_name in ("global_idx", "idx"):
        if key_name in pred:
            try:
                row_idx = int(pred[key_name])
            except (TypeError, ValueError):
                row_idx = -1
            if 0 <= row_idx < len(records):
                return records[row_idx]
            record = index.get(f"row:{pred[key_name]}")
            if record is not None:
                return record
    return find_record_for_pred(pred, index)


def find_image_path(name, record=None, image_root=None, data_path=None):
    candidates = []
    if record and record.get("image_path"):
        candidates.append(record["image_path"])
    text = str(name)
    name_variants = [text]
    if "_gtFine_leftImg8bit" in text:
        name_variants.append(text.replace("_gtFine_leftImg8bit", "_leftImg8bit"))
    if "_leftImg8bit" in text and "_gtFine_leftImg8bit" not in text:
        name_variants.append(text.replace("_leftImg8bit", "_gtFine_leftImg8bit"))
    coco_match = re.search(r"(COCO_(?:train|val)2014_\d{12})", text)
    if coco_match:
        coco_name = coco_match.group(1) + ".jpg"
        for base in (data_path, image_root):
            if base:
                candidates.append(os.path.join(base, "datas/ref_seg_data/images/coco2014/train2014", coco_name))
                candidates.append(os.path.join(base, "coco/train2014", coco_name))
                candidates.append(os.path.join(base, coco_name))
    for base in (image_root, data_path):
        if not base:
            continue
        for variant in name_variants:
            candidates.append(os.path.join(base, variant))
            basename = os.path.basename(variant)
            stem = os.path.splitext(basename)[0]
            city = basename.split("_", 1)[0] if "_" in basename else ""
            for city_base in (
                os.path.join(base, "leftImg8bit", "val", city) if city else None,
                os.path.join(base, "val", city) if city else None,
                os.path.join(base, city) if city else None,
            ):
                if city_base:
                    candidates.append(os.path.join(city_base, basename))
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(os.path.join(base, stem + ext))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def output_dir(root, args, suffix=None):
    if args.vis_dir:
        path = args.vis_dir
    else:
        path = os.path.join(root if os.path.isdir(root) else os.path.dirname(root), f"vis_concat{args.concat_col}")
    if suffix:
        path = os.path.join(path, suffix)
    os.makedirs(path, exist_ok=True)
    return path


def should_skip_missing(path, args, what):
    if path and os.path.exists(path):
        return False
    if args.skip_missing:
        return True
    raise FileNotFoundError(f"{what} not found: {path}")


def panoptic_png_path(pred_dir, file_name):
    candidates = [
        os.path.join(pred_dir, file_name),
        os.path.join(pred_dir, "panoptic_eval", file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def visualize_panoptic_root(root, args, palette):
    data, paths = load_panoptic_predictions(root)
    anns = data.get("annotations", [])
    categories = data.get("categories", [])
    records, index = load_record_index(args.input_jsonl, args.data_path)
    pred_dir = os.path.dirname(paths[0]) if os.path.isfile(root) else root
    out_dir = output_dir(root, args)
    config = make_config(args)

    written = 0
    skipped = 0
    for ann in tqdm(anns[: args.limit] if args.limit else anns, desc="panoptic"):
        file_name = ann.get("file_name")
        panoptic_path = panoptic_png_path(pred_dir, file_name)
        if should_skip_missing(panoptic_path, args, "panoptic png"):
            skipped += 1
            continue
        record = find_record_for_ann(ann, index)
        image_path = find_image_path(file_name, record, args.image_root, args.data_path)
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        pred = visualize_panoptic_prediction(
            image,
            {"png": panoptic_path, "annotation": ann, "categories": categories},
            categories=categories,
            palette=palette,
            config=config,
        )
        gt_panel = None
        if args.concat_col == 3 and record and record.get("seg_path") and os.path.exists(record["seg_path"]):
            gt_text = record.get("gt_rich_caption") or record.get("caption") or record.get("prompt")
            gt_panel = visualize_gcg_segmentation(
                image,
                record["seg_path"],
                gt_text,
                palette=palette,
                config=config,
            )
        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel)
        final.save(os.path.join(out_dir, safe_stem(Path(file_name).stem) + ".png"))
        written += 1
    return {"task": "panoptic", "out_dir": out_dir, "written": written, "skipped": skipped}


def prediction_category_path_from_dir(root):
    paths = prediction_jsons_from_root(root)
    for path in paths:
        data = load_json_or_jsonl(path)
        if isinstance(data, dict) and data.get("categories"):
            return path
    return None


def semantic_category_path(root):
    root = os.path.abspath(root)
    candidate_dirs = [
        root,
        os.path.dirname(root),
        os.path.join(root, "panoptic_eval"),
        os.path.join(os.path.dirname(root), "panoptic_eval"),
    ]
    for directory in candidate_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        path = prediction_category_path_from_dir(directory)
        if path:
            return path
    return None


def semantic_mask_values(paths, limit=8):
    values = set()
    for path in paths[:limit]:
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        for value in np.unique(arr):
            value = int(value)
            if value != 255:
                values.add(value)
    return values


def semantic_mask_path_for_name(pred_dir, name):
    basename = os.path.basename(str(name))
    stem = os.path.splitext(basename)[0]
    stems = [stem]
    if "_gtFine_leftImg8bit" in stem:
        stems.append(stem.replace("_gtFine_leftImg8bit", "_leftImg8bit"))
    if "_leftImg8bit" in stem and "_gtFine_leftImg8bit" not in stem:
        stems.append(stem.replace("_leftImg8bit", "_gtFine_leftImg8bit"))
    for item in stems:
        path = os.path.join(pred_dir, item + ".png")
        if os.path.exists(path):
            return path
    return os.path.join(pred_dir, stems[0] + ".png")


def select_semantic_categories(category_path, paths, key_mode="auto"):
    contiguous_categories, category_id_categories = semantic_category_maps_from_panoptic_json(category_path)
    if key_mode == "contiguous":
        return contiguous_categories
    if key_mode == "id":
        return category_id_categories
    if key_mode != "auto":
        raise ValueError(f"Unsupported semantic category key mode: {key_mode}")

    values = semantic_mask_values(paths)
    if values:
        contiguous_keys = set(contiguous_categories)
        category_id_keys = set(category_id_categories)
        if values.issubset(category_id_keys) and not values.issubset(contiguous_keys):
            return category_id_categories
    return contiguous_categories


def visualize_semantic_root(root, args, palette):
    category_path = args.category_json or semantic_category_path(root)
    if not category_path:
        raise FileNotFoundError("Cannot locate panoptic_eval/predictions.json for semantic categories.")
    records, index = load_record_index(args.input_jsonl, args.data_path)
    pred_dir = os.path.join(root, "semantic_eval") if os.path.isdir(os.path.join(root, "semantic_eval")) else root
    paths = sorted(glob.glob(os.path.join(pred_dir, "*.png"))) if os.path.isdir(pred_dir) else [pred_dir]
    categories = select_semantic_categories(category_path, paths, args.semantic_category_key)
    out_dir = output_dir(pred_dir, args)
    config = make_config(args)

    written = 0
    skipped = 0
    if records:
        items = records[: args.limit] if args.limit else records
    else:
        items = paths[: args.limit] if args.limit else paths

    for item in tqdm(items, desc="semantic"):
        if isinstance(item, dict):
            record = item
            name = record["file_name"]
            sem_path = semantic_mask_path_for_name(pred_dir, name)
        else:
            sem_path = item
            name = os.path.basename(sem_path)
            record = index.get(os.path.splitext(name)[0]) or index.get(name)
        if should_skip_missing(sem_path, args, "semantic png"):
            skipped += 1
            continue
        image_path = find_image_path(name, record, args.semantic_image_dir or args.image_root, args.data_path)
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        pred = visualize_semantic_segmentation(image, sem_path, categories, palette=palette, config=config)
        gt_panel = None
        if args.semantic_gt_dir:
            gt_path = semantic_mask_path_for_name(args.semantic_gt_dir, name)
            if gt_path and os.path.exists(gt_path):
                gt_panel = visualize_semantic_segmentation(image, gt_path, categories, palette=palette, config=config)
        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel)
        final.save(os.path.join(out_dir, safe_stem(Path(name).stem) + "_semantic.png"))
        written += 1
    return {"task": "semantic", "out_dir": out_dir, "written": written, "skipped": skipped}


def mask_paths(root):
    if os.path.isfile(root):
        return [root]
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(sorted(glob.glob(os.path.join(root, ext))))
    return paths


def visualize_mask_root(root, args, palette):
    json_paths = prediction_jsons_from_root(root)
    if args.task_type == "gcg" and json_paths:
        return visualize_gcg_json_root(root, args, palette)
    if args.task_type in INTERACTIVE_PROMPT_STYLES and json_paths:
        return visualize_interactive_json_root(root, args, palette)
    if json_paths:
        return visualize_binary_json_root(root, args, palette)

    records, index = load_record_index(args.input_jsonl, args.data_path)
    pred_mask_index = load_pred_mask_sidecar_index(root)
    paths = mask_paths(root)
    if args.limit:
        paths = paths[: args.limit]
    out_dir = output_dir(root, args)
    config = make_config(args)

    written = 0
    skipped = 0
    for idx, pred_path in enumerate(tqdm(paths, desc=args.task_type)):
        name = os.path.basename(pred_path)
        sidecar = pred_mask_index.get(sample_index_from_mask_name(name), {})
        record = index.get(os.path.splitext(name)[0]) or index.get(name) or index.get(f"row:{idx}")
        image_path = find_image_path(name, record, args.image_root, args.data_path)
        if sidecar.get("file_name"):
            image_path = find_image_path(sidecar["file_name"], record, args.image_root, args.data_path)
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        label = label_from_prediction_item(sidecar) or (record.get("prompt") if record else label_from_mask_name(name))
        if args.task_type == "gcg":
            pred = visualize_gcg_segmentation(image, pred_path, label, palette=palette, config=config)
        else:
            pred = visualize_binary_segmentation(image, pred_path, label=label, palette=palette, config=config)

        prompt_panel = None
        gt_panel = None
        if args.task_type in INTERACTIVE_PROMPT_STYLES and record:
            prompt_path = record.get("visual_prompt_path")
            if prompt_path and os.path.exists(prompt_path):
                prompt_panel = draw_visual_prompt(
                    image,
                    prompt_path,
                    prompt_style=INTERACTIVE_PROMPT_STYLES[args.task_type],
                )
            gt_path = record.get("seg_path")
            if gt_path and os.path.exists(gt_path):
                gt_panel = visualize_binary_segmentation(image, gt_path, label=label, palette=palette, config=config)
        elif sidecar.get("gt_name"):
            gt_path = resolve_path(sidecar.get("gt_name"), args.data_path)
            if gt_path and os.path.exists(gt_path):
                gt_panel = visualize_binary_segmentation(
                    image,
                    gt_path,
                    label=label,
                    palette=palette,
                    color=(255, 48, 48),
                    config=config,
                )
        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel, prompt=prompt_panel)
        final.save(os.path.join(out_dir, safe_stem(Path(name).stem) + ".png"))
        written += 1
    return {"task": args.task_type, "out_dir": out_dir, "written": written, "skipped": skipped}


def sample_index_from_mask_name(name):
    match = re.search(r"sample_(\d+)_", str(name))
    return int(match.group(1)) if match else None


def label_from_mask_name(name):
    stem = Path(str(name)).stem
    stem = re.sub(r"^sample_\d+_COCO_(?:train|val)2014_\d{12}_\d+_", "", stem)
    stem = re.sub(r"_pred$", "", stem)
    return stem.replace("_", " ") or Path(str(name)).stem


def label_from_prediction_item(item):
    if not item:
        return ""
    categories = item.get("categories") or item.get("category")
    if isinstance(categories, (list, tuple)):
        return ", ".join(str(x) for x in categories)
    if categories:
        return str(categories)
    prompt = item.get("prompt")
    return str(prompt) if prompt else ""


def load_pred_mask_sidecar_index(pred_mask_root):
    if os.path.basename(os.path.abspath(pred_mask_root)) != "pred_masks":
        return {}
    parent = os.path.dirname(os.path.abspath(pred_mask_root))
    json_paths = prediction_jsons_from_root(parent)
    if not json_paths:
        return {}
    preds, _ = load_prediction_jsons(parent)
    index = {}
    for item in preds:
        key = item.get("global_idx", item.get("idx"))
        if key is not None:
            index[int(key)] = item
    return index


def visualize_gcg_json_root(root, args, palette):
    preds, paths = load_prediction_jsons(root)
    if args.limit:
        preds = preds[: args.limit]
    records, index = load_record_index(args.input_jsonl, args.data_path)
    out_dir = output_dir(root, args)
    config = make_config(args)

    written = 0
    skipped = 0
    for row_idx, pred_item in enumerate(tqdm(preds, desc=args.task_type)):
        record = find_record_for_ann(pred_item, index)
        image_path = record.get("image_path") if record else None
        if not image_path:
            image_path = find_image_path(str(pred_item.get("file_name", "")), None, args.image_root, args.data_path)
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        pred = visualize_gcg_prediction(image, pred_item, palette=palette, config=config)

        gt_panel = None
        if args.concat_col == 3 and record and record.get("seg_path") and os.path.exists(record["seg_path"]):
            gt_text = record.get("gt_rich_caption") or record.get("caption") or record.get("prompt")
            gt_panel = visualize_gcg_segmentation(
                image,
                record["seg_path"],
                gt_text,
                palette=palette,
                config=config,
            )

        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel)
        sample_idx = int(pred_item.get("global_idx", pred_item.get("idx", row_idx)))
        image_id = pred_item.get("image_id") or (record.get("image_id") if record else Path(image_path).stem)
        final.save(os.path.join(out_dir, f"sample_{sample_idx:06d}_{safe_stem(image_id)}_gcg.png"))
        written += 1
    return {"task": "gcg", "out_dir": out_dir, "written": written, "skipped": skipped}


def visualize_interactive_json_root(root, args, palette):
    preds, paths = load_prediction_jsons(root)
    if args.limit:
        preds = preds[: args.limit]
    records, index = load_record_index(args.input_jsonl, args.data_path)
    record_indexes = interactive_record_index(records)
    out_dir = output_dir(root, args)
    config = make_config(args)

    written = 0
    skipped = 0
    for row_idx, pred_item in enumerate(tqdm(preds, desc=args.task_type)):
        record = find_record_for_interactive_pred(pred_item, records, index, record_indexes)
        if record is None:
            if args.skip_missing:
                skipped += 1
                continue
            raise KeyError(f"Cannot locate interactive prediction row {row_idx}: {paths[0]}")
        image_path = record.get("image_path")
        prompt_path = record.get("visual_prompt_path")
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        if should_skip_missing(prompt_path, args, "visual prompt"):
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        prompt_image = draw_visual_prompt(
            image,
            Image.open(prompt_path).convert("L"),
            prompt_style=INTERACTIVE_PROMPT_STYLES.get(args.task_type, "fill"),
        )
        categories = pred_item.get("categories") or ["mask"]
        label = categories if isinstance(categories, str) else ", ".join(str(x) for x in categories)
        pred_mask = pred_item.get("pred_mask") or pred_item.get("mask") or pred_item.get("segmentation")
        pred_base = prompt_image if args.concat_col == 1 else image
        pred = visualize_binary_segmentation(pred_base, pred_mask, label=label or "mask", palette=palette, config=config)

        gt_panel = None
        if args.concat_col == 3 and record.get("seg_path") and os.path.exists(record["seg_path"]):
            gt_panel = visualize_binary_segmentation(
                image,
                record["seg_path"],
                label=label or "GT",
                palette=palette,
                color=(255, 48, 48),
                config=config,
            )

        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel, prompt=prompt_image)
        sample_idx = int(pred_item.get("global_idx", pred_item.get("idx", record.get("row_idx", row_idx))))
        final.save(
            os.path.join(
                out_dir,
                f"sample_{sample_idx:06d}_{safe_stem(Path(record['file_name']).stem)}_{args.task_type}.png",
            )
        )
        written += 1
    return {"task": args.task_type, "out_dir": out_dir, "written": written, "skipped": skipped}


def binary_record_index(records):
    index = {}
    for record in records:
        for value in (record.get("raw_image"), record.get("image_path"), record.get("file_name")):
            if not value:
                continue
            index[normalize_prediction_path(value)] = record
            index[os.path.basename(value)] = record
    return index


def find_binary_record_for_pred(pred, record_index):
    raw_file = pred.get("file_name")
    if not raw_file:
        return None
    return record_index.get(normalize_prediction_path(raw_file)) or record_index.get(os.path.basename(raw_file))


def visualize_binary_json_root(root, args, palette):
    preds, paths = load_prediction_jsons(root)
    if args.limit:
        preds = preds[: args.limit]
    records, _index = load_record_index(args.input_jsonl, args.data_path)
    records_by_file = binary_record_index(records)
    out_dir = output_dir(root, args)
    config = make_config(args)

    written = 0
    skipped = 0
    for row_idx, pred_item in enumerate(tqdm(preds, desc=args.task_type)):
        record = find_binary_record_for_pred(pred_item, records_by_file) if records_by_file else None
        image_value = pred_item.get("file_name") or pred_item.get("image_path") or pred_item.get("image")
        image_path = record.get("image_path") if record else None
        if image_value:
            image_path = find_image_path(str(image_value), record, args.image_root, args.data_path)
        if should_skip_missing(image_path, args, "source image"):
            skipped += 1
            continue
        pred_mask = pred_item.get("pred_mask") or pred_item.get("segmentation") or pred_item.get("mask")
        if pred_mask is None:
            if args.skip_missing:
                skipped += 1
                continue
            raise KeyError(f"No pred_mask/segmentation/mask in {paths[0]} row {row_idx}")
        image = Image.open(image_path).convert("RGB")
        categories = pred_item.get("categories") or pred_item.get("category") or ["prediction"]
        if isinstance(categories, (list, tuple)):
            label = ", ".join(str(x) for x in categories)
        else:
            label = str(categories)
        if (not label or label == "prediction") and record:
            label = record.get("prompt") or label
        pred = visualize_binary_segmentation(image, pred_mask, label=label, palette=palette, config=config)

        gt_panel = None
        gt_name = pred_item.get("gt_name") or pred_item.get("gt_path") or (record.get("seg_path") if record else None)
        if gt_name:
            gt_path = gt_name if os.path.isabs(str(gt_name)) else resolve_path(gt_name, args.data_path)
            if gt_path and os.path.exists(gt_path):
                gt_panel = visualize_binary_segmentation(
                    image,
                    gt_path,
                    label=label,
                    palette=palette,
                    color=(255, 48, 48),
                    config=config,
                )
        final = visualize_concat_col(image, pred, concat_col=args.concat_col, gt=gt_panel)
        stem = safe_stem(Path(str(image_value)).stem)
        final.save(os.path.join(out_dir, f"{row_idx:06d}_{stem}.png"))
        written += 1
    return {"task": args.task_type, "out_dir": out_dir, "written": written, "skipped": skipped}


def make_config(args):
    return VisualizationConfig(
        alpha=args.alpha,
        min_label_area=args.min_label_area,
        max_labels=args.max_labels,
        font_size=args.font_size,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SenseNova-Vision segmentation-family outputs.")
    parser.add_argument("--task_type", required=True, choices=["panoptic", "semantic", "gcg", "ref", "rea", "point", "box", "mask", "scribble"])
    parser.add_argument("--prediction_root", nargs="+", required=True)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--semantic_image_dir", default=None)
    parser.add_argument("--semantic_gt_dir", default=None)
    parser.add_argument("--semantic_category_key", default="auto", choices=["auto", "contiguous", "id"])
    parser.add_argument("--category_json", default=None)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--concat_col", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--min_label_area", type=int, default=80)
    parser.add_argument("--max_labels", type=int, default=120)
    parser.add_argument(
        "--font_size",
        type=int,
        default=0,
        help="Override dynamic label sizing. 0 uses image- and mask-aware sizing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    palette = build_palette()
    roots = parse_root_args(args.prediction_root)
    summaries = []
    for root in roots:
        root = resolve_path(root, args.data_path)
        if args.task_type == "panoptic":
            summaries.append(visualize_panoptic_root(root, args, palette))
        elif args.task_type == "semantic":
            summaries.append(visualize_semantic_root(root, args, palette))
        else:
            summaries.append(visualize_mask_root(root, args, palette))
    for summary in summaries:
        print(
            f"{summary['task']} -> {summary['out_dir']} "
            f"written={summary['written']} skipped={summary['skipped']}"
        )


if __name__ == "__main__":
    main()
