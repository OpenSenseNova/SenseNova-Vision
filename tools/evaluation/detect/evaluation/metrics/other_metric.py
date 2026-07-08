import argparse
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon
from tqdm import tqdm


def save_metrics_jsonl(all_results: dict, data_path: str, output_path: str):
    """
    Save flattened metrics as JSONL for auto table-fill.

    Naming rule (STRICT):
      metrics filename is based on output_path basename:
        <output_path basename>_metrics.jsonl
      e.g. output_path = ".../output/IC15_nw.jsonl"
           -> metrics/.../IC15_nw_metrics.jsonl

    Location rule:
      metrics directory is sibling of output_path's directory:
        <dirname(output_path)>/../metrics/
      This matches your original convention "some_dir/metrics/xxx_metrics.jsonl",
      but now anchored on output_path (so it won't overwrite when data_path is shared).
    """
    # -------- Resolve output-based naming & directory --------
    out_abs = os.path.abspath(output_path)
    out_dir_path = os.path.dirname(out_abs)              # e.g. .../detection/output
    out_parent = os.path.dirname(out_dir_path)           # e.g. .../detection
    base_name = os.path.splitext(os.path.basename(out_abs))[0]  # e.g. IC15_nw

    metrics_dir = os.path.join(out_parent, "metrics")    # e.g. .../detection/metrics
    os.makedirs(metrics_dir, exist_ok=True)

    out_jsonl = os.path.join(metrics_dir, f"{base_name}_metrics.jsonl")

    # -------- Prevent overwrite: auto versioning --------
    if os.path.exists(out_jsonl):
        idx = 2
        while True:
            candidate = os.path.join(metrics_dir, f"{base_name}_metrics_v{idx}.jsonl")
            if not os.path.exists(candidate):
                out_jsonl = candidate
                break
            idx += 1

    # -------- Aggregate per task_dataset across IoUs (basic metrics) --------
    dataset_metrics_by_iou = {}

    for iou, results in all_results.items():
        basic_metrics = results.get("basic_metrics", {})
        for key, metrics in basic_metrics.items():
            recalls = metrics.get("recalls", [])
            precisions = metrics.get("precisions", [])
            if not recalls:
                continue

            if key not in dataset_metrics_by_iou:
                dataset_metrics_by_iou[key] = {}

            avg_recall = mean(recalls)
            avg_precision = mean(precisions)
            denom = avg_precision + avg_recall
            avg_f1 = (2 * avg_precision * avg_recall / denom) if denom > 0 else 0.0

            dataset_metrics_by_iou[key][float(iou)] = {
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1": float(avg_f1),
                "samples": int(len(recalls)),
            }

    # -------- Extra metrics aggregated across IoUs --------
    vp_mae = defaultdict(list)
    inst_follow = defaultdict(list)
    halluc_acc = defaultdict(list)
    gui_acc = defaultdict(list)
    wrong_rej = defaultdict(lambda: {"wrong": [], "total": []})

    for iou, results in all_results.items():
        for k, v in results.get("visual_prompt_metrics", {}).items():
            vp_mae[k].extend(v.get("maes", []))
        for k, v in results.get("instruction_following_metrics", {}).items():
            inst_follow[k].extend(v.get("ratios", []))
        for k, v in results.get("hallucination_metrics", {}).items():
            halluc_acc[k].extend(v.get("accuracies", []))
        for k, v in results.get("gui_metrics", {}).items():
            gui_acc[k].extend(v.get("accuracies", []))
        for k, v in results.get("wrong_rejection_metrics", {}).items():
            wrong_rej[k]["wrong"].extend(v.get("wrong_rejections", []))
            wrong_rej[k]["total"].extend(v.get("total_samples", []))

    def mean_or_none(xs):
        return float(mean(xs)) if xs else None

    def rate_or_none(wrong_list, total_list):
        if not total_list:
            return None
        denom = sum(total_list)
        return float(sum(wrong_list) / denom) if denom > 0 else None

    # -------- Key set: union of everything so we still write rows even if basic empty --------
    all_keys = set(dataset_metrics_by_iou.keys())
    all_keys |= set(vp_mae.keys())
    all_keys |= set(inst_follow.keys())
    all_keys |= set(halluc_acc.keys())
    all_keys |= set(gui_acc.keys())
    all_keys |= set(wrong_rej.keys())

    # -------- Write JSONL: one line per task_dataset --------
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for key in sorted(all_keys):
            iou_map = dataset_metrics_by_iou.get(key, {})
            iou05 = iou_map.get(0.5)
            iou095 = iou_map.get(0.95)

            # mIoU mean across all IoUs that exist for this key
            ps = [m["precision"] for m in iou_map.values()]
            rs = [m["recall"] for m in iou_map.values()]
            fs = [m["f1"] for m in iou_map.values()]

            row = {
                "task_dataset": key,
                "iou05_precision": iou05["precision"] if iou05 else None,
                "iou05_recall": iou05["recall"] if iou05 else None,
                "iou05_f1": iou05["f1"] if iou05 else None,
                "iou095_precision": iou095["precision"] if iou095 else None,
                "iou095_recall": iou095["recall"] if iou095 else None,
                "iou095_f1": iou095["f1"] if iou095 else None,
                "miou_precision": float(mean(ps)) if ps else None,
                "miou_recall": float(mean(rs)) if rs else None,
                "miou_f1": float(mean(fs)) if fs else None,
                "samples": int(max((m["samples"] for m in iou_map.values()), default=0)),
                "mae": mean_or_none(vp_mae.get(key, [])),
                "inst_follow": mean_or_none(inst_follow.get(key, [])),
                "halluc_acc": mean_or_none(halluc_acc.get(key, [])),
                "gui_acc": mean_or_none(gui_acc.get(key, [])),
                "wrong_rej": rate_or_none(
                    wrong_rej.get(key, {}).get("wrong", []),
                    wrong_rej.get(key, {}).get("total", []),
                ),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n💾 Saved auto-table metrics JSONL to: {out_jsonl}")


def save_keypoint_metrics_jsonl(all_results: dict, output_path: str):
    out_abs = os.path.abspath(output_path)
    out_dir_path = os.path.dirname(out_abs)
    out_parent = os.path.dirname(out_dir_path)
    base_name = os.path.splitext(os.path.basename(out_abs))[0]
    metrics_dir = os.path.join(out_parent, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out_jsonl = os.path.join(metrics_dir, f"{base_name}_metrics.jsonl")

    keypoint_results = {}
    for results in all_results.values():
        for key, kp_metrics in results.get("keypoint_metrics", {}).items():
            if key not in keypoint_results:
                keypoint_results[key] = {
                    "prf1_results_list": [],
                    "ap_scores": [],
                    "avg_oks": [],
                }
            keypoint_results[key]["prf1_results_list"].extend(
                kp_metrics.get("prf1_results_list", [])
            )
            keypoint_results[key]["ap_scores"].extend(kp_metrics.get("ap_scores", []))
            keypoint_results[key]["avg_oks"].extend(kp_metrics.get("avg_oks", []))

    def avg(values):
        return float(mean(values)) if values else None

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for key in sorted(keypoint_results):
            item = keypoint_results[key]
            prf1 = item["prf1_results_list"]
            ap_scores = item["ap_scores"]
            avg_oks = item["avg_oks"]
            row = {
                "task_dataset": key,
                "keypoint_f1_50": avg([r.get("F1@0.50", 0.0) for r in prf1]),
                "keypoint_f1_95": avg([r.get("F1@0.95", 0.0) for r in prf1]),
                "keypoint_f1_moks": avg([r.get("F1@mOKS", 0.0) for r in prf1]),
                "iou05_f1": avg([r.get("F1@0.50", 0.0) for r in prf1]),
                "iou095_f1": avg([r.get("F1@0.95", 0.0) for r in prf1]),
                "miou_f1": avg([r.get("F1@mOKS", 0.0) for r in prf1]),
                "avg_oks": avg(avg_oks),
            }
            if row["keypoint_f1_50"] is None and ap_scores:
                ap50 = [ap.get("AP@0.50", 0.0) for ap in ap_scores if "AP@0.50" in ap]
                row["keypoint_f1_50"] = avg(ap50)
                row["iou05_f1"] = row["keypoint_f1_50"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved keypoint metrics JSONL to: {out_jsonl}")


def decode_rle(rle_str, size):
    """Decode RLE (Run Length Encoding) string to binary mask using pycocotools"""
    try:
        # Create RLE dict format expected by pycocotools
        rle = {"counts": rle_str.encode("utf-8"), "size": size}

        # Use pycocotools to decode
        mask = coco_mask.decode(rle)
        return mask

    except Exception as e:
        print(f"Error decoding RLE: {e}, RLE string: {rle_str[:50]}...")
        # Return empty mask if decoding fails
        height, width = size
        return np.zeros((height, width), dtype=np.uint8)


def is_point_in_mask(point, mask):
    """Check if a point falls within a binary mask"""
    x, y = int(point[0]), int(point[1])

    # Check bounds
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False

    return mask[y, x] == 1


def is_point_in_bbox(point, bbox):
    """
    Return whether a point is inside a bounding box (for GUI tasks).

    Args:
        point: [x, y] point coordinates
        bbox: [x1, y1, x2, y2] bounding-box coordinates (xyxy format)

    Returns:
        bool: whether the point is inside the bounding box
    """
    if len(point) != 2 or len(bbox) != 4:
        return False

    x, y = point
    x1, y1, x2, y2 = bbox

    # Ensure the bbox coordinate order is valid.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_center(box):
    """
    Compute the bounding-box center point (for GUI tasks).

    Args:
        box: [x1, y1, x2, y2] bounding box (xyxy format)

    Returns:
        List[float]: [center_x, center_y] center-point coordinates
    """
    if len(box) != 4:
        return []

    x1, y1, x2, y2 = box

    # Ensure the coordinate order is valid.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return [center_x, center_y]


def extract_gui_predictions(extracted_predictions):
    """
    Extract all point and box coordinates from the extracted_predictions dictionary (for GUI tasks).

    Args:
        extracted_predictions: for example {"street sign": [[209.45945945945945, 269.2692692692693]]} or
                              {"button": [[x1, y1, x2, y2]]}

    Returns:
        Tuple[List[List[float]], List[List[float]]]: (point coordinate list, box coordinate list)
    """
    points = []
    boxes = []

    if not isinstance(extracted_predictions, dict):
        return points, boxes

    for key, value in extracted_predictions.items():
        if isinstance(value, list):
            for coord in value:
                if isinstance(coord, list):
                    try:
                        if len(coord) == 2:
                            # point coordinatesFormat [x, y]
                            x, y = float(coord[0]), float(coord[1])
                            points.append([x, y])
                        elif len(coord) == 4:
                            # box coordinate format [x1, y1, x2, y2]
                            x1, y1, x2, y2 = (
                                float(coord[0]),
                                float(coord[1]),
                                float(coord[2]),
                                float(coord[3]),
                            )
                            boxes.append([x1, y1, x2, y2])
                    except (ValueError, TypeError):
                        continue

    return points, boxes


def has_wrong_rejection(pred_boxes_dict):
    """
    Check whether predictions contain an invalid refusal: predicting None outside hallucination tasks.

    Args:
        pred_boxes_dict: prediction dictionary，for example {"category": [boxes_or_None]}

    Returns:
        bool: whether an invalid refusal is present
    """
    for category, predictions in pred_boxes_dict.items():
        if not predictions:  # empty list
            continue
        for pred in predictions:
            if pred is None or (isinstance(pred, str) and pred.lower() == "none"):
                return True
    return False


def calculate_gui_metrics(gt_box, extracted_predictions):
    """
    Compute accuracy for GUI localization tasks.

    Args:
        gt_box: Ground truthbounding box [x1, y1, x2, y2]
        extracted_predictions: prediction dictionary

    Returns:
        bool: whether the prediction is correct
    """
    # Extract predicted point and box coordinates.
    predicted_points, predicted_boxes = extract_gui_predictions(extracted_predictions)

    # If no point or box is predicted, treat the prediction as failed.
    if not predicted_points and not predicted_boxes:
        return False

    # If the GT is empty or malformed, treat the prediction as failed.
    if not gt_box or len(gt_box) != 4:
        return False

    # Check whether any predicted point falls inside the GT box.
    for point in predicted_points:
        if is_point_in_bbox(point, gt_box):
            return True

    # Check whether any predicted box center falls inside the GT box.
    for box in predicted_boxes:
        center_point = get_box_center(box)
        if center_point and is_point_in_bbox(center_point, gt_box):
            return True

    return False


def calculate_pointing_metrics(gt_masks_dict, pred_points_dict):
    """Calculate metrics for pointing task"""
    # Count total GT masks and predictions
    total_gt_count = sum(len(masks) for masks in gt_masks_dict.values())
    total_pred_count = sum(len(points) for points in pred_points_dict.values())

    if total_gt_count == 0:
        if total_pred_count == 0:
            return 1.0, 1.0, total_gt_count, total_pred_count
        else:
            return 0.0, 0.0, total_gt_count, total_pred_count

    if total_pred_count == 0:
        return 0.0, 0.0, total_gt_count, total_pred_count

    # Flatten all masks and points for matching
    all_gt_masks = []
    all_pred_points = []
    all_gt_categories = []
    all_pred_categories = []

    for category in gt_masks_dict:
        for mask_info in gt_masks_dict[category]:
            all_gt_masks.append(mask_info)
            all_gt_categories.append(category)

    for category in pred_points_dict:
        for point in pred_points_dict[category]:
            all_pred_points.append(point)
            all_pred_categories.append(category)

    # Pre-decode all masks to avoid repeated decoding
    decoded_masks = []
    for gt_mask_info in all_gt_masks:
        mask = decode_rle(gt_mask_info["counts"], gt_mask_info["size"])
        decoded_masks.append(mask)

    matches = 0
    used_preds = set()

    # For each GT mask, find the best matching prediction
    for i, (decoded_mask, gt_category) in enumerate(
        zip(decoded_masks, all_gt_categories)
    ):
        best_match = -1
        best_score = 0

        for j, (pred_point, pred_category) in enumerate(
            zip(all_pred_points, all_pred_categories)
        ):
            if j in used_preds:
                continue

            # Only match if categories match
            if gt_category != pred_category:
                continue

            # Check if point is in mask (using pre-decoded mask)
            if is_point_in_mask(pred_point, decoded_mask):
                # For pointing task, we consider it a match if the point is in the mask
                # and it's the first match for this GT mask
                if best_match == -1:
                    best_match = j
                    best_score = 1.0

        if best_match != -1:
            matches += 1
            used_preds.add(best_match)

    recall = matches / total_gt_count if total_gt_count > 0 else 0.0
    precision = matches / total_pred_count if total_pred_count > 0 else 0.0

    return recall, precision, total_gt_count, total_pred_count


def calculate_area(box):
    """Calculate area of a bounding box or polygon"""
    if is_polygon_format(box):
        # For polygon, use shapely to calculate area
        try:
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            polygon = Polygon(coords)
            return polygon.area if polygon.is_valid else 0.0
        except:
            return 0.0
    else:
        # For bounding box
        return (box[2] - box[0]) * (box[3] - box[1])


def get_size_category(area):
    """Categorize object by size according to COCO standards"""
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"


def get_gt_count_range(gt_count):
    """Categorize by GT count ranges"""
    if gt_count == 0:
        return "0"
    elif gt_count <= 5:
        return "1-5"
    elif gt_count <= 10:
        return "6-10"
    elif gt_count <= 15:
        return "11-15"
    elif gt_count <= 20:
        return "16-20"
    else:
        return "20+"


def normalize_category_name(category, task_name, dataset_name=None):
    """Normalize category name based on task type"""
    if task_name in [
        "common_object_detection",
        "long_tailed_object_detection",
        "dense_object_detection",
    ]:
        # For these tasks: lowercase and keep only letters
        return "".join(c.lower() for c in category if c.isalpha())
    else:
        # For other tasks: remove underscores and spaces
        return category.replace("_", "").replace(" ", "")


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)


def calculate_polygon_iou(poly1, poly2):
    """Calculate IoU between two polygons"""
    try:
        # Convert polygon coordinates to shapely Polygon objects
        # poly1 and poly2 are lists of coordinates [x0, y0, x1, y1, x2, y2, ...]
        if len(poly1) < 6 or len(poly2) < 6:  # Need at least 3 points (6 coordinates)
            return 0.0

        # Reshape coordinates to pairs
        coords1 = [(poly1[i], poly1[i + 1]) for i in range(0, len(poly1), 2)]
        coords2 = [(poly2[i], poly2[i + 1]) for i in range(0, len(poly2), 2)]

        # Create shapely polygons
        polygon1 = Polygon(coords1)
        polygon2 = Polygon(coords2)

        # Check if polygons are valid
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0

        # Calculate intersection and union
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        # If there's any error in polygon calculation, return 0
        return 0.0


def is_polygon_format(coords):
    """Check if coordinates are in polygon format (more than 4 values)"""
    return len(coords) > 4


def calculate_detection_metrics(
    gt_boxes_dict,
    pred_boxes_dict,
    iou_threshold=0.5,
    use_polygon_iou=False,
    match_by_category=False,
):
    """Calculate recall and precision for object detection tasks"""
    if match_by_category:
        total_gt = 0
        total_pred = 0
        total_matches = 0
        all_categories = set(gt_boxes_dict.keys()) | set(pred_boxes_dict.keys())

        for category in all_categories:
            gt_boxes = gt_boxes_dict.get(category, [])
            pred_boxes = pred_boxes_dict.get(category, [])
            pred_boxes = [
                box for box in pred_boxes if box is not None and box != "None"
            ]

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            if not gt_boxes or not pred_boxes:
                continue

            used_preds = set()
            matches = 0

            for gt_box in gt_boxes:
                best_iou = 0
                best_pred_idx = -1

                for i, pred_box in enumerate(pred_boxes):
                    if i in used_preds:
                        continue
                    try:
                        if use_polygon_iou:
                            iou = calculate_polygon_iou(gt_box, pred_box)
                        else:
                            iou = calculate_iou(gt_box, pred_box)
                    except Exception:
                        iou = 0.0

                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_pred_idx = i

                if best_pred_idx != -1:
                    matches += 1
                    used_preds.add(best_pred_idx)

            total_matches += matches

        recall = total_matches / total_gt if total_gt > 0 else 0.0
        precision = total_matches / total_pred if total_pred > 0 else 0.0
        return recall, precision, total_gt, total_pred

    # Count total GT boxes across all categories
    total_gt_count = sum(len(boxes) for boxes in gt_boxes_dict.values())
    # Count total predicted boxes, excluding None values
    total_pred_count = sum(
        len([box for box in boxes if box is not None and box != "None"])
        for boxes in pred_boxes_dict.values()
    )

    if len(gt_boxes_dict) == 0:
        if total_pred_count == 0:
            return 1.0, 1.0, total_gt_count, total_pred_count
        else:
            return 0.0, 0.0, total_gt_count, total_pred_count

    if total_pred_count == 0:
        return 0.0, 0.0, total_gt_count, total_pred_count

    # Flatten all boxes for matching
    all_gt_boxes = []
    all_pred_boxes = []

    for category in gt_boxes_dict:
        all_gt_boxes.extend(gt_boxes_dict[category])
        # Filter out None values from predictions
        pred_boxes_for_category = pred_boxes_dict.get(category, [])
        valid_pred_boxes = [
            box for box in pred_boxes_for_category if box is not None and box != "None"
        ]
        all_pred_boxes.extend(valid_pred_boxes)

    matches = 0
    used_preds = set()

    for gt_box in all_gt_boxes:
        best_iou = 0
        best_pred_idx = -1

        for i, pred_box in enumerate(all_pred_boxes):
            if i in used_preds:
                continue

            # Choose IoU calculation method based on format
            if use_polygon_iou:
                iou = calculate_polygon_iou(gt_box, pred_box)
            else:
                iou = calculate_iou(gt_box, pred_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = i

        if best_pred_idx != -1:
            matches += 1
            used_preds.add(best_pred_idx)

    recall = matches / total_gt_count if total_gt_count > 0 else 0.0
    precision = matches / total_pred_count if total_pred_count > 0 else 0.0

    return recall, precision, total_gt_count, total_pred_count


def calculate_visual_prompt_metrics(gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5):
    """Calculate metrics for visual prompt detection task"""
    # Calculate basic detection metrics
    recall, precision, gt_count, pred_count = calculate_detection_metrics(
        gt_boxes_dict, pred_boxes_dict, iou_threshold
    )

    # Calculate MAE (Mean Absolute Error) for box count
    mae = abs(pred_count - gt_count)

    # Calculate duplicate prediction ratio
    all_pred_boxes = []
    for category in pred_boxes_dict:
        all_pred_boxes.extend(pred_boxes_dict[category])

    duplicate_count = 0
    total_predictions = len(all_pred_boxes)

    if total_predictions > 1:
        for i in range(len(all_pred_boxes)):
            for j in range(i + 1, len(all_pred_boxes)):
                # Choose IoU calculation method based on format
                if is_polygon_format(all_pred_boxes[i]) or is_polygon_format(
                    all_pred_boxes[j]
                ):
                    iou = calculate_polygon_iou(all_pred_boxes[i], all_pred_boxes[j])
                else:
                    iou = calculate_iou(all_pred_boxes[i], all_pred_boxes[j])

                if iou > 0.9:  # High IoU threshold for duplicates
                    duplicate_count += 1

    duplicate_ratio = (
        duplicate_count / total_predictions if total_predictions > 0 else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "mae": mae,
        "duplicate_ratio": duplicate_ratio,
        "gt_count": gt_count,
        "pred_count": pred_count,
    }


def calculate_size_metrics(
    gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5, use_polygon_iou=False
):
    """Calculate recall and precision for different object sizes"""
    size_metrics = {
        "small": {"gt": [], "pred": [], "matches": 0},
        "medium": {"gt": [], "pred": [], "matches": 0},
        "large": {"gt": [], "pred": [], "matches": 0},
    }

    # Process each category separately
    for category in gt_boxes_dict:
        gt_boxes = gt_boxes_dict[category]
        pred_boxes = pred_boxes_dict.get(category, [])

        # Categorize ground truth boxes by size
        for gt_box in gt_boxes:
            area = calculate_area(gt_box)
            size = get_size_category(area)
            size_metrics[size]["gt"].append(gt_box)

        # Categorize prediction boxes by size
        for pred_box in pred_boxes:
            area = calculate_area(pred_box)
            size = get_size_category(area)
            size_metrics[size]["pred"].append(pred_box)

    # Calculate matches for each size category
    for size in size_metrics:
        gt_boxes = size_metrics[size]["gt"]
        pred_boxes = size_metrics[size]["pred"]

        if len(gt_boxes) == 0:
            if len(pred_boxes) == 0:
                size_metrics[size]["recall"] = 1.0
                size_metrics[size]["precision"] = 1.0
            else:
                size_metrics[size]["recall"] = 0.0
                size_metrics[size]["precision"] = 0.0
            continue

        if len(pred_boxes) == 0:
            size_metrics[size]["recall"] = 0.0
            size_metrics[size]["precision"] = 0.0
            continue

        matches = 0
        used_preds = set()

        for gt_box in gt_boxes:
            best_iou = 0
            best_pred_idx = -1

            for i, pred_box in enumerate(pred_boxes):
                if i in used_preds:
                    continue

                # Choose IoU calculation method based on format
                if use_polygon_iou:
                    iou = calculate_polygon_iou(gt_box, pred_box)
                else:
                    iou = calculate_iou(gt_box, pred_box)

                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = i

            if best_pred_idx != -1:
                matches += 1
                used_preds.add(best_pred_idx)

        size_metrics[size]["recall"] = (
            matches / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        )
        size_metrics[size]["precision"] = (
            matches / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        )

    return size_metrics


def calculate_gt_count_metrics(
    gt_boxes_dict,
    pred_boxes_dict,
    iou_threshold=0.5,
    use_polygon_iou=False,
    match_by_category=False,
):
    """Calculate metrics for different GT count ranges"""
    total_gt_count = sum(len(boxes) for boxes in gt_boxes_dict.values())
    gt_count_range = get_gt_count_range(total_gt_count)

    recall, precision, gt_count, pred_count = calculate_detection_metrics(
        gt_boxes_dict,
        pred_boxes_dict,
        iou_threshold,
        use_polygon_iou,
        match_by_category,
    )

    return gt_count_range, recall, precision, gt_count, pred_count


class UniversalMetricsCalculator:
    """Universal metrics calculator for multiple tasks and datasets"""

    def __init__(self, match_by_category=False):
        self.match_by_category = match_by_category
        self.results = defaultdict(lambda: defaultdict(list))
        self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
        self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
        self.keypoint_metrics = defaultdict(
            lambda: {
                "ap_scores": [],
                "avg_oks": [],
                "gt_counts": [],
                "pred_counts": [],
                "prf1_results_list": [],
            }
        )
        self.hallucination_metrics = defaultdict(
            lambda: {"accuracies": [], "pred_counts": []}
        )
        self.gui_metrics = defaultdict(
            lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
        )
        self.wrong_rejection_metrics = defaultdict(
            lambda: {"wrong_rejections": [], "total_samples": []}
        )

    def calculate_metrics_for_sample(self, sample, iou_threshold=0.5):
        """Calculate metrics for a single sample"""
        task_name = sample["task_name"]
        dataset_name = sample["dataset_name"]
        gt_boxes = sample["gt"]
        pred_boxes = sample["extracted_predictions"]

        # Special handling for GUI task - gt is a list, not a dict
        if task_name == "gui":
            # For GUI task, keep gt_boxes as is (it's a list [x1, y1, x2, y2])
            # Only lowercase the prediction keys
            pred_boxes = {k.lower(): v for k, v in pred_boxes.items()}
        else:
            # Lowercase all the category keys before task-specific normalization,
            gt_boxes = {k.lower(): v for k, v in gt_boxes.items()}
            pred_boxes = {k.lower(): v for k, v in pred_boxes.items()}

        # Special handling for different task types
        if task_name == "referring_object_detection":
            # For referring_object_detection, extract all boxes regardless of category names
            all_gt_boxes = []
            all_pred_boxes = []

            for category in gt_boxes:
                all_gt_boxes.extend(gt_boxes[category])

            for category in pred_boxes:
                all_pred_boxes.extend(pred_boxes[category])

            # Calculate metrics directly with flattened boxes
            gt_count = len(all_gt_boxes)
            pred_count = len(all_pred_boxes)

            if gt_count == 0:
                if pred_count == 0:
                    recall, precision = 1.0, 1.0
                else:
                    recall, precision = 0.0, 0.0
            elif pred_count == 0:
                recall, precision = 0.0, 0.0
            else:
                # Calculate matches
                matches = 0
                used_preds = set()

                for gt_box in all_gt_boxes:
                    best_iou = 0
                    best_pred_idx = -1

                    for i, pred_box in enumerate(all_pred_boxes):
                        if i in used_preds:
                            continue
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_pred_idx = i

                    if best_pred_idx != -1:
                        matches += 1
                        used_preds.add(best_pred_idx)

                recall = matches / gt_count if gt_count > 0 else 0.0
                precision = matches / pred_count if pred_count > 0 else 0.0
        elif task_name in [
            "task4_detect_all_in_polygon",
            "task2_detect_all_in_polygon_and_recog",
        ]:
            # For polygon tasks, use polygon IoU calculation
            # Process category names based on task type
            processed_gt_boxes = {}
            processed_pred_boxes = {}

            for category in gt_boxes:
                processed_category = normalize_category_name(
                    category, task_name, dataset_name
                )
                processed_gt_boxes[processed_category] = gt_boxes[category]

            for category in pred_boxes:
                processed_category = normalize_category_name(
                    category, task_name, dataset_name
                )
                processed_pred_boxes[processed_category] = pred_boxes[category]

            # Calculate basic detection metrics for polygon tasks
            recall, precision, gt_count, pred_count = calculate_detection_metrics(
                processed_gt_boxes,
                processed_pred_boxes,
                iou_threshold,
                use_polygon_iou=True,
                match_by_category=self.match_by_category,
            )
        elif task_name == "pointing":
            # For pointing task, use pointing metrics
            recall, precision, gt_count, pred_count = calculate_pointing_metrics(
                gt_boxes, pred_boxes
            )
        elif task_name == "pointing_referring":
            # For pointing task, use pointing metrics
            for gt_cate_names, things in gt_boxes.items():
                break

            if len(pred_boxes) > 0:
                for pred_cate_names, things in pred_boxes.items():
                    break
                new_pred_boxes = {gt_cate_names: pred_boxes[pred_cate_names]}
            else:
                new_pred_boxes = pred_boxes

            recall, precision, gt_count, pred_count = calculate_pointing_metrics(
                gt_boxes, new_pred_boxes
            )
        elif task_name == "keypoint":
            # For keypoint detection task, use keypoint metrics
            keypoint_results = calculate_keypoint_metrics_for_sample(
                gt_boxes, pred_boxes
            )

            # Store keypoint-specific metrics
            key = f"{task_name}_{dataset_name}"
            self.keypoint_metrics[key]["ap_scores"].append(
                keypoint_results["ap_results"]
            )
            self.keypoint_metrics[key]["avg_oks"].append(keypoint_results["avg_oks"])
            self.keypoint_metrics[key]["gt_counts"].append(
                keypoint_results["total_gt_instances"]
            )
            self.keypoint_metrics[key]["pred_counts"].append(
                keypoint_results["total_pred_instances"]
            )
            self.keypoint_metrics[key]["prf1_results_list"].append(
                keypoint_results["prf1_results"]
            )

            recall = keypoint_results["prf1_results"].get("R@0.50", 0.0)
            precision = keypoint_results["prf1_results"].get("P@0.50", 0.0)
            gt_count = keypoint_results["total_gt_instances"]
            pred_count = keypoint_results["total_pred_instances"]
        elif task_name == "hallucination":
            # For hallucination task, model should NOT predict any boxes
            # Count total predicted boxes across all categories
            total_pred_count = sum(len(boxes) for boxes in pred_boxes.values())

            # Correct if no predictions are made (model didn't hallucinate)
            is_correct = 1.0 if total_pred_count == 0 else 0.0

            # For hallucination task:
            # - recall = accuracy (proportion of samples where model correctly predicted nothing)
            # - precision = accuracy (same as recall for this binary task)
            # - gt_count = 0 (no ground truth boxes since we're testing non-existent objects)
            # - pred_count = total predictions made
            recall = is_correct
            precision = is_correct
            gt_count = 0
            pred_count = total_pred_count
        elif task_name == "gui":
            # For GUI task, calculate if predicted points/boxes fall within GT box
            # GT format should be a single box [x1, y1, x2, y2] in sample["gt"]
            gt_box = sample.get("gt", [])

            # Calculate GUI accuracy
            is_correct = calculate_gui_metrics(gt_box, pred_boxes)

            # For GUI task:
            # - recall = accuracy (whether the prediction is correct)
            # - precision = accuracy (same as recall for this binary task)
            # - gt_count = 1 (always have one GT box)
            # - pred_count = total number of predictions made
            recall = 1.0 if is_correct else 0.0
            precision = 1.0 if is_correct else 0.0
            gt_count = 1
            pred_count = sum(len(coords) for coords in pred_boxes.values())
        else:
            # Process category names based on task type
            processed_gt_boxes = {}
            processed_pred_boxes = {}

            for category in gt_boxes:
                processed_category = normalize_category_name(
                    category, task_name, dataset_name
                )
                processed_gt_boxes[processed_category] = gt_boxes[category]

            for category in pred_boxes:
                processed_category = normalize_category_name(
                    category, task_name, dataset_name
                )
                processed_pred_boxes[processed_category] = pred_boxes[category]

            # Calculate basic detection metrics for other tasks
            recall, precision, gt_count, pred_count = calculate_detection_metrics(
                processed_gt_boxes,
                processed_pred_boxes,
                iou_threshold,
                match_by_category=self.match_by_category,
            )

        # Store basic metrics
        key = f"{task_name}_{dataset_name}"
        self.results[key]["recalls"].append(recall)
        self.results[key]["precisions"].append(precision)
        self.results[key]["gt_counts"].append(gt_count)
        self.results[key]["pred_counts"].append(pred_count)

        # Store hallucination-specific metrics
        if task_name == "hallucination":
            if key not in self.hallucination_metrics:
                self.hallucination_metrics[key] = {"accuracies": [], "pred_counts": []}
            self.hallucination_metrics[key]["accuracies"].append(
                recall
            )  # recall = accuracy for hallucination task
            self.hallucination_metrics[key]["pred_counts"].append(pred_count)

        # Store GUI-specific metrics
        if task_name == "gui":
            if key not in self.gui_metrics:
                self.gui_metrics[key] = {
                    "accuracies": [],
                    "correct_counts": [],
                    "total_counts": [],
                }
            self.gui_metrics[key]["accuracies"].append(
                recall
            )  # recall = accuracy for GUI task
            self.gui_metrics[key]["correct_counts"].append(1 if recall == 1.0 else 0)
            self.gui_metrics[key]["total_counts"].append(1)

        # Calculate wrong rejection metrics (for all tasks except hallucination)
        if task_name != "hallucination":
            wrong_rejection = has_wrong_rejection(pred_boxes)
            if key not in self.wrong_rejection_metrics:
                self.wrong_rejection_metrics[key] = {
                    "wrong_rejections": [],
                    "total_samples": [],
                }
            self.wrong_rejection_metrics[key]["wrong_rejections"].append(
                1 if wrong_rejection else 0
            )
            self.wrong_rejection_metrics[key]["total_samples"].append(1)

        # Calculate additional metrics for visual_prompt_detection
        if task_name == "visual_prompt_detection" or task_name == "visual_prompting":
            mae = abs(pred_count - gt_count)
            if key not in self.visual_prompt_metrics:
                self.visual_prompt_metrics[key] = {"maes": []}
            self.visual_prompt_metrics[key]["maes"].append(mae)

        # Calculate InstructionFollowing metric
        # Count unique categories in GT and predictions
        if task_name == "gui":
            # For GUI task, there's no category concept, skip instruction following metric
            gt_categories = set()
            pred_categories = set(pred_boxes.keys())
        else:
            gt_categories = set(gt_boxes.keys())
            pred_categories = set(pred_boxes.keys())

        # Special processing for common_object_detection and long_tailed_object_detection
        if task_name in ["common_object_detection", "long_tailed_object_detection"]:
            # Normalize categories: lowercase and keep only letters
            gt_categories_normalized = {
                normalize_category_name(cat, task_name, dataset_name): cat
                for cat in gt_categories
            }
            pred_categories_normalized = {
                normalize_category_name(cat, task_name, dataset_name): cat
                for cat in pred_categories
            }

            # Find matching categories based on normalized names
            matching_categories = set()
            for gt_norm, gt_orig in gt_categories_normalized.items():
                if gt_norm in pred_categories_normalized:
                    matching_categories.add(gt_orig)
        else:
            # For other tasks, use exact matching
            matching_categories = gt_categories.intersection(pred_categories)

        # Calculate the ratio of matching category counts
        if len(gt_categories) == 0:
            if len(pred_categories) == 0:
                instruction_following_ratio = 1.0
            else:
                instruction_following_ratio = 0.0
        else:
            instruction_following_ratio = len(matching_categories) / len(gt_categories)

        # Store instruction following metric
        if key not in self.instruction_following_metrics:
            self.instruction_following_metrics[key] = {"ratios": []}
        self.instruction_following_metrics[key]["ratios"].append(
            instruction_following_ratio
        )

    def calculate_all_metrics(self, data, iou_thresholds=[0.5]):
        """Calculate metrics for all samples with multiple IoU thresholds"""
        all_results = {}

        # Separate pointing, keypoint, hallucination and GUI tasks from other tasks for efficiency
        pointing_samples = []
        keypoint_samples = []
        hallucination_samples = []
        gui_samples = []
        other_samples = []

        for sample in data:
            if "task_name" not in sample:
                sample["task_name"] = "referring_object_detection"
            if sample["task_name"] == "pointing":
                pointing_samples.append(sample)
            elif sample["task_name"] == "keypoint":
                keypoint_samples.append(sample)
            elif sample["task_name"] == "hallucination":
                hallucination_samples.append(sample)
            elif sample["task_name"] == "gui":
                gui_samples.append(sample)
            else:
                other_samples.append(sample)

        print(
            f"Found {len(pointing_samples)} pointing samples, {len(keypoint_samples)} keypoint samples, {len(hallucination_samples)} hallucination samples, {len(gui_samples)} GUI samples, and {len(other_samples)} other samples"
        )

        # Process pointing tasks only once (IoU not relevant)
        if pointing_samples:
            print("Processing pointing tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(pointing_samples, desc="Pointing tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for pointing sample: {e}")

            # Store pointing results for all IoU thresholds (they're the same)
            pointing_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process keypoint tasks with different OKS thresholds (mapped from IoU thresholds)
        keypoint_results_by_iou = {}
        if keypoint_samples:
            print("Processing keypoint tasks with different OKS thresholds...")
            for iou in iou_thresholds:
                print(f"  Processing keypoint tasks for IoU/OKS threshold: {iou}")
                self.results = defaultdict(lambda: defaultdict(list))
                self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
                self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
                self.hallucination_metrics = defaultdict(
                    lambda: {"accuracies": [], "pred_counts": []}
                )
                self.gui_metrics = defaultdict(
                    lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
                )
                self.wrong_rejection_metrics = defaultdict(
                    lambda: {"wrong_rejections": [], "total_samples": []}
                )
                self.keypoint_metrics = defaultdict(
                    lambda: {
                        "ap_scores": [],
                        "avg_oks": [],
                        "gt_counts": [],
                        "pred_counts": [],
                        "prf1_results_list": [],
                    }
                )

                for sample in tqdm(
                    keypoint_samples, desc=f"Keypoint tasks (OKS threshold {iou})"
                ):
                    try:
                        self.calculate_metrics_for_sample(
                            sample, iou
                        )  # Use current IoU/OKS threshold
                    except Exception as e:
                        print(f"Error calculating metrics for keypoint sample: {e}")

                # Store keypoint results for this specific IoU/OKS threshold
                keypoint_results_by_iou[iou] = {
                    "basic_metrics": dict(self.results),
                    "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                    "instruction_following_metrics": dict(
                        self.instruction_following_metrics
                    ),
                    "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
                    "keypoint_metrics": dict(self.keypoint_metrics),
                }

        # Process hallucination tasks only once (IoU not relevant)
        if hallucination_samples:
            print("Processing hallucination tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(hallucination_samples, desc="Hallucination tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for hallucination sample: {e}")

            # Store hallucination results for all IoU thresholds (they're the same)
            hallucination_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "hallucination_metrics": dict(self.hallucination_metrics),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process GUI tasks only once (IoU not relevant)
        if gui_samples:
            print("Processing GUI tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(gui_samples, desc="GUI tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for GUI sample: {e}")

            # Store GUI results for all IoU thresholds (they're the same)
            gui_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "gui_metrics": dict(self.gui_metrics),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process other tasks with multiple IoU thresholds
        for iou in iou_thresholds:
            print(f"Calculating metrics with IoU threshold: {iou}")

            # Reset results for this IoU threshold
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            # Process non-pointing samples
            for sample in tqdm(other_samples, desc=f"IoU {iou}"):
                try:
                    self.calculate_metrics_for_sample(sample, iou)
                except Exception as e:
                    print(f"Error calculating metrics for sample: {e}")

            # Store results for this IoU threshold
            all_results[iou] = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

            # Add pointing and keypoint results to this IoU threshold if they exist
            if pointing_samples:
                for key, value in pointing_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in pointing_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in pointing_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in pointing_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if keypoint_samples and iou in keypoint_results_by_iou:
                keypoint_results = keypoint_results_by_iou[iou]
                for key, value in keypoint_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in keypoint_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in keypoint_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in keypoint_results["keypoint_metrics"].items():
                    if "keypoint_metrics" not in all_results[iou]:
                        all_results[iou]["keypoint_metrics"] = {}
                    all_results[iou]["keypoint_metrics"][key] = value
                for key, value in keypoint_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if hallucination_samples:
                for key, value in hallucination_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in hallucination_results[
                    "visual_prompt_metrics"
                ].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in hallucination_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in hallucination_results[
                    "hallucination_metrics"
                ].items():
                    if "hallucination_metrics" not in all_results[iou]:
                        all_results[iou]["hallucination_metrics"] = {}
                    all_results[iou]["hallucination_metrics"][key] = value
                for key, value in hallucination_results[
                    "wrong_rejection_metrics"
                ].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if gui_samples:
                for key, value in gui_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in gui_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in gui_results["instruction_following_metrics"].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in gui_results["gui_metrics"].items():
                    if "gui_metrics" not in all_results[iou]:
                        all_results[iou]["gui_metrics"] = {}
                    all_results[iou]["gui_metrics"][key] = value
                for key, value in gui_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

        return all_results

    def print_results(self, all_results):
        """Print simplified results with COCO-style metrics"""
        print("\n" + "=" * 180)
        print("UNIVERSAL METRICS CALCULATION RESULTS".center(180))
        print("=" * 180)

        # Organize metrics by dataset and IoU threshold
        dataset_metrics_by_iou = {}

        for iou, results in all_results.items():
            for key, metrics in results["basic_metrics"].items():
                if metrics["recalls"]:
                    if key not in dataset_metrics_by_iou:
                        dataset_metrics_by_iou[key] = {}

                    avg_recall = mean(metrics["recalls"])
                    avg_precision = mean(metrics["precisions"])
                    f1_score = (
                        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                        if (avg_precision + avg_recall) > 0
                        else 0.0
                    )

                    dataset_metrics_by_iou[key][iou] = {
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1": f1_score,
                        "samples": len(metrics["recalls"]),
                    }

        # Calculate MAE for visual prompt tasks (aggregated across all IoU thresholds)
        vp_mae_results = {}
        for iou, results in all_results.items():
            for key, vp_metrics in results["visual_prompt_metrics"].items():
                if vp_metrics["maes"]:
                    if key not in vp_mae_results:
                        vp_mae_results[key] = []
                    vp_mae_results[key].extend(vp_metrics["maes"])

        # Calculate InstructionFollowing metrics (aggregated across all IoU thresholds)
        instruction_following_results = {}
        for iou, results in all_results.items():
            for key, if_metrics in results["instruction_following_metrics"].items():
                if if_metrics["ratios"]:
                    if key not in instruction_following_results:
                        instruction_following_results[key] = []
                    instruction_following_results[key].extend(if_metrics["ratios"])

        # Calculate Keypoint metrics (aggregated across all IoU thresholds)
        keypoint_results = {}
        for iou, results in all_results.items():
            if "keypoint_metrics" in results:
                for key, kp_metrics in results["keypoint_metrics"].items():
                    if kp_metrics.get("prf1_results_list"):
                        if key not in keypoint_results:
                            keypoint_results[key] = {
                                "prf1_results_list": [],
                                "ap_scores": [],
                                "avg_oks": [],
                            }
                        keypoint_results[key]["prf1_results_list"].extend(
                            kp_metrics["prf1_results_list"]
                        )
                        keypoint_results[key]["ap_scores"].extend(
                            kp_metrics.get("ap_scores", [])
                        )
                        keypoint_results[key]["avg_oks"].extend(
                            kp_metrics.get("avg_oks", [])
                        )
                    elif kp_metrics.get("ap_scores"):
                        if key not in keypoint_results:
                            keypoint_results[key] = {"ap_scores": [], "avg_oks": []}
                        keypoint_results[key]["ap_scores"].extend(
                            kp_metrics["ap_scores"]
                        )
                        keypoint_results[key]["avg_oks"].extend(
                            kp_metrics.get("avg_oks", [])
                        )

        # Calculate Hallucination metrics (aggregated across all IoU thresholds)
        hallucination_results = {}
        for iou, results in all_results.items():
            if "hallucination_metrics" in results:
                for key, hall_metrics in results["hallucination_metrics"].items():
                    if hall_metrics["accuracies"]:
                        if key not in hallucination_results:
                            hallucination_results[key] = {
                                "accuracies": [],
                                "pred_counts": [],
                            }
                        hallucination_results[key]["accuracies"].extend(
                            hall_metrics["accuracies"]
                        )
                        hallucination_results[key]["pred_counts"].extend(
                            hall_metrics["pred_counts"]
                        )

        # Calculate GUI metrics (aggregated across all IoU thresholds)
        gui_results = {}
        for iou, results in all_results.items():
            if "gui_metrics" in results:
                for key, gui_metrics in results["gui_metrics"].items():
                    if gui_metrics["accuracies"]:
                        if key not in gui_results:
                            gui_results[key] = {
                                "accuracies": [],
                                "correct_counts": [],
                                "total_counts": [],
                            }
                        gui_results[key]["accuracies"].extend(gui_metrics["accuracies"])
                        gui_results[key]["correct_counts"].extend(
                            gui_metrics["correct_counts"]
                        )
                        gui_results[key]["total_counts"].extend(
                            gui_metrics["total_counts"]
                        )

        # Calculate Wrong Rejection metrics (aggregated across all IoU thresholds)
        wrong_rejection_results = {}
        for iou, results in all_results.items():
            if "wrong_rejection_metrics" in results:
                for key, wr_metrics in results["wrong_rejection_metrics"].items():
                    if wr_metrics["wrong_rejections"]:
                        if key not in wrong_rejection_results:
                            wrong_rejection_results[key] = {
                                "wrong_rejections": [],
                                "total_samples": [],
                            }
                        wrong_rejection_results[key]["wrong_rejections"].extend(
                            wr_metrics["wrong_rejections"]
                        )
                        wrong_rejection_results[key]["total_samples"].extend(
                            wr_metrics["total_samples"]
                        )

        # Find the maximum length of task_dataset names for proper alignment
        max_name_length = (
            max(len(key) for key in dataset_metrics_by_iou.keys())
            if dataset_metrics_by_iou
            else 30
        )
        name_width = min(max_name_length + 5, 60)  # Cap at 60 characters

        # Print header with IoU thresholds
        header = f"{'Task_Dataset':<{name_width}} | {'IoU=0.5':<20} | {'IoU=0.9':<20} | {'mIoU':<20} | {'MAE':<12} | {'InstFollow':<12} | {'Keypoint':<15} | {'Halluc_Acc':<12} | {'GUI_Acc':<10} | {'WrongRej':<10}"
        print(header)
        print("-" * len(header))

        for key, iou_metrics in dataset_metrics_by_iou.items():
            # Check if this is a pointing task
            is_pointing_task = key.startswith("pointing_")

            if is_pointing_task:
                # For pointing tasks, use any available IoU threshold (they should all be the same)
                any_iou = list(iou_metrics.keys())[0] if iou_metrics else 0.5
                pointing_metrics = iou_metrics.get(
                    any_iou, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )

                # For pointing tasks, we only show the main metrics (no IoU variations)
                pointing_str = f"P:{pointing_metrics['precision']:.3f} R:{pointing_metrics['recall']:.3f} F1:{pointing_metrics['f1']:.3f}"

                # Get Wrong Rejection metrics for pointing tasks
                wrong_rej_str = "-"
                if key in wrong_rejection_results:
                    wrong_rejections = wrong_rejection_results[key]["wrong_rejections"]
                    total_samples = wrong_rejection_results[key]["total_samples"]
                    if total_samples:
                        wrong_rej_rate = sum(wrong_rejections) / sum(total_samples)
                        wrong_rej_str = f"{wrong_rej_rate:.4f}"

                # Format the line for pointing tasks
                line = f"{key:<{name_width}} | {pointing_str:<20} | {'N/A':<20} | {'N/A':<20} | {'N/A':<12} | {'N/A':<12} | {'N/A':<15} | {'N/A':<12} | {'N/A':<10} | {wrong_rej_str:<10}"
                print(line)
            else:
                # Get metrics for IoU=0.5
                iou_05_metrics = iou_metrics.get(
                    0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )
                iou_05_str = f"P:{iou_05_metrics['precision']:.3f} R:{iou_05_metrics['recall']:.3f} F1:{iou_05_metrics['f1']:.3f}"

                # Get metrics for IoU=0.95
                iou_095_metrics = iou_metrics.get(
                    0.95, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )
                iou_095_str = f"P:{iou_095_metrics['precision']:.3f} R:{iou_095_metrics['recall']:.3f} F1:{iou_095_metrics['f1']:.3f}"

                # Calculate mIoU (mean across all IoU thresholds)
                all_precisions = [
                    metrics["precision"] for metrics in iou_metrics.values()
                ]
                all_recalls = [metrics["recall"] for metrics in iou_metrics.values()]
                all_f1s = [metrics["f1"] for metrics in iou_metrics.values()]

                mprecision = mean(all_precisions) if all_precisions else 0.0
                mrecall = mean(all_recalls) if all_recalls else 0.0
                mf1 = mean(all_f1s) if all_f1s else 0.0
                miou_str = f"P:{mprecision:.3f} R:{mrecall:.3f} F1:{mf1:.3f}"

                # Get MAE for visual prompt tasks
                mae_str = "-"
                if key in vp_mae_results:
                    avg_mae = mean(vp_mae_results[key])
                    mae_str = f"{avg_mae:.4f}"

                # Get InstructionFollowing metric
                inst_follow_str = "-"
                if key in instruction_following_results:
                    avg_inst_follow = mean(instruction_following_results[key])
                    inst_follow_str = f"{avg_inst_follow:.4f}"

                # Get Keypoint metrics
                keypoint_str = "-"
                if key in keypoint_results:
                    if keypoint_results[key].get("prf1_results_list"):
                        prf1_list = keypoint_results[key]["prf1_results_list"]
                        f1_50s = [r.get("F1@0.50", 0.0) for r in prf1_list]
                        f1_95s = [r.get("F1@0.95", 0.0) for r in prf1_list]
                        f1_moks = [r.get("F1@mOKS", 0.0) for r in prf1_list]
                        avg_f1_50 = mean(f1_50s) if f1_50s else 0.0
                        avg_f1_95 = mean(f1_95s) if f1_95s else 0.0
                        avg_f1_moks = mean(f1_moks) if f1_moks else 0.0
                        keypoint_str = (
                            f"F1@0.5:{avg_f1_50:.3f} "
                            f"F1@0.95:{avg_f1_95:.3f} "
                            f"mOKS:{avg_f1_moks:.3f}"
                        )
                    elif keypoint_results[key].get("ap_scores"):
                        ap_scores = keypoint_results[key]["ap_scores"]
                        avg_oks = keypoint_results[key]["avg_oks"]
                        # Extract AP@0.5 from each sample
                        ap_05_scores = []
                        for ap_dict in ap_scores:
                            if "AP@0.50" in ap_dict:
                                ap_05_scores.append(ap_dict["AP@0.50"])

                        avg_ap_05 = mean(ap_05_scores) if ap_05_scores else 0.0
                        avg_oks_score = mean(avg_oks) if avg_oks else 0.0
                        keypoint_str = f"AP:{avg_ap_05:.3f} OKS:{avg_oks_score:.3f}"

                # Get Hallucination metrics
                hallucination_str = "-"
                if key in hallucination_results:
                    # Calculate average accuracy and average prediction count
                    accuracies = hallucination_results[key]["accuracies"]
                    pred_counts = hallucination_results[key]["pred_counts"]

                    if accuracies:
                        avg_accuracy = mean(accuracies)
                        avg_pred_count = mean(pred_counts) if pred_counts else 0.0
                        hallucination_str = f"{avg_accuracy:.4f}"

                # Get GUI metrics
                gui_str = "-"
                if key in gui_results:
                    # Calculate average accuracy
                    accuracies = gui_results[key]["accuracies"]
                    correct_counts = gui_results[key]["correct_counts"]
                    total_counts = gui_results[key]["total_counts"]

                    if accuracies:
                        avg_accuracy = mean(accuracies)
                        total_correct = sum(correct_counts)
                        total_samples = sum(total_counts)
                        gui_str = f"{avg_accuracy:.4f}"

                # Get Wrong Rejection metrics
                wrong_rej_str = "-"
                if key in wrong_rejection_results:
                    wrong_rejections = wrong_rejection_results[key]["wrong_rejections"]
                    total_samples = wrong_rejection_results[key]["total_samples"]
                    if total_samples:
                        wrong_rej_rate = sum(wrong_rejections) / sum(total_samples)
                        wrong_rej_str = f"{wrong_rej_rate:.4f}"

                # Format the line with proper alignment
                line = f"{key:<{name_width}} | {iou_05_str:<20} | {iou_095_str:<20} | {miou_str:<20} | {mae_str:<12} | {inst_follow_str:<12} | {keypoint_str:<15} | {hallucination_str:<12} | {gui_str:<10} | {wrong_rej_str:<10}"
                print(line)

        print("=" * len(header))


def get_args():
    parser = argparse.ArgumentParser(
        description="Universal metrics calculator for multiple tasks"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/RefCOCOg_test/answer.jsonl",
        help="Path to prediction JSONL file",
    )
    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="IoU thresholds to evaluate (default: [0.5, 0.75, 0.9] for faster evaluation)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/RefCOCOg_test/eval_results.json",
        help="Path to save detailed results JSON (optional)",
    )
    parser.add_argument(
        "--auto_detect_pointing",
        action="store_true",
        default=True,
        help="Automatically detect pointing tasks and use single IoU threshold",
    )
    parser.add_argument(
        "--match_by_category",
        action="store_true",
        help="Match detection boxes within each normalized category; use for OCR/text detection datasets.",
    )
    parser.add_argument(
        "--save_keypoint_metrics",
        action="store_true",
        help="Write keypoint F1/mOKS metrics JSONL for auto_table.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples")

    # Check if auto-detect pointing is enabled
    if args.auto_detect_pointing:
        # Check if any sample is a pointing task
        pointing_tasks = [
            sample
            for sample in data
            if sample.get("task_name") in ["pointing", "pointing_referring"]
        ]
        if pointing_tasks:
            print(
                f"Detected {len(pointing_tasks)} pointing tasks. Using single IoU threshold [0.5] for efficiency."
            )
            iou_thresholds = [0.5]
        else:
            print("No pointing tasks detected. Using full IoU threshold range.")
            iou_thresholds = args.iou_thresholds
    else:
        # Always use the full IoU threshold range when auto-detect is disabled
        # The calculate_all_metrics method will handle mixed tasks properly
        iou_thresholds = args.iou_thresholds
        pointing_tasks = [
            sample
            for sample in data
            if sample.get("task_name") in ["pointing", "pointing_referring"]
        ]
        if pointing_tasks:
            print(
                f"Found {len(pointing_tasks)} pointing tasks mixed with other tasks. "
                f"Will calculate metrics for all IoU thresholds: {iou_thresholds}"
            )

    # Initialize calculator
    calculator = UniversalMetricsCalculator(match_by_category=args.match_by_category)

    # # Preprocess data before use.
    # processed_data = preprocess_data(data)
    # Calculate metrics
    all_results = calculator.calculate_all_metrics(data, iou_thresholds)

    # Print results
    calculator.print_results(all_results)
    if args.save_keypoint_metrics:
        save_keypoint_metrics_jsonl(all_results, args.output_path)
    else:
        save_metrics_jsonl(all_results, args.data_path, args.output_path)
    # Save detailed results if requested
    if args.output_path:
        print(f"\nSaving detailed results to: {args.output_path}")
        if not os.path.exists(os.path.dirname(args.output_path)):
            os.makedirs(os.path.dirname(args.output_path))
        with open(args.output_path, "w") as f:
            json.dump(
                all_results,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (int, float)) else str(x),
            )


def calculate_keypoint_distance(pred_point, gt_point):
    """Calculate Euclidean distance between two keypoints"""
    if pred_point is None or gt_point is None:
        return float("inf")
    return np.sqrt(
        (pred_point[0] - gt_point[0]) ** 2 + (pred_point[1] - gt_point[1]) ** 2
    )


def calculate_oks(gt_bbox, gt_keypoints, pred_bbox, pred_keypoints, sigma=0.025):
    """
    Calculate Object Keypoint Similarity (OKS)

    Args:
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
        gt_keypoints: Ground truth keypoints dict
        pred_bbox: Predicted bounding box [x1, y1, x2, y2]
        pred_keypoints: Predicted keypoints dict
        sigma: Standard deviation for OKS calculation

    Returns:
        OKS score
    """
    if not gt_keypoints or not pred_keypoints:
        return 0.0

    # Calculate bbox area for normalization
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    if gt_area <= 0:
        return 0.0

    # Only consider keypoints that exist in GT (don't penalize extra predictions)
    gt_keypoint_names = set(gt_keypoints.keys())

    total_weight = 0
    total_score = 0

    for kp_name in gt_keypoint_names:
        gt_kp = gt_keypoints.get(kp_name)
        pred_kp = pred_keypoints.get(kp_name)

        # Skip if GT keypoint is missing (shouldn't happen since we iterate over GT keys)
        # or if predicted keypoint is missing (model didn't predict this GT keypoint)
        if gt_kp is None or pred_kp is None or pred_kp == "unvisible":
            continue

        # Calculate distance
        distance = calculate_keypoint_distance(pred_kp, gt_kp)

        # Calculate OKS for this keypoint
        # Using a simplified approach where all keypoints have equal weight
        weight = 1.0
        kp_score = weight * np.exp(-(distance**2) / (2 * sigma**2 * gt_area))

        total_score += kp_score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight


def calculate_keypoint_ap(
    gt_instances, pred_instances, oks_thresholds=[0.5, 0.75, 0.9, 0.95]
):
    """
    Calculate Average Precision for keypoint detection

    Args:
        gt_instances: List of ground truth instances with bbox and keypoints
        pred_instances: List of predicted instances with bbox and keypoints
        oks_thresholds: List of OKS thresholds for AP calculation

    Returns:
        Dictionary with AP scores for each threshold
    """
    if not gt_instances:
        # If no GT instances with keypoints, return perfect score regardless of predictions
        # This means we don't penalize the model for predicting keypoints when GT has none
        return {f"AP@{thresh:.2f}": 1.0 for thresh in oks_thresholds}

    if not pred_instances:
        return {f"AP@{thresh:.2f}": 0.0 for thresh in oks_thresholds}

    ap_results = {}

    for threshold in oks_thresholds:
        # Calculate OKS for all GT-prediction pairs
        all_scores = []
        all_matches = []

        for gt_idx, gt_instance in enumerate(gt_instances):
            gt_bbox = gt_instance.get("bbox", [0, 0, 1, 1])
            gt_keypoints = gt_instance.get("keypoints", {})

            best_score = 0.0
            best_pred_idx = -1

            for pred_idx, pred_instance in enumerate(pred_instances):
                pred_bbox = pred_instance.get("bbox", [0, 0, 1, 1])
                pred_keypoints = pred_instance.get("keypoints", {})

                oks_score = calculate_oks(
                    gt_bbox, gt_keypoints, pred_bbox, pred_keypoints
                )

                if oks_score > best_score:
                    best_score = oks_score
                    best_pred_idx = pred_idx

            all_scores.append(best_score)
            all_matches.append(best_score >= threshold)

        # Calculate precision and recall
        if len(all_matches) == 0:
            ap = 0.0
        else:
            # Simple AP calculation: fraction of GT instances that have a match
            ap = sum(all_matches) / len(all_matches)

        ap_results[f"AP@{threshold:.2f}"] = ap

    return ap_results


def calculate_keypoint_prf1(
    gt_instances: List[Dict],
    pred_instances: List[Dict],
    oks_thresholds: List[float] = None,
):
    """Calculate one-to-one keypoint precision, recall, and F1 at OKS thresholds."""
    if oks_thresholds is None:
        oks_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if not gt_instances and not pred_instances:
        results = {}
        for threshold in oks_thresholds:
            results[f"P@{threshold:.2f}"] = 1.0
            results[f"R@{threshold:.2f}"] = 1.0
            results[f"F1@{threshold:.2f}"] = 1.0
        results["F1@mOKS"] = 1.0
        return results

    if not gt_instances or not pred_instances:
        results = {}
        for threshold in oks_thresholds:
            results[f"P@{threshold:.2f}"] = 0.0
            results[f"R@{threshold:.2f}"] = 0.0
            results[f"F1@{threshold:.2f}"] = 0.0
        results["F1@mOKS"] = 0.0
        return results

    num_gt = len(gt_instances)
    num_pred = len(pred_instances)
    oks_matrix = np.zeros((num_gt, num_pred))

    for i, gt_instance in enumerate(gt_instances):
        for j, pred_instance in enumerate(pred_instances):
            oks_matrix[i, j] = calculate_oks(
                gt_instance.get("bbox", [0, 0, 1, 1]),
                gt_instance.get("keypoints", {}),
                pred_instance.get("bbox", [0, 0, 1, 1]),
                pred_instance.get("keypoints", {}),
            )

    results = {}
    f1_scores = []
    gt_areas = [
        (gt.get("bbox", [0, 0, 1, 1])[2] - gt.get("bbox", [0, 0, 1, 1])[0])
        * (gt.get("bbox", [0, 0, 1, 1])[3] - gt.get("bbox", [0, 0, 1, 1])[1])
        for gt in gt_instances
    ]
    gt_order = np.argsort(gt_areas)[::-1]

    for threshold in oks_thresholds:
        matched_preds = set()
        true_positives = 0

        for gt_idx in gt_order:
            best_pred_idx = -1
            best_oks = -1
            for pred_idx in range(num_pred):
                if pred_idx in matched_preds:
                    continue
                if oks_matrix[gt_idx, pred_idx] > best_oks:
                    best_oks = oks_matrix[gt_idx, pred_idx]
                    best_pred_idx = pred_idx
            if best_oks >= threshold and best_pred_idx != -1:
                true_positives += 1
                matched_preds.add(best_pred_idx)

        precision = true_positives / num_pred if num_pred > 0 else 0.0
        recall = true_positives / num_gt if num_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[f"P@{threshold:.2f}"] = precision
        results[f"R@{threshold:.2f}"] = recall
        results[f"F1@{threshold:.2f}"] = f1
        f1_scores.append(f1)

    results["F1@mOKS"] = float(np.mean(f1_scores))
    return results


def calculate_keypoint_metrics_for_sample(
    gt_data: Dict[str, List[Dict]],
    pred_data: Dict[str, List[Dict]],
):
    """Calculate keypoint metrics for a single sample."""
    gt_instances = []
    pred_instances = []

    for instances in gt_data.values():
        for instance in instances:
            keypoints = instance.get("keypoints", {})
            if isinstance(keypoints, dict) and keypoints:
                gt_instances.append(instance)

    for instances in pred_data.values():
        pred_instances.extend(instances)

    oks_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    prf1_results = calculate_keypoint_prf1(
        gt_instances, pred_instances, oks_thresholds
    )
    ap_results = {
        f"AP@{threshold:.2f}": prf1_results.get(f"R@{threshold:.2f}", 0.0)
        for threshold in [0.5, 0.75, 0.9, 0.95]
    }

    avg_oks = 0.0
    matched_pairs = 0
    for gt_instance in gt_instances:
        gt_bbox = gt_instance.get("bbox", [0, 0, 1, 1])
        gt_keypoints = gt_instance.get("keypoints", {})
        best_oks = 0.0
        for pred_instance in pred_instances:
            pred_bbox = pred_instance.get("bbox", [0, 0, 1, 1])
            pred_keypoints = pred_instance.get("keypoints", {})
            best_oks = max(
                best_oks,
                calculate_oks(gt_bbox, gt_keypoints, pred_bbox, pred_keypoints),
            )
        if best_oks > 0.5:
            avg_oks += best_oks
            matched_pairs += 1

    avg_oks = avg_oks / matched_pairs if matched_pairs > 0 else 0.0

    return {
        "prf1_results": prf1_results,
        "ap_results": ap_results,
        "avg_oks": avg_oks,
        "total_gt_instances": len(gt_instances),
        "total_pred_instances": len(pred_instances),
        "matched_pairs": matched_pairs,
    }


if __name__ == "__main__":
    main()
