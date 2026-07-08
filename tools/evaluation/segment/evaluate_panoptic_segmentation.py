import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from panopticapi.utils import rgb2id
from pycocotools import mask as mask_utils
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import normalize_category
from utils.mask import compute_confusion_matrix
from segment_utils.coco import COCO
from segment_utils.map import derive_coco_results, evaluate_predictions_on_coco
from segment_utils.pq import pq_compute, print_panoptic_results


def _coco_segm_to_rle(segm, height: int, width: int):
    if segm is None:
        return None
    if isinstance(segm, dict) and "counts" in segm and "size" in segm:
        counts = segm.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segm, height, width)
            return rle[0] if isinstance(rle, list) else rle
        return segm
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, height, width)
        return mask_utils.merge(rles)
    return None


def _rle_to_iou_compatible(rle):
    if rle is None or not isinstance(rle, dict):
        return None
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle = dict(rle)
        rle["counts"] = counts.encode("utf-8")
    return rle


def compute_instance_f1_by_gt_order(
    gt_instance_json_path: str,
    instance_results: list,
    iou_threshold: float = 0.5,
):
    with open(gt_instance_json_path, "r") as f:
        gt = json.load(f)

    valid_category_ids = {int(c["id"]) for c in gt.get("categories", []) if "id" in c}
    img_hw = {img["id"]: (int(img["height"]), int(img["width"])) for img in gt.get("images", [])}

    gt_by_image = {}
    total_gt = 0
    for ann in gt.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if category_id is None:
            continue
        category_id = int(category_id)
        if valid_category_ids and category_id not in valid_category_ids:
            continue
        if image_id not in img_hw:
            continue
        h, w = img_hw[image_id]
        rle = _coco_segm_to_rle(ann.get("segmentation"), h, w)
        if rle is None:
            continue
        item = {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "iscrowd": int(ann.get("iscrowd", 0)),
        }
        gt_by_image.setdefault(image_id, []).append(item)
        total_gt += 1

    pred_by_image = {}
    total_pred = 0
    for pred in instance_results:
        image_id = pred.get("image_id")
        segm = pred.get("segmentation")
        category_id = pred.get("category_id")
        if category_id is None:
            continue
        category_id = int(category_id)
        if valid_category_ids and category_id not in valid_category_ids:
            continue
        if image_id is None or segm is None or image_id not in img_hw:
            continue
        h, w = img_hw[image_id]
        rle = _coco_segm_to_rle(segm, h, w)
        if rle is None:
            continue
        item = {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
        }
        pred_by_image.setdefault(image_id, []).append(item)
        total_pred += 1

    matches = 0
    for image_id, gt_list in gt_by_image.items():
        pred_list = pred_by_image.get(image_id, [])
        if not gt_list or not pred_list:
            continue

        used = [False] * len(pred_list)
        for gt_ann in gt_list:
            best_iou = 0.0
            best_j = -1
            gt_rle = _rle_to_iou_compatible(gt_ann["segmentation"])
            gt_cat = gt_ann["category_id"]
            gt_iscrowd = gt_ann.get("iscrowd", 0)

            for j, pred_ann in enumerate(pred_list):
                if used[j] or pred_ann["category_id"] != gt_cat:
                    continue
                pred_rle = _rle_to_iou_compatible(pred_ann["segmentation"])
                if pred_rle is None or gt_rle is None:
                    continue
                iou = float(mask_utils.iou([pred_rle], [gt_rle], [gt_iscrowd])[0][0])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                used[best_j] = True
                matches += 1

    precision = matches / total_pred if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
    recall = matches / total_gt if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "iou_threshold": float(iou_threshold),
        "matches": int(matches),
        "gt_count": int(total_gt),
        "pred_count": int(total_pred),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def parse_start_idx(filename: str, prefix: str) -> int:
    """
    Parse start_idx from a file name, for example:
    metrics_00000000_00001000.csv
    conf_matrix_00000000_00001000.npy
    """
    match = re.search(rf"{prefix}_(\d+)_(\d+)", filename)
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    return int(match.group(1))


def _maybe_append_candidate(candidates: list, value: str | None):
    if value and value not in candidates:
        candidates.append(value)


def _to_cityscapes_label_name(name: str | None):
    if not name:
        return None

    basename = os.path.basename(str(name))
    if basename.endswith("_gtFine_labelIds.png"):
        return basename
    if basename.endswith("_leftImg8bit.png"):
        return basename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    if basename.endswith("_gtFine_leftImg8bit.png"):
        return basename.replace("_gtFine_leftImg8bit.png", "_gtFine_labelIds.png")
    if basename.endswith("_gtFine_panoptic.png"):
        return basename.replace("_gtFine_panoptic.png", "_gtFine_labelIds.png")
    if "." not in basename:
        return f"{basename}_gtFine_labelIds.png"
    return None


def _to_semseg_candidate_names(name: str | None):
    candidates = []

    def add(value: str | None):
        if value and value not in candidates:
            candidates.append(value)

    add(_to_cityscapes_label_name(name))
    if not name:
        return candidates

    basename = os.path.basename(str(name))
    stem, ext = os.path.splitext(basename)
    if stem.isdigit():
        add(f"{int(stem):012d}.png")
        add(f"{stem}.png")
    if ext.lower() in {".jpg", ".jpeg"}:
        add(f"{stem}.png")
    elif ext.lower() == ".png":
        add(basename)
    return candidates


def resolve_gt_semseg_path(gt_semseg_folder: str, pred_ann: dict, gt_ann: dict | None = None, gt_img: dict | None = None):
    candidate_names = []
    for value in (
        pred_ann.get("file_name"),
        pred_ann.get("image_id"),
        gt_ann.get("file_name") if gt_ann else None,
        gt_ann.get("image_id") if gt_ann else None,
        gt_img.get("file_name") if gt_img else None,
        gt_img.get("id") if gt_img else None,
    ):
        for candidate_name in _to_semseg_candidate_names(value):
            _maybe_append_candidate(candidate_names, candidate_name)

    roots = []
    for root in (gt_semseg_folder, os.path.join(gt_semseg_folder, "val")):
        root = os.path.normpath(root)
        if root not in roots:
            roots.append(root)

    tried_paths = []
    for candidate_name in candidate_names:
        city = candidate_name.split("_")[0]
        for root in roots:
            for path in (
                os.path.join(root, candidate_name),
                os.path.join(root, city, candidate_name),
            ):
                tried_paths.append(path)
                if os.path.isfile(path):
                    return path

    raise FileNotFoundError(
        "Unable to resolve semantic GT png. "
        f"pred_file={pred_ann.get('file_name')} image_id={pred_ann.get('image_id')} "
        f"searched={tried_paths[:8]}"
    )


def convert_gt_semseg_to_contiguous(
    gt_semseg: np.ndarray,
    id_to_contiguous_id: dict,
    num_classes: int,
):
    gt_semseg = gt_semseg.astype(np.int64, copy=False)
    valid_pixels = gt_semseg[gt_semseg != 255]
    if valid_pixels.size == 0:
        return gt_semseg.astype(np.uint32, copy=False)

    max_valid = int(valid_pixels.max())
    min_valid = int(valid_pixels.min())
    if min_valid >= 0 and max_valid < num_classes:
        return gt_semseg.astype(np.uint32, copy=False)

    lut_size = max(max_valid, max(id_to_contiguous_id.keys(), default=0), 255) + 1
    lut = np.full(lut_size, 255, dtype=np.uint16)
    for category_id, contiguous_id in id_to_contiguous_id.items():
        if 0 <= int(category_id) < lut_size:
            lut[int(category_id)] = int(contiguous_id)

    mapped = np.full(gt_semseg.shape, 255, dtype=np.uint32)
    in_range = (gt_semseg >= 0) & (gt_semseg < lut_size)
    mapped[in_range] = lut[gt_semseg[in_range]]
    return mapped


def convert_panseg_to_semseg(pred_seg: np.ndarray, segments_info: list, id_to_contiguous_id: dict):
    """
    Convert panoptic segmentation to semantic segmentation.

    pred_seg: (H, W) panoptic id map decoded by rgb2id.
    segments_info: list. Each item contains:
        - "id": panoptic segment id
        - "category_id": COCO category id
    id_to_contiguous_id: dict mapping COCO category id to contiguous id.

    Return the semantic segmentation map (H, W), whose pixel values are contiguous class ids.
    """
    semseg = np.full(pred_seg.shape, 255, dtype=np.uint32)

    # pred_seg can contain large ids, so a dictionary is safer than an array.
    mapping = {seg["id"]: id_to_contiguous_id.get(seg["category_id"], 255) for seg in segments_info}

    flat = pred_seg.reshape(-1)
    out_flat = semseg.reshape(-1)

    for pan_id, cls in mapping.items():
        mask = (flat == pan_id)
        out_flat[mask] = cls

    return semseg


def panoptic_to_instance_segments(panoptic_json, panoptic_mask_dir, output_file):
    """
    Convert panoptic segmentation results to COCO instance segmentation format.
    """
    with open(panoptic_json, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    instance_results = []

    for ann in annotations:
        image_id = ann["image_id"]
        file_name = ann["file_name"]
        segments_info = ann["segments_info"]

        mask_path = os.path.join(panoptic_mask_dir, file_name)
        pan_mask = np.array(Image.open(mask_path), dtype=np.uint32)
        pan_mask = rgb2id(pan_mask)

        for seg in segments_info:
            seg_id = seg["id"]
            category_id = seg["category_id"]

            binary_mask = (pan_mask == seg_id).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            instance_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": seg.get("score", 1.0),  # Default score to 1.0 when absent.
                }
            )

    print(f"Converted {len(instance_results)} instance masks.")

    with open(output_file, "w") as f:
        json.dump(instance_results, f)

    print(f"Saved instance segmentation results to: {output_file}")
    return instance_results


def compute_metrics_from_conf_matrix(conf_matrix: np.ndarray, cat_id2name: dict):
    """
    Compute IoU, ACC, and pACC from the aggregated confusion matrix.
    Supports multi-class evaluation and skips NaN values.
    """
    num_classes = conf_matrix.shape[0]
    assert conf_matrix.shape[0] == conf_matrix.shape[1], f"Confusion matrix must be square, got {conf_matrix.shape}"

    tp = np.diag(conf_matrix).astype(np.float64)
    pos_gt = np.sum(conf_matrix, axis=0).astype(np.float64)  # GT count per class
    pos_pred = np.sum(conf_matrix, axis=1).astype(np.float64)  # prediction count per class

    results = {}
    ious = []

    for cls in range(num_classes):
        # Check whether this is a background class.
        if cls in cat_id2name:
            name = cat_id2name[cls]
            is_bg = False
        else:
            name = f"bg_{cls}"
            is_bg = True

        inter = tp[cls]
        union = pos_gt[cls] + pos_pred[cls] - tp[cls]
        if union > 0:
            iou = inter / union
            results[f"IoU_{name}"] = iou
            if not is_bg:  # Only count non-background classes.
                ious.append(iou)

        # Compute ACC only when pos_gt > 0.
        if pos_gt[cls] > 0:
            acc = tp[cls] / pos_gt[cls]
            results[f"ACC_{name}"] = acc

    # Global pixel accuracy.
    if np.sum(pos_gt) > 0:
        results["pACC"] = np.sum(tp) / np.sum(pos_gt)

    # Mean IoU over valid classes only.
    if ious:
        results["mIoU"] = np.mean(ious)

    return results


def aggregate_caption_metrics(csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    tp = df["TP"].sum()
    fn = df["FN"].sum()
    fp = df["FP"].sum()
    irrelevant_fp = df.get("irrelevant_FP", pd.Series([0])).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fnr = fn / max(tp + fn, 1)
    fpr = fp / max(tp + fp, 1)

    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "irrelevant_FP": int(irrelevant_fp),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "FNR": float(fnr),
        "FPR": float(fpr),
        "num_images": len(df),
    }


def should_run_instance_eval(gt_instance_json_path: str, pred_annotations: list):
    if not os.path.isfile(gt_instance_json_path):
        print(f"[WARN] gt_instance_json not found: {gt_instance_json_path}; skip instance evaluation.")
        return False

    with open(gt_instance_json_path, "r") as f:
        gt_instance_data = json.load(f)

    pred_image_ids = {ann.get("image_id") for ann in pred_annotations}
    gt_image_ids = {img.get("id") for img in gt_instance_data.get("images", [])}
    overlap = len(pred_image_ids & gt_image_ids)
    if overlap == 0:
        print(
            f"[WARN] gt_instance_json={gt_instance_json_path} has no overlapping image ids "
            "with current predictions; skip instance evaluation."
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Directory containing predictions_*.json files for each split."
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        default="./datas/gen_seg_data/coco2017/annotations/panoptic_val2017.json",
    )
    parser.add_argument("--gt_folder", type=str, default="./datas/gen_seg_data/coco2017/panoptic_val2017/")
    parser.add_argument(
        "--gt_semseg_folder", type=str, default="./datas/gen_seg_data/coco2017/panoptic_semseg_val2017/"
    )
    parser.add_argument(
        "--gt_instance_json",
        type=str,
        default="./datas/gen_seg_data/coco2017/annotations/instances_val2017.json",
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Canonical dataset name written into auto_table.csv, e.g. pan_coco_val / pan_coco_test")
    parser.add_argument("--task", type=str, default="Segmentation",
                        help="Task name written into auto_table.csv (default: segmentation)")
    parser.add_argument("--metrics_dir", type=str, default="",
                        help="Directory for merged metric outputs. Defaults to result_dir.")


    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    metrics_dir = args.metrics_dir.strip() or args.result_dir
    os.makedirs(metrics_dir, exist_ok=True)

    pred_folder = os.path.join(args.result_dir, "panoptic_eval")
    pred_semseg_folder = os.path.join(args.result_dir, "semantic_eval")

    pred_json_file = os.path.join(pred_folder, "predictions.json")
    pred_instance_file = os.path.join(metrics_dir, "instance.json")

    # ===== Merge split JSON files. =====
    if os.path.exists(os.path.join(args.result_dir, "predictions.json")):
        json_files = [os.path.join(args.result_dir, "predictions.json")]
    else:
        json_files = sorted(
            glob.glob(os.path.join(args.result_dir, "predictions_*.json")),
            key=lambda f: parse_start_idx(f, "predictions"),
        )
    if not json_files:
        raise FileNotFoundError(f"No json files found in {args.result_dir}")

    with open(json_files[0], "r") as f:
        predictions = json.load(f)

    pre_annotations = []
    for jf in json_files:
        with open(jf, "r") as f:
            pre_annotations.extend(json.load(f)["annotations"])
    predictions["annotations"] = pre_annotations

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(pred_semseg_folder, exist_ok=True)

    with open(pred_json_file, "w") as f:
        json.dump(predictions, f)

    # ===== Optional split-level panoptic caption metrics =====
    caption_summary = None
    metric_csv_files = sorted(
        glob.glob(os.path.join(args.result_dir, "metrics_*.csv")),
        key=lambda f: parse_start_idx(f, "metrics"),
    )
    if metric_csv_files:
        caption_summary = aggregate_caption_metrics(metric_csv_files)
        caption_table = [
            ["#Images", caption_summary["num_images"]],
            ["TP", caption_summary["TP"]],
            ["FN", caption_summary["FN"]],
            ["FP", caption_summary["FP"]],
            ["Irrelevant FP", caption_summary["irrelevant_FP"]],
            ["Precision", f"{caption_summary['precision'] * 100:.2f}"],
            ["Recall", f"{caption_summary['recall'] * 100:.2f}"],
            ["F1", f"{caption_summary['F1'] * 100:.2f}"],
            ["FNR", f"{caption_summary['FNR'] * 100:.2f}"],
            ["FPR", f"{caption_summary['FPR'] * 100:.2f}"],
        ]
        print("\n===== Caption Metrics (Instance-level) =====")
        print(tabulate(caption_table, headers=["Metric", "Value"], tablefmt="github"))

        out_caption_txt = os.path.join(metrics_dir, "caption_instance_metrics.txt")
        with open(out_caption_txt, "w") as f:
            f.write(tabulate(caption_table, headers=["Metric", "Value"], tablefmt="github"))
            f.write("\n")
        print(f"Saved caption instance metrics to {out_caption_txt}")
    else:
        print("\n[WARN] No metrics_*.csv found; skip panoptic caption metrics.")

    # ===== Panoptic PQ =====
    print("\n===== Panoptic Results (from pq_compute) =====")
    pq_res = pq_compute(
        gt_json_file=args.gt_json,
        pred_json_file=pred_json_file,
        gt_folder=args.gt_folder,
        pred_folder=pred_folder,
    )

    pq_table = print_panoptic_results(pq_res)
    print(pq_table)

    out_pq_txt = os.path.join(metrics_dir, "panoptic_pq_results.txt")
    with open(out_pq_txt, "w") as f:
        f.write("===== Panoptic Results (pq_compute) =====\n")
        f.write(pq_table)
        f.write("\n")
    print(f"Saved PQ results to {out_pq_txt}")

    # ===== Semantic (Confusion Matrix) =====
    with open(args.gt_json, "r") as f:
        gt_json_data = json.load(f)
    gt_ann_by_image_id = {ann["image_id"]: ann for ann in gt_json_data.get("annotations", [])}
    gt_img_by_id = {img["id"]: img for img in gt_json_data.get("images", [])}

    contiguous_id_to_name = {}
    id_to_contiguous_id = {}
    for contiguous_id, cat in enumerate(predictions["categories"]):
        name = normalize_category(cat["name"])
        contiguous_id_to_name[contiguous_id] = name
        id_to_contiguous_id[cat["id"]] = contiguous_id

    num_classes = len(predictions["categories"])
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for ann in pre_annotations:
        file_name_semseg = ann["file_name"]
        image_id = ann.get("image_id")
        gt_ann = gt_ann_by_image_id.get(image_id)
        gt_img = gt_img_by_id.get(image_id)
        gt_semseg_path = resolve_gt_semseg_path(args.gt_semseg_folder, ann, gt_ann=gt_ann, gt_img=gt_img)
        gt = np.array(Image.open(gt_semseg_path), dtype=np.uint32)
        gt = convert_gt_semseg_to_contiguous(gt, id_to_contiguous_id, num_classes)

        pred_semseg_path = os.path.join(pred_semseg_folder, file_name_semseg)
        if not os.path.isfile(pred_semseg_path):
            pred_seg_path = os.path.join(pred_folder, file_name_semseg)
            pred_seg = rgb2id(np.array(Image.open(pred_seg_path), dtype=np.uint32))
            segments_info = ann["segments_info"]
            pred_semseg = convert_panseg_to_semseg(pred_seg, segments_info, id_to_contiguous_id)
        else:
            pred_semseg = np.array(Image.open(pred_semseg_path), dtype=np.uint32)

        gt[gt == 255] = num_classes
        pred_semseg[pred_semseg == 255] = num_classes

        valid_mask = gt != num_classes
        conf_matrix += compute_confusion_matrix(
            gt[valid_mask], pred_semseg[valid_mask], num_classes=num_classes + 1
        )

    cm_results = compute_metrics_from_conf_matrix(conf_matrix, contiguous_id_to_name)
    miou = float(cm_results.get("mIoU", np.nan))

    print("\n===== Semantic Results (from Confusion Matrix) =====")
    for k, v in cm_results.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    out_cm_path = os.path.join(metrics_dir, "final_conf_matrix.npy")
    np.save(out_cm_path, conf_matrix)
    print(f"Saved merged confusion matrix to {out_cm_path}")

    # per-class + overall csv
    rows = []
    for cid, cname in contiguous_id_to_name.items():
        iou_key = f"IoU_{cname}"
        acc_key = f"ACC_{cname}"
        iou_val = cm_results.get(iou_key, None)
        acc_val = cm_results.get(acc_key, None)
        if iou_val is not None or acc_val is not None:
            rows.append({"class": cname, "IoU": iou_val, "ACC": acc_val})
    rows.append({"class": "overall", "IoU": cm_results.get("mIoU", None), "ACC": cm_results.get("pACC", None)})

    out_cm_csv = os.path.join(metrics_dir, "conf_matrix_metrics.csv")
    pd.DataFrame(rows).to_csv(out_cm_csv, index=False)
    print(f"Saved per-class + overall metrics to {out_cm_csv}")

    # ===== Instance (COCO segm) with try/except =====
    print("\n===== Instance Results (from coco_eval) =====")

    ap_50_95 = np.nan
    inst_err = None

    inst_f1 = np.nan
    inst_precision = np.nan
    inst_recall = np.nan

    try:
        if not should_run_instance_eval(args.gt_instance_json, pre_annotations):
            raise RuntimeError("Instance evaluation skipped because GT is unavailable or has no overlapping image ids.")

        instance_result = panoptic_to_instance_segments(pred_json_file, pred_folder, pred_instance_file)
        coco_api = COCO(args.gt_instance_json)
        coco_eval = evaluate_predictions_on_coco(coco_api, instance_result, iou_type="segm")
        table = derive_coco_results(coco_eval, iou_type="segm")
        print(table)

        ap_50_95 = float(coco_eval.stats[0])

        out_inst_txt = os.path.join(metrics_dir, "instance_coco_results.txt")
        with open(out_inst_txt, "w") as f:
            f.write("===== Instance Results (COCO segm) =====\n")
            f.write(str(table))
            f.write("\n")
        print(f"Saved instance results table to {out_inst_txt}")

        print("\n===== Instance F1 (GT-order matching, no confidence) =====")
        f1_res = compute_instance_f1_by_gt_order(
            gt_instance_json_path=args.gt_instance_json,
            instance_results=instance_result,
            iou_threshold=0.5,
        )
        f1_table = [
            ["IoU thr", f1_res["iou_threshold"]],
            ["GT count", f1_res["gt_count"]],
            ["Pred count", f1_res["pred_count"]],
            ["Matches", f1_res["matches"]],
            ["Precision", f"{f1_res['precision'] * 100:.2f}"],
            ["Recall", f"{f1_res['recall'] * 100:.2f}"],
            ["F1", f"{f1_res['f1'] * 100:.2f}"],
        ]
        print(tabulate(f1_table, headers=["Metric", "Value"], tablefmt="github"))

        out_f1_txt = os.path.join(metrics_dir, "instance_f1_results.txt")
        with open(out_f1_txt, "w") as f:
            f.write(tabulate(f1_table, headers=["Metric", "Value"], tablefmt="github"))
            f.write("\n")
        print(f"Saved instance F1 results to {out_f1_txt}")

        inst_precision = float(f1_res["precision"])
        inst_recall = float(f1_res["recall"])
        inst_f1 = float(f1_res["f1"])

    except Exception as e:
        inst_err = repr(e)
        print(f"[WARN] Instance COCO eval failed, skip it. Error: {inst_err}")

        out_inst_err = os.path.join(metrics_dir, "instance_coco_results.error.txt")
        with open(out_inst_err, "w") as f:
            f.write("===== Instance COCO eval failed =====\n")
            f.write(inst_err + "\n")
        print(f"Saved instance error log to {out_inst_err}")

        # Also write a placeholder txt file to avoid missing downstream dependencies.
        out_inst_txt = os.path.join(metrics_dir, "instance_coco_results.txt")
        with open(out_inst_txt, "w") as f:
            f.write("===== Instance Results (COCO segm) =====\n")
            f.write(f"FAILED: {inst_err}\n")
        print(f"Saved placeholder instance results to {out_inst_txt}")

    # ===== Summary CSV (WIDE, 1 row) =====
    pq_all = float(pq_res["All"]["pq"])

    def f4(x):
        """Segmentation: scale to percent if looks like ratio (<=1.5), then keep 4 decimals."""
        try:
            if x is None:
                return ""
            x = float(x)
            if x != x:  # NaN
                return ""
            if x <= 1.1:
                x *= 100.0
            return float(f"{x:.4f}")
        except Exception:
            return ""

    summary_wide = {
        "task": args.task,
        "dataset": args.dataset_name,
        "mIoU_conf_matrix": f4(miou),
        "AP_50_95_instance": f4(ap_50_95),
        "PQ_panoptic_all": f4(pq_all),
        "Instance_Precision": f4(inst_precision),
        "Instance_Recall": f4(inst_recall),
        "Instance_F1": f4(inst_f1),
    }

    if caption_summary is not None:
        summary_wide.update(
            {
                "Caption_Precision": f4(caption_summary["precision"]),
                "Caption_Recall": f4(caption_summary["recall"]),
                "Caption_F1": f4(caption_summary["F1"]),
                "Caption_FNR": f4(caption_summary["FNR"]),
                "Caption_FPR": f4(caption_summary["FPR"]),
            }
        )

    out_summary_csv = os.path.join(metrics_dir, "auto_table.csv")
    pd.DataFrame([summary_wide]).to_csv(out_summary_csv, index=False)
    print(f"Saved summary metrics (wide) to {out_summary_csv}")

if __name__ == "__main__":
    main()
