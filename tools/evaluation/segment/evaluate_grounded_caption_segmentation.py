import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from copy import deepcopy
import csv
import numpy as np
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mask import decode_mask
from segment_utils.coco import COCO
from segment_utils.coco_cap_eval import COCOEvalCap
from segment_utils.miou import compute_miou


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


def eval_miou(preds, gt_json):
    with open(gt_json, "r") as f:
        gt_data = json.load(f)
        gt_data["info"] = {"description": "GCGSeg dataset"}
    coco_gt = COCO(dataset=gt_data)
    coco_dt = coco_gt.loadRes(preds)
    imgids = sorted(list(set([pred["image_id"] for pred in preds])))
    imgids = sorted(list(coco_gt.imgs.keys()))

    mious = []
    for imgid in imgids:
        imginfo = coco_gt.loadImgs([imgid])[0]
        height, width = imginfo["height"], imginfo["width"]

        gt_ann_ids = coco_gt.getAnnIds(imgIds=[imgid])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        dt_ann_ids = coco_dt.getAnnIds(imgIds=[imgid])
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        gt_masks = [decode_mask(ann["segmentation"], height, width) for ann in gt_anns]
        dt_masks = [decode_mask(ann["segmentation"], height, width) for ann in dt_anns]
        mious.append(compute_miou(dt_masks, gt_masks))

    miou_res = float(np.mean(mious) * 100) if mious else 0.0
    return miou_res


def eval_map(preds, gt_json):
    with open(gt_json, "r") as f:
        gt_data = json.load(f)
        gt_data["info"] = {"description": "GCGSeg dataset"}
    coco_gt = COCO(dataset=gt_data)
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    imgids = sorted(list(coco_gt.imgs.keys()))
    coco_eval.params.imgIds = imgids
    coco_eval.params.catIds = [1]

    print("map results:")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist() if hasattr(coco_eval.stats, "tolist") else list(coco_eval.stats)
    assert len(stats) == 12, f"Unexpected COCOeval.stats length: {len(stats)}"

    return {
        "AP": float(stats[0]) * 100.0,
        "AP50": float(stats[1]) * 100.0,
        "AP75": float(stats[2]) * 100.0,
        "APs": float(stats[3]) * 100.0,
        "APm": float(stats[4]) * 100.0,
        "APl": float(stats[5]) * 100.0,
        "AR1": float(stats[6]) * 100.0,
        "AR10": float(stats[7]) * 100.0,
        "AR100": float(stats[8]) * 100.0,
        "ARs": float(stats[9]) * 100.0,
        "ARm": float(stats[10]) * 100.0,
        "ARl": float(stats[11]) * 100.0,
    }


def eval_caption(preds, gt_json):
    with open(gt_json, "r") as f:
        gt_data = json.load(f)
        gt_data["info"] = {"description": "GCGSeg dataset"}
    coco_gt = COCO(dataset=gt_data)
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOEvalCap(coco_gt, coco_dt)
    imgids = sorted(list(set([pred["image_id"] for pred in preds])))
    coco_eval.params["image_id"] = imgids

    coco_eval.evaluate()

    print("caption results:")
    for metric, value in coco_eval.eval.items():
        print(f"{metric}, {value * 100:.3f}")

    # Persist values after converting them to percentages.
    return {k: float(v) * 100.0 for k, v in coco_eval.eval.items()}


def append_metrics_csv(out_csv: str, row: dict):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # If the file does not exist, use the current row keys as the header.
    if not os.path.exists(out_csv):
        fieldnames = list(row.keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    # If the file exists, ensure the header contains all keys and backfill old rows when new fields are added.
    with open(out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        old_rows = list(reader)

    new_keys = [k for k in row.keys() if k not in old_fieldnames]
    if new_keys:
        fieldnames = old_fieldnames + new_keys
        # Backfill old and new rows with empty values where needed.
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in old_rows:
                for k in new_keys:
                    r.setdefault(k, "")
                writer.writerow(r)
            for k in old_fieldnames:
                row.setdefault(k, "")
            writer.writerow(row)
    else:
        # Append directly.
        with open(out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_fieldnames)
            # Fill missing fields with empty values; this should not normally happen.
            for k in old_fieldnames:
                row.setdefault(k, "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing predictions_*.json files for each split.")
    parser.add_argument("--gt_json", type=str,
                        default="datas/gcg_seg_data/annotations/val_test/val_gcg_coco_mask_gt.json")
    parser.add_argument("--cap_gt_json", type=str,
                        default="datas/gcg_seg_data/annotations/val_test/val_gcg_coco_caption_gt.json")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Canonical dataset name written into auto_table.csv, e.g. gcg_val / gcg_test")
    parser.add_argument("--task", type=str, default="Segmentation",
                        help="Task name written into auto_table.csv (default: segmentation)")
    parser.add_argument("--metrics_dir", type=str, default="",
                        help="Directory for merged metric outputs. Defaults to result_dir.")


    args = parser.parse_args()
    metrics_dir = args.metrics_dir.strip() or args.result_dir
    os.makedirs(metrics_dir, exist_ok=True)

    gt_json = args.gt_json
    cap_gt_json = args.cap_gt_json

    # ===== Process the CSV/JSON merge stage. =====
    if os.path.exists(os.path.join(args.result_dir, "predictions.json")):
        json_files = [os.path.join(args.result_dir, "predictions.json")]
    else:
        json_files = sorted(
            glob.glob(os.path.join(args.result_dir, "predictions_*.json")),
            key=lambda f: parse_start_idx(f, "predictions")
        )
    if not json_files:
        raise FileNotFoundError(f"No json files found in {args.result_dir}")

    predictions = list()
    for json_file in json_files:
        predictions.extend(json.load(open(json_file)))

    mask_preds = []
    caption_preds = []
    failed_cnt = 0
    for pred in predictions:
        if len(pred["gcg_phrases"]) == 0:
            failed_cnt += 1
        for seg in pred["segmentation"]:
            cur_mask_pred = {
                "image_id": pred["image_id"],
                "category_id": 1,
                "segmentation": seg,
                "score": 1.0,
            }
            mask_preds.append(cur_mask_pred)
        cur_caption_pred = {
            "image_id": pred["image_id"],
            "caption": pred["gcg_caption"],
            "labels": pred["gcg_phrases"],
        }
        caption_preds.append(cur_caption_pred)

    miou_res = eval_miou(deepcopy(mask_preds), gt_json)
    map_res = eval_map(deepcopy(mask_preds), gt_json)          # 12 items
    cap_res = eval_caption(deepcopy(caption_preds), cap_gt_json)  # METEOR/CIDEr

    print(f"mIoU results:\n{miou_res:.3f}")
    print(f"Failed {failed_cnt}/{len(predictions)} cases!")

    # ---- Assemble one row with all metrics.
    row = {
        "task": args.task,
        "dataset": args.dataset_name,
        "num_images": len(set([p["image_id"] for p in predictions])),
        "failed_cnt": failed_cnt,
        "mIoU": float(miou_res),
    }

    # Write all 12 COCOeval metrics.
    row.update(map_res)

    # Caption metrics: write all available metrics so future additions are included automatically.
    # Caption metrics: write METEOR / CIDEr.
    row["CAP_METEOR"] = cap_res.get("METEOR", "")
    row["CAP_CIDEr"] = cap_res.get("CIDEr", "")
    def f4(x):
        try:
            if x is None:
                return ""
            x = float(x)
            if x != x:  # NaN
                return ""
            return float(f"{x:.4f}")
        except Exception:
            return x  # Return non-numeric values, such as empty strings, unchanged.
    # ---- Write one appended CSV row.
    out_csv = os.path.join(metrics_dir, "auto_table.csv")
    for k, v in list(row.items()):
        if isinstance(v, (int, float, np.floating)):
            row[k] = f4(v)
    append_metrics_csv(out_csv, row)
    print(f"[Saved] metrics -> {out_csv}")




if __name__ == '__main__':
    main()
