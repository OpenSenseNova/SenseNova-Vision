import argparse
import json
import os

import fastevaluate as fe
import numpy as np


def safe_mean(values):
    vals = [v for v in values if v > 0]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def format_table(rows, headers):
    # Compute column widths
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO/LVIS predictions (FastEval)."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/coco/instances_val2017.json",
        help="Path to GT json (COCO/LVIS)",
    )
    parser.add_argument(
        "--pred_tsv",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/eval_results/box_eval/COCO/fast_eval.tsv",
        help="Path to predictions (TSV/JSON)",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="auto",
        choices=["auto", "coco", "lvis"],
        help="Evaluation type: auto (detect from filename), coco, or lvis",
    )
    return parser.parse_args()


def detect_eval_type(gt_path, pred_path, eval_type):
    """Detect evaluation type from file paths if auto is specified"""
    if eval_type != "auto":
        return eval_type

    # Check for LVIS indicators in paths
    lvis_indicators = ["lvis", "LVIS"]
    for indicator in lvis_indicators:
        if indicator in gt_path or indicator in pred_path:
            return "lvis"

    # Default to COCO
    return "coco"


def main():
    args = parse_args()

    # Detect evaluation type
    eval_type = detect_eval_type(args.gt, args.pred_tsv, args.eval_type)
    print(f"🔍 Detected evaluation type: {eval_type.upper()}")

    res = fe.evaluate(args.gt, args.pred_tsv, 0, 0, eval_type)

    # Basic means
    avg_precision = safe_mean(res.get("precision", []))
    avg_recall = safe_mean(res.get("recall", []))
    avg_f1 = f1(avg_precision, avg_recall)

    # IoU=0.50
    precision50 = safe_mean(res.get("precision50", []))
    recall50 = safe_mean(res.get("recall50", []))
    f1_50 = f1(precision50, recall50)

    # IoU=0.95
    precision95 = safe_mean(res.get("precision95", []))
    recall95 = safe_mean(res.get("recall95", []))
    f1_95 = f1(precision95, recall95)

    headers = ["Metric", "Value"]
    rows = [
        ["Avg Precision", f"{avg_precision:.4f}"],
        ["Avg Recall", f"{avg_recall:.4f}"],
        ["Avg F1", f"{avg_f1:.4f}"],
        ["Precision@0.50", f"{precision50:.4f}"],
        ["Recall@0.50", f"{recall50:.4f}"],
        ["F1@0.50", f"{f1_50:.4f}"],
        ["Precision@0.95", f"{precision95:.4f}"],
        ["Recall@0.95", f"{recall95:.4f}"],
        ["F1@0.95", f"{f1_95:.4f}"],
    ]

    print(format_table(rows, headers))
    # ---- save metrics json next to pred file ----
    pred_path = os.path.abspath(args.pred_tsv)
    pred_dir = os.path.dirname(pred_path)
    base_name = os.path.splitext(os.path.basename(pred_path))[0]

    out_dir = os.path.join(os.path.dirname(pred_dir) if os.path.basename(pred_dir)=="output" else pred_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    out_jsonl = os.path.join(out_dir, base_name + "_metrics.jsonl")

    row = {
        "task_dataset": base_name,
        "iou05_precision": precision50,
        "iou05_recall": recall50,
        "iou05_f1": f1_50,
        "iou095_precision": precision95,
        "iou095_recall": recall95,
        "iou095_f1": f1_95,
        "miou_precision": avg_precision,
        "miou_recall": avg_recall,
        "miou_f1": avg_f1,
        "mae": None,
        "inst_follow": None,
        "keypoint_ap05": None,
        "halluc_acc": None,
        "gui_acc": None,
        "wrong_rej": None,
        "samples": None,
    }

    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"💾 Saved metrics to: {out_jsonl}")




if __name__ == "__main__":
    main()
