#!/usr/bin/env python3
"""Compute Ref/Rea binary segmentation metrics from saved JSON predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mask import decode_mask


DATASETS = (
    ("refcoco_val", "ref/refcoco_val"),
    ("refcocop_val", "ref/refcocop_val"),
    ("refcocog_val", "ref/refcocog_val"),
    ("rea_val", "rea/val"),
    ("rea_test", "rea/test"),
)


def resolve_path(path: str | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    expanded = Path(os.path.expanduser(path))
    if expanded.is_absolute():
        return expanded
    return (repo_root / expanded).resolve()


def binary_from_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def binary_from_rle(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    return decode_mask(rle, height, width) > 0


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(image) > 0


def iou_counts(gt: np.ndarray, pred: np.ndarray) -> tuple[float, int, int, int, int]:
    pred = resize_bool(pred, gt.shape)
    inter = int(np.logical_and(gt, pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    bg_tp = int(np.logical_and(~gt, ~pred).sum())
    total = int(gt.size)
    iou = float("nan") if union == 0 else inter / union
    return iou, inter, union, bg_tp, total


def new_accumulator() -> dict:
    return {
        "ious": [],
        "inter": 0,
        "union": 0,
        "bg_tp": 0,
        "pixels": 0,
        "samples": 0,
        "missing": 0,
    }


def update_accumulator(acc: dict, gt: np.ndarray, pred: np.ndarray) -> None:
    iou, inter, union, bg_tp, total = iou_counts(gt, pred)
    acc["ious"].append(iou)
    acc["inter"] += inter
    acc["union"] += union
    acc["bg_tp"] += bg_tp
    acc["pixels"] += total
    acc["samples"] += 1


def summarize_accumulator(acc: dict) -> dict:
    valid_ious = [x for x in acc["ious"] if not math.isnan(x)]
    giou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    ciou = acc["inter"] / acc["union"] if acc["union"] else float("nan")
    pacc = (
        (acc["inter"] + acc["bg_tp"]) / acc["pixels"] if acc["pixels"] else float("nan")
    )
    return {
        "samples": int(acc["samples"]),
        "missing": int(acc["missing"]),
        "gIoU": giou * 100,
        "cIoU": ciou * 100,
        "pACC": pacc * 100,
        "intersection": int(acc["inter"]),
        "union": int(acc["union"]),
    }


def load_prediction_items(dataset_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(dataset_dir.glob("predictions_*.json")):
        with path.open("r") as f:
            items.extend(json.load(f))
    return items


def compute_dataset(dataset_name: str, dataset_dir: Path, repo_root: Path) -> dict:
    acc = new_accumulator()
    for item in load_prediction_items(dataset_dir):
        gt_path = resolve_path(item.get("gt_name"), repo_root)
        pred_rle = item.get("pred_mask")
        if gt_path is None or not gt_path.exists() or not pred_rle:
            acc["missing"] += 1
            continue
        update_accumulator(acc, binary_from_image(gt_path), binary_from_rle(pred_rle))
    return {"dataset": dataset_name, **summarize_accumulator(acc)}


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# Ref/Rea Segmentation Metrics",
        "",
        "| Dataset | Samples | Missing | gIoU | cIoU | pACC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['samples']} | {row['missing']} | "
            f"{fmt(row['gIoU'])} | {fmt(row['cIoU'])} | {fmt(row['pACC'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_auto_tables(results: list[dict], segmentation_root: Path) -> None:
    for row in results:
        dataset = row["dataset"]
        if dataset.startswith("ref"):
            out_dir = segmentation_root / "metrics" / "ref" / dataset
        elif dataset == "rea_val":
            out_dir = segmentation_root / "metrics" / "rea" / "val"
        elif dataset == "rea_test":
            out_dir = segmentation_root / "metrics" / "rea" / "test"
        else:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "auto_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task", "dataset", "gIoU", "num_images", "cIoU", "pACC"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "task": "Segmentation",
                    "dataset": dataset,
                    "gIoU": f"{row['gIoU']:.4f}",
                    "num_images": row["samples"],
                    "cIoU": f"{row['cIoU']:.4f}",
                    "pACC": f"{row['pACC']:.4f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--segmentation_root",
        required=True,
        help="Benchmark segmentation output directory, for example output/benchmark/segmentation.",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Repository root used to resolve relative gt_name paths. Default: current directory.",
    )
    parser.add_argument(
        "--out_json",
        default="",
        help="Output JSON path. Default: <segmentation_root>/metrics/ref_rea_metrics.json.",
    )
    parser.add_argument(
        "--out_md",
        default="",
        help="Output Markdown path. Default: <segmentation_root>/metrics/ref_rea_metrics.md.",
    )
    parser.add_argument(
        "--write_auto_tables",
        action="store_true",
        help="Write compatible auto_table.csv files under <segmentation_root>/metrics/ref and metrics/rea.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmentation_root = Path(args.segmentation_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    metrics_dir = segmentation_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset_name, rel_dir in DATASETS:
        dataset_dir = segmentation_root / rel_dir
        if dataset_dir.exists():
            results.append(compute_dataset(dataset_name, dataset_dir, repo_root))

    out_json = Path(args.out_json) if args.out_json else metrics_dir / "ref_rea_metrics.json"
    out_md = Path(args.out_md) if args.out_md else metrics_dir / "ref_rea_metrics.md"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results, out_md)
    if args.write_auto_tables:
        write_auto_tables(results, segmentation_root)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
