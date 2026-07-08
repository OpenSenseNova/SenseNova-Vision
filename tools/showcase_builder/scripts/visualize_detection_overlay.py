import argparse
import glob
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.colormap import build_palette  # noqa: E402
from utils import image_path_for_item, load_jsonl, safe_stem  # noqa: E402
from utils.visualize import (  # noqa: E402
    VisualizationConfig,
    visualize_concat_col,
    visualize_detection,
)


def parse_root_args(values):
    roots = []
    for value in values or []:
        for part in str(value).replace(",", " ").split():
            part = part.strip().strip("[]").strip("'\"")
            if part:
                roots.append(part)
    return roots


def prediction_jsonls_from_roots(roots, data_path=None):
    paths = []
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isabs(root) and data_path:
            root = os.path.join(data_path, root)
        if os.path.isdir(root):
            paths.extend(sorted(glob.glob(os.path.join(root, "*.jsonl"))))
        else:
            paths.extend(sorted(glob.glob(root)))
    return [path for path in paths if path.endswith(".jsonl")]


def output_dir_for_file(jsonl_path, args, multi_file=False):
    if args.vis_dir:
        base = args.vis_dir
    else:
        parent = os.path.dirname(jsonl_path)
        base = os.path.join(parent, f"vis_concat{args.concat_col}")
    if multi_file:
        base = os.path.join(base, safe_stem(Path(jsonl_path).stem))
    os.makedirs(base, exist_ok=True)
    return base


def prediction_value(item):
    for key in ("extracted_predictions", "predictions", "prediction", "raw_response"):
        if key in item and item[key] is not None:
            return item[key]
    return {}


def gt_value(item):
    for key in ("gt", "ground_truth"):
        if key in item and item[key] is not None:
            return item[key]
    return None


def visualize_jsonl(jsonl_path, args, palette, multi_file=False):
    records = load_jsonl(jsonl_path)
    if args.limit is not None:
        records = records[: args.limit]
    out_dir = output_dir_for_file(jsonl_path, args, multi_file=multi_file)
    config = VisualizationConfig(
        alpha=args.alpha,
        font_size=args.font_size,
        draw_width=args.draw_width,
        point_radius=args.point_radius,
        keypoint_radius=args.keypoint_radius,
        max_labels=args.max_labels,
        max_label_chars=args.max_label_chars,
        prefer_unwrapped_labels=args.prefer_unwrapped_labels,
    )

    written = 0
    skipped = 0
    for row_idx, item in enumerate(tqdm(records, desc=Path(jsonl_path).name)):
        image_path = image_path_for_item(item, image_root=args.image_root, data_path=args.data_path)
        if not image_path or not os.path.exists(image_path):
            if args.skip_missing:
                skipped += 1
                continue
            raise FileNotFoundError(f"Image not found for row {row_idx}: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = item.get("question") or item.get("prompt") or ""
        task_name = item.get("task_name") or Path(jsonl_path).stem

        pred_panel = visualize_detection(
            image,
            prediction_value(item),
            task_name=task_name,
            prompt=prompt,
            palette=palette,
            config=config,
            include_prompt=True,
        )

        gt_panel = None
        gt = gt_value(item)
        if gt is not None:
            gt_panel = visualize_detection(
                image,
                gt,
                task_name=task_name,
                prompt=prompt,
                palette=palette,
                config=config,
                include_prompt=True,
            )

        final = visualize_concat_col(
            image,
            pred_panel,
            concat_col=args.concat_col,
            gt=gt_panel,
            source_label="Image",
            gt_label="GT",
            pred_label="Prediction",
        )
        dataset = safe_stem(item.get("dataset_name") or Path(jsonl_path).stem)
        stem = safe_stem(Path(image_path).stem)
        out_name = f"{row_idx:06d}_{dataset}_{stem}.png"
        final.save(os.path.join(out_dir, out_name))
        written += 1
    return {"jsonl": jsonl_path, "out_dir": out_dir, "written": written, "skipped": skipped}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SenseNova-Vision detection-family JSONL outputs.")
    parser.add_argument("--prediction_root", nargs="+", required=True, help="JSONL file(s), glob(s), or directories.")
    parser.add_argument("--image_root", default=None, help="Base directory for relative image paths.")
    parser.add_argument("--data_path", default=None, help="Base directory for relative prediction/image paths.")
    parser.add_argument("--vis_dir", default=None, help="Output directory.")
    parser.add_argument("--concat_col", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--color_csv", default=None)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--font_size", type=int, default=0)
    parser.add_argument("--draw_width", type=int, default=0)
    parser.add_argument("--point_radius", type=int, default=0)
    parser.add_argument("--keypoint_radius", type=int, default=0)
    parser.add_argument("--max_labels", type=int, default=120)
    parser.add_argument("--max_label_chars", type=int, default=96)
    parser.add_argument("--prefer_unwrapped_labels", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    roots = parse_root_args(args.prediction_root)
    jsonls = prediction_jsonls_from_roots(roots, data_path=args.data_path)
    if not jsonls:
        raise FileNotFoundError(f"No JSONL files found under: {roots}")
    palette = build_palette(args.color_csv)
    multi_file = len(jsonls) > 1
    summaries = [visualize_jsonl(path, args, palette, multi_file=multi_file) for path in jsonls]
    for summary in summaries:
        print(
            f"{summary['jsonl']} -> {summary['out_dir']} "
            f"written={summary['written']} skipped={summary['skipped']}"
        )


if __name__ == "__main__":
    main()
