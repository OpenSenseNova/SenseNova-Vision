# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import argparse
import logging
import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import (
    DatasetMode,
    get_dataset,
)
from src.util import metric
from src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from src.util.metric import MetricTracker

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Metrics Evaluation"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory with predictions obtained from inference.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base path to the datasets.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    cfg_data = OmegaConf.load(args.dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    for data in tqdm(dataloader, desc="Evaluating"):
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]
        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        depth_raw_ts = depth_raw_ts.to(device)
        valid_mask_ts = valid_mask_ts.to(device)

        extension = 'png'
        pred_basename = rgb_name[:-4] + '.' + extension

        pred_path = os.path.join(args.prediction_dir, pred_basename)
        if not os.path.exists(pred_path):
            logging.warning(f"Can't find prediction: {pred_path}")
            continue

        if pred_path.endswith('.npz') or pred_path.endswith('.npy'):
            if pred_path.endswith('.npz'):
                depth_pred_raw = np.load(pred_path)['depth'].astype(np.float32)
            elif pred_path.endswith('.npy'):
                depth_pred_raw = np.load(pred_path)
            if cfg_data.name == 'kitti_depth':
                KB_CROP_HEIGHT = 352
                KB_CROP_WIDTH = 1216
                height, width = depth_pred_raw.shape[-2:]
                top_margin = int(height - KB_CROP_HEIGHT)
                left_margin = int((width - KB_CROP_WIDTH) / 2)
                depth_pred = depth_pred_raw[
                    top_margin : top_margin + KB_CROP_HEIGHT,
                    left_margin : left_margin + KB_CROP_WIDTH,
                ]

            else:
                depth_pred = depth_pred_raw
        elif pred_path.endswith('.png') or pred_path.endswith('.jpg') or pred_path.endswith('.JPG'):
            depth_pred_raw = cv2.imread(pred_path)

            if cfg_data.name == 'kitti_depth':
                KB_CROP_HEIGHT = 352
                KB_CROP_WIDTH = 1216
                depth_pred_raw = cv2.resize(depth_pred_raw,(KB_CROP_WIDTH,KB_CROP_HEIGHT),interpolation=cv2.INTER_NEAREST)
                depth_pred = np.mean(depth_pred_raw,axis=2)/255
            elif cfg_data.name in {'nyu_depth', 'eth3d_depth', 'scannet_depth', 'diode_depth'}:
                raw_h,raw_w, = depth_raw_ts.shape
                depth_pred_raw = cv2.resize(depth_pred_raw,(raw_w,raw_h),interpolation=cv2.INTER_NEAREST)
                depth_pred = np.mean(depth_pred_raw,axis=2)/255

        if "least_square" == args.alignment:
            depth_pred, _, _ = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=args.alignment_max_res,
            )
        elif "least_square_disparity" == args.alignment:
            gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_raw, return_mask=True
            )
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, _, _ = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=args.alignment_max_res,
            )
            disparity_pred = np.clip(
                disparity_pred, a_min=1e-3, a_max=None
            )
            depth_pred = disparity2depth(disparity_pred)

        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        with open(per_sample_filename, "a+") as f:
            f.write(pred_basename + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    eval_text = f"Evaluation metrics:\n\
    of predictions: {args.prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"
    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    if args.alignment:
        metrics_filename += f"-{args.alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(args.output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
    summary = metric_tracker.result()

    row = {
        "dataset": str(cfg_data.get("name", dataset.disp_name)),
        "dataset_disp_name": str(dataset.disp_name),
        "prediction_dir": os.path.abspath(args.prediction_dir),
        "dataset_config": os.path.abspath(args.dataset_config),
        "base_data_dir": os.path.abspath(args.base_data_dir),
        "alignment": str(args.alignment) if args.alignment is not None else "",
        "alignment_max_res": int(args.alignment_max_res) if args.alignment_max_res is not None else "",
        "min_depth": float(dataset.min_depth),
        "max_depth": float(dataset.max_depth),
        "num_samples": int(len(dataset)),
    }

    for k, v in summary.items():
        try:
            row[k] = float(v)
        except Exception:
            row[k] = v

    auto_csv = os.path.join(args.output_dir, "auto_table.csv")
    new_df = pd.DataFrame([row])

    if os.path.exists(auto_csv):
        old_df = pd.read_csv(auto_csv)
        for c in old_df.columns:
            if c not in new_df.columns:
                new_df[c] = ""
        for c in new_df.columns:
            if c not in old_df.columns:
                old_df[c] = ""
        new_df = new_df[old_df.columns]
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        meta_cols = list(row.keys())
        metric_cols = [k for k in summary.keys()]
        cols = [c for c in meta_cols if c not in metric_cols] + metric_cols
        out_df = new_df.reindex(columns=cols)

    out_df.to_csv(auto_csv, index=False)
    logging.info(f"Summary metrics saved to {auto_csv}")
