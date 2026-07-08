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
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import cv2
import argparse
import logging
import numpy as np
import os
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import DatasetMode, get_dataset
from src.util import metric
from src.util.metric import MetricTracker, compute_cosine_error

eval_metrics = [
    "mean_angular_error",
    "median_angular_error",
    "sub5_error",
    "sub7_5_error",
    "sub11_25_error",
    "sub22_5_error",
    "sub30_error",
]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Surface Normals Estimation : Metrics Evaluation"
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
        "--use_mask", action="store_true", help="Evaluate only in the masked region."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(args.dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    # import pdb;pdb.set_trace()
    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Results Dictionary --------------------
    results = {}

    # -------------------- Per-sample metrics file --------------------
    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        rgb_name = data["rgb_relative_path"][0]

        normals_gt = data["normals"].to(device)  # [1,3,H,W]


        extension = rgb_name.split('.')[-1]
        # extension ='npy'
        # Load predictions
      
        if cfg_data.name == 'scannet_normals':
            rgb_basename = rgb_name.split('/')[-2] + '-' + rgb_name.split('/')[-1]
            rgb_basename_without_extension = os.path.splitext(rgb_basename)[0]
            pred_basename = rgb_basename_without_extension + f'.{extension}'
        elif cfg_data.name == 'diode_normals':
            rgb_basename = rgb_name.split('/')[-3] + '-' + rgb_name.split('/')[-1]
            rgb_basename_without_extension = os.path.splitext(rgb_basename)[0]
            pred_basename = rgb_basename_without_extension + f'.{extension}'
        else:
            rgb_basename = os.path.basename(rgb_name)
            # scene_dir = os.path.join(args.prediction_dir, os.path.dirname(rgb_name))
            rgb_basename_without_extension = os.path.splitext(rgb_basename)[0]
            pred_basename = rgb_basename_without_extension + f'.{extension}'
        # pred_path = os.path.join(scene_dir, pred_basename)
        pred_path = os.path.join(args.prediction_dir, pred_basename)

        if not os.path.exists(pred_path):
            logging.warning(f"Can't find prediction: {pred_path}")
            continue


        # norm_pred_numpy = np.load((pred_path))[:,:,::-1]
        # import pdb;pdb.set_trace()
        if pred_path.endswith('.npy'):
            norm_pred_numpy = np.load((pred_path))
            # norm_pred_numpy = norm_pred_numpy*2 -1
        else:
            norm_img = cv2.imread(pred_path)
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
            norm_pred_numpy = norm_img/255
            # import pdb;pdb.set_trace()
            if cfg_data.name == 'nyu_normals':
                image_dir = os.environ.get(
                    "NORMAL_NYU_IMAGE_DIR",
                    os.path.join(args.base_data_dir, "nyuv2/test"),
                )
                image_ori = cv2.imread(os.path.join(image_dir,rgb_basename))
                h,w,_ = image_ori.shape
                norm_pred_numpy = cv2.resize(norm_pred_numpy,(w,h),interpolation=cv2.INTER_NEAREST)
            elif cfg_data.name == 'scannet_normals':
                image_dir = os.environ.get(
                    "NORMAL_SCANNET_IMAGE_DIR",
                    os.path.join(args.base_data_dir, "scannet"),
                )
                image_ori = cv2.imread(os.path.join(image_dir,rgb_basename.split('-')[0],rgb_basename.split('-')[1]))
                h,w,_ = image_ori.shape
                norm_pred_numpy = cv2.resize(norm_pred_numpy,(w,h),interpolation=cv2.INTER_NEAREST)
            elif cfg_data.name == 'ibims_normals':
                image_dir = os.environ.get(
                    "NORMAL_IBIMS_IMAGE_DIR",
                    os.path.join(args.base_data_dir, "ibims/ibims"),
                )
                image_ori = cv2.imread(os.path.join(image_dir,rgb_basename))
                h,w,_ = image_ori.shape
                norm_pred_numpy = cv2.resize(norm_pred_numpy,(w,h),interpolation=cv2.INTER_NEAREST)
        normals_pred = (
            torch.from_numpy(norm_pred_numpy.astype(np.float32))
            .permute(2,0,1).unsqueeze(0)
            .to(device)
        )  # [1,3,H,W]


        # import pdb;pdb.set_trace()
        normals_pred = normals_pred*2 - 1
        normals_pred[:,0,:,:] = -normals_pred[:,0,:,:]
        cosine_error = compute_cosine_error(normals_pred, normals_gt, masked=True)
        sample_metric = []

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(cosine_error).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(rgb_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {args.prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    metrics_filename += ".txt"

    _save_to = os.path.join(args.output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
    # -------------------- Save summary metrics to CSV (auto_table.csv) ---------------
    if hasattr(metric_tracker, "result") and callable(metric_tracker.result):
        summary = metric_tracker.result()
    elif hasattr(metric_tracker, "results"):
        summary = metric_tracker.results
    elif hasattr(metric_tracker, "avg"):
        summary = metric_tracker.avg
    elif hasattr(metric_tracker, "get_results") and callable(metric_tracker.get_results):
        summary = metric_tracker.get_results()
    else:
        raise RuntimeError("MetricTracker has no known method/field to fetch summary results.")

    row = {
        "task": "normal",
        "dataset": str(cfg_data.get("name", dataset.disp_name)),
        "dataset_disp_name": str(dataset.disp_name),
        "prediction_dir": os.path.abspath(args.prediction_dir),
        "dataset_config": os.path.abspath(args.dataset_config),
        "base_data_dir": os.path.abspath(args.base_data_dir),
        "use_mask": int(bool(args.use_mask)),
        "num_samples": int(len(dataset)),
    }

    # Write tabulated summary metrics from the txt file into the same row
    for k, v in summary.items():
        try:
            row[k] = float(v)
        except Exception:
            row[k] = v

    auto_csv = os.path.join(args.output_dir, "auto_table.csv")
    new_df = pd.DataFrame([row])

    if os.path.exists(auto_csv):
        old_df = pd.read_csv(auto_csv)
        # Align columns by filling missing columns in both the old and new tables.
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
        metric_cols = list(summary.keys())
        cols = [c for c in meta_cols if c not in metric_cols] + metric_cols
        out_df = new_df.reindex(columns=cols)

    out_df.to_csv(auto_csv, index=False)
    logging.info(f"Summary metrics saved to {auto_csv}")
