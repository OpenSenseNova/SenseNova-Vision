import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging

from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mv_recon.utils import umeyama, accuracy_and_precison, completion_and_recall
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    eval_model_name: str = hydra_cfg.eval_model_name  # see configs/evaluation/mv_recon.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_predict_results: ListConfig = hydra_cfg.predict_results  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig = hydra_cfg.data  # see configs/data

    logger = logging.getLogger("mv_recon-eval")

    model_keyname = eval_model_name

    assert len(all_predict_results) == len(all_eval_datasets)

    model_logger = logging.getLogger(f"mv_recon-eval-{model_keyname}")
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):

        results_path = all_predict_results[idx_dataset - 1]
        if not osp.isdir(results_path):
            model_logger.warning(
                f"[{idx_dataset}/{len(all_eval_datasets)}] Skip {dataset_name} since directory not exists: {results_path}."
            )
            continue

        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(
                f"Unknown dataset in global data information: {dataset_name}"
            )
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.2 ready for output directory & metrics
        output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        all_data_dict = {
            "model": model_keyname,
            "Acc-mean":  0.0,
            "Comp-mean": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F-score": 0.0,
        }

        # 1.3 load pre-sampled seq-id-map
        model_logger.info(
            f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction on dataset {dataset_name}..."
        )
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        model_logger.info(f"Evaluating {dataset_name} with {model_keyname}...")
        if osp.exists(osp.join(output_root, "_all_samples.csv")):
            os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. align predicted pointcloud to ground truth (data_h, data_w)
            data_h, data_w = images.shape[-2:]
            results_path = all_predict_results[idx_dataset - 1]
            pred_pts = np.stack([
                np.load(osp.join(results_path, seq_name.replace("/", "_") + f"_frame-{id:06d}.npy"))
                for id in ids
            ], axis=0)
            # align to gt
            pred_pts = F.interpolate(
                torch.from_numpy(pred_pts).permute(0, 3, 1, 2), (data_h, data_w),
                mode="bilinear", align_corners=False, antialias=True
            ).permute(0, 2, 3, 1).cpu().numpy()
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)

            # 10. compute metrics
            dist_th = dataset_info.dist_threshold
            acc, prec = accuracy_and_precison(pcd_gt.points, pcd.points, dist_th=dist_th)
            comp, rcl = completion_and_recall(pcd_gt.points, pcd.points, dist_th=dist_th)
            f_score = 2 * prec * rcl / (prec + rcl) if prec + rcl > 0 else 0.0

            model_logger.info(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, Prec: {prec}, Recall: {rcl}, F-score: {f_score}"
                f". ICP: fitness {reg_p2p.fitness} inlier_rmse {reg_p2p.inlier_rmse}"
            )

            # 11. save metrics to csv
            write_csv(osp.join(output_root, f"_all_samples.csv"), {
                "seq":       seq_name,
                "Acc-mean":  acc,
                "Comp-mean": comp,
                "Precision": prec,
                "Recall":    rcl,
                "F-score":   f_score,
                "ICP-fitness": reg_p2p.fitness,
                "ICP-inlier_rmse": reg_p2p.inlier_rmse,
            })
            all_data_dict["Acc-mean"]  += acc
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Precision"] += prec
            all_data_dict["Recall"]    += rcl
            all_data_dict["F-score"]   += f_score

        num_samples = len(dataset)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, {"model": model_keyname, **metric_dict})

    model_logger.info(f"Finished evaluating {model_keyname} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
