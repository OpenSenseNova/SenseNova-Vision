import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.messages import set_default_arg, write_csv
from relpose.metric import se3_to_relative_pose_error, calculate_auc_np


def transform_jsonl_extrs(pose_info, expected_len, inverse=False):
    extri_list = []
    if isinstance(pose_info, dict):
        poses = np.tile(
            np.eye(4, dtype=np.float32),
            (len(pose_info["rotation"]), 1, 1),
        )
        poses[:, :3, :3] = Rotation.from_quat(pose_info["rotation"]).as_matrix()
        poses[:, :3, 3] = pose_info["translation"]
    else:
        poses = pose_info

    if len(poses) == expected_len - 1:
        # the first camera is a reference with identity pose
        extri_list.append(np.eye(4, dtype=np.float32))

    for pose in poses:
        extri = np.array(pose, dtype=np.float32)
        if inverse:
            extri = np.linalg.inv(extri)
        extri_list.append(extri)
    return torch.tensor(extri_list)


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):

    logger = logging.getLogger("relpose-angle")

    eval_model_name: str = hydra_cfg.eval_model_name  # see configs/evaluation/relpose-angular.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-angular.yaml
    all_predict_results: ListConfig = hydra_cfg.predict_results  # see configs/evaluation/relpose-angular.yaml
    all_data_info: DictConfig = hydra_cfg.data  # see configs/data
    pose_type: str = hydra_cfg.pose_type

    assert pose_type in ["c2w", "w2c"]
    assert len(all_predict_results) == len(all_eval_datasets)
    model_keyname = eval_model_name

    model_logger = logging.getLogger(f"relpose-angle-{model_keyname}")

    err_scene_count = 0
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        pred_result_file = all_predict_results[idx_dataset - 1]
        if not osp.isfile(pred_result_file):
            model_logger.warning(
                f"[{idx_dataset}/{len(all_eval_datasets)}] Skip {dataset_name} since file not exists: {pred_result_file}."
            )
            continue
        # 2. ready to read, and look up sampled ids from sequence name
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map = json.load(f)
        seq_id_map = {
            k: [
                (item["idx"] if isinstance(item, dict) else item)
                for item in ids
            ]
            for k, ids in seq_id_map.items()
        }

        # 3. prepare for metrics
        result_pose_dict = {}
        rError = []
        tError = []
        metric_dict: dict = {}
        model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating {dataset_name} with {model_keyname}...")
        tbar = tqdm(dataset.sequence_list, desc=f"[{dataset_name} eval]")
        with open(pred_result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                for key, value in data.items():
                    result_pose_dict[key] = value

        process_scenes = 0
        for seq_name in tbar:
            if seq_name not in result_pose_dict:
                continue  # NOTE: we only evaluate a subset of Re10K; refer to datasets/seq-id-maps/Re10K_relpose_seq-id-map_seed42.json

            ids = seq_id_map[seq_name]
            # 5. load data sample (only extrinsics are used)
            batch = dataset.get_data(sequence_name=seq_name, ids=ids)
            gt_extrs = batch["extrs"]  # <- world-to-camera

            process_scenes += 1
            pred_extrs = transform_jsonl_extrs(  # <- ensures world-to-camera
                result_pose_dict[seq_name],
                expected_len=gt_extrs.shape[0],
                inverse=(pose_type == "c2w"),
            )
            # 7. compute metrics
            if pred_extrs.shape[0] != gt_extrs.shape[0]:
                err_scene_count += 1
                continue

            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                pred_se3   = pred_extrs,
                gt_se3     = gt_extrs,
                num_frames = len(ids),
            )
            # 8. update metric for a sequence
            tbar.set_postfix_str(
                f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}"
            )

            rError.extend(rel_rangle_deg.cpu().numpy())
            tError.extend(rel_tangle_deg.cpu().numpy())

        print(f'{model_keyname} unmatch scenes: {err_scene_count}')
        rError = np.array(rError)
        tError = np.array(tError)
        # 9. arrange all intermediate results to metrics
        for threshold in dataset_info.metric_thresholds:
            metric_dict[f"Racc_{threshold}"] = np.mean(rError < threshold).item() * 100
            metric_dict[f"Tacc_{threshold}"] = np.mean(tError < threshold).item() * 100
            Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
            metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100

        model_logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

        # 9. save evaluation metrics to csv
        statistics_data = {"model": model_keyname, **metric_dict}
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, statistics_data)
        print(f'total process {process_scenes} scenes')
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating model {model_keyname} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
