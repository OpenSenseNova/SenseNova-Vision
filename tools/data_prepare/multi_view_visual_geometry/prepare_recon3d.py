# Copyright (c) 2026 SenseTime Group Inc. and/or its affiliates.

import argparse
import json
import os
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "OFF"
import h5py
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R

from data.prompts import mv_recon3d_prompt_templates
from data.recon3d.camera import Camera, CameraTrajMixin


def dump_camera_pose(cam: Camera, filepath: str):
    assert cam.intrinsic is not None
    fx = cam.intrinsic[0][0]
    fy = cam.intrinsic[1][1]
    cx = cam.intrinsic[0][2]
    cy = cam.intrinsic[1][2]
    q = R.from_matrix(cam.c2w_R).as_quat()
    t = cam.c2w_t
    with Path(filepath).open("w") as f:
        f.write(
            f"{fx} {fy} {cx} {cy} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]}\n"
        )


class BlendedMVSCameraTraj(CameraTrajMixin):
    def __init__(self, scene_dir: Path):
        from .toolkit_blendedmvs import load_cam

        # NOTE: frame ids in BlendedMVS are not always continguous (a small amout of frames are dropped)
        frame_ids = [
            int(x.name[: -len("_cam.txt")])
            for x in (scene_dir / "cams").iterdir()
            if str(x).endswith("_cam.txt")
        ]

        intrinsics = {}
        c2w_R = {}
        c2w_t = {}
        # depth_info: depth_min, depth_interval, depth_num, depth_max
        depth_info = {}

        for frame_id in frame_ids:
            cam_path = scene_dir / "cams" / f"{frame_id:08d}_cam.txt"
            with cam_path.open() as f:
                cam_meta = load_cam(f).astype(np.float32)
            T_c2w = np.linalg.inv(cam_meta[0])
            c2w_R[frame_id] = T_c2w[:3, :3]
            c2w_t[frame_id] = T_c2w[:3, 3]
            intrinsics[frame_id] = cam_meta[1, :3, :3]
            depth_info[frame_id] = cam_meta[1, 3]

        self.c2w_R = c2w_R  # dict{int: (3, 3)}
        self.c2w_t = c2w_t  # dict{int: (3,)}
        self.intrinsic = intrinsics  # dict{int: (3, 3)}
        self._depth_info = depth_info  # dict{int: (4,)}

    def get_mvs_min_max_depth(self, frame_id: int):
        return self._depth_info[frame_id][0], self._depth_info[frame_id][3]


class HypersimCameraTraj(CameraTrajMixin):
    def __init__(self, hypersim_cam_dir: Path):
        with (hypersim_cam_dir / "camera_keyframe_orientations.hdf5").open("rb") as bio:
            with h5py.File(bio, locking=False) as fd:
                self.c2w_R = fd["dataset"][:]  # (n, 3, 3)
        with (hypersim_cam_dir / "camera_keyframe_positions.hdf5").open("rb") as bio:
            with h5py.File(bio, locking=False) as fd:
                self.c2w_t = fd["dataset"][:]  # (n, 3)
        self.cam_dir = hypersim_cam_dir
        self.intrinsic = None
        self._cvt_coord_sys()

    def _cvt_coord_sys(self):
        """Convert from Hypersim's coordinate system to that of OpenCV's."""
        self.c2w_R[:, :, 1:] = -self.c2w_R[:, :, 1:]

    def validate_camera_frame_indices(self):
        assert self.c2w_t.shape[0] == self.c2w_R.shape[0]
        with (self.cam_dir / "camera_keyframe_frame_indices.hdf5").open("rb") as bio:
            with h5py.File(bio, locking=False) as fd:
                frame_indices = fd["dataset"][:]
        assert frame_indices.tolist() == list(range(len(frame_indices)))
        assert len(frame_indices) == self.c2w_R.shape[0]


class IRSCameraTraj(CameraTrajMixin):
    def __init__(self, irs_cam_dir: Path):
        with (irs_cam_dir / "UE_Trace.txt").open() as fd:
            data = np.loadtxt(
                fd, dtype=np.float32
            )  # (n, 10): tx ty tz qx qy qz qw, _, _, _
        # UE4 to OpenCV
        matT = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        t_scale = 1 / 100.0  # centimeter to meter

        self.c2w_R = matT @ R.from_quat(data[:, 3:7]).as_matrix() @ matT.T  # (n, 3, 3)
        self.c2w_t = (matT @ (data[:, :3, None] * t_scale))[..., 0]  # (n, 3)

        # intrinsic
        with (irs_cam_dir / "Camera.txt").open() as fd:
            rows = fd.readlines()
            assert len(rows) == 4
        intrinsic = np.loadtxt(rows[:3], dtype=np.float32)
        self.intrinsic = intrinsic
        self.baseline = float(rows[-1])


class MetaSynthCameraTraj(CameraTrajMixin):
    def __init__(self, scene_dir: Path):
        with (scene_dir / "hanwen/opencv_cameras.json").open() as f:
            cam_info = json.load(f)["frames"]
            num_frames = len(cam_info)

        Ts_c2w = []
        intrinsic_mats = []
        for frame_id, row in enumerate(cam_info):
            f_x = row["fx"]
            f_y = row["fy"]
            c_x = row["cx"]
            c_y = row["cy"]
            extrinsic = np.array(row["w2c"], dtype=np.float32)
            file_path = row["file_path"]
            assert file_path == f"renderings/{frame_id:08d}_rgba.png"

            intrinsic = np.array(
                [
                    [f_x, 0.0, c_x],
                    [0.0, f_y, c_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            Ts_c2w.append(np.linalg.inv(extrinsic))
            intrinsic_mats.append(intrinsic)

        Ts_c2w = np.asarray(Ts_c2w, dtype=np.float32)
        self.c2w_R = Ts_c2w[..., :3, :3]  # (n, 3, 3)
        self.c2w_t = Ts_c2w[..., :3, 3]  # (n, 3)
        # hint: width, height = 512, 512
        self.intrinsic = np.asarray(intrinsic_mats, dtype=np.float32)


class MVSSynthCameraTraj(CameraTrajMixin):
    def __init__(self, scene_dir: Path):
        coordT_w = np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        num_frames = len(
            [x for x in (scene_dir / "poses").iterdir() if x.suffix == ".json"]
        )

        Ts_c2w = []
        intrinsic_mats = []
        for frame_id in range(num_frames):
            with (scene_dir / "poses" / f"{frame_id:04d}.json").open() as f:
                _info = json.load(f)
                c_x = _info["c_x"]
                c_y = _info["c_y"]
                f_x = _info["f_x"]
                f_y = _info["f_y"]
                extrinsic = np.array(_info["extrinsic"], dtype=np.float32)
                extrinsic = extrinsic @ coordT_w.T
            intrinsic = np.array(
                [
                    [f_x, 0.0, c_x],
                    [0.0, f_y, c_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            Ts_c2w.append(np.linalg.inv(extrinsic))
            intrinsic_mats.append(intrinsic)

        Ts_c2w = np.asarray(Ts_c2w, dtype=np.float32)
        self.c2w_R = Ts_c2w[..., :3, :3]  # (n, 3, 3)
        self.c2w_t = Ts_c2w[..., :3, 3]  # (n, 3)
        self.intrinsic = np.asarray(intrinsic_mats, dtype=np.float32)


class OmniObject3DCameraTraj(CameraTrajMixin):
    def __init__(self, scene_dir: Path):
        with (scene_dir / "transforms.json").open() as fp:
            meta = json.load(fp)

        poses = []
        for idx, frame in enumerate(meta["frames"]):
            fname = frame["file_path"].split("/")[-1]
            assert fname == f"r_{idx}"
            pose = np.array(frame["transform_matrix"])
            pose[:, 1:3] *= -1
            poses.append(pose)
        poses = np.array(poses).astype(np.float32)

        self.c2w_R = poses[..., :3, :3]  # (n, 3, 3)
        self.c2w_t = poses[..., :3, 3]  # (n, 3)
        width = 800
        height = 800
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
        intrinsic = np.array(
            [
                [focal, 0.0, height / 2.0],
                [0.0, focal, width / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.intrinsic = intrinsic

        # gt_scale := normalized object size / real-world object size
        # unit: 1/mm (?)
        self._gt_scale = float(meta["frames"][0]["scale"])


class TartanAirCameraTraj(CameraTrajMixin):
    def __init__(self, tartanair_scene_dir: Path):
        with (tartanair_scene_dir / "pose_left.txt").open() as fd:
            data = np.loadtxt(fd, dtype=np.float32)  # (n, 7): tx ty tz qx qy qz qw
        data = data[:, [1, 2, 0, 4, 5, 3, 6]]  # coordinate system: NED to OpenCV
        self.c2w_R = R.from_quat(data[:, 3:]).as_matrix()  # (n, 3, 3)
        self.c2w_t = data[:, :3]  # (n, 3)
        fx = 320.0
        fy = 320.0
        cx = 320.0
        cy = 240.0
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        self.intrinsic = intrinsic


class TartanAirDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)

    def build_annotations(
        self,
        output_jsonl_path: Path,
        output_anno_dir: Path,
        random_seed: int = 42,
    ):
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        output_anno_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed=random_seed)
        with Path(output_jsonl_path).open("w") as jsonl_f:
            for scene_id in tqdm.tqdm(self._iter_scenes(), desc="Preparing TartanAir"):
                scene_dir = self.dataset_dir / scene_id
                camera_traj = TartanAirCameraTraj(scene_dir)
                frame_ids = range(len(camera_traj))
                output_traj_dir = output_anno_dir / "trajectory" / scene_id
                output_traj_dir.mkdir(parents=True, exist_ok=True)
                image_list = []
                depth_list = []
                camera_list = []
                w_frame_id = 0
                for r_frame_id in frame_ids:
                    color_path = scene_dir / "image_left" / f"{r_frame_id:06d}_left.png"
                    depth_path = (
                        scene_dir / "depth_left" / f"{r_frame_id:06d}_left_depth.npy"
                    )

                    camera_path = output_traj_dir / f"{w_frame_id:06d}.txt"
                    dump_camera_pose(camera_traj.get_camera(r_frame_id), camera_path)
                    image_list.append(str(color_path.relative_to(self.dataset_dir)))
                    depth_list.append(str(depth_path.relative_to(self.dataset_dir)))
                    camera_list.append(str(camera_path.relative_to(output_anno_dir)))
                    w_frame_id += 1
                prompt_samples = rng.choice(
                    mv_recon3d_prompt_templates, size=3, replace=False
                )
                record = {
                    "id": scene_id,
                    "image": image_list,
                    "depth": depth_list,
                    "trajectory": camera_list,
                    "conversations": [
                        {"from": "human", "value": str(prompt)}
                        for prompt in prompt_samples
                    ],
                }
                jsonl_f.write(json.dumps(record) + "\n")

    def _iter_scenes(self):
        for env_dir in self.dataset_dir.iterdir():
            if not env_dir.is_dir():
                continue
            for level_dir in env_dir.iterdir():
                if not level_dir.is_dir() or level_dir.name not in ["Easy", "Hard"]:
                    continue
                for traj_dir in level_dir.glob("P*"):
                    if not traj_dir.is_dir():
                        continue
                    yield f"{env_dir.name}/{level_dir.name}/{traj_dir.name}"


DATASET_CLASSES = {
    "tartanair": TartanAirDataset,
}
"""We only provide an example for preparing the TartanAir dataset.
Refer to [README.md](./README.md) for how to process other datasets.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Prepare dataset annotations for multi-view reconstruction."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CLASSES.keys()),
        help="the dataset to process",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        type=Path,
        help="path to the dataset root directory",
    )
    parser.add_argument(
        "--out-jsonl", required=True, type=Path, help="output path to save JSONL"
    )
    parser.add_argument(
        "--out-anno",
        required=True,
        type=Path,
        help="output directiry to save camera trajectories and/or depths",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    DATASET_CLASSES[args.dataset](
        dataset_dir=args.root_dir,
    ).build_annotations(
        output_jsonl_path=args.out_jsonl,
        output_anno_dir=args.out_anno,
        random_seed=args.seed,
    )
