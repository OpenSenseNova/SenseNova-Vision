from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


PROMPT = (
    'With the first frame as the reference frame, output the relative pose of all subsequent frames (excluding the first frame) with respect to the first frame, '
    'following the input order and adhering to the strict format below:Rotation: Represented by a quaternion in the format <quat>[x,y,z,w], '
    'enclosed in <quat> tags;Translation: Represented by a unit vector (direction) in the format <offset>[x,y,z], enclosed in <offset> tags (the vector has no absolute physical meaning, '
    'only directional information);Scale: Represented by a numerical value in the format <scale>value</scale> tags, where the value denotes the magnitude of translation '
    '(corresponding to the length of the translation unit vector);Enclose the result of each frame in <frame> tags, with no extra characters, spaces, or line breaks outside the tags.'
)
CO3DV2_CATEGORIES = (
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
)


def env_path(env_name: str, default: str) -> Path:
    """Resolve a dataset path without hard-coding local absolute paths."""
    return Path(os.environ.get(env_name, default)).expanduser()


def relative_to_root(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def clean_path(path: str | Path) -> str:
    return str(path).replace(os.sep, "/")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def format_vector(values: Sequence[float], decimals: int) -> List[float]:
    values_array = np.asarray(values, dtype=np.float64)
    values_array = np.where(np.abs(values_array) < 1e-7, 0.0, values_array)
    return np.round(values_array, decimals=decimals).tolist()


def format_token_values(values: Sequence[float], scale: float = 1000.0) -> str:
    token_str = "".join(f"<{round(float(value) * scale)}>" for value in values)
    return token_str.replace("-0", "0")


def format_scale_frame(
    quat: Sequence[float],
    offset_unit: Sequence[float],
    scale: float,
) -> str:
    quat_str = format_token_values(quat)
    offset_str = format_token_values(offset_unit)
    scale_str = f"<{round(float(scale) * 100.0)}>"
    return (
        f"<frame><quat>{quat_str}</quat><offset>{offset_str}</offset>"
        f"<scale>{scale_str}</scale></frame>\n"
    )


def count_jsonl_records(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r", encoding="utf-8") as reader:
        return sum(1 for line in reader if line.strip())


def load_txt_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def build_record(record_id: int, image_list: Sequence[str], prompt: str, answer: str) -> dict:
    return {
        "id": record_id,
        "image": [clean_path(image_path) for image_path in image_list],
        "conversations": [
            {"from": "human", "value": "<image>" * len(image_list) + prompt},
            {"from": "gpt", "value": answer},
        ],
    }


@dataclass
class FrameInfo:
    image_path: str
    extrinsic: np.ndarray


@dataclass
class CameraPoseSequence:
    dataset: str
    name: str
    frames: Dict[int, FrameInfo]


@dataclass
class CameraPoseDataset:
    """Common interface for multi-frame camera-pose JSONL generation."""

    root_dir: Path
    name: str = "base"
    samples_per_sequence: int = 10
    samples_per_sequence_divisor: Optional[int] = None
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 10
    seed: int = 42
    decimals: int = 3
    prompt: str = PROMPT
    min_num_images: int = 2
    max_scale: float = 10.0
    min_scale: float = 1e-2
    _rng: random.Random = field(init=False, repr=False)
    _np_rng: np.random.Generator = field(init=False, repr=False)
    sample_stats: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser()
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        self.reset_sample_stats()

    def reset_sample_stats(self) -> None:
        self.sample_stats = {
            "sequences_seen": 0,
            "sequences_too_short": 0,
            "sample_candidates": 0,
            "skipped_nearby": 0,
            "skipped_build_sample": 0,
            "yielded_samples": 0,
        }

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        raise NotImplementedError

    def iter_samples(self) -> Iterator[dict]:
        self.reset_sample_stats()
        for sequence in self.iter_sequences():
            self.sample_stats["sequences_seen"] += 1
            frame_ids = sorted(sequence.frames)
            if len(frame_ids) < self.min_num_images:
                self.sample_stats["sequences_too_short"] += 1
                continue

            start_candidates = frame_ids.copy()
            self._rng.shuffle(start_candidates)
            sample_count = min(
                self.get_samples_per_sequence(frame_ids), len(start_candidates)
            )
            for start_id in start_candidates[:sample_count]:
                self.sample_stats["sample_candidates"] += 1
                sequence_len = self._rng.randint(
                    self.min_sequence_len, self.max_sequence_len
                )
                ids = self.sample_nearby_ids(start_id, sequence_len, frame_ids)
                if ids is None:
                    self.sample_stats["skipped_nearby"] += 1
                    continue
                sample = self.build_sample(ids, sequence.frames)
                if sample is not None:
                    self.sample_stats["yielded_samples"] += 1
                    yield sample
                else:
                    self.sample_stats["skipped_build_sample"] += 1

    def sample_nearby_ids(
        self, start_id: int, sequence_len: int, frame_ids: Sequence[int]
    ) -> Optional[List[int]]:
        expand_range = self.expand_ratio * sequence_len
        low_bound = max(frame_ids[0], start_id - expand_range)
        high_bound = min(frame_ids[-1], start_id + expand_range)
        valid_ids = [
            frame_id
            for frame_id in frame_ids
            if low_bound <= frame_id <= high_bound and frame_id != start_id
        ]
        if len(valid_ids) < sequence_len - 1:
            return None
        sampled_ids = self._np_rng.choice(
            np.asarray(valid_ids), size=sequence_len - 1, replace=False
        )
        return [start_id] + [int(frame_id) for frame_id in sampled_ids]

    def get_samples_per_sequence(self, frame_ids: Sequence[int]) -> int:
        if self.samples_per_sequence_divisor is not None:
            return max(0, len(frame_ids) // self.samples_per_sequence_divisor)
        return self.samples_per_sequence

    def build_sample(
        self, ids: Sequence[int], frames: Dict[int, FrameInfo]
    ) -> Optional[dict]:
        return self.build_scale_sample(ids, frames)

    def build_scale_sample(
        self, ids: Sequence[int], frames: Dict[int, FrameInfo]
    ) -> Optional[dict]:
        first_pose = frames[ids[0]].extrinsic
        image_list = [frames[ids[0]].image_path]
        frame_outputs: List[str] = []
        for frame_id in ids[1:]:
            frame = frames[frame_id]

            relative_pose = np.linalg.inv(first_pose) @ frame.extrinsic
            quat = format_vector(
                np.clip(R.from_matrix(relative_pose[:3, :3]).as_quat(), -1.0, 1.0),
                self.decimals,
            )
            offset = np.asarray(relative_pose[:3, 3], dtype=np.float64)
            offset = np.where(np.abs(offset) < 1e-7, 0.0, offset)
            scale = float(np.linalg.norm(offset))
            if scale > self.max_scale or scale < self.min_scale:
                return None
            offset_unit = format_vector(np.clip(offset / scale, -1.0, 1.0), self.decimals)
            frame_outputs.append(format_scale_frame(quat, offset_unit, scale))
            image_list.append(frame.image_path)

        return {"image": image_list, "answer": "".join(frame_outputs), "prompt": self.prompt}

@dataclass
class ScannetV2CameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("SCANNETV2_ROOT", "scannet_v2_data")
    name: str = "scannetv2"
    samples_per_sequence: int = 200
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 40

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            pose_dir = scene_dir / "pose"
            if not pose_dir.exists():
                continue
            frames: Dict[int, FrameInfo] = {}
            for pose_path in sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem)):
                frame_id = int(pose_path.stem)
                image_path = f"{scene_dir.name}/color/{frame_id}.jpg"
                frames[frame_id] = FrameInfo(
                    image_path=image_path,
                    extrinsic=load_txt_matrix(pose_path),
                )
            yield CameraPoseSequence(self.name, scene_dir.name, frames)


@dataclass
class ScannetPPCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("SCANNETPP_ROOT", "scannet_raw_output")
    name: str = "scannetpp"
    samples_per_sequence: int = 200
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 40

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(self.root_dir.iterdir()):
            meta_path = scene_dir / "scene_metadata.npz"
            if not meta_path.exists():
                continue
            meta = np.load(meta_path, allow_pickle=True)
            frames: Dict[int, FrameInfo] = {}
            for image_key, trajectory in zip(meta["images"], meta["trajectories"]):
                image_name = str(image_key)
                match = re.search(r"(\d+)$", image_name)
                if match is None:
                    continue
                frame_id = int(match.group(1))
                frames[frame_id] = FrameInfo(
                    image_path=f"{scene_dir.name}/images/frame_{frame_id:06d}.png",
                    extrinsic=np.asarray(trajectory, dtype=np.float64),
                )
            yield CameraPoseSequence(self.name, scene_dir.name, frames)


@dataclass
class TartanAirCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("TARTANAIR_ROOT", "tartanair")
    name: str = "tartanair"
    samples_per_sequence: int = 400
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 5

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            for split in ("Easy", "Hard"):
                split_dir = scene_dir / split
                if not split_dir.exists():
                    continue
                for batch_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    for pose_name, camera_name in (
                        ("pose_left.txt", "left"),
                        ("pose_right.txt", "right"),
                    ):
                        pose_path = batch_dir / pose_name
                        if not pose_path.exists():
                            continue
                        poses = np.loadtxt(pose_path)
                        frames = {
                            frame_id: FrameInfo(
                                image_path=clean_path(
                                    Path(scene_dir.name)
                                    / split
                                    / batch_dir.name
                                    / f"image_{camera_name}"
                                    / f"{frame_id:06d}_{camera_name}.png"
                                ),
                                extrinsic=pose_to_tartanair_extrinsic(poses[frame_id]),
                            )
                            for frame_id in range(len(poses))
                        }
                        seq_name = f"{scene_dir.name}/{split}/{batch_dir.name}/{camera_name}"
                        yield CameraPoseSequence(self.name, seq_name, frames)


@dataclass
class WildRGBDCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("WILDRGBD_ROOT", "wildrgbd")
    name: str = "wildrgbd"
    samples_per_sequence: int = 10
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 3
    prompt: str = PROMPT

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            scenes_dir = scene_dir / "scenes"
            if not scenes_dir.exists():
                continue
            for batch_dir in sorted(path for path in scenes_dir.iterdir() if path.is_dir()):
                pose_path = batch_dir / "cam_poses.txt"
                if not pose_path.exists():
                    continue
                data = np.loadtxt(pose_path)
                if data.ndim != 2 or data.shape[1] != 17:
                    continue
                poses = data[:, 1:].reshape(-1, 4, 4)
                frames = {
                    frame_id: FrameInfo(
                        image_path=f"{scene_dir.name}/scenes/{batch_dir.name}/rgb/{frame_id:05d}.png",
                        extrinsic=np.asarray(poses[frame_id], dtype=np.float64),
                    )
                    for frame_id in range(len(poses))
                }
                yield CameraPoseSequence(self.name, f"{scene_dir.name}/{batch_dir.name}", frames)


@dataclass
class DL3DVCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("DL3DV_ROOT", "DL3DV-10K/ALL-960P")
    name: str = "dl3dv"
    prompt: str = PROMPT
    samples_per_sequence: int = 0
    samples_per_sequence_divisor: Optional[int] = 10
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    min_num_images: int = 20
    expand_ratio: int = 3

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for batch_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            for scene_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
                transform_path = scene_dir / "colmap" / "dense" / "scaled_transforms.json"
                if not transform_path.exists():
                    continue
                transforms = read_json(transform_path)
                frames: Dict[int, FrameInfo] = {}
                for item in transforms.get("frames", []):
                    frame_id = int(item["colmap_im_id"])
                    extrinsic = np.asarray(item["transform_matrix"], dtype=np.float64)
                    extrinsic[2, :] *= -1
                    extrinsic = extrinsic[np.array([1, 0, 2, 3]), :]
                    extrinsic[:3, 1:3] *= -1
                    frames[frame_id] = FrameInfo(
                        image_path=clean_path(
                            Path(batch_dir.name)
                            / scene_dir.name
                            / "colmap"
                            / "dense"
                            / "images"
                            / f"frame_{frame_id:05d}.png"
                        ),
                        extrinsic=extrinsic,
                    )
                yield CameraPoseSequence(
                    self.name, f"{batch_dir.name}/{scene_dir.name}", frames
                )


@dataclass
class DemonCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("DEMON_ROOT", "demon-mve")
    name: str = "demon"
    scene_names: Tuple[str, ...] = ("mvs_breisach", "mvs_citywall")
    samples_per_sequence: int = 0
    samples_per_sequence_divisor: Optional[int] = 10
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    min_num_images: int = 20
    expand_ratio: int = 3

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            if self.scene_names and scene_dir.name not in self.scene_names:
                continue
            for batch_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                pose_dir = batch_dir / "poses"
                if not pose_dir.exists():
                    continue
                frames = self.read_pose_json_sequence(
                    pose_dir=pose_dir,
                    image_base=Path(scene_dir.name) / batch_dir.name / "images",
                    coord_transform=None,
                )
                yield CameraPoseSequence(self.name, f"{scene_dir.name}/{batch_dir.name}", frames)

    @staticmethod
    def read_pose_json_sequence(
        pose_dir: Path,
        image_base: Path,
        coord_transform: Optional[np.ndarray],
    ) -> Dict[int, FrameInfo]:
        frames: Dict[int, FrameInfo] = {}
        for pose_path in sorted(pose_dir.glob("*.json")):
            info = read_json(pose_path)
            extrinsic = np.asarray(info["extrinsic"], dtype=np.float64)
            if coord_transform is not None:
                extrinsic = extrinsic @ coord_transform.T
            extrinsic = np.linalg.inv(extrinsic)
            frame_id = int(pose_path.stem)
            frames[frame_id] = FrameInfo(
                image_path=clean_path(image_base / f"{frame_id:04d}.png"),
                extrinsic=extrinsic,
            )
        return frames


@dataclass
class MVSSynthCameraPoseDataset(DemonCameraPoseDataset):
    root_dir: Path = env_path("MVS_SYNTH_ROOT", "MVS-Synth/GTAV_540")
    name: str = "mvs_synth"
    samples_per_sequence: int = 15
    samples_per_sequence_divisor: Optional[int] = None
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 6

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        coord_transform = np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            pose_dir = scene_dir / "poses"
            if not pose_dir.exists():
                continue
            frames = self.read_pose_json_sequence(
                pose_dir=pose_dir,
                image_base=Path(scene_dir.name) / "images",
                coord_transform=coord_transform,
            )
            yield CameraPoseSequence(self.name, scene_dir.name, frames)


@dataclass
class MegaSynthCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("MEGASYNTH_ROOT", "MegaSynth")
    name: str = "megasynth"
    samples_per_sequence: int = 5
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 4

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        split_dirs = [
            path
            for path in sorted(self.root_dir.iterdir())
            if path.is_dir() and "split" in path.name
        ]
        for split_dir in split_dirs:
            for batch_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                camera_json = batch_dir / "hanwen" / "opencv_cameras.json"
                if not camera_json.exists():
                    continue
                camera_info = read_json(camera_json).get("frames", [])
                frames: Dict[int, FrameInfo] = {}
                for frame_id, row in enumerate(camera_info):
                    frames[frame_id] = FrameInfo(
                        image_path=clean_path(
                            Path(split_dir.name)
                            / batch_dir.name
                            / "hanwen"
                            / row["file_path"]
                        ),
                        extrinsic=np.linalg.inv(np.asarray(row["w2c"], dtype=np.float64)),
                    )
                yield CameraPoseSequence(
                    self.name, f"{split_dir.name}/{batch_dir.name}", frames
                )


@dataclass
class OmniObject3DCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("OMNIOBJECT3D_ROOT", "OmniObject3D/raw/blender_renders")
    name: str = "omniobject3d"
    samples_per_sequence: int = 5
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 10

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            for batch_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                transform_path = batch_dir / "render" / "transforms.json"
                if not transform_path.exists():
                    continue
                meta = read_json(transform_path)
                frames: Dict[int, FrameInfo] = {}
                for frame in meta.get("frames", []):
                    pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
                    pose[:, 1:3] *= -1
                    gt_scale = 1.0 / (float(frame["scale"]) * 1000.0)
                    pose[:3, 3] *= gt_scale
                    frame_id = int(str(frame["file_path"]).split("_")[-1])
                    frames[frame_id] = FrameInfo(
                        image_path=clean_path(
                            Path(scene_dir.name)
                            / batch_dir.name
                            / "render"
                            / "images"
                            / f"{frame['file_path']}.png"
                        ),
                        extrinsic=pose,
                    )
                yield CameraPoseSequence(
                    self.name, f"{scene_dir.name}/{batch_dir.name}", frames
                )


@dataclass
class ObjaverseCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("OBJAVERSE_ROOT", "objaverse_v1")
    name: str = "objaverse"
    samples_per_sequence: int = 2
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 2

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        trans_mat = np.eye(4)
        trans_mat[1, 1] = -1
        trans_mat[2, 2] = -1
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            frames: Dict[int, FrameInfo] = {}
            for rt_path in sorted(scene_dir.glob("*_rt.npy")):
                match = re.match(r"(\d+)_rt\.npy", rt_path.name)
                if match is None:
                    continue
                frame_id = int(match.group(1))
                extrinsic = np.load(rt_path)
                extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
                extrinsic = np.linalg.inv(extrinsic) @ trans_mat
                frames[frame_id] = FrameInfo(
                    image_path=f"{scene_dir.name}/{frame_id:03d}_rgb0001.png",
                    extrinsic=extrinsic,
                )
            yield CameraPoseSequence(self.name, scene_dir.name, frames)


@dataclass
class IRSCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("IRS_ROOT", "IRS")
    name: str = "irs"
    samples_per_sequence: int = 200
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 10

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for split in ("Home", "Office", "Restaurant", "Store"):
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                pose_path = self.root_dir / "Auxiliary" / "CameraPos" / split / scene_dir.name / "UE_Trace.txt"
                if not pose_path.exists():
                    continue
                pose_data = np.loadtxt(pose_path)
                frames = {
                    frame_id: FrameInfo(
                        image_path=f"{split}/{scene_dir.name}/l_{frame_id}.png",
                        extrinsic=ue_pose_to_camera(pose_data[frame_id]),
                    )
                    for frame_id in range(len(pose_data))
                }
                yield CameraPoseSequence(self.name, f"{split}/{scene_dir.name}", frames)


@dataclass
class HypersimCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("HYPERSIM_ROOT", "hypersim")
    name: str = "hypersim"
    samples_per_sequence: int = 25
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 2

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for cam_dir in sorted(self.root_dir.glob("*/_detail/cam_*")):
            if not cam_dir.is_dir():
                continue
            scene_dir = cam_dir.parent.parent
            scene_name = scene_dir.name
            cam_name = cam_dir.name
            orientation_path = cam_dir / "camera_keyframe_orientations.hdf5"
            look_at_path = cam_dir / "camera_keyframe_look_at_positions.hdf5"
            metadata_path = cam_dir.parent / "metadata_scene.csv"
            if (
                not orientation_path.exists()
                or not look_at_path.exists()
                or not metadata_path.exists()
            ):
                continue

            rotations, translations = self.read_camera_keyframes(
                orientation_path, look_at_path
            )
            asset_unit_value = self.read_asset_unit_value(metadata_path)
            frames = {}
            for frame_id, (rotation, translation) in enumerate(
                zip(rotations, translations)
            ):
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :3] = rotation
                extrinsic[:3, 3] = translation * asset_unit_value
                frames[frame_id] = FrameInfo(
                    image_path=(
                        f"{scene_name}/images/scene_{cam_name}_final_preview/"
                        f"frame.{frame_id:04d}.tonemap.jpg"
                    ),
                    extrinsic=extrinsic,
                )
            yield CameraPoseSequence(self.name, f"{scene_name}/{cam_name}", frames)

    @staticmethod
    def read_camera_keyframes(
        orientation_path: Path, look_at_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Hypersim camera-pose reading requires h5py.") from exc

        with h5py.File(orientation_path, "r") as h5f:
            rotations = np.asarray(h5f["dataset"], dtype=np.float64)
        with h5py.File(look_at_path, "r") as h5f:
            translations = np.asarray(h5f["dataset"], dtype=np.float64)
        return rotations, translations

    @staticmethod
    def read_asset_unit_value(metadata_path: Path) -> float:
        with metadata_path.open("r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            first_row = next(reader)
        return float(first_row[1])


@dataclass
class SceneNetRGBDCameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("SCENENET_RGBD_ROOT", "ScenenetRGBD/train")
    name: str = "scenenet"
    samples_per_sequence: int = 5
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 10
    prompt: str = PROMPT

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        sn = self.load_scenenet_module()
        protobuf_dir = self.get_protobuf_dir()
        for protobuf_path in sorted(protobuf_dir.glob("*.pb")):
            trajectories = sn.Trajectories()
            with protobuf_path.open("rb") as protobuf_file:
                trajectories.ParseFromString(protobuf_file.read())
            for traj_index, traj in enumerate(trajectories.trajectories):
                frames: Dict[int, FrameInfo] = {}
                for view in traj.views:
                    pose = interpolate_scenenet_pose(view.shutter_open, view.shutter_close, 0.5, sn)
                    world_to_camera = scenenet_world_to_camera(pose)
                    frames[int(view.frame_num)] = FrameInfo(
                        image_path=clean_path(Path(traj.render_path) / "photo" / f"{view.frame_num}.jpg"),
                        extrinsic=np.linalg.inv(world_to_camera),
                    )
                yield CameraPoseSequence(
                    self.name, f"{protobuf_path.stem}/{traj_index}", frames
                )

    def load_scenenet_module(self) -> Any:
        protobuf_dir = self.get_protobuf_dir()
        if str(protobuf_dir) not in sys.path:
            sys.path.insert(0, str(protobuf_dir))
        try:
            import scenenet_pb2 as sn
        except ImportError as exc:
            raise ImportError(
                "SceneNet RGB-D camera pose requires scenenet_pb2. "
                "Put scenenet_pb2.py under root_dir/scenenet_protobuf or root_dir."
            ) from exc
        return sn

    def get_protobuf_dir(self) -> Path:
        candidate = self.root_dir / "scenenet_protobuf"
        if candidate.exists():
            return candidate
        return self.root_dir


@dataclass
class CO3Dv2CameraPoseDataset(CameraPoseDataset):
    root_dir: Path = env_path("CO3DV2_ANNOTATION_ROOT", "co3d_v2_annotations")
    name: str = "co3dv2"
    samples_per_sequence: int = 10
    min_sequence_len: int = 2
    max_sequence_len: int = 10
    expand_ratio: int = 10
    min_num_images: int = 50
    split_name: str = "train"
    categories: Tuple[str, ...] = CO3DV2_CATEGORIES
    translation_threshold: float = 1e5

    def iter_sequences(self) -> Iterator[CameraPoseSequence]:
        for category in self.categories:
            annotation_path = self.root_dir / f"{category}_{self.split_name}.jgz"
            if not annotation_path.exists():
                continue
            with gzip.open(annotation_path, "rb") as reader:
                annotation = json.loads(reader.read())
            for seq_name, seq_data in annotation.items():
                frames = self.prepare_sequence(seq_data)
                if frames is not None:
                    yield CameraPoseSequence(self.name, f"{category}/{seq_name}", frames)

    def prepare_sequence(self, seq_data: list) -> Optional[Dict[int, FrameInfo]]:
        if len(seq_data) < self.min_num_images:
            return None
        frames: Dict[int, FrameInfo] = {}
        for data in seq_data:
            translation = np.asarray(data["T"], dtype=np.float64)
            if np.linalg.norm(translation) > self.translation_threshold:
                return None
            frame_id = extract_frame_number(data["filepath"])
            if frame_id is None:
                continue
            frames[frame_id] = FrameInfo(
                image_path=data["filepath"],
                extrinsic=np.linalg.inv(convert_co3dv2_rt_to_opencv(data["R"], data["T"])),
            )
        if len(frames) < self.min_num_images:
            return None
        return frames

def pose_to_tartanair_extrinsic(pose: np.ndarray) -> np.ndarray:
    pose = pose[[1, 2, 0, 4, 5, 3, 6]]
    quat = pose[3:]
    translation = np.asarray(pose[:3], dtype=np.float64).reshape(3, 1)
    rotation = R.from_quat(quat).as_matrix()
    extrinsic = np.hstack((rotation, translation))
    return np.vstack((extrinsic, [0, 0, 0, 1]))


def pos_quat_to_se_matrix(quat_data: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(quat_data[3:7]).as_matrix()
    matrix[:3, 3] = quat_data[:3]
    return matrix


def ue_pose_to_camera(pose: np.ndarray) -> np.ndarray:
    quat_data = np.asarray(pose[:7], dtype=np.float64).copy()
    quat_data[:3] = quat_data[:3] / 100.0
    transform = np.array(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return transform @ pos_quat_to_se_matrix(quat_data) @ np.linalg.inv(transform)


def extract_frame_number(filepath: str) -> Optional[int]:
    match = re.search(r"frame(\d+)", os.path.basename(filepath))
    if match is None:
        return None
    return int(match.group(1))


def convert_co3dv2_rt_to_opencv(
    rot: Sequence[Sequence[float]], trans: Sequence[float]
) -> np.ndarray:
    rot_pt3d = np.asarray(rot, dtype=np.float64)
    trans_pt3d = np.asarray(trans, dtype=np.float64) / 30.0
    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot_pt3d
    extrinsic[:3, 3] = trans_pt3d
    return extrinsic


def scenenet_normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def scenenet_position_to_array(position: Any) -> np.ndarray:
    return np.array([position.x, position.y, position.z])


def scenenet_world_to_camera(view_pose: Any) -> np.ndarray:
    lookat_pose = scenenet_position_to_array(view_pose.lookat)
    camera_pose = scenenet_position_to_array(view_pose.camera)
    up = np.array([0, 1, 0])
    rotation = np.diag(np.ones(4))
    rotation[2, :3] = scenenet_normalize(lookat_pose - camera_pose)
    rotation[0, :3] = scenenet_normalize(np.cross(rotation[2, :3], up))
    rotation[1, :3] = -scenenet_normalize(np.cross(rotation[0, :3], rotation[2, :3]))
    translation = np.diag(np.ones(4))
    translation[:3, 3] = -camera_pose
    return rotation.dot(translation)


def interpolate_scenenet_pose(start_pose: Any, end_pose: Any, alpha: float, sn: Any) -> Any:
    camera_pose = alpha * scenenet_position_to_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * scenenet_position_to_array(start_pose.camera)
    lookat_pose = alpha * scenenet_position_to_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * scenenet_position_to_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose


DATASET_CLASSES = {
    "co3dv2": CO3Dv2CameraPoseDataset,
    "demon": DemonCameraPoseDataset,
    "dl3dv": DL3DVCameraPoseDataset,
    "hypersim": HypersimCameraPoseDataset,
    "irs": IRSCameraPoseDataset,
    "megasynth": MegaSynthCameraPoseDataset,
    "mvs_synth": MVSSynthCameraPoseDataset,
    "objaverse": ObjaverseCameraPoseDataset,
    "omniobject3d": OmniObject3DCameraPoseDataset,
    "scenenet": SceneNetRGBDCameraPoseDataset,
    "scannetpp": ScannetPPCameraPoseDataset,
    "scannetv2": ScannetV2CameraPoseDataset,
    "tartanair": TartanAirCameraPoseDataset,
    "wildrgbd": WildRGBDCameraPoseDataset,
}


def create_dataset(dataset_name: str, root_dir: Optional[Path] = None) -> CameraPoseDataset:
    dataset_cls = DATASET_CLASSES[dataset_name.lower()]
    if root_dir is None:
        return dataset_cls()
    return dataset_cls(root_dir=root_dir)


def apply_common_args(dataset: CameraPoseDataset, args: argparse.Namespace) -> None:
    dataset.seed = args.seed
    if args.decimals is not None:
        dataset.decimals = args.decimals
    dataset.__post_init__()

    if args.co3dv2_split is not None and hasattr(dataset, "split_name"):
        dataset.split_name = args.co3dv2_split
    if args.co3dv2_categories is not None and hasattr(dataset, "categories"):
        dataset.categories = parse_categories(args.co3dv2_categories)


def parse_categories(value: str) -> Tuple[str, ...]:
    if value == "all":
        return tuple(CO3DV2_CATEGORIES)
    categories = tuple(category.strip() for category in value.split(",") if category.strip())
    if not categories:
        raise ValueError("--co3dv2-categories must be 'all' or comma-separated names.")
    return categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified camera-pose dataset readers and JSONL generator."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_CLASSES), help="Dataset reader.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Override dataset root.")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Optional output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum records to print/write. Use -1 for all available records.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=None,
        help=(
            "Print one progress line every N written records. "
            "Defaults to 1 for limited runs and 1000 when --limit is -1. "
            "Use 0 to disable per-record progress lines."
        ),
    )
    parser.add_argument("--append", action="store_true", help="Append to --out-jsonl.")
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Start id. Defaults to 0, or existing JSONL line count with --append.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="Override numeric output decimals.",
    )
    parser.add_argument("--co3dv2-split", default=None)
    parser.add_argument("--co3dv2-categories", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_every is not None and args.print_every < 0:
        raise ValueError("--print-every must be non-negative.")

    dataset = create_dataset(args.dataset, args.root_dir)
    apply_common_args(dataset, args)
    print_every = args.print_every
    if print_every is None:
        print_every = 1000 if args.limit < 0 else 1

    writer = None
    start_id = args.start_id
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        writer = args.out_jsonl.open("a" if args.append else "w", encoding="utf-8")
        if start_id is None:
            start_id = count_jsonl_records(args.out_jsonl) if args.append else 0
    if start_id is None:
        start_id = 0

    record_count = 0
    if args.limit == 0:
        if writer is not None:
            writer.close()
        print(f"jsonl_written: 0, jsonl_path: {args.out_jsonl}" if args.out_jsonl else "samples: 0")
        return

    try:
        for sample in dataset.iter_samples():
            if args.limit >= 0 and record_count >= args.limit:
                break
            record = build_record(
                record_id=start_id + record_count,
                image_list=sample["image"],
                prompt=sample["prompt"],
                answer=sample["answer"],
            )
            if writer is not None:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            if print_every and record_count % print_every == 0:
                print(f"{record['id']} sample: {record['image']}")
            record_count += 1
    finally:
        if writer is not None:
            writer.close()

    if args.out_jsonl is not None:
        print(f"jsonl_written: {record_count}, jsonl_path: {args.out_jsonl}")
    else:
        print(f"samples: {record_count}")
    print(f"sample_stats: {json.dumps(dataset.sample_stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
