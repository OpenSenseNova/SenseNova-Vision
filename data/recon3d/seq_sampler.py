# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

import math
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .camera import Camera
from .mixed_path import MixedPath
from .types import SeqSamplerProtocol
from ..data_utils import pil_img2rgb


def solve_unit_cube_skybox(
        sky_points,  # (k, 3)
        c2w_t,  # (3,)
        *,
        clip=True,
    ):
    if sky_points.size == 0:
        return sky_points
    a = sky_points - c2w_t
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = (np.sign(a) - c2w_t) / a
    scale = np.min(roots, axis=-1, where=(roots > 0), initial=np.inf, keepdims=True)
    scale[~np.isfinite(scale)] = 0.0
    skybox_pts = scale * a + c2w_t
    if clip:
        mask = np.abs(skybox_pts) > 1.0
        skybox_pts[mask] = np.sign(skybox_pts[mask])
    return skybox_pts


def crop_center_and_optional_rot90(
    rgb: Image.Image,
    frame: dict,
    *,
    rot90: bool,
    clockwise: bool,
    aspect_ratio: float,
):
    """Crop center and optional rotote 90 degrees on image plane.

    The center-crop is performed in the intersection area of the images before
    and after rotation (even when `rot90` is false).
    See implementation for details.
    """
    frame = deepcopy(frame)
    if rot90:
        rgb = rgb.transpose(Image.Transpose.ROTATE_270 if clockwise else Image.Transpose.ROTATE_90)
        np_rot90_k = (-1 if clockwise else 1)
        frame["world_points"] = np.rot90(frame["world_points"], k=np_rot90_k)
        if frame["valid_mask"] is not None:
            frame["valid_mask"] = np.rot90(frame["valid_mask"], k=np_rot90_k)
        ## If "intrinsic" in frame: also need to pass-in intrinsic to adjust the intrinsic matrix
        cam = Camera(frame["c2w_R"], frame["c2w_t"], intrinsic=None).rot90(clockwise=clockwise)
        frame["c2w_R"] = cam.c2w_R
        frame["c2w_t"] = cam.c2w_t

    width, height = rgb.size
    short_side = min(width, height)
    out_long_side = short_side
    if aspect_ratio > 1.0:
        out_w = out_long_side
        out_h = round(out_long_side / aspect_ratio)
    else:
        out_w = round(out_long_side * aspect_ratio)
        out_h = out_long_side

    start_x = math.floor((width - out_w) * 0.5)
    start_y = math.floor((height - out_h) * 0.5)
    end_x = start_x + out_w
    end_y = start_y + out_h

    rgb = rgb.crop(box=(start_x, start_y, end_x, end_y))

    frame["world_points"] = frame["world_points"][start_y:end_y, start_x:end_x]
    if frame["valid_mask"] is not None:
        frame["valid_mask"] = frame["valid_mask"][start_y:end_y, start_x:end_x]
    ## If "intrinsic" in frame: also need to shift the principal point
    # intrinsic[1, 2] = intrinsic[1, 2] - start_x
    # intrinsic[0, 2] = intrinsic[0, 2] - start_y

    return rgb, frame


class SeqSamplerBase(SeqSamplerProtocol):
    def __init__(
        self,
        *,
        # sampling
        seq_num_frames_range: list = [2, 4],  # inclusive
        expand_ratio: int = 4,
        # augmentation
        aug_orientation_ratio: float = -1.0,
        # normalization
        seq_norm_mean: str | float | List[float] = 0.0,
        seq_norm_std: str | float | List[float] = "max_l2",
        # misc
        rng: np.random.Generator = None,
        eps: float = 1.0e-6,
    ):
        if isinstance(seq_norm_mean, str):
            assert seq_norm_mean in ["mean_xyz", "minmaxcenter_xyz"]
        if isinstance(seq_norm_std, str):
            assert seq_norm_std in ["max_l2", "maxabs_scalar"]

        self.min_num_frames = seq_num_frames_range[0]
        self.max_num_frames = seq_num_frames_range[1]
        self.expand_ratio = expand_ratio
        self.aug_orientation_ratio = aug_orientation_ratio
        self.seq_norm_mean = seq_norm_mean
        self.seq_norm_std = seq_norm_std
        self.rng = rng or np.random.default_rng(2025)
        self.eps = eps

    def choose_frame_seq(self, frame_ids: NDArray):
        seq_len = self.rng.integers(self.min_num_frames, self.max_num_frames + 1)
        if seq_len > len(frame_ids):
            return None
        first = self.rng.choice(frame_ids)
        left = first - self.expand_ratio * seq_len
        right = first + self.expand_ratio * seq_len
        neighbor_ids = frame_ids[(frame_ids >= left) & (frame_ids <= right) & (frame_ids != first)]
        if len(neighbor_ids) < seq_len - 1:
            return None
        arr2 = self.rng.choice(neighbor_ids, size=seq_len - 1, replace=False)
        return [first] + arr2.tolist()

    def _aug_image_orientation_with_crop(
        self,
        rgb_or_path_seq: List[str | Path | Image.Image],
        frame_seq: List[dict],
    ):
        """Apply random orientation (rot90) on image plane to RGBs, world_points,
        valid_masks and extrinsics.

        Images and maps will be cropped to keep identical shapes and close
        intrinsics within the sample sequence.
        """
        n_frames = len(frame_seq)
        rot90_augs = self.rng.random(size=(n_frames,)) < self.aug_orientation_ratio
        if not rot90_augs.any():
            return rgb_or_path_seq, frame_seq
        out_rgb_seq = []
        out_frame_seq = []
        for rot90, rgb_or_path, frame in zip(rot90_augs, rgb_or_path_seq, frame_seq):
            if isinstance(rgb_or_path, (str, Path)):
                with MixedPath(rgb_or_path).open("rb") as fd:
                    rgb = pil_img2rgb(Image.open(fd))
            else:
                rgb = pil_img2rgb(rgb_or_path)

            clockwise = (self.rng.random() < 0.5) if rot90 else False
            # TODO: Here we choose to use input's aspect ratio; alternatively we can introduce an augmentation on aspect ratio.
            aspect_ratio = rgb.width / rgb.height

            rgb, frame = crop_center_and_optional_rot90(
                rgb, frame, rot90=rot90, clockwise=clockwise, aspect_ratio=aspect_ratio)

            out_rgb_seq.append(rgb)
            out_frame_seq.append(frame)

        # re-transform the sequence to cam0's viewpoint if the cam0 is rotated
        if rot90_augs[0]:
            # use cam0 as the new world origin
            cam0 = Camera(out_frame_seq[0]["c2w_R"], out_frame_seq[0]["c2w_t"], intrinsic=None)
            for frame in out_frame_seq:
                frame["world_points"] = cam0.transform_points_w2c(frame["world_points"])
                frame["c2w_R"], frame["c2w_t"] = cam0.transform_R_t(frame["c2w_R"], frame["c2w_t"])

        return out_rgb_seq, out_frame_seq

    def augment_sequence(
        self,
        rgb_or_path_seq: List[str | Path | Image.Image],
        frame_seq: List[dict],
    ):
        return self._aug_image_orientation_with_crop(rgb_or_path_seq, frame_seq)

    def normalize_sequence(self, frame_seq: List[dict]):
        if self.seq_norm_mean == "mean_xyz":
            mean_vecs = []
            for frame in frame_seq:
                valid_mask = frame["valid_mask"]  # could be None
                mean_vecs.append(frame["world_points"][valid_mask].reshape(-1, 3).mean(axis=0))
            norm_mean = np.mean(mean_vecs, axis=0, dtype=np.float32)
        elif self.seq_norm_mean == "minmaxcenter_xyz":
            mean_vecs = []
            for frame in frame_seq:
                valid_mask = frame["valid_mask"]  # could be None
                _wpts = frame["world_points"][valid_mask].reshape(-1, 3)
                mean_vecs.append((_wpts.max(axis=0) + _wpts.min(axis=0)) * 0.5)
            norm_mean = np.mean(mean_vecs, axis=0, dtype=np.float32)
        else:
            assert isinstance(self.seq_norm_mean, (float, list))
            norm_mean = np.asarray(self.seq_norm_mean, dtype=np.float32)

        # subtract `norm_mean` before calculating `norm_std`
        for frame in frame_seq:
            frame["world_points"] = frame["world_points"] - norm_mean
            frame["c2w_t"] = frame["c2w_t"] - norm_mean

        if self.seq_norm_std == "max_l2":
            frame0_cam_center = frame_seq[0]["c2w_t"]
            dist_from_cam0 = []
            for frame in frame_seq:
                valid_mask = frame["valid_mask"]  # could be None
                dist_from_cam0.append(
                    np.linalg.norm(frame["world_points"][valid_mask] - frame0_cam_center, ord=2, axis=-1).reshape(-1))
            dist_from_cam0 = np.concatenate(dist_from_cam0)
            kth = int(len(dist_from_cam0) * 0.005 + 0.5)
            norm_std = np.partition(dist_from_cam0, -kth)[-kth]
        elif self.seq_norm_std == "maxabs_scalar":
            max_vals = []
            for frame in frame_seq:
                valid_mask = frame["valid_mask"]  # could be None
                max_vals.append(np.abs(frame["world_points"][valid_mask]).max())
            norm_std = max(max_vals)
        else:
            assert isinstance(self.seq_norm_std, (float, list))
            norm_std = np.asarray(self.seq_norm_std, dtype=np.float32)

        for frame in frame_seq:
            valid_mask = frame["valid_mask"]  # could be None
            frame["world_points"] = frame["world_points"] / (norm_std + self.eps)
            if valid_mask is not None:
                # NOTE: move invalid points along camera ray to skybox
                frame["world_points"][~valid_mask] = solve_unit_cube_skybox(
                    frame["world_points"][~valid_mask], frame["c2w_t"], clip=True)
            frame["c2w_t"] = frame["c2w_t"] / (norm_std + self.eps)

        return frame_seq
