# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


class Camera:
    def __init__(
        self,
        c2w_R: NDArray,
        c2w_t: NDArray,
        intrinsic: NDArray | None,
        *,
        image_size: Tuple[int, int] | None = None,
    ):
        self.c2w_R = c2w_R  # (3, 3)
        self.c2w_t = c2w_t  # (3,)
        self.intrinsic = intrinsic  # (3, 3)
        self.image_size = image_size  # width, height

    def rot90(self, clockwise: bool) -> "Camera":
        intrinsic, image_size = None, None
        if self.intrinsic is not None:
            if self.image_size is None:
                raise ValueError("image_size should be provided when rotating intrinsic matrix by 90 degrees.")
            intrinsic = adjust_intrinsic_matrix_rot90(
                self.intrinsic,
                image_width=self.image_size[0],
                image_height=self.image_size[1],
                clockwise=clockwise,
            )
            image_size = self.image_size[::-1]
        c2w_R, c2w_t = adjust_extrinsic_matrix_rot90(self.c2w_R, self.c2w_t, clockwise=clockwise)
        return Camera(c2w_R, c2w_t, intrinsic=intrinsic, image_size=image_size)

    def transform_points_c2w(self, cam_points: NDArray):
        # expects: cam_points.shape == (..., 3)
        return (self.c2w_R @ cam_points[..., None])[..., 0] + self.c2w_t

    def transform_points_w2c(self, world_points: NDArray):
        # expects: world_points.shape == (..., 3)
        matR = self.c2w_R.T
        t = -matR @ self.c2w_t
        return (matR @ world_points[..., None])[..., 0] + t

    def transform_R_t(self, in_R: NDArray, in_t: NDArray):
        # expects in_R.shape == (..., 3, 3) and in_t.shape == (..., 3)
        matR = self.c2w_R
        t = self.c2w_t
        out_R = matR.T @ in_R
        out_t = matR.T @ (in_t[..., None] - t[..., None])[..., 0]
        return out_R, out_t


class CameraTrajMixin:
    # typing hints
    c2w_R: NDArray | Dict[int, NDArray]  # (n, 3, 3)
    c2w_t: NDArray | Dict[int, NDArray]  # (n, 3)
    intrinsic: NDArray | Dict[int, NDArray] | None  # (3, 3) or (n, 3, 3)

    def get_camera(self, frame_id: int):
        if self.intrinsic is None:
            intrinsic = None
        elif isinstance(self.intrinsic, np.ndarray) and self.intrinsic.ndim == 2:
            intrinsic = self.intrinsic
        else:
            intrinsic = self.intrinsic[frame_id]
        return Camera(self.c2w_R[frame_id], self.c2w_t[frame_id], intrinsic)

    def __len__(self):
        return len(self.c2w_t)


def unproject_z_depth(z_depth: NDArray, intrinsic: NDArray) -> NDArray:
    """Unproject z-depth to camera points."""
    assert intrinsic.shape == (3, 3)
    height, width = z_depth.shape
    y, x = np.indices((height, width))
    pixel_coords_h = np.stack((x, y, np.ones_like(x)), axis=-1)
    inv_K = np.linalg.inv(intrinsic)
    ray_dir = (inv_K @ pixel_coords_h[..., None])[..., 0]
    points_3d = ray_dir * z_depth[..., None]
    return points_3d


def unproject_ray_depth(ray_depth: NDArray, intrinsic: NDArray) -> NDArray:
    """Unproject ray-depth to camera points."""
    assert intrinsic.shape == (3, 3)
    height, width = ray_depth.shape
    y, x = np.indices((height, width))
    pixel_coords_h = np.stack((x, y, np.ones_like(x)), axis=-1)
    inv_K = np.linalg.inv(intrinsic)
    ray_dir = (inv_K @ pixel_coords_h[..., None])[..., 0]
    points_3d = ray_dir / np.linalg.norm(ray_dir, ord=2, axis=-1, keepdims=True) * ray_depth[..., None]
    return points_3d


def adjust_extrinsic_matrix_rot90(c2w_R_opencv, c2w_t_opencv, clockwise):
    """
    Adjusts the extrinsic matrix (R, t) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        c2w_R_opencv (np.ndarray):
            Camera-to-world ratation matrix (3x3) in OpenCV convention.
        c2w_t_opencv (np.ndarray):
            Camera-to-world translation (3) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.
    """
    if clockwise:
        R_rotation = np.array([
            [0,  1, 0],
            [-1, 0, 0],
            [0,  0, 1]
        ], dtype=c2w_R_opencv.dtype)
    else:
        R_rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=c2w_R_opencv.dtype)

    new_R = R_rotation @ c2w_R_opencv
    return new_R, c2w_t_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3, dtype=intri_opencv.dtype)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv
