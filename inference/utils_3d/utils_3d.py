# Copyright (c) 2026 SenseNova-Vision contributors.

from typing import List

import numpy as np
import open3d as o3d
import trimesh
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial.transform import Rotation

from .geometry import (
    normals_edge,
    points_to_normals,
)


def postprocess_reconstruction(
    pts3d: List[NDArray[np.float32]],  # N, H, W, 3
    colors: List[Image.Image],
    *,
    mask_edge: bool = True,  # whether to compute an edge mask based on normals and apply it to the output
    edge_normal_threshold: float = 5.0,
    mask_sky: bool = False,  # TODO: not implemented yet
    mask_black_bg: bool = False,  # mask out black background pixels
    mask_white_bg: bool = False,  # mask out white background pixels
    voxel_downsample: bool = False,  # optional voxel downsample before filtering
    voxel_size: float = 4.0 / 512.0,
) -> trimesh.Scene:
    rgb_colors = [
        np.asarray(rgb.convert("RGB"))
        for rgb in colors
    ]

    final_masks = [
        np.ones(pts.shape[:2], dtype=np.bool_)
        for pts in pts3d
    ]

    # ---- mask normal edges ----
    if mask_edge:
        # Compute normals and normal-based edge mask
        edge_masks = []
        for pts, mask in zip(pts3d, final_masks):
            normals, normals_mask = points_to_normals(pts, mask=mask)
            normal_edges = normals_edge(
                normals, tol=edge_normal_threshold, mask=normals_mask
            )
            edge_masks.append(normal_edges)

        final_masks = [
            (m1 & m2) for m1, m2 in zip(final_masks, edge_masks)
        ]

    # ---- mask sky ----
    if mask_sky:
        pass  # TODO

    # ---- merge ----
    merged_pts3d = np.concatenate(
        [pts[mask] for pts, mask in zip(pts3d, final_masks)],
    )  # K, 3
    merged_colors = np.concatenate(
        [rgb[mask] for rgb, mask in zip(rgb_colors, final_masks)],
    )  # K, 3

    # ---- mask white / black backgrounds ----
    bg_mask = np.ones(merged_pts3d.shape[0], dtype=np.bool_)
    if mask_black_bg:
        black_bg_mask = (merged_colors >= 20).any(axis=-1)
        bg_mask = bg_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 235
        white_bg_mask = (merged_colors <= 235).any(axis=-1)
        bg_mask = bg_mask & white_bg_mask

    if mask_black_bg or mask_white_bg:
        merged_pts3d = merged_pts3d[bg_mask]
        merged_colors = merged_colors[bg_mask]

    # ---- to Open3D pointcloud ----
    pcd = o3d.t.geometry.PointCloud(merged_pts3d)
    pcd.point["colors"] = merged_colors

    # ---- optional voxel downsample before filtering ----
    if voxel_downsample:
        pcd = pcd.voxel_down_sample(voxel_size)

    # ---- statistical outlier removal ----
    pcd, _ = pcd.remove_statistical_outliers(
        nb_neighbors=20,
        std_ratio=1.5,  # -OPTION-
    )

    # ---- to trimesh Scene ----
    scene_3d = trimesh.Scene()

    point_cloud_data = trimesh.PointCloud(
        vertices=pcd.point["positions"].numpy(),
        colors=pcd.point["colors"].numpy())

    scene_3d.add_geometry(point_cloud_data)


    # --- coordinate system conversion: OpenCV to OpenGL ----
    opencv2opengl = np.identity(4)
    # flip the y and z axes
    opencv2opengl[1, 1] = -1
    opencv2opengl[2, 2] = -1
    scene_3d.apply_transform(opencv2opengl)

    return scene_3d
