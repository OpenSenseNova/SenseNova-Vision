import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import cv2
import numpy as np

NORMAL_PROMPT_TEMPLATES = [
    "Estimate surface normals and encode as an RGB image. Each channel corresponds to a direction component (X, Y, Z) with continuous value variations, creating smooth color gradients distinct from other task outputs.",
    "Generate an RGB normal map where R, G, B channels represent X, Y, Z surface directions. The output should show continuous color variations with no discrete regions, unlike segmentation results.",
    "Estimate the surface orientation at each pixel. Output an RGB image where the red, green, and blue channels continuously represent the X, Y, and Z components of the unit normal vector, creating smooth color gradients.",
    "Infer the geometric surface orientation for each point in the image. The final representation is an RGB image, with the three channels independently and continuously representing the normalized X, Y, and Z coordinates.",
    "Estimate the per-pixel surface normal, capturing local orientation. The visualization should be an RGB image where the three color channels hold continuous, independently varying values for the normal vector components.",
    "Calculate the 3D surface normal vectors for the entire scene. Visualize as an RGB image with continuous color variations in all three channels. Unlike a depth map is single grayscale channel, this uses full RGB to encode complex orientation data.",
    "Compute the orientation of surfaces throughout the image. Output an RGB image with smooth, complex color gradients across all three channels, clearly differing from a grayscale depth map or a segmentation map with uniform colored segments.",
    "Analyze the image to estimate a continuous field of surface normals. The output is an RGB map where colors correspond to normal components. This creates complex, smooth gradients-visually distinct from segmentation discrete color blocks.",
]

def env_path(env_name: str, default: str) -> Path:
    """Resolve a dataset path without hard-coding local absolute paths."""
    return Path(os.environ.get(env_name, default)).expanduser()


def relative_to_root(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def normal_to_image(normal: np.ndarray, invalid_threshold: float = 0.1) -> Optional[np.ndarray]:
    """Convert normal vectors in [-1, 1] to a uint8 BGR/RGB-like normal image."""
    normal = normal.astype(np.float32).copy()
    invalid_mask = np.linalg.norm(normal, axis=2) < invalid_threshold
    if invalid_mask.sum() > 0:
        return None
    normal_image = ((normal + 1.0) * 255.0 / 2.0).clip(0, 255).astype(np.uint8)
    return normal_image


@dataclass(frozen=True)
class NormalSample:
    """One RGB/normal pair from a dataset."""

    dataset: str
    image_path: Path
    normal_path: Path
    output_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class NormalDataset:
    """Common interface for reading original images and normal maps."""

    name = "base"

    def iter_samples(self) -> Iterator[NormalSample]:
        raise NotImplementedError

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        raise NotImplementedError

    def read_image(self, sample: NormalSample, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        image = cv2.imread(str(sample.image_path), flags)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        return image

    def read_normal_image(self, sample: NormalSample) -> Optional[np.ndarray]:
        return normal_to_image(self.read_normal(sample))

    def save_sample(self, sample: NormalSample, out_normal_dir: Optional[Path] = None) -> bool:
        if out_normal_dir is None:
            return True

        normal_image = self.read_normal_image(sample)
        if normal_image is None:
            return False

        out_normal_dir.mkdir(parents=True, exist_ok=True)
        out_normal_path = out_normal_dir / sample.output_name
        if not cv2.imwrite(str(out_normal_path), normal_image):
            raise RuntimeError(f"Failed to write normal image: {out_normal_path}")
        return True


@dataclass
class IRSNormalDataset(NormalDataset):
    """IRS: RGB is l_*.png, normal is the matching n_*.exr."""

    root_dir: Path = env_path("IRS_ROOT", "IRS")
    image_pattern: str = "*/*/l_*.png"
    name: str = "IRS"

    def iter_samples(self) -> Iterator[NormalSample]:
        for image_path in sorted(self.root_dir.glob(self.image_pattern)):
            output_name = "_".join(
                [image_path.parts[-3], image_path.parts[-2], image_path.name]
            )
            yield NormalSample(
                dataset=self.name,
                image_path=image_path,
                normal_path=self.get_normal_path(image_path),
                output_name=output_name,
            )

    def get_normal_path(self, image_path: Path) -> Path:
        return Path(str(image_path).replace("/l_", "/n_")).with_suffix(".exr")

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        try:
            import Imath
            import OpenEXR
        except ImportError as exc:
            raise ImportError("IRS normal reading requires OpenEXR and Imath.") from exc

        exr_file = OpenEXR.InputFile(str(sample.normal_path))
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

        nx = np.frombuffer(exr_file.channel("R", pixel_type), dtype=np.float32).reshape(
            (height, width)
        )
        ny = np.frombuffer(exr_file.channel("G", pixel_type), dtype=np.float32).reshape(
            (height, width)
        )
        nz = np.frombuffer(exr_file.channel("B", pixel_type), dtype=np.float32).reshape(
            (height, width)
        )
        normals = np.stack([nx, ny, nz], axis=-1)
        return 1.0 - normals * 2.0

    def read_normal_image(self, sample: NormalSample) -> Optional[np.ndarray]:
        normal_image = normal_to_image(self.read_normal(sample))
        if normal_image is None:
            return None
        return cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)


@dataclass
class HypersimNormalDataset(NormalDataset):
    """Hypersim: normal is *_geometry_preview/*.normal_cam.png with flipped z-axis."""

    root_dir: Path = env_path("HYPERSIM_NORMAL_ROOT", "hypersim_normal_data")
    scene_keyword: str = "ai_"
    normal_pattern: str = "*_geometry_preview/*.normal_cam.png"
    name: str = "Hypersim"

    def iter_scene_dirs(self) -> Iterable[Path]:
        for scene_dir in sorted(self.root_dir.iterdir()):
            if scene_dir.is_dir() and self.scene_keyword in scene_dir.name:
                yield scene_dir

    def iter_samples(self) -> Iterator[NormalSample]:
        for scene_dir in self.iter_scene_dirs():
            image_root = scene_dir / "images"
            for normal_path in sorted(image_root.glob(self.normal_pattern)):
                image_path = self.get_image_path(normal_path)
                yield NormalSample(
                    dataset=self.name,
                    image_path=image_path,
                    normal_path=normal_path,
                    output_name=self.get_output_name(scene_dir.name, normal_path),
                )

    def get_output_name(self, scene_name: str, normal_path: Path) -> str:
        out_name = f"{scene_name}_{normal_path.parent.name}_{normal_path.name}"
        return out_name.replace(".normal_cam.png", "_normal.png")

    def get_image_path(self, normal_path: Path) -> Path:
        image_parent = normal_path.parent.name.replace(
            "_geometry_preview", "_final_preview"
        )
        image_name = normal_path.name.replace(".normal_cam.png", ".tonemap.jpg")
        return normal_path.parent.parent / image_parent / image_name

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        normal_image = cv2.imread(str(sample.normal_path))
        if normal_image is None:
            raise FileNotFoundError(f"Cannot read normal image: {sample.normal_path}")
        normal = normal_image.astype(np.float32) * 2.0 / 255.0 - 1.0
        normal[:, :, 2] = -normal[:, :, 2]
        return normal


@dataclass
class InteriorVerseNormalDataset(NormalDataset):
    """InteriorVerse: RGB and normal are stored in per-frame NPZ files."""

    root_dir: Path = env_path("INTERIORVERSE_ROOT", "InteriorVerse_85")
    npz_pattern: str = "*/*/*.npz"
    name: str = "InteriorVerse"

    def iter_samples(self) -> Iterator[NormalSample]:
        for npz_path in sorted(self.root_dir.glob(self.npz_pattern)):
            output_name = f"{npz_path.parent.name}_{npz_path.stem}.png"
            yield NormalSample(
                dataset=self.name,
                image_path=npz_path,
                normal_path=npz_path,
                output_name=output_name,
            )

    def read_image(self, sample: NormalSample, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        base_name = sample.normal_path.stem
        data = np.load(sample.normal_path)
        image = data[f"{base_name}_color"]
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        return image[:, :, ::-1]

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        base_name = sample.normal_path.stem
        data = np.load(sample.normal_path)
        normal = data[f"{base_name}_normal"].astype(np.float32)
        normal[:, :, 0] = -normal[:, :, 0]
        return normal

    def read_normal_image(self, sample: NormalSample) -> Optional[np.ndarray]:
        normal_image = normal_to_image(self.read_normal(sample))
        if normal_image is None:
            return None
        return normal_image[:, :, ::-1]


class HDF5ImageNormalProcessor:
    """Adapter used by TartanAir to estimate normals from metric depth."""

    def __init__(self, cam_worlds: np.ndarray):
        self.point_cloud = cam_worlds
        self.mask: Optional[np.ndarray] = None
        self.raw_normals: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None

    def read_hdf5_image_point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        self.mask = np.logical_and(
            np.linalg.norm(self.point_cloud, axis=-1) > 1e-3,
            np.isfinite(self.point_cloud).all(axis=-1),
        )
        self.point_cloud[~self.mask] = 0.0
        return self.point_cloud, self.mask

    def compute_normals(self, window_size: int = 10) -> np.ndarray:
        self.read_hdf5_image_point_cloud()
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("TartanAir normal generation requires open3d.") from exc

        valid_y, valid_x = np.where(self.mask)
        valid_points = self.point_cloud[valid_y, valid_x]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        search_radius = window_size / 100.0
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=50
            )
        )
        pcd.orient_normals_towards_camera_location()
        normals = np.zeros_like(self.point_cloud)
        normals[valid_y, valid_x] = np.asarray(pcd.normals)
        self.raw_normals = normals
        return normals

    def smooth_normals(self, sigma_s: float = 30.0, sigma_r: float = 1.0) -> np.ndarray:
        if self.raw_normals is None:
            raise RuntimeError("Call compute_normals() before smooth_normals().")
        normals = cv2.bilateralFilter(
            self.raw_normals.astype(np.float32),
            d=5,
            sigmaColor=sigma_r,
            sigmaSpace=sigma_s,
        )
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        normals[~self.mask] = 0.0
        self.normals = normals
        return normals


def get_normal_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    cam_pts = np.stack(
        [
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        ],
        axis=-1,
    )
    processor = HDF5ImageNormalProcessor(cam_pts)
    processor.compute_normals(window_size=10)
    return processor.smooth_normals()


@dataclass
class TartanAirNormalDataset(NormalDataset):
    """TartanAir: estimate normal from matching depth_*_depth.npy files."""

    root_dir: Path = env_path("TARTANAIR_ROOT", "tartanair")
    image_pattern: str = "*/*/*/*/*.png"
    min_depth: float = 1e-3
    max_depth: float = 80.0
    max_invalid_ratio: float = 0.01
    name: str = "TartanAir"

    def iter_samples(self) -> Iterator[NormalSample]:
        for image_path in sorted(self.root_dir.glob(self.image_pattern)):
            relative_parts = image_path.relative_to(self.root_dir).parts
            if len(relative_parts) < 4:
                continue
            output_name = "_".join(
                [relative_parts[0], relative_parts[1], relative_parts[2], image_path.name]
            )
            yield NormalSample(
                dataset=self.name,
                image_path=image_path,
                normal_path=self.get_depth_path(image_path),
                output_name=output_name,
            )

    def get_depth_path(self, image_path: Path) -> Path:
        depth_path_str = str(image_path).replace("image_", "depth_")
        depth_path = Path(depth_path_str)
        return depth_path.with_suffix("").with_name(depth_path.stem + "_depth.npy")

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        depth = np.load(sample.normal_path).astype(np.float32)
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        invalid_ratio = float((~valid_mask).sum()) / depth.size
        if invalid_ratio > self.max_invalid_ratio:
            raise ValueError(f"Invalid depth ratio too high: {invalid_ratio:.4f}")
        depth = np.clip(depth, self.min_depth, self.max_depth)

        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = 320
        intrinsic[1, 1] = 320
        intrinsic[0, 2] = 320
        intrinsic[1, 2] = 240
        return -get_normal_from_depth(depth, intrinsic)

    def read_normal_image(self, sample: NormalSample) -> Optional[np.ndarray]:
        normal_image = normal_to_image(self.read_normal(sample))
        if normal_image is None:
            return None
        return normal_image[:, :, ::-1]


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-8
    return v / norm


def pixel_to_ray(
    pixel: tuple[int, int],
    vfov: float = 45,
    hfov: float = 60,
    pixel_width: int = 320,
    pixel_height: int = 240,
) -> tuple[float, float, float]:
    x, y = pixel
    x_vect = math.tan(math.radians(hfov / 2.0)) * (
        (2.0 * ((x + 0.5) / pixel_width)) - 1.0
    )
    y_vect = math.tan(math.radians(vfov / 2.0)) * (
        (2.0 * ((y + 0.5) / pixel_height)) - 1.0
    )
    return (x_vect, y_vect, 1.0)


def normalised_pixel_to_ray_array(width: int = 320, height: int = 240) -> np.ndarray:
    pixel_to_ray_array = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y, x] = normalize(
                np.array(pixel_to_ray((x, y), pixel_height=height, pixel_width=width))
            )
    return pixel_to_ray_array


def points_in_camera_coords(
    depth_map: np.ndarray, pixel_to_ray_array: np.ndarray
) -> np.ndarray:
    camera_relative_xyz = np.ones(
        (depth_map.shape[0], depth_map.shape[1], 4), dtype=np.float32
    )
    camera_relative_xyz[:, :, :3] = depth_map[:, :, np.newaxis] * pixel_to_ray_array
    return camera_relative_xyz


def scenenet_surface_normal(points: np.ndarray) -> np.ndarray:
    d = 2
    lookups = {
        0: (-d, 0),
        1: (-d, d),
        2: (0, d),
        3: (d, d),
        4: (d, 0),
        5: (d, -d),
        6: (0, -d),
        7: (-d, -d),
    }
    height, width = points.shape[:2]
    surface_normals = np.zeros((height, width, 3), dtype=np.float32)
    lookup_list = [lookups[k] for k in range(8)]
    points_3d = points[:, :, :3].astype(np.float32)

    for i in range(height):
        for j in range(width):
            min_diff = None
            normal = np.zeros(3, dtype=np.float32)
            point1 = points_3d[i, j]
            for k in range(8):
                dy1, dx1 = lookup_list[k]
                dy2, dx2 = lookup_list[(k + 2) % 8]
                i2, j2 = i + dy1, j + dx1
                i3, j3 = i + dy2, j + dx2
                if not (0 <= i2 < height and 0 <= j2 < width):
                    continue
                if not (0 <= i3 < height and 0 <= j3 < width):
                    continue
                point2 = points_3d[i2, j2]
                point3 = points_3d[i3, j3]
                diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                if min_diff is None or diff < min_diff:
                    cross_result = np.cross(point2 - point1, point3 - point1)
                    norm = np.linalg.norm(cross_result)
                    normal = (
                        np.zeros_like(cross_result)
                        if norm < 1e-8
                        else cross_result / norm
                    )
                    min_diff = diff
            surface_normals[i, j] = normal
    return surface_normals


@dataclass
class SceneNetRGBDNormalDataset(NormalDataset):
    """SceneNet RGB-D: estimate normal from depth png referenced by protobuf views."""

    root_dir: Path = env_path("SCENENET_RGBD_ROOT", "ScenenetRGBD/train")
    protobuf_path: Path = env_path(
        "SCENENET_RGBD_PROTOBUF",
        "ScenenetRGBD/train_protobufs/scenenet_rgbd_train_16.pb",
    )
    name: str = "SceneNetRGBD"

    def iter_samples(self) -> Iterator[NormalSample]:
        trajectories = self.load_trajectories()
        for traj in trajectories.trajectories:
            render_parts = Path(traj.render_path).parts
            if len(render_parts) < 2:
                continue
            patch, scene = render_parts[0], render_parts[1]
            for view_index, view in enumerate(traj.views):
                image_path = self.root_dir / traj.render_path / "photo" / f"{view.frame_num}.jpg"
                depth_path = self.root_dir / traj.render_path / "depth" / f"{view.frame_num}.png"
                yield NormalSample(
                    dataset=self.name,
                    image_path=image_path,
                    normal_path=depth_path,
                    output_name=f"{patch}_{scene}_{view.frame_num}_normal.png",
                    metadata={
                        "render_path": traj.render_path,
                        "frame_num": view.frame_num,
                        "view_index": view_index,
                    },
                )

    def load_trajectories(self) -> Any:
        try:
            import scenenet_pb2 as sn
        except ImportError as exc:
            raise ImportError(
                "SceneNet RGB-D sample iteration requires scenenet_pb2 in PYTHONPATH."
            ) from exc

        trajectories = sn.Trajectories()
        with self.protobuf_path.open("rb") as protobuf_file:
            trajectories.ParseFromString(protobuf_file.read())
        return trajectories

    def read_normal(self, sample: NormalSample) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("SceneNet RGB-D normal generation requires Pillow.") from exc

        depth_map = np.array(Image.open(sample.normal_path)).astype(np.float32) * 0.001
        depth_map[depth_map == 0.0] = 100.0
        points = points_in_camera_coords(depth_map, normalised_pixel_to_ray_array())
        return scenenet_surface_normal(points)

    def read_normal_image(self, sample: NormalSample) -> Optional[np.ndarray]:
        normal = self.read_normal(sample)
        return ((normal + 1.0) * 128.0).clip(0, 255).astype(np.uint8)


DATASET_CLASSES = {
    "irs": IRSNormalDataset,
    "hypersim": HypersimNormalDataset,
    "interiorverse": InteriorVerseNormalDataset,
    "tartanair": TartanAirNormalDataset,
    "scenenet": SceneNetRGBDNormalDataset,
}


def build_normal_jsonl_record(
    record_id: int,
    rgb_image: str,
    normal_image: str,
    prompt: str,
) -> dict:
    return {
        "id": record_id,
        "image": [rgb_image, normal_image],
        "conversations": [
            {"from": "human", "value": "<image>" + prompt},
            {"from": "gpt", "value": "<image>"},
        ],
    }


def count_jsonl_records(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r", encoding="utf-8") as reader:
        return sum(1 for line in reader if line.strip())


def get_rgb_jsonl_path(sample: NormalSample, dataset: NormalDataset) -> str:
    try:
        return relative_to_root(sample.image_path, dataset.root_dir)
    except ValueError:
        return sample.image_path.name


def get_normal_jsonl_path(sample: NormalSample, out_normal_dir: Path) -> str:
    normal_path = out_normal_dir / sample.output_name
    try:
        return relative_to_root(normal_path, out_normal_dir)
    except ValueError:
        return sample.output_name


def create_dataset(dataset_name: str, root_dir: Optional[Path] = None) -> NormalDataset:
    dataset_cls = DATASET_CLASSES[dataset_name.lower()]
    if root_dir is None:
        return dataset_cls()
    return dataset_cls(root_dir=root_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified normal/original-image readers for dense geometry datasets."
    )
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_CLASSES),
        help="Dataset reader to inspect.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Override dataset root directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Maximum number of successful outputs to produce. "
            "Use -1 to process all available samples; use 0 to process none."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=None,
        help=(
            "Print one progress line every N visited samples. "
            "Defaults to 1 for limited runs and 1000 when --limit is -1. "
            "Use 0 to disable per-sample progress lines."
        ),
    )
    parser.add_argument(
        "--check-read",
        action="store_true",
        help="Also read image/normal arrays for the printed samples.",
    )
    parser.add_argument(
        "--check-normal-image",
        action="store_true",
        help="Also convert/read normal image arrays for the printed samples.",
    )
    parser.add_argument(
        "--out-normal-dir",
        type=Path,
        default=None,
        help="Optional directory to save converted normal images.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Optional output JSONL path. Requires --out-normal-dir.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --out-jsonl instead of overwriting it.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Start id for JSONL records. Defaults to 0 or existing line count with --append.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection. Use -1 for non-deterministic prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_jsonl is not None and args.out_normal_dir is None:
        raise ValueError("--out-jsonl requires --out-normal-dir so normal_image paths exist.")
    if args.print_every is not None and args.print_every < 0:
        raise ValueError("--print-every must be non-negative.")

    dataset = create_dataset(args.dataset, args.root_dir)
    save_outputs = args.out_normal_dir is not None
    write_jsonl = args.out_jsonl is not None
    print_every = args.print_every
    if print_every is None:
        print_every = 1000 if args.limit < 0 else 1
    rng = random.Random(None if args.seed == -1 else args.seed)
    jsonl_writer = None
    jsonl_start_id = args.start_id
    if write_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_mode = "a" if args.append else "w"
        jsonl_writer = args.out_jsonl.open(jsonl_mode, encoding="utf-8")
        if jsonl_start_id is None:
            jsonl_start_id = count_jsonl_records(args.out_jsonl) if args.append else 0

    saved_count = 0
    skipped_count = 0
    jsonl_count = 0
    inspected_count = 0

    def output_count() -> int:
        if write_jsonl:
            return jsonl_count
        if save_outputs:
            return saved_count
        return inspected_count

    if args.limit == 0:
        if jsonl_writer is not None:
            jsonl_writer.close()
        if not save_outputs and not write_jsonl:
            print("samples: 0")
        if save_outputs:
            print("saved: 0, skipped: 0")
        if write_jsonl:
            print(f"jsonl_written: 0, jsonl_path: {args.out_jsonl}")
        return

    try:
        for index, sample in enumerate(dataset.iter_samples()):
            # Negative limit means no truncation; non-negative limits count successful outputs.
            if args.limit >= 0 and output_count() >= args.limit:
                break

            if print_every and index % print_every == 0:
                print(
                    f"[{index}] {sample.dataset} output={sample.output_name} "
                    f"image={sample.image_path} normal={sample.normal_path}"
                )
            if args.check_read:
                image = dataset.read_image(sample)
                normal = dataset.read_normal(sample)
                print(
                    f"    image_shape={image.shape} normal_shape={normal.shape} "
                    f"normal_dtype={normal.dtype}"
                )
            if args.check_normal_image:
                normal_image = dataset.read_normal_image(sample)
                if normal_image is None:
                    print("    normal_image=None")
                else:
                    print(
                        f"    normal_image_shape={normal_image.shape} "
                        f"normal_image_dtype={normal_image.dtype}"
                    )
            sample_saved = True
            if save_outputs:
                try:
                    if dataset.save_sample(sample, out_normal_dir=args.out_normal_dir):
                        saved_count += 1
                    else:
                        sample_saved = False
                        skipped_count += 1
                        print(f"    skip save: invalid normal image for {sample.output_name}")
                except Exception as exc:
                    sample_saved = False
                    skipped_count += 1
                    print(f"    skip save: {sample.output_name}: {exc}")

            if write_jsonl and sample_saved:
                if not sample.image_path.exists():
                    skipped_count += 1
                    print(f"    skip jsonl: missing rgb image {sample.image_path}")
                    continue
                rgb_image = get_rgb_jsonl_path(sample, dataset)
                normal_image = get_normal_jsonl_path(sample, args.out_normal_dir)
                record = build_normal_jsonl_record(
                    record_id=jsonl_start_id + jsonl_count,
                    rgb_image=rgb_image,
                    normal_image=normal_image,
                    prompt=rng.choice(NORMAL_PROMPT_TEMPLATES),
                )
                jsonl_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_count += 1
            elif not save_outputs:
                inspected_count += 1
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()

    if not save_outputs and not write_jsonl:
        print(f"samples: {inspected_count}")
    if save_outputs:
        print(f"saved: {saved_count}, skipped: {skipped_count}")
    if write_jsonl:
        print(f"jsonl_written: {jsonl_count}, jsonl_path: {args.out_jsonl}")


if __name__ == "__main__":
    main()
