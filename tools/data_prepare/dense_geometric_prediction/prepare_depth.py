import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import cv2
import numpy as np

DEPTH_PROMPT_TEMPLATES = [
    "Estimate relative depth for each pixel in the image, with closer objects appearing brighter and distant objects appearing darker. Output is a grayscale image with pixel values ranging from 0-255.",
    "Predict per-pixel relative depth, visualized as a continuous grayscale image where brightness indicates proximity (brighter for closer, darker for farther). The output should be a smooth gray image with varying shades of gray, representing a continuous depth field.",
    "Generate a continuous depth map as a grayscale image with smooth variations in gray levels (0-255). Closer regions appear brighter, farther regions appear darker, creating a visually continuous depth representation. Output is an image.",
    "Predict relative depth visualized as a grayscale image with subtle, continuous shading transitions. The output should show gradual value changes throughout, not discrete color regions like segmentation masks.",
    "Output a grayscale image with continuous tone variations representing depth. Unlike segmentation discrete color blocks, this should show smooth brightness gradients across the entire image.",
    "Visualize relative depth as a continuous grayscale image. The output should show gradual value transitions without discrete boundaries, distinguishing it from segmentation blocky color regions.",
    "Estimate relative depth per pixel. Output a grayscale image where brighter pixels represent closer distances. The result should show continuous tone variations without discrete color jumps, distinguishing it from other task outputs.",
]



def env_path(env_name: str, default: str) -> Path:
    """Resolve a dataset path without hard-coding local absolute paths."""
    return Path(os.environ.get(env_name, default)).expanduser()


def relative_to_root(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def depth_to_image(
    depth: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    max_invalid_ratio: float = 0.01,
    inverse: bool = True,
) -> Optional[np.ndarray]:
    """Convert a metric depth map to an 8-bit grayscale visualization image."""
    valid_mask = (depth >= min_depth) & (depth <= max_depth) & np.isfinite(depth)
    invalid_ratio = float((~valid_mask).sum()) / depth.size
    if invalid_ratio > max_invalid_ratio:
        return None

    clipped_depth = np.clip(depth.astype(np.float32), min_depth, max_depth)
    image_depth = 1.0 / clipped_depth if inverse else clipped_depth
    valid_values = image_depth[valid_mask]
    if valid_values.size == 0:
        return None

    depth_min = valid_values.min()
    depth_max = valid_values.max()
    if depth_max - depth_min < 1e-8:
        depth_image = np.zeros_like(image_depth, dtype=np.uint8)
    else:
        depth_image = (image_depth - depth_min) / (depth_max - depth_min) * 255.0
        depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)

    depth_image[~valid_mask] = 0
    return depth_image


@dataclass(frozen=True)
class DepthSample:
    """One RGB/depth pair from a dataset."""

    dataset: str
    image_path: Path
    depth_path: Path
    output_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DepthDataset:
    """Common interface for reading original images and metric depth maps."""

    name = "base"
    min_depth = 1e-3
    max_depth = 80.0
    max_invalid_ratio = 0.01
    inverse_depth_image = True

    def iter_samples(self) -> Iterator[DepthSample]:
        raise NotImplementedError

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        raise NotImplementedError

    def read_image(self, sample: DepthSample, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        image = cv2.imread(str(sample.image_path), flags)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        return image

    def has_sample_files(self, sample: DepthSample) -> bool:
        return sample.image_path.exists() and sample.depth_path.exists()

    def read_depth_image(self, sample: DepthSample) -> Optional[np.ndarray]:
        return self.depth_to_image(self.read_depth(sample))

    def save_sample(
        self,
        sample: DepthSample,
        out_depth_dir: Optional[Path] = None,
    ) -> bool:
        if out_depth_dir is None:
            return True

        depth_image = self.read_depth_image(sample)
        if depth_image is None:
            return False
        out_depth_dir.mkdir(parents=True, exist_ok=True)
        out_depth_path = out_depth_dir / sample.output_name
        if not cv2.imwrite(str(out_depth_path), depth_image):
            raise RuntimeError(f"Failed to write depth image: {out_depth_path}")

        return True

    def depth_to_image(self, depth: np.ndarray) -> Optional[np.ndarray]:
        return depth_to_image(
            depth=depth,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            max_invalid_ratio=self.max_invalid_ratio,
            inverse=self.inverse_depth_image,
        )

    @staticmethod
    def inverse_depth_to_uint8(*args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        return depth_to_image(*args, inverse=True, **kwargs)


@dataclass
class IRSDepthDataset(DepthDataset):
    """IRS: RGB is l_*.png, depth is the matching d_*.exr G channel."""

    root_dir: Path = env_path("IRS_ROOT", "IRS")
    image_pattern: str = "*/*/l_*.png"
    depth_channel: str = "G"
    name: str = "IRS"
    min_depth: float = 1 / 80
    max_depth: float = 1000.0
    max_invalid_ratio: float = 0.01
    inverse_depth_image: bool = False

    def iter_samples(self) -> Iterator[DepthSample]:
        for image_path in sorted(self.root_dir.glob(self.image_pattern)):
            depth_path = self.get_depth_path(image_path)
            output_name = "_".join(
                [image_path.parts[-3], image_path.parts[-2], image_path.name]
            )
            yield DepthSample(
                dataset=self.name,
                image_path=image_path,
                depth_path=depth_path,
                output_name=output_name,
            )

    def get_depth_path(self, image_path: Path) -> Path:
        return Path(str(image_path).replace("/l_", "/d_")).with_suffix(".exr")

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        try:
            import Imath
            import OpenEXR
        except ImportError as exc:
            raise ImportError("IRS depth reading requires OpenEXR and Imath.") from exc

        if not sample.depth_path.exists():
            raise FileNotFoundError(f"EXR file does not exist: {sample.depth_path}")
        if not OpenEXR.isOpenExrFile(str(sample.depth_path)):
            raise ValueError(f"Not a valid EXR file: {sample.depth_path}")

        exr_file = OpenEXR.InputFile(str(sample.depth_path))
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1

        if self.depth_channel not in header["channels"]:
            available_channels = list(header["channels"].keys())
            raise ValueError(
                f"Depth channel {self.depth_channel!r} does not exist. "
                f"Available channels: {available_channels}"
            )

        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_bytes = exr_file.channel(self.depth_channel, pixel_type)
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
        depth = depth.copy()
        depth[depth <= 0] = 0
        return depth


@dataclass
class TartanAirDepthDataset(DepthDataset):
    """TartanAir: RGB image_* png, depth is matching depth_*_depth.npy."""

    root_dir: Path = env_path("TARTANAIR_ROOT", "tartanair")
    image_pattern: str = "*/*/*/*/*.png"
    name: str = "TartanAir"
    min_depth: float = 1e-3
    max_depth: float = 80.0
    max_invalid_ratio: float = 0.01
    inverse_depth_image: bool = True

    def iter_samples(self) -> Iterator[DepthSample]:
        for image_path in sorted(self.root_dir.glob(self.image_pattern)):
            relative_parts = image_path.relative_to(self.root_dir).parts
            if len(relative_parts) < 4:
                continue
            output_name = "_".join(
                [relative_parts[0], relative_parts[1], relative_parts[2], image_path.name]
            )
            yield DepthSample(
                dataset=self.name,
                image_path=image_path,
                depth_path=self.get_depth_path(image_path),
                output_name=output_name,
                metadata={"image_id": self.get_image_id(output_name)},
            )

    def get_depth_path(self, image_path: Path) -> Path:
        depth_path_str = str(image_path).replace("image_", "depth_")
        depth_path = Path(depth_path_str)
        return depth_path.with_suffix("").with_name(depth_path.stem + "_depth.npy")

    def get_image_id(self, image_name: str) -> Optional[str]:
        if "left" in image_name:
            return image_name.replace("left", "")
        if "right" in image_name:
            return image_name.replace("right", "")
        return None

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        return np.load(sample.depth_path).astype(np.float32)


@dataclass
class HypersimDepthDataset(DepthDataset):
    """Hypersim: RGB is *_final_preview/*.tonemap.jpg, depth is HDF5 dataset."""

    root_dir: Path = env_path("HYPERSIM_ROOT", "hypersim")
    scene_keyword: str = "ai_"
    image_pattern: str = "*_final_preview/*.tonemap.jpg"
    depth_pattern: str = "*_geometry_hdf5/*.depth_meters.hdf5"
    name: str = "Hypersim"
    min_depth: float = 1e-3
    max_depth: float = 80.0
    max_invalid_ratio: float = 0.01
    inverse_depth_image: bool = True

    def iter_scene_dirs(self) -> Iterable[Path]:
        for scene_dir in sorted(self.root_dir.iterdir()):
            if scene_dir.is_dir() and self.scene_keyword in scene_dir.name:
                yield scene_dir

    def iter_samples(self) -> Iterator[DepthSample]:
        for scene_dir in self.iter_scene_dirs():
            image_root = scene_dir / "images"
            for image_path in sorted(image_root.glob(self.image_pattern)):
                depth_path = self.get_depth_path(image_path)
                yield DepthSample(
                    dataset=self.name,
                    image_path=image_path,
                    depth_path=depth_path,
                    output_name=self.get_output_name(scene_dir.name, image_path),
                )

    def iter_depth_samples(self) -> Iterator[DepthSample]:
        """Yield depth files even when the paired preview image is not needed."""
        for scene_dir in self.iter_scene_dirs():
            image_root = scene_dir / "images"
            for depth_path in sorted(image_root.glob(self.depth_pattern)):
                image_path = self.get_image_path(depth_path)
                yield DepthSample(
                    dataset=self.name,
                    image_path=image_path,
                    depth_path=depth_path,
                    output_name=self.get_depth_output_name(scene_dir.name, depth_path),
                )

    def get_output_name(self, scene_name: str, image_path: Path) -> str:
        out_name = f"{scene_name}_{image_path.parent.name}_{image_path.name}"
        return out_name.replace(".tonemap.jpg", ".jpg")

    def get_depth_output_name(self, scene_name: str, depth_path: Path) -> str:
        out_name = f"{scene_name}_{depth_path.parent.name}_{depth_path.name}"
        return out_name.replace(".depth_meters.hdf5", ".npz")

    def get_depth_path(self, image_path: Path) -> Path:
        depth_parent = image_path.parent.name.replace(
            "_final_preview", "_geometry_hdf5"
        )
        depth_name = image_path.name.replace(".tonemap.jpg", ".depth_meters.hdf5")
        return image_path.parent.parent / depth_parent / depth_name

    def get_image_path(self, depth_path: Path) -> Path:
        image_parent = depth_path.parent.name.replace(
            "_geometry_hdf5", "_final_preview"
        )
        image_name = depth_path.name.replace(".depth_meters.hdf5", ".tonemap.jpg")
        return depth_path.parent.parent / image_parent / image_name

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Hypersim depth reading requires h5py.") from exc

        with h5py.File(sample.depth_path, "r") as h5f:
            return np.array(h5f["dataset"], dtype=np.float32)


@dataclass
class SceneNetRGBDDepthDataset(DepthDataset):
    """SceneNet RGB-D: protobuf views point to photo jpg and depth png in millimeters."""

    root_dir: Path = env_path("SCENENET_RGBD_ROOT", "ScenenetRGBD/train")
    protobuf_path: Path = env_path(
        "SCENENET_RGBD_PROTOBUF",
        "ScenenetRGBD/train_protobufs/scenenet_rgbd_train_16.pb",
    )
    name: str = "SceneNetRGBD"
    min_depth: float = 1e-3
    max_depth: float = 80.0
    max_invalid_ratio: float = 0.1
    inverse_depth_image: bool = True

    def iter_samples(self) -> Iterator[DepthSample]:
        trajectories = self.load_trajectories()
        for traj in trajectories.trajectories:
            render_parts = Path(traj.render_path).parts
            if len(render_parts) < 2:
                continue
            patch, scene = render_parts[0], render_parts[1]
            for view_index, view in enumerate(traj.views):
                image_path = self.root_dir / traj.render_path / "photo" / f"{view.frame_num}.jpg"
                depth_path = self.root_dir / traj.render_path / "depth" / f"{view.frame_num}.png"
                output_name = f"{patch}_{scene}_{view.frame_num}.png"
                yield DepthSample(
                    dataset=self.name,
                    image_path=image_path,
                    depth_path=depth_path,
                    output_name=output_name,
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

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("SceneNet RGB-D depth reading requires Pillow.") from exc

        pixel = np.array(Image.open(sample.depth_path))
        return pixel.astype(np.float32) * 0.001


@dataclass
class KittiDepthDataset(DepthDataset):
    """Virtual KITTI: depth png is stored as centimeter-scaled uint image."""

    root_dir: Path = env_path("KITTI_DEPTH_ROOT", "vkitti_depth")
    depth_pattern: str = "*/*/*/*/*/*.png"
    include_keyword: str = "-deg-"
    min_depth: float = 0.1
    max_depth: float = 80.0
    normalize_min_depth: float = 1.0
    depth_scale: float = 100.0
    max_invalid_ratio: float = 0.01
    inverse_depth_image: bool = True
    name: str = "KITTI"

    def iter_samples(self) -> Iterator[DepthSample]:
        for depth_path in sorted(self.root_dir.glob(self.depth_pattern)):
            if self.include_keyword and self.include_keyword not in str(depth_path):
                continue
            output_name = self.get_output_name(depth_path)
            yield DepthSample(
                dataset=self.name,
                image_path=self.get_image_path(depth_path),
                depth_path=depth_path,
                output_name=output_name,
            )

    def get_output_name(self, depth_path: Path) -> str:
        relative_parts = depth_path.relative_to(self.root_dir).parts
        if len(relative_parts) < 3:
            raise ValueError(f"Unexpected KITTI depth path: {depth_path}")
        return "_".join(
            [
                relative_parts[0],
                relative_parts[1],
                depth_path.parent.name,
                depth_path.name,
            ]
        )

    def get_image_path(self, depth_path: Path) -> Path:
        # VKITTI conversion script only needs depth pngs. Keep a best-effort RGB path.
        return Path(str(depth_path).replace("depth", "rgb")).with_suffix(".png")

    def read_depth(self, sample: DepthSample) -> np.ndarray:
        depth = cv2.imread(str(sample.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth image: {sample.depth_path}")
        return depth.astype(np.float32) / self.depth_scale

    def depth_to_image(self, depth: np.ndarray) -> Optional[np.ndarray]:
        valid_mask = (
            (depth >= self.min_depth)
            & (depth <= self.max_depth)
            & np.isfinite(depth)
        )
        invalid_ratio = float((~valid_mask).sum()) / depth.size
        if invalid_ratio > self.max_invalid_ratio:
            return None

        reciprocal_depth = np.zeros_like(depth, dtype=np.float32)
        reciprocal_depth[valid_mask] = 1.0 / depth[valid_mask]
        reciprocal_min = 1.0 / self.max_depth
        reciprocal_max = 1.0 / self.normalize_min_depth

        if reciprocal_max - reciprocal_min < 1e-8:
            normalized = np.zeros_like(reciprocal_depth, dtype=np.float32)
            normalized[valid_mask] = 128.0
        else:
            normalized = (
                (reciprocal_depth - reciprocal_min)
                / (reciprocal_max - reciprocal_min)
                * 255.0
            )

        depth_image = np.clip(normalized, 0, 255).astype(np.uint8)
        depth_image[~valid_mask] = 0
        return depth_image


DATASET_CLASSES = {
    "irs": IRSDepthDataset,
    "tartanair": TartanAirDepthDataset,
    "hypersim": HypersimDepthDataset,
    "scenenet": SceneNetRGBDDepthDataset,
    "vkitti": KittiDepthDataset,
}


def build_depth_jsonl_record(
    record_id: int,
    rgb_image: str,
    depth_image: str,
    prompt: str,
) -> dict:
    return {
        "id": record_id,
        "image": [rgb_image, depth_image],
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


def get_rgb_jsonl_path(sample: DepthSample, dataset: DepthDataset) -> str:
    try:
        return relative_to_root(sample.image_path, dataset.root_dir)
    except ValueError:
        return sample.image_path.name


def get_depth_jsonl_path(
    sample: DepthSample,
    out_depth_dir: Path,
) -> str:
    depth_path = out_depth_dir / sample.output_name
    try:
        return relative_to_root(depth_path, out_depth_dir)
    except ValueError:
        return sample.output_name


def create_dataset(dataset_name: str, root_dir: Optional[Path] = None) -> DepthDataset:
    dataset_cls = DATASET_CLASSES[dataset_name.lower()]
    if root_dir is None:
        return dataset_cls()
    return dataset_cls(root_dir=root_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified depth/original-image readers for dense geometry datasets."
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
        help="Also read image/depth arrays for the printed samples.",
    )
    parser.add_argument(
        "--check-depth-image",
        action="store_true",
        help="Also convert depth to an 8-bit image for the printed samples.",
    )
    parser.add_argument(
        "--out-depth-dir",
        type=Path,
        default=None,
        help="Optional directory to save converted 8-bit depth images.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Optional output JSONL path. Requires --out-depth-dir.",
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
    if args.out_jsonl is not None and args.out_depth_dir is None:
        raise ValueError("--out-jsonl requires --out-depth-dir so depth_image paths exist.")
    if args.print_every is not None and args.print_every < 0:
        raise ValueError("--print-every must be non-negative.")

    dataset = create_dataset(args.dataset, args.root_dir)
    save_outputs = args.out_depth_dir is not None
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

    try:
        for index, sample in enumerate(dataset.iter_samples()):
            # Negative limit means no truncation; non-negative limits count successful outputs.
            if args.limit >= 0 and output_count() >= args.limit:
                break

            if print_every and index % print_every == 0:
                print(
                    f"[{index}] {sample.dataset} output={sample.output_name} "
                    f"image={sample.image_path} depth={sample.depth_path}"
                )
            if args.check_read:
                image = dataset.read_image(sample)
                depth = dataset.read_depth(sample)
                print(
                    f"    image_shape={image.shape} depth_shape={depth.shape} "
                    f"depth_dtype={depth.dtype}"
                )
            if args.check_depth_image:
                depth_image = dataset.read_depth_image(sample)
                if depth_image is None:
                    print("    depth_image=None")
                else:
                    print(
                        f"    depth_image_shape={depth_image.shape} "
                        f"depth_image_dtype={depth_image.dtype}"
                    )
            sample_saved = True
            if save_outputs:
                try:
                    if dataset.save_sample(sample, out_depth_dir=args.out_depth_dir):
                        saved_count += 1
                    else:
                        sample_saved = False
                        skipped_count += 1
                        print(f"    skip save: invalid depth image for {sample.output_name}")
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
                depth_image = get_depth_jsonl_path(
                    sample,
                    out_depth_dir=args.out_depth_dir,
                )
                record = build_depth_jsonl_record(
                    record_id=jsonl_start_id + jsonl_count,
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    prompt=rng.choice(DEPTH_PROMPT_TEMPLATES),
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
