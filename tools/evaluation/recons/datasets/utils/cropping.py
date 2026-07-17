import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from typing import Tuple

try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC


# GT_PREPROCESSING_FOR_TASK = "VGGT"
GT_PREPROCESSING_FOR_TASK = "SenseNova-Vision"


class MaxLongEdgeMinShortEdgeResize:
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def __call__(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


def resize_image(image: Image.Image, output_resolution: Tuple[int, int]) -> Image.Image:
    max_resize_scale = max(output_resolution[0] / image.size[0], output_resolution[1] / image.size[1])
    return image.resize(output_resolution, resample=lanczos if max_resize_scale < 1 else bicubic)


def resize_image_depth_and_intrinsic(
    image: Image.Image,
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
    pixel_center: bool = True,
) ->  Tuple[Image.Image, np.ndarray, np.ndarray]:
    if len(depth_map.shape) != 2:
        raise ValueError(f"Depth map must be a 2D array, but found depthmap.shape = {depth_map.shape}")

    if GT_PREPROCESSING_FOR_TASK == "VGGT":
        input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (H, W) -> (W, H)
        # output_resolution = np.array([output_width, round(input_resolution[1] * (output_width / input_resolution[0]))])
        output_resolution = np.array([output_width, round(input_resolution[1] * (output_width / input_resolution[0]) / 14) * 14])

        image = resize_image(image, tuple(output_resolution))
    elif GT_PREPROCESSING_FOR_TASK == "SenseNova-Vision":
        input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (H, W) -> (W, H)
        image = MaxLongEdgeMinShortEdgeResize(512, 256, 16)(image)
        output_resolution = np.array(image.size)
    else:
        raise ValueError(f"unknown GT_PREPROCESSING_FOR_TASK: {GT_PREPROCESSING_FOR_TASK}")

    depth_map = cv2.resize(
        depth_map,
        output_resolution,
        interpolation = cv2.INTER_NEAREST,
    )

    intrinsic = np.copy(intrinsic)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    resize_scale = np.max(output_resolution / input_resolution)
    intrinsic[:2, :] = intrinsic[:2, :] * resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.size == depth_map.shape[::-1], f"Image size {image.size} does not match depth map shape {depth_map.shape[::-1]}"
    return image, depth_map, intrinsic
