# Copyright (c) 2026 SenseNova-Vision contributors.

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Recon3DImageBatch:
    images: list[Image.Image]
    valid_masks: list[NDArray[np.bool_]]


def prepare_recon3d_images(
    images: Sequence[Image.Image],
    resize_transform: Callable[[Image.Image], Image.Image],
    *,
    pad_value: int = 127,
) -> Recon3DImageBatch:
    """Resize Recon3D inputs and center-pad them to one shared canvas."""

    if not images:
        raise ValueError("`images` must contain at least one image.")
    if not 0 <= pad_value <= 255:
        raise ValueError(f"`pad_value` must be in [0, 255], got {pad_value}.")

    resized_images = [resize_transform(image.convert("RGB")) for image in images]
    canvas_width = max(image.width for image in resized_images)
    canvas_height = max(image.height for image in resized_images)

    padded_images = []
    valid_masks = []
    for image in resized_images:
        horizontal_padding = canvas_width - image.width
        vertical_padding = canvas_height - image.height
        left = horizontal_padding // 2
        right = horizontal_padding - left
        top = vertical_padding // 2
        bottom = vertical_padding - top

        padded_images.append(
            ImageOps.expand(
                image,
                border=(left, top, right, bottom),
                fill=(pad_value, pad_value, pad_value),
            )
        )
        valid_mask = np.zeros((canvas_height, canvas_width), dtype=np.bool_)
        valid_mask[top : top + image.height, left : left + image.width] = True
        valid_masks.append(valid_mask)

    return Recon3DImageBatch(images=padded_images, valid_masks=valid_masks)
