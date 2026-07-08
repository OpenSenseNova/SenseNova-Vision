# Copyright (c) 2026 SenseNova-Vision contributors.

import re
from typing import Optional

import numpy as np


__all__ = ["resolve_pose_string"]


def _extract_tag_values(input_str: str, tag: str, expected_count: int) -> np.ndarray:
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, got {type(input_str)}")

    contents = re.findall(
        rf"<{tag}>(.*?)</{tag}>",
        input_str,
        flags=re.DOTALL,
    )

    if not contents:
        return np.array([])

    values = []
    for idx, content in enumerate(contents):
        numbers = re.findall(r"-?\d+", content)

        if len(numbers) != expected_count:
            print(
                f"Warning: invalid {tag} content #{idx + 1}: "
                f"expected {expected_count} values, got {len(numbers)}. Skipped."
            )
            continue

        values.append([int(x) / 1000 for x in numbers])

    return np.array(values) if values else np.array([])

def extract_scale_to_array(input_str: str) -> np.ndarray:
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, got {type(input_str)}")

    contents = re.findall(
        r"<scale>(.*?)</scale>",
        input_str,
        flags=re.DOTALL,
    )

    scales = []
    for idx, content in enumerate(contents):
        numbers = re.findall(r"-?\d+", content)

        if not numbers:
            print(f"Warning: invalid scale content #{idx + 1}: {content}. Skipped.")
            continue

        scales.append(int(numbers[0]))

    return np.array(scales) if scales else np.array([])


def resolve_pose_string(pose_str: str) -> Optional[dict]:
    """
    Parse a pose string and extract quaternion, offset, and scale data.

    Args:
        pose_str (str): String containing <quat>, <offset>, and <scale> tags.

    Returns:
        Optional[dict]:
            {
                "rotation": list[list[float]],   # N x 4 quaternion values
                "translation": list[list[float]],  # N x 3 translation
            }
            Returns None if validation fails or no valid data remains.
    """

    quat_list = _extract_tag_values(pose_str, "quat", 4).tolist()
    offset_array = _extract_tag_values(pose_str, "offset", 3)
    scale_array = extract_scale_to_array(pose_str)

    if offset_array.size == 0 or scale_array.size == 0:
        print("Warning: offset or scale is empty. Skipping sample.")
        return None

    if offset_array.shape[0] != scale_array.shape[0]:
        print(
            f"Warning: offset/scale count mismatch: "
            f"offset={offset_array.shape[0]}, scale={scale_array.shape[0]}. "
            "Skipping sample."
        )
        return None

    try:
        offset_array = np.asarray(offset_array, dtype=np.float32)
        scale_array = np.asarray(scale_array, dtype=np.float32)
    except (TypeError, ValueError):
        print("Warning: offset or scale contains invalid values. Skipping sample.")
        return None

    valid_mask = np.isfinite(offset_array).all(axis=1) & np.isfinite(scale_array)

    if not np.all(valid_mask):
        invalid_count = int((~valid_mask).sum())
        print(f"Warning: {invalid_count} invalid offset/scale records skipped.")
        offset_array = offset_array[valid_mask]
        scale_array = scale_array[valid_mask]

    if offset_array.size == 0 or scale_array.size == 0:
        print("Warning: no valid offset/scale values remain. Skipping sample.")
        return None

    translation_array = offset_array * scale_array[:, None] / 100.0

    return {
        "rotation": quat_list,
        "translation": translation_array.tolist(),
        # "scale": scale_array.tolist(),
    }
