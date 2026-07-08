# Copyright (c) Facebook, Inc. and its affiliates.

"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""

import csv
import random
from hashlib import md5

import numpy as np

__all__ = [
    "build_palette",
    "color_for_label",
    "color_for_segment",
    "colormap",
    "load_extra_palette",
    "random_color",
    "random_colors",
    "stable_index",
    "stable_jitter",
]

# fmt: off
# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret


def load_extra_palette(csv_path):
    """Load optional RGB colors from a CSV file with R/G/B columns."""
    colors = []
    if not csv_path:
        return colors

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {str(k).strip().lower(): v for k, v in row.items()}
            if not all(k in lower for k in ("r", "g", "b")):
                continue
            try:
                colors.append(
                    (
                        int(float(lower["r"])),
                        int(float(lower["g"])),
                        int(float(lower["b"])),
                    )
                )
            except (TypeError, ValueError):
                continue
    return colors


def build_palette(extra_csv=None, rgb=True, maximum=255):
    """Return a list of RGB tuples, optionally prepending colors from CSV."""
    base = [tuple(int(x) for x in c) for c in colormap(rgb=rgb, maximum=maximum)]
    return load_extra_palette(extra_csv) + base


def stable_index(text, modulo):
    digest = md5(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % max(1, modulo)


def stable_jitter(color, seed, strength=42):
    """Apply deterministic per-instance jitter while keeping colors readable."""
    digest = md5(str(seed).encode("utf-8")).digest()
    jitter = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
    jitter = (jitter / 255.0 - 0.5) * float(strength)
    arr = np.asarray(color, dtype=np.float32) + jitter
    return tuple(int(x) for x in np.clip(arr, 0, 255))


def color_for_segment(palette, category_id, index=0, segment_id=None):
    if not palette:
        palette = build_palette()
    base = palette[int(category_id) % len(palette)]
    seed = segment_id if segment_id is not None else f"{category_id}-{index}"
    return stable_jitter(base, seed)


def color_for_label(palette, label, index=0):
    if not palette:
        palette = build_palette()
    palette_idx = stable_index(label, len(palette))
    return color_for_segment(palette, palette_idx, index, f"{label}-{index}")


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)
