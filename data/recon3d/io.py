# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
from pathlib import Path

import OpenEXR
from numpy.typing import NDArray

from .mixed_path import spooled_file_name


def read_exr(
        exr_path: Path,
        *,
        channel: str,
        dtype = None,
        separate_channels: bool = False,
    ) -> NDArray:
    with spooled_file_name(exr_path) as pathname:
        with OpenEXR.File(pathname, separate_channels=separate_channels) as exr_file:
            # header = exr_file.header()
            ch = exr_file.channels()[channel].pixels
    if dtype is not None:
        ch = ch.astype(dtype)
    return ch
