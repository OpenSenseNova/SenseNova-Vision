# Copyright (c) 2026 SenseNova-Vision contributors.

from .camera_pose_parser import resolve_pose_string

__all__ = ["postprocess_reconstruction", "resolve_pose_string"]


def __getattr__(name: str):
    """Load optional 3D postprocessing dependencies only when requested."""
    if name == "postprocess_reconstruction":
        from .utils_3d import postprocess_reconstruction

        return postprocess_reconstruction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
