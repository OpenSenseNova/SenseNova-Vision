# Copyright 2026 SenseTime Group Inc. and/or its affiliates.

from contextlib import contextmanager
from pathlib import Path

__all__ = ["MixedPath", "spooled_file_name"]


class MixedPath:
    def __new__(cls, *args, **kwargs) -> Path:
        if kwargs:
            raise TypeError("MixedPath only accepts positional path parts")
        return Path(*args)


@contextmanager
def spooled_file_name(mixed_path: Path):
    yield str(mixed_path)
