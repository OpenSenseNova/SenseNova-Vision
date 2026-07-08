# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
from pathlib import Path
from typing import Any, Dict, Hashable, Protocol, Sequence, Tuple, TypeAlias

import numpy.random
from numpy.typing import NDArray
from pandas import Index, DataFrame
from PIL import Image


AnyPath: TypeAlias = Path
_FrameSequence: TypeAlias = Sequence[Dict[str, Any]]
_RGBOrPathSequence: TypeAlias = Sequence[str | AnyPath | Image.Image]


class SceneProtocol(Protocol):
    """Protocol defining the interface for a recon3d scene."""

    def get_color_path(self, frame_id: int) -> AnyPath:
        """Return the file path to the color image for the given frame ID."""
        ...

    def get_color_image(self, frame_id: int) -> Image.Image:
        """Return the loaded color image for the given frame ID."""
        ...

    def read_seq_and_transform(
        self,
        frame_ids: Sequence[int],
        *args,
        **kwargs,
    ) -> _FrameSequence:
        """Read a sampled sequence and express the world points in the first frame's camera coordinate system."""
        ...


class SeqSamplerProtocol(Protocol):
    """Protocol defining the interface for a recon3d sequence sampler."""

    use_single_pointmap_split: bool
    """Controls how multi-view pointmaps are represented in latent space.
    True: Use a single <pointmap_seq> split.
    False: Use multiple <pointmap> splits with next-token prediction.
    """

    scene_keys: Index
    """Index of the dataframe composed of unique identifiers for each scene."""

    metadata_df: DataFrame
    """Dataframe of metadata records."""

    rng: numpy.random.Generator
    """RNG for data sampling."""

    def get_scene(self, scene_id: Hashable) -> SceneProtocol:
        """Return the scene corresponding to the given scene identifier."""
        ...

    def choose_frame_seq(self, frame_ids: NDArray) -> Sequence[int] | None:
        """Randomly select a sequence of frame indices, or return None if not possible."""
        ...

    def augment_sequence(
        self,
        rgb_or_path_seq: _RGBOrPathSequence,
        frame_seq: _FrameSequence,
    ) -> Tuple[_RGBOrPathSequence, _FrameSequence]:
        """Apply augmentation to a sampled sequence."""
        ...

    def normalize_sequence(self, frame_seq: _FrameSequence) -> _FrameSequence:
        """Normalize the sampled sequence."""
        ...
