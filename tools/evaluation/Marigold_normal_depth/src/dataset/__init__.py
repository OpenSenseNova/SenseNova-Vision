# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import os
from typing import Union

from .base_depth_dataset import (
    BaseDepthDataset,
    get_pred_name,  # noqa: F401
    DatasetMode,
)  # noqa: F401
from .base_normals_dataset import BaseNormalsDataset  # noqa: F401
from .diode_dataset import DIODEDepthDataset
from .eth3d_dataset import ETH3DDepthDataset
from .ibims_dataset import IBimsNormalsDataset
from .kitti_dataset import KITTIDepthDataset
from .nyu_dataset import NYUDepthDataset, NYUNormalsDataset
from .scannet_dataset import ScanNetDepthDataset, ScanNetNormalsDataset

dataset_name_class_dict = {
    "nyu_depth": NYUDepthDataset,
    "kitti_depth": KITTIDepthDataset,
    "eth3d_depth": ETH3DDepthDataset,
    "diode_depth": DIODEDepthDataset,
    "scannet_depth": ScanNetDepthDataset,
    "ibims_normals": IBimsNormalsDataset,
    "nyu_normals": NYUNormalsDataset,
    "scannet_normals": ScanNetNormalsDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> Union[
    BaseDepthDataset,
    BaseNormalsDataset,
]:
    if cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
