# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import logging

import pyarrow.fs as pf
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            data_paths_per_dir = [
                file for file in files if file.endswith(".parquet")
            ]
        else:
            files = os.listdir(data_dir)
            data_paths_per_dir = [
                os.path.join(data_dir, name)
                for name in files
                if name.endswith(".parquet")
            ]
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


# NOTE: cumtomize this function for your cluster
def get_hdfs_host():
    return "hdfs://xxx"


# NOTE: cumtomize this function for your cluster
def get_hdfs_block_size():
    return 134217728


# NOTE: cumtomize this function for your cluster
def get_hdfs_extra_conf():
    return None


def init_arrow_pf_fs(parquet_file_path):
    if parquet_file_path.startswith("hdfs://"):
        fs = pf.HadoopFileSystem(
            host=get_hdfs_host(),
            port=0,
            buffer_size=get_hdfs_block_size(),
            extra_conf=get_hdfs_extra_conf(),
        )
    else:
        fs = pf.LocalFileSystem()
    return fs


def hdfs_ls_cmd(dir):
    result = subprocess.run(["hdfs", "dfs", "ls", dir], capture_output=True, text=True).stdout
    return ['hdfs://' + i.split('hdfs://')[-1].strip() for i in result.split('\n') if 'hdfs://' in i]
