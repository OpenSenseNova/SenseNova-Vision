# Copyright (c) 2026 SenseTime Group Inc. and/or its affiliates.

import re

import numpy as np


def load_cam(file, interval_scale=1, flags_max_d=128):
    """read camera txt file"""
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = flags_max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        raise ValueError("invalid camera txt")

    return cam


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode("UTF-8").rstrip()

    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")
    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode("UTF-8").rstrip())
    if scale < 0:  # little-endian
        data_type = "<f"
    else:
        data_type = ">f"  # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0)
    return data
