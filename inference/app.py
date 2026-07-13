import argparse
import base64
import importlib
import json
import math
import os
import re
import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from packaging.version import Version

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
import sys

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _patch_huggingface_hub_hffolder() -> None:
    """Restore the legacy HfFolder symbol for older gradio/transformers builds."""
    try:
        import huggingface_hub
    except ImportError:
        return

    if hasattr(huggingface_hub, "HfFolder"):
        return

    class HfFolder:
        @staticmethod
        def get_token() -> Optional[str]:
            get_token = getattr(huggingface_hub, "get_token", None)
            if callable(get_token):
                return get_token()
            return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        @staticmethod
        def save_token(token: str) -> None:
            login = getattr(huggingface_hub, "login", None)
            if callable(login):
                login(token=token, add_to_git_credential=False)
                return
            raise RuntimeError("huggingface_hub.login is unavailable; cannot save token.")

        @staticmethod
        def delete_token() -> None:
            logout = getattr(huggingface_hub, "logout", None)
            if callable(logout):
                logout()

    huggingface_hub.HfFolder = HfFolder


_patch_huggingface_hub_hffolder()

import gradio as gr
from PIL import Image, ImageOps


def _patch_gradio_client_bool_schema() -> None:
    """Allow older gradio_client to parse JSON Schema boolean shortcuts."""
    try:
        import gradio_client.utils as client_utils
    except ImportError:
        return

    original = getattr(client_utils, "_json_schema_to_python_type", None)
    if not callable(original) or getattr(original, "_sensenova_bool_schema_patch", False):
        return

    def patched_json_schema_to_python_type(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return original(schema, defs)

    patched_json_schema_to_python_type._sensenova_bool_schema_patch = True
    client_utils._json_schema_to_python_type = patched_json_schema_to_python_type


_patch_gradio_client_bool_schema()


APP_TITLE = "SenseNova-Vision"
LOG_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "log.webp")
DEFAULT_MODEL_PATH = "sensenova/SenseNova-Vision-7B-MoT"
DEFAULT_OUTPUT_DIR = "examples/output/demo/"
TASK_ORDER = [
    "raw_query",
    "depth",
    "normal",
    "binary_seg",
    "pan_seg",
    "gcg_seg",
    "bbox_detection",
    "point_detection",
    "keypoint",
    "ocr",
    "recon3d",
    "camera_pose",
]
TASK_HELP: Dict[str, str] = {
    "raw_query": "raw prompt, query needed",
    "depth": "relative depth estimation, no further query needed",
    "normal": "surface normal estimation, no further query needed",
    "binary_seg": "binary segmentation, query reference needed",
    "pan_seg": "panoptic segmentation, query categories needed",
    "gcg_seg": "Grounded Conversation Generation Segmentation, no further query needed",
    "bbox_detection": "bounding-box detection, query categories needed",
    "point_detection": "point detection, query categories needed",
    "keypoint": "keypoint detection, query categories needed",
    "ocr": "word-level OCR, no further query needed",
    "recon3d": "multi-view 3D reconstruction, multi-view images needed, no further query needed",
    "camera_pose": "relative camera pose estimation, multi-view images needed, no further query needed",
}
TASK_TO_MODE: Dict[str, str] = {
    "raw_query": "dense_perception",
    "depth": "dense_perception",
    "normal": "dense_perception",
    "binary_seg": "dense_perception",
    "pan_seg": "caption_generate",
    "gcg_seg": "caption_generate",
    "bbox_detection": "dense_detection",
    "point_detection": "dense_detection",
    "keypoint": "dense_detection",
    "ocr": "dense_OCR",
    "recon3d": "recon3d",
    "camera_pose": "understanding",
}
BASE_PARAMS: Dict[str, Dict[str, Any]] = {
    "generate": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    ),
    "think_generate": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        think=True,
    ),
    "caption_generate": dict(
        max_think_token_n=8192,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        caption=True,
    ),
    "dense_perception": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "edit": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "think_edit": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        think=True,
    ),
    "understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "think_understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
        think=True,
    ),
    "dense_detection": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "dense_OCR": dict(
        max_think_token_n=20000,
        do_sample=False,
        understanding_output=True,
    ),
    "recon3d": dict(
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
}
TUNABLE_PARAM_EXCLUDE = {
    "caption",
    "coco_json",
    "do_sample",
    "file_name",
    "image_id",
    "mode",
    "task",
    "task_name",
    "think",
    "think_flag",
    "understanding_output",
    "max_think_token_n",
}
BASE_PARAM_CONTROL_KEYS = (
    "cfg_text_scale",
    "cfg_img_scale",
    "cfg_interval_start",
    "cfg_interval_end",
    "timestep_shift",
    "num_timesteps",
    "cfg_renorm_min",
    "max_length_token",
    "seed",
)
BASE_PARAM_CONTROL_DEFAULTS: Dict[str, Any] = {
    "cfg_text_scale": 1.0,
    "cfg_img_scale": 1.0,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "timestep_shift": 1.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 1.0,
    "max_length_token": 8192,
    "seed": 42,
}
MAX_LENGTH_TOKEN_ACTIONS = {
    "Open understanding",
    "Pan seg",
    "GCG Seg",
    "Bbox detection",
    "Point detection",
    "OCR (word/text line)",
    "keypoint",
    "Camera pose",
}
PARAM_RANGE_HELP = {
    "cfg_text_scale": "cfg_text_scale >= 1.0",
    "cfg_img_scale": "cfg_img_scale >= 1.0",
    "cfg_interval": "0.0 <= cfg_interval_start <= cfg_interval_end <= 1.0",
    "timestep_shift": "timestep_shift > 0.0",
    "num_timesteps": "num_timesteps <= 50",
    "cfg_renorm_min": "0.0 <= cfg_renorm_min <= 1.0",
    "max_length_token": "max_length_token <= 20000",
}
TASK_CHOICES = [task for task in TASK_ORDER if task != "raw_query"] + ["raw_query"]
MULTI_IMAGE_TASKS = {"recon3d", "camera_pose"}
CATEGORY_QUERY_TASKS = {"bbox_detection", "point_detection", "keypoint", "binary_seg", "pan_seg"}
MAX_INPUT_IMAGES = 12
RAW_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEMO_EXAMPLE_ROOT = os.environ.get(
    "SENSENOVA_DEMO_EXAMPLE_ROOT",
    os.path.join(PROJECT_ROOT, "examples"),
)
DEMO_INFERENCE_EXAMPLE_ROOT = os.path.join(CURRENT_DIR, "examples")

TASK_ACTION_INFO: Dict[str, Dict[str, str]] = {
    "Binary seg (ref/reason/inter)": {
        "label": "Binary Seg",
        "task": "binary_seg",
        "mode": "dense_perception",
        "sample_class": "dense_prediction",
        "placeholder": "person",
    },
    "depth": {
        "label": "Depth",
        "task": "depth",
        "mode": "dense_perception",
        "sample_class": "dense_prediction",
        "placeholder": "",
    },
    "normal": {
        "label": "Normal",
        "task": "normal",
        "mode": "dense_perception",
        "sample_class": "dense_prediction",
        "placeholder": "",
    },
    "Pan seg": {
        "label": "Pan Seg",
        "task": "pan_seg",
        "mode": "caption_generate",
        "sample_class": "caption_seg",
        "placeholder": "person, car, road, sky",
    },
    "GCG Seg": {
        "label": "GCG Seg",
        "task": "gcg_seg",
        "mode": "caption_generate",
        "sample_class": "caption_seg",
        "placeholder": "",
    },
    "Bbox detection": {
        "label": "BBox Det",
        "task": "bbox_detection",
        "mode": "dense_detection",
        "sample_class": "detection",
        "placeholder": "person, car",
    },
    "Point detection": {
        "label": "Point Det",
        "task": "point_detection",
        "mode": "dense_detection",
        "sample_class": "detection",
        "placeholder": "person",
    },
    "OCR (word/text line)": {
        "label": "OCR",
        "task": "ocr",
        "mode": "dense_OCR",
        "sample_class": "detection",
        "placeholder": "",
    },
    "keypoint": {
        "label": "Keypoint",
        "task": "keypoint",
        "mode": "dense_detection",
        "sample_class": "detection",
        "placeholder": "person",
    },
    "3D reconstruction": {
        "label": "3D Recon",
        "task": "recon3d",
        "mode": "recon3d",
        "sample_class": "geometry",
        "placeholder": (
            "Reconstruct a scene from multiple input images and output one dense 3D "
            "coordinate map per view, all aligned to the first camera's perspective."
        ),
    },
    "Camera pose": {
        "label": "Camera pose",
        "task": "camera_pose",
        "mode": "understanding",
        "sample_class": "geometry",
        "placeholder": (
            "With the first frame as the reference frame, output the relative pose of "
            "all subsequent frames with respect to the first frame."
        ),
    },
    "Open generate": {
        "label": "Generate",
        "task": "raw_query",
        "mode": "generate",
        "sample_class": "generate",
        "placeholder": "A city street scene with cars, pedestrians, buildings, road, sidewalk, trees, and sky.",
    },
    "Open edit": {
        "label": "Edit",
        "task": "raw_query",
        "mode": "edit",
        "sample_class": "edit",
        "placeholder": "Turn the apples in the picture green.",
    },
    "Open understanding": {
        "label": "Understand",
        "task": "raw_query",
        "mode": "understanding",
        "sample_class": "understanding",
        "placeholder": "Describe the image in detail.",
    },
}

GALLERY_LABELS: Dict[str, str] = {
    "depth": "Depth",
    "normal": "Normal",
    "Binary seg (ref/reason/inter)": "Binary Seg",
    "Pan seg": "Panoptic Seg",
    "GCG Seg": "GCG Seg",
    "Bbox detection": "Bbox Det",
    "Point detection": "Point Det",
    "OCR (word/text line)": "OCR",
    "keypoint": "Keypoint",
    "Open understanding": "General Understanding",
    "Open edit": "General Editing",
    "Open generate": "General Generation",
    "3D reconstruction": "3D Recon",
    "Camera pose": "Camera Pose",
}

OUTPUT_TYPE_BY_ACTION: Dict[str, str] = {
    "depth": "image",
    "normal": "image",
    "Binary seg (ref/reason/inter)": "image",
    "Pan seg": "text+image",
    "GCG Seg": "text+image",
    "Open generate": "image",
    "Open edit": "image",
    "Open understanding": "text",
    "Bbox detection": "text",
    "Point detection": "text",
    "OCR (word/text line)": "text",
    "keypoint": "text",
    "Camera pose": "text",
    "3D reconstruction": "image",
}

TASK_ACTION_GROUPS = (
    ("Bbox detection", "Point detection", "OCR (word/text line)", "keypoint"),
    ("Binary seg (ref/reason/inter)", "GCG Seg", "Pan seg"),
    ("depth", "normal"),
    ("3D reconstruction", "Camera pose"),
    ("Open generate", "Open edit", "Open understanding"),
)
TASK_ACTION_CHOICES = [
    (TASK_ACTION_INFO[action]["label"], action)
    for group in TASK_ACTION_GROUPS
    for action in group
]
TASK_ACTION_ALIASES = {
    "binary seg": "Binary seg (ref/reason/inter)",
    "binary segmentation": "Binary seg (ref/reason/inter)",
    "binary seg (ref/reason/inter)": "Binary seg (ref/reason/inter)",
    "depth": "depth",
    "normal": "normal",
    "pan seg": "Pan seg",
    "panoptic": "Pan seg",
    "gcg seg": "GCG Seg",
    "gcg": "GCG Seg",
    "bbox detection": "Bbox detection",
    "bbox": "Bbox detection",
    "detection": "Bbox detection",
    "point detection": "Point detection",
    "point": "Point detection",
    "ocr": "OCR (word/text line)",
    "ocr (word/text line)": "OCR (word/text line)",
    "keypoint": "keypoint",
    "open understanding": "Open understanding",
    "understanding": "Open understanding",
    "general understanding": "Open understanding",
    "open generate": "Open generate",
    "generate": "Open generate",
    "general generation": "Open generate",
    "open edit": "Open edit",
    "edit": "Open edit",
    "general editing": "Open edit",
    "3d reconstruction": "3D reconstruction",
    "3d recon": "3D reconstruction",
    "recon3d": "3D reconstruction",
    "camera pose": "Camera pose",
    "pose": "Camera pose",
}

TASK_TO_ACTION = {
    info["task"]: action
    for action, info in TASK_ACTION_INFO.items()
    if info["task"] != "raw_query"
}
MODE_TO_ACTION = {
    "generate": "Open generate",
    "think_generate": "Open generate",
    "caption_generate": "Pan seg",
    "dense_perception": "depth",
    "edit": "Open edit",
    "think_edit": "Open edit",
    "understanding": "Open understanding",
    "think_understanding": "Open understanding",
    "dense_detection": "Bbox detection",
    "dense_OCR": "OCR (word/text line)",
}

TASK_EXAMPLES = {
    task: TASK_ACTION_INFO[action]["placeholder"]
    for task, action in TASK_TO_ACTION.items()
}
TASK_EXAMPLES["raw_query"] = TASK_ACTION_INFO["Open understanding"]["placeholder"]
TASK_EXAMPLES["recon3d"] = TASK_ACTION_INFO["3D reconstruction"]["placeholder"]
TASK_EXAMPLES["camera_pose"] = TASK_ACTION_INFO["Camera pose"]["placeholder"]

DEPTH_DISPLAY_PROMPT = (
    "Estimate relative depth for each pixel in the image, with closer objects "
    "appearing brighter and distant objects appearing darker. Output is a "
    "grayscale image with pixel values ranging from 0-255."
)
NORMAL_DISPLAY_PROMPT = (
    "Estimate surface normals and encode as an RGB image. Each channel "
    "corresponds to a direction component (X, Y, Z) with continuous value "
    "variations, creating smooth color gradients distinct from other task outputs."
)
OCR_DISPLAY_PROMPT = (
    "Perform word-level text detection and recognition on the entire image. "
    "Output a structured text list containing every detected word, its bounding "
    "box coordinates with <bbox> format, and the recognized text content."
)
BBOX_DISPLAY_PROMPT_TEMPLATE = (
    "Detect all instances of {categories} in the image. Output the results "
    "as a structured text list with each detection including category and "
    "bounding box coordinates in <bbox> format."
)
POINT_DISPLAY_PROMPT_TEMPLATE = (
    "Locate and identify {categories} within the scene. Output detection "
    "results as text entries, each containing the object class and pixel "
    "coordinates defining the object point location."
)
KEYPOINT_DISPLAY_PROMPT_TEMPLATE = (
    "Detect all instances of {categories} in the image. For each instance, "
    "output a bounding box in <bbox> format and keypoints in <kpt>[x,y]</kpt> "
    "format. Return results as a structured list."
)
BINARY_SEG_DISPLAY_PROMPT_TEMPLATE = (
    "Can you segment the image based on the following categories: {categories}? "
    "Please output the binary segmentation masks."
)
PAN_SEG_DISPLAY_PROMPT_TEMPLATE = (
    "Can you segment the image based on the following categories: {categories}? "
    "Please output the panoptic segmentation masks. Please find all instances "
    "in the image and assign color to each instance in the EXACT format: "
    "<p>instance-no<color>(R,G,B)</color></p>, then respond with panoptic "
    "segmentation masks."
)
GCG_DISPLAY_PROMPT = (
    "Please briefly describe the contents of the image. Please respond with "
    "interleaved segmentation masks for the corresponding parts of the answer."
)
KEYPOINT_ANTELOPE_DEMO_PROMPT = (
    "Detect all instances of <p>antelope</p> in the image. For each instance, "
    "output a bounding box in <bbox> format and the coordinates of its "
    "left eye, right eye, nose, neck, root of tail, left shoulder, left "
    "elbow, left front paw, right shoulder, right elbow, right front paw, "
    "left hip, left knee, left back paw, right hip, right knee, right back "
    "paw in <kpt>[x,y]</kpt> format. Return results as a structured list."
)
PAN_SEG_COCO_DEMO_CATEGORIES = (
    "<p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, "
    "<p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, "
    "<p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, "
    "<p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, "
    "<p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, "
    "<p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, "
    "<p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, "
    "<p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, "
    "<p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, "
    "<p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, "
    "<p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, "
    "<p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, "
    "<p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, "
    "<p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, "
    "<p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, "
    "<p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, "
    "<p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, "
    "<p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, "
    "<p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, "
    "<p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, "
    "<p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, "
    "<p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, "
    "<p>curtain</p>, <p>door</p>, <p>floor-wood</p>, <p>flower</p>, "
    "<p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, "
    "<p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, "
    "<p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, "
    "<p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, "
    "<p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall-brick</p>, "
    "<p>wall-stone</p>, <p>wall-tile</p>, <p>wall-wood</p>, "
    "<p>water</p>, <p>window-blind</p>, <p>window</p>, <p>tree</p>, "
    "<p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, "
    "<p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, "
    "<p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, "
    "<p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p>"
)
PAN_SEG_COCO_DEMO_PROMPT = PAN_SEG_DISPLAY_PROMPT_TEMPLATE.format(
    categories=PAN_SEG_COCO_DEMO_CATEGORIES
)
RECON3D_DISPLAY_PROMPT = (
    "Reconstruct a scene from multiple input images and output one dense 3D "
    "coordinate map per view, all aligned to the first camera's perspective."
)
CAMERA_POSE_DISPLAY_PROMPT = (
    "With the first frame as the reference frame, output the relative pose of "
    "all subsequent frames (excluding the first frame) with respect to the "
    "first frame, following the input order and adhering to the strict format "
    "below:Rotation: Represented by a quaternion in the format "
    "<quat>[x,y,z,w], enclosed in <quat> tags;Translation: Represented by a "
    "unit vector (direction) in the format <offset>[x,y,z], enclosed in "
    "<offset> tags (the vector has no absolute physical meaning, only "
    "directional information);Scale: Represented by a numerical value in the "
    "format <scale>value</scale> tags, where the value denotes the magnitude "
    "of translation (corresponding to the length of the translation unit "
    "vector);Enclose the result of each frame in <frame> tags, with no extra "
    "characters, spaces, or line breaks outside the tags."
)


DEMO_CASE_SPECS: List[Dict[str, Any]] = [
    {
        "case_dir": "depth",
        "action": "depth",
    },
    {
        "case_dir": "normal",
        "action": "normal",
    },
    {
        "case_dir": "binary_seg",
        "action": "Binary seg (ref/reason/inter)",
    },
    {
        "case_dir": "pan_val",
        "action": "Pan seg",
        "display_prompt": PAN_SEG_COCO_DEMO_PROMPT,
    },
    {
        "case_dir": "gcg_val",
        "action": "GCG Seg",
        "display_prompt": GCG_DISPLAY_PROMPT,
    },
    {
        "case_dir": "bbox",
        "action": "Bbox detection",
    },
    {
        "case_dir": "point",
        "action": "Point detection",
    },
    {
        "case_dir": "OCR",
        "action": "OCR (word/text line)",
    },
    {
        "case_dir": "keypoint",
        "action": "keypoint",
        "display_prompt": KEYPOINT_ANTELOPE_DEMO_PROMPT,
    },
    {
        "case_dir": "open_editing",
        "action": "Open edit",
    },
    {
        "case_dir": "open_generate",
        "action": "Open generate",
    },
    {
        "case_dir": "recon3d",
        "action": "3D reconstruction",
        "prompt": TASK_ACTION_INFO["3D reconstruction"]["placeholder"],
        "multi_images": [
            "47204575_4847.103.png",
            "47204575_4852.001.png",
            "47204575_4871.692.png",
            "47204575_4873.692.png",
            "47204575_4875.791.png",
        ],
    },
    {
        "case_dir": "recon3d",
        "action": "Camera pose",
        "prompt": TASK_ACTION_INFO["Camera pose"]["placeholder"],
        "multi_images": [
            "47204575_4847.103.png",
            "47204575_4852.001.png",
            "47204575_4871.692.png",
            "47204575_4873.692.png",
            "47204575_4875.791.png",
        ],
    },
]


def _clean_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path).strip().strip("'\"")))


def _normalize_model_path(path: str) -> str:
    path = str(path or "").strip().strip("'\"")
    if not path:
        return DEFAULT_MODEL_PATH
    expanded = os.path.expanduser(path)
    if expanded.startswith(("/", "./", "../")) or os.path.exists(expanded):
        return os.path.abspath(expanded)
    return path


def _file_value_to_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return _clean_path(value) if value else None
    if isinstance(value, dict):
        path = value.get("path") or value.get("name")
        if isinstance(path, str) and path.strip():
            return _clean_path(path)
        image_value = value.get("image")
        if isinstance(image_value, str) and image_value.strip():
            return _clean_path(image_value)
        if isinstance(image_value, dict):
            return _file_value_to_path(image_value)
        if isinstance(image_value, (list, tuple)):
            return _file_value_to_path(image_value[0]) if image_value else None
        return None
    path = getattr(value, "path", None) or getattr(value, "name", None)
    return _clean_path(path) if path else None


def _image_collection_to_paths(value: Any) -> List[str]:
    paths: List[str] = []

    def add_item(item: Any) -> None:
        if item is None:
            return
        path = _file_value_to_path(item)
        if path:
            paths.append(path)
            return
        if isinstance(item, tuple):
            if item:
                first_path = _file_value_to_path(item[0])
                if first_path:
                    paths.append(first_path)
                    return
            for child in item:
                add_item(child)
            return
        if isinstance(item, list):
            for child in item:
                add_item(child)

    add_item(value)
    deduped_paths: List[str] = []
    seen = set()
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_paths.append(path)
    return deduped_paths


def normalize_task_action(task_action: str) -> str:
    action = str(task_action or "").strip()
    return TASK_ACTION_ALIASES.get(action.lower(), action) if action else "Open understanding"


def output_type_for_action(action: str) -> str:
    return OUTPUT_TYPE_BY_ACTION.get(normalize_task_action(action), "")


def _is_numeric_param_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    return False


def _filter_tunable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if key not in TUNABLE_PARAM_EXCLUDE and _is_numeric_param_value(value)
    }


def base_params_for_action(action: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = normalize_task_action(action)
    _, mode = action_to_task_mode(normalized)
    params = dict(BASE_PARAMS.get(mode, {}))
    if overrides:
        params.update(overrides)
    output_type = output_type_for_action(normalized)
    if output_type == "image":
        params.setdefault("seed", 42)
    elif output_type == "data":
        params.setdefault("seed", 123456)
    tunable_params = _filter_tunable_params(params)
    if normalized in MAX_LENGTH_TOKEN_ACTIONS and _is_numeric_param_value(params.get("max_think_token_n")):
        tunable_params["max_length_token"] = params["max_think_token_n"]
    return tunable_params


def base_params_json_for_action(action: str, overrides: Optional[Dict[str, Any]] = None) -> str:
    return _safe_json(base_params_for_action(action, overrides))


def _control_values_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    values = dict(BASE_PARAM_CONTROL_DEFAULTS)
    if "cfg_interval" in params and isinstance(params["cfg_interval"], list) and len(params["cfg_interval"]) >= 2:
        values["cfg_interval_start"] = params["cfg_interval"][0]
        values["cfg_interval_end"] = params["cfg_interval"][1]
    for key in BASE_PARAM_CONTROL_KEYS:
        if key in params:
            values[key] = params[key]
    return values


def _base_param_control_updates(params: Dict[str, Any]) -> Tuple[Any, ...]:
    values = _control_values_from_params(params)
    visible_keys = set(params)
    if "cfg_interval" in params:
        visible_keys.update({"cfg_interval_start", "cfg_interval_end"})
    return tuple(
        gr.update(value=values[key], visible=key in visible_keys)
        for key in BASE_PARAM_CONTROL_KEYS
    )


def base_param_control_updates_for_action(
    action: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, ...]:
    return _base_param_control_updates(base_params_for_action(action, overrides))


def _format_param_error(param_name: str, current_value: Any) -> str:
    return f"{param_name}: correct range is {PARAM_RANGE_HELP[param_name]}. Current value: {current_value!r}."


def _read_float_param(param_name: str, value: Any, errors: List[str]) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(_format_param_error(param_name, value))
        return None
    if not math.isfinite(number):
        errors.append(_format_param_error(param_name, value))
        return None
    return number


def _read_int_param(param_name: str, value: Any, errors: List[str]) -> Optional[int]:
    number = _read_float_param(param_name, value, errors)
    if number is None:
        return None
    if not number.is_integer():
        errors.append(_format_param_error(param_name, value))
        return None
    return int(number)


def _raise_param_validation_errors(errors: List[str]) -> None:
    if errors:
        raise ValueError("Invalid parameter value:\n" + "\n".join(errors))


def params_from_base_param_controls(
    action: str,
    cfg_text_scale: Any,
    cfg_img_scale: Any,
    cfg_interval_start: Any,
    cfg_interval_end: Any,
    timestep_shift: Any,
    num_timesteps: Any,
    cfg_renorm_min: Any,
    max_length_token: Any,
    seed: Any,
) -> Dict[str, Any]:
    params = base_params_for_action(action)
    validation_errors: List[str] = []
    if "cfg_text_scale" in params:
        value = _read_float_param("cfg_text_scale", cfg_text_scale, validation_errors)
        if value is not None:
            params["cfg_text_scale"] = value
            if value < 1.0:
                validation_errors.append(_format_param_error("cfg_text_scale", value))
    if "cfg_img_scale" in params:
        value = _read_float_param("cfg_img_scale", cfg_img_scale, validation_errors)
        if value is not None:
            params["cfg_img_scale"] = value
            if value < 1.0:
                validation_errors.append(_format_param_error("cfg_img_scale", value))
    if "cfg_interval" in params:
        start = _read_float_param("cfg_interval", cfg_interval_start, validation_errors)
        end = _read_float_param("cfg_interval", cfg_interval_end, validation_errors)
        if start is not None and end is not None:
            params["cfg_interval"] = [start, end]
            if not (0.0 <= start <= end <= 1.0):
                validation_errors.append(_format_param_error("cfg_interval", [start, end]))
    if "timestep_shift" in params:
        value = _read_float_param("timestep_shift", timestep_shift, validation_errors)
        if value is not None:
            params["timestep_shift"] = value
            if value <= 0.0:
                validation_errors.append(_format_param_error("timestep_shift", value))
    if "num_timesteps" in params:
        value = _read_int_param("num_timesteps", num_timesteps, validation_errors)
        if value is not None:
            params["num_timesteps"] = value
            if value > 50:
                validation_errors.append(_format_param_error("num_timesteps", value))
    if "cfg_renorm_min" in params:
        value = _read_float_param("cfg_renorm_min", cfg_renorm_min, validation_errors)
        if value is not None:
            params["cfg_renorm_min"] = value
            if not (0.0 <= value <= 1.0):
                validation_errors.append(_format_param_error("cfg_renorm_min", value))
    if "max_length_token" in params:
        value = _read_int_param("max_length_token", max_length_token, validation_errors)
        if value is not None:
            params["max_think_token_n"] = value
            if value > 20000:
                validation_errors.append(_format_param_error("max_length_token", value))
        params.pop("max_length_token", None)
    if "seed" in params:
        params["seed"] = int(seed)
    _raise_param_validation_errors(validation_errors)
    return params


def action_to_task_mode(task_action: str) -> Tuple[str, str]:
    action = normalize_task_action(task_action)
    info = TASK_ACTION_INFO.get(action)
    if info is None:
        raise ValueError(f"Unsupported task action: {task_action}")
    return info["task"], info["mode"]


def task_mode_to_action(task: str, mode: str) -> str:
    if task == "raw_query":
        return MODE_TO_ACTION.get(mode, "Open understanding")
    return TASK_TO_ACTION.get(task, "Open understanding")


def resolve_request_mode(task: str, requested_mode: str) -> str:
    if task != "raw_query":
        return TASK_TO_MODE[task]
    mode = (requested_mode or TASK_TO_MODE["raw_query"]).strip()
    if mode not in BASE_PARAMS:
        raise ValueError(f"Unsupported mode: {mode}. Available modes: {', '.join(BASE_PARAMS)}")
    return mode


@lru_cache(maxsize=1)
def _backend_module() -> Any:
    return importlib.import_module("inference.inference_demo")


def _collect_image_paths(
    input_images: Any,
    allow_empty: bool = False,
) -> List[str]:
    paths = _image_collection_to_paths(input_images)

    if not paths and allow_empty:
        return []
    if not paths:
        raise ValueError("Please upload at least one image.")

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Image not found: {missing[0]}")

    return paths


def _load_input_image(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _image_data_uri(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/webp" if ext == ".webp" else "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _first_existing_file(directory: str, names: List[str]) -> str:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return ""


def _read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _localize_demo_path(case_dir: str, raw_path: str, fallback_names: List[str]) -> str:
    local_dir = os.path.join(DEMO_EXAMPLE_ROOT, case_dir)
    fallback = _first_existing_file(local_dir, fallback_names)
    if fallback:
        return fallback
    path = str(raw_path or "").strip()
    if path and os.path.isfile(path):
        return path
    return ""


def _localize_demo_paths(case_dir: str, names: List[str]) -> List[str]:
    local_dir = os.path.join(DEMO_EXAMPLE_ROOT, case_dir)
    paths = []
    for name in names:
        path = str(name or "").strip()
        if not path:
            continue
        candidate = path if os.path.isabs(path) else os.path.join(local_dir, path)
        if os.path.isfile(candidate):
            paths.append(candidate)
    return paths


def _first_demo_file_with_ext(case_dir: str, extensions: Tuple[str, ...]) -> str:
    local_dir = os.path.join(DEMO_EXAMPLE_ROOT, case_dir)
    if not os.path.isdir(local_dir):
        return ""
    skip_names = {"label.json"}
    for root, _, filenames in os.walk(local_dir):
        for filename in sorted(filenames):
            if filename in skip_names:
                continue
            if os.path.splitext(filename)[1].lower() in extensions:
                return os.path.join(root, filename)
    return ""


def _first_file_with_ext(directory: str, extensions: Tuple[str, ...]) -> str:
    if not os.path.isdir(directory):
        return ""
    for root, _, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() in extensions:
                return os.path.join(root, filename)
    return ""


def _demo_model3d_path(case_dir: str) -> str:
    return (
        _first_demo_file_with_ext(case_dir, (".glb", ".gltf"))
        or _first_file_with_ext(
            os.path.join(DEMO_INFERENCE_EXAMPLE_ROOT, "3d_reconstruction_output"),
            (".glb", ".gltf"),
        )
    )


def _extract_prompt_categories(prompt: str) -> str:
    names = []
    for match in re.finditer(r"<p>(.*?)</p>", prompt or "", flags=re.IGNORECASE | re.DOTALL):
        name = " ".join(match.group(1).split())
        if name:
            names.append(name)
    return ", ".join(names)


def _demo_query_from_label(action: str, label: Dict[str, Any]) -> str:
    action = normalize_task_action(action)
    info = TASK_ACTION_INFO[action]
    prompt = str(label.get("prompt") or "").strip()

    return prompt or info["placeholder"]


def _request_query_from_prompt(task: str, prompt_or_query: str) -> str:
    text = str(prompt_or_query or "").strip()
    if task in CATEGORY_QUERY_TASKS and "<p>" in text.lower() and "</p>" in text.lower():
        return _extract_prompt_categories(text) or text
    return text


def _format_display_categories(query: str) -> str:
    text = str(query or "").strip()
    if "<p>" in text.lower() and "</p>" in text.lower():
        text = _extract_prompt_categories(text) or text
    categories = [item.strip() for item in text.split(",") if item.strip()]
    return ", ".join(f"<p>{category}</p>" for category in categories)


def _display_prompt_for_task(task: str, query: Optional[str] = None) -> str:
    query_text = TASK_EXAMPLES.get(task, "") if query is None else str(query or "")
    categories = _format_display_categories(query_text)
    if task == "depth":
        return DEPTH_DISPLAY_PROMPT
    if task == "normal":
        return NORMAL_DISPLAY_PROMPT
    if task == "binary_seg":
        return BINARY_SEG_DISPLAY_PROMPT_TEMPLATE.format(categories=categories)
    if task == "pan_seg":
        return PAN_SEG_DISPLAY_PROMPT_TEMPLATE.format(categories=categories)
    if task == "gcg_seg":
        return GCG_DISPLAY_PROMPT
    if task == "bbox_detection":
        return BBOX_DISPLAY_PROMPT_TEMPLATE.format(categories=categories)
    if task == "point_detection":
        return POINT_DISPLAY_PROMPT_TEMPLATE.format(categories=categories)
    if task == "keypoint":
        return KEYPOINT_DISPLAY_PROMPT_TEMPLATE.format(categories=categories)
    if task == "ocr":
        return OCR_DISPLAY_PROMPT
    if task == "recon3d":
        return RECON3D_DISPLAY_PROMPT
    if task == "camera_pose":
        return CAMERA_POSE_DISPLAY_PROMPT
    return query_text


def _base_params_for_action(action: str, label: Dict[str, Any], mode: str) -> Dict[str, Any]:
    base_param = label.get("base_param") or {}
    if isinstance(base_param, dict):
        mode_params = base_param.get(mode)
        if isinstance(mode_params, dict):
            return dict(mode_params)
        return dict(base_param)
    return {}


def _load_demo_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for spec in DEMO_CASE_SPECS:
        case_dir = spec["case_dir"]
        action = normalize_task_action(spec["action"])
        task, mode = action_to_task_mode(action)
        local_dir = os.path.join(DEMO_EXAMPLE_ROOT, case_dir)
        label = _read_json_file(os.path.join(local_dir, "label.json"))
        if spec.get("prompt") and not label.get("prompt"):
            label["prompt"] = spec["prompt"]
        multi_images = _localize_demo_paths(case_dir, list(spec.get("multi_images") or []))
        image_path = _localize_demo_path(
            case_dir,
            str(label.get("input") or ""),
            ["image.jpg", "image.png", "image.webp", "original.jpg"],
        )
        if multi_images:
            image_path = multi_images[0]
        gt_path = _localize_demo_path(
            case_dir,
            str(label.get("output") or ""),
            ["gt.png", "gt.jpg", "gt.jpeg", "gt.webp"],
        )
        gt_model3d_path = _demo_model3d_path(case_dir) if task == "recon3d" else ""
        gt_text_path = _first_demo_file_with_ext(case_dir, (".json", ".txt")) if task == "camera_pose" else ""
        query = _demo_query_from_label(action, label)

        if not image_path and TASK_ACTION_INFO[action]["sample_class"] != "generate":
            continue

        cases.append(
            {
                "label": TASK_ACTION_INFO[action]["label"],
                "gallery_label": GALLERY_LABELS.get(action, TASK_ACTION_INFO[action]["label"]),
                "output_type": output_type_for_action(action),
                "action": action,
                "image": image_path,
                "images": multi_images or ([image_path] if image_path else []),
                "gt": gt_path,
                "gt_model3d": gt_model3d_path,
                "gt_text": gt_text_path,
                "task": task,
                "mode": mode,
                "query": query,
                "display_prompt": str(spec.get("display_prompt") or ""),
                "source_prompt": str(label.get("prompt") or ""),
                "base_params": _base_params_for_action(action, label, mode),
                "sample_class": TASK_ACTION_INFO[action]["sample_class"],
                "source": local_dir,
            }
        )
    return cases


DEMO_CASES = _load_demo_cases()
DEMO_CASE_LABELS = [case["label"] for case in DEMO_CASES]


def _safe_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


@lru_cache(maxsize=2)
def _get_runner(
    model_path: str,
    device: str,
    dtype: str,
    output_dir: str,
) -> Any:
    backend = _backend_module()
    return backend.WebTaskRunner(
        model_path=model_path,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
    )


def _run_request(
    runner: Any,
    request: Any,
    prompt: str,
) -> Tuple[Dict[str, Any], Image.Image, List[str]]:
    backend = _backend_module()
    if request.image_paths:
        input_image = _load_input_image(request.image_paths[0])
        output_context = backend.make_output_context(runner.output_dir, request)
    else:
        task_dir, vis_dir = backend.ensure_output_dirs(runner.output_dir, request.task)
        output_context = backend.OutputContext(
            task_dir=task_dir,
            vis_dir=vis_dir,
            save_prefix=f"{request.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        input_image = Image.new("RGB", (1024, 1024), (245, 245, 245))

    return runner.run_web(request, prompt, input_image, output_context)


def _file_output_update(paths: List[str]) -> Any:
    existing_paths = [path for path in paths if path and os.path.isfile(path)]
    return gr.update(value=existing_paths or None, visible=True)


def _is_raw_image_path(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.splitext(path)[1].lower() in RAW_IMAGE_EXTENSIONS


def _raw_image_output_update(paths: List[str]) -> Any:
    image_paths = [path for path in paths if _is_raw_image_path(path)]
    return gr.update(value=image_paths or None, visible=bool(image_paths))


def _raw_output_files_for_display(paths: List[str], task: str) -> Tuple[List[str], List[str]]:
    existing_paths = [path for path in paths if path and os.path.isfile(path)]
    if task == "recon3d":
        return (
            [path for path in existing_paths if os.path.splitext(path)[1].lower() == ".npy"],
            [],
        )
    return existing_paths, [path for path in existing_paths if _is_raw_image_path(path)]


def update_raw_output_display(metadata_json: str) -> Tuple[Any, Any]:
    try:
        metadata = json.loads(str(metadata_json or "{}"))
    except json.JSONDecodeError:
        metadata = {}
    request = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    task = str(request.get("task") or metadata.get("task") or "")
    raw_output_files = metadata.get("raw_output_files") or []
    if not isinstance(raw_output_files, list):
        raw_output_files = []
    downloadable_files, raw_image_files = _raw_output_files_for_display(raw_output_files, task)
    return _file_output_update(downloadable_files), _raw_image_output_update(raw_image_files)


def clear_raw_image_output() -> Any:
    return _raw_image_output_update([])


def _raise_gradio_error(exc: Exception) -> None:
    message = str(exc) or "Unknown error."
    raise gr.Error(f"{type(exc).__name__}: {message}") from exc


def _model3d_output_update(path: str, *, visible_when_empty: bool = False) -> Any:
    model_path = str(path or "").strip()
    if model_path and os.path.isfile(model_path):
        return gr.update(value=model_path, visible=True)
    return gr.update(value=None, visible=visible_when_empty)


def _image_output_update(image: Optional[Image.Image], task: str) -> Any:
    if task == "recon3d":
        return gr.update(value=None, visible=False)
    return gr.update(value=image, visible=True)


def _text_output_for_display(task: str, text_output: str) -> str:
    text = str(text_output or "")
    if task != "recon3d":
        return text
    marker = "Model Text:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return ""


def _output_visibility_for_action(task_action: str) -> Tuple[Any, Any]:
    task, _ = action_to_task_mode(task_action)
    return (
        gr.update(value=None, visible=task != "recon3d"),
        gr.update(value=None, visible=task == "recon3d"),
    )


def _input_image_slot_updates(paths: Any) -> List[Any]:
    image_paths = _image_collection_to_paths(paths)[:MAX_INPUT_IMAGES]
    updates: List[Any] = [gr.update(visible=not image_paths)]
    for idx in range(MAX_INPUT_IMAGES):
        if idx < len(image_paths):
            updates.extend(
                [
                    gr.update(visible=True),
                    gr.update(value=image_paths[idx], visible=True),
                    gr.update(visible=True),
                ]
            )
        else:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(visible=False),
                ]
            )
    return updates


def append_input_images(current_images: Any, uploaded_files: Any) -> Tuple[Any, ...]:
    image_paths = _image_collection_to_paths(current_images)
    image_paths.extend(_image_collection_to_paths(uploaded_files))
    image_paths = _image_collection_to_paths(image_paths)[:MAX_INPUT_IMAGES]
    return (image_paths, *_input_image_slot_updates(image_paths), gr.update(value=None))


def delete_input_image(current_images: Any, delete_index: Any) -> Tuple[Any, ...]:
    image_paths = _image_collection_to_paths(current_images)
    index = int(delete_index)
    if 0 <= index < len(image_paths):
        image_paths.pop(index)
    return (image_paths, *_input_image_slot_updates(image_paths))


def _read_text_preview(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def predict(
    input_images: Any,
    task_action: str,
    task: str,
    mode: str,
    query: str,
    cfg_text_scale: Any,
    cfg_img_scale: Any,
    cfg_interval_start: Any,
    cfg_interval_end: Any,
    timestep_shift: Any,
    num_timesteps: Any,
    cfg_renorm_min: Any,
    max_length_token: Any,
    seed: Any,
) -> Tuple[Optional[Image.Image], str, Any, Any, str]:
    start = time.time()
    action = normalize_task_action(task_action)
    action_task, action_mode = action_to_task_mode(action)
    if action != "Open understanding":
        task = action_task
        mode = action_mode
    else:
        task = (task or action_task).strip()
        mode = (mode or action_mode).strip()
    backend = _backend_module()
    mode = backend.resolve_request_mode(task, mode)
    allow_empty_images = TASK_ACTION_INFO[action]["sample_class"] == "generate"
    image_paths = _collect_image_paths(
        input_images,
        allow_empty=allow_empty_images,
    )
    if task not in MULTI_IMAGE_TASKS and len(image_paths) > 1:
        image_paths = image_paths[:1]

    prompt = str(query or "").strip()
    request = backend.InferenceRequest(
        image_paths=image_paths,
        task=task,
        mode=mode,
        query=prompt,
        params=params_from_base_param_controls(
            action,
            cfg_text_scale,
            cfg_img_scale,
            cfg_interval_start,
            cfg_interval_end,
            timestep_shift,
            num_timesteps,
            cfg_renorm_min,
            max_length_token,
            seed,
        ),
    )
    runner = _get_runner(
        _normalize_model_path(os.environ.get("SENSENOVA_MODEL_PATH", DEFAULT_MODEL_PATH)),
        os.environ.get("SENSENOVA_DEVICE", "cuda").strip(),
        os.environ.get("SENSENOVA_DTYPE", "bf16").strip(),
        _clean_path(os.environ.get("SENSENOVA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    output, visualization, saved_files = _run_request(runner, request, prompt)
    output_text = str(output.get("text") or "")
    glb_path = str(output.get("glb_output") or "")
    raw_output_files = list(output.get("raw_output_files") or [])
    metadata = {
        "task_action": action,
        "request": asdict(request),
        "prompt": prompt,
        "elapsed_sec": round(time.time() - start, 3),
        "saved_files": saved_files,
        "raw_output_files": raw_output_files,
    }
    if glb_path:
        metadata["glb_path"] = glb_path
    return (
        _image_output_update(visualization, task),
        _text_output_for_display(task, output_text),
        _file_output_update(raw_output_files),
        _model3d_output_update(glb_path),
        _safe_json(metadata),
    )


def _validate_run_inputs(
    input_images: Any,
    task_action: str,
    cfg_text_scale: Any,
    cfg_img_scale: Any,
    cfg_interval_start: Any,
    cfg_interval_end: Any,
    timestep_shift: Any,
    num_timesteps: Any,
    cfg_renorm_min: Any,
    max_length_token: Any,
    seed: Any,
) -> None:
    action = normalize_task_action(task_action)
    allow_empty_images = TASK_ACTION_INFO[action]["sample_class"] == "generate"
    _collect_image_paths(input_images, allow_empty=allow_empty_images)
    params_from_base_param_controls(
        action,
        cfg_text_scale,
        cfg_img_scale,
        cfg_interval_start,
        cfg_interval_end,
        timestep_shift,
        num_timesteps,
        cfg_renorm_min,
        max_length_token,
        seed,
    )


def validate_params_for_error(
    task_action: str,
    cfg_text_scale: Any,
    cfg_img_scale: Any,
    cfg_interval_start: Any,
    cfg_interval_end: Any,
    timestep_shift: Any,
    num_timesteps: Any,
    cfg_renorm_min: Any,
    max_length_token: Any,
    seed: Any,
) -> None:
    try:
        params_from_base_param_controls(
            task_action,
            cfg_text_scale,
            cfg_img_scale,
            cfg_interval_start,
            cfg_interval_end,
            timestep_shift,
            num_timesteps,
            cfg_renorm_min,
            max_length_token,
            seed,
        )
    except gr.Error:
        raise
    except Exception as exc:
        _raise_gradio_error(exc)


def run_with_validation(
    input_images: Any,
    task_action: str,
    task: str,
    mode: str,
    query: str,
    cfg_text_scale: Any,
    cfg_img_scale: Any,
    cfg_interval_start: Any,
    cfg_interval_end: Any,
    timestep_shift: Any,
    num_timesteps: Any,
    cfg_renorm_min: Any,
    max_length_token: Any,
    seed: Any,
) -> Tuple[Optional[Image.Image], str, Any, Any, str]:
    try:
        _validate_run_inputs(
            input_images,
            task_action,
            cfg_text_scale,
            cfg_img_scale,
            cfg_interval_start,
            cfg_interval_end,
            timestep_shift,
            num_timesteps,
            cfg_renorm_min,
            max_length_token,
            seed,
        )
        return predict(
            input_images,
            task_action,
            task,
            mode,
            query,
            cfg_text_scale,
            cfg_img_scale,
            cfg_interval_start,
            cfg_interval_end,
            timestep_shift,
            num_timesteps,
            cfg_renorm_min,
            max_length_token,
            seed,
        )
    except gr.Error:
        raise
    except Exception as exc:
        _raise_gradio_error(exc)


def update_task(task: str) -> Tuple[Any, ...]:
    task = task or "raw_query"
    default_mode = resolve_request_mode(task, "understanding")
    task_action = task_mode_to_action(task, default_mode)
    placeholder = _display_prompt_for_task(task)
    return (
        gr.update(value=task_action),
        gr.update(value=placeholder, placeholder=placeholder),
        gr.update(value=default_mode),
        *base_param_control_updates_for_action(task_action),
    )


def update_task_action(task_action: str) -> Tuple[Any, ...]:
    task, mode = action_to_task_mode(task_action)
    info = TASK_ACTION_INFO[normalize_task_action(task_action)]
    prompt = _display_prompt_for_task(task, info["placeholder"])
    image_update, model3d_update = _output_visibility_for_action(task_action)
    return (
        gr.update(value=task),
        gr.update(value=mode),
        gr.update(value=prompt, placeholder=prompt),
        *base_param_control_updates_for_action(task_action),
        current_sample_html(
            {
                "label": info["label"],
                "sample_class": info["sample_class"],
                "output_type": output_type_for_action(task_action),
            }
        ),
        image_update,
        model3d_update,
    )


def load_task_action_sample(task_action: str) -> Tuple[Any, ...]:
    action = normalize_task_action(task_action)
    case = next((item for item in DEMO_CASES if item["action"] == action), None)
    if case is not None:
        return (*load_demo_case(str(case["label"])), current_sample_html(case))

    task, mode = action_to_task_mode(action)
    info = TASK_ACTION_INFO[normalize_task_action(action)]
    prompt = _display_prompt_for_task(task, info["placeholder"])
    image_update, model3d_update = _output_visibility_for_action(action)
    return (
        [],
        *_input_image_slot_updates([]),
        gr.update(value=action),
        gr.update(value=task),
        gr.update(value=mode),
        gr.update(value=prompt, placeholder=prompt),
        *base_param_control_updates_for_action(action),
        image_update,
        "",
        _file_output_update([]),
        model3d_update,
        _safe_json(
            {
                "task_action": action,
                "task": task,
                "mode": mode,
                "prompt": prompt,
                "source": "task_action",
            }
        ),
        current_sample_html(
            {
                "label": info["label"],
                "sample_class": info["sample_class"],
                "output_type": output_type_for_action(action),
            }
        ),
    )


def _make_multiview_preview(image_paths: List[str]) -> Optional[Image.Image]:
    paths = [path for path in image_paths if path and os.path.isfile(path)]
    if not paths:
        return None
    thumbs = []
    for path in paths[:6]:
        try:
            image = _load_input_image(path)
        except (OSError, ValueError):
            continue
        image.thumbnail((220, 160))
        tile = Image.new("RGB", (220, 160), (245, 247, 251))
        x = (220 - image.width) // 2
        y = (160 - image.height) // 2
        tile.paste(image, (x, y))
        thumbs.append(tile)
    if not thumbs:
        return None
    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    preview = Image.new("RGB", (cols * 220, rows * 160), (255, 255, 255))
    for idx, tile in enumerate(thumbs):
        x = (idx % cols) * 220
        y = (idx // cols) * 160
        preview.paste(tile, (x, y))
    return preview


def load_demo_case(label: str) -> Tuple[Any, ...]:
    case = next((item for item in DEMO_CASES if item["label"] == label), None)
    if case is None:
        return (
            [],
            *_input_image_slot_updates([]),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            *base_param_control_updates_for_action("Open understanding"),
            gr.update(value=None, visible=True),
            "",
            _file_output_update([]),
            _model3d_output_update(""),
            _safe_json({"error": f"Demo case not found: {label}"}),
        )

    preview = _load_input_image(case["gt"]) if case.get("gt") and os.path.isfile(case["gt"]) else None
    if preview is None:
        preview = _make_multiview_preview(list(case.get("images") or []))
    is_recon3d = case["task"] == "recon3d"
    gt_model3d = "" if is_recon3d else str(case.get("gt_model3d") or "")
    gt_downloads = [path for path in [gt_model3d, str(case.get("gt_text") or "")] if path and os.path.isfile(path)]
    input_image_paths = list(case.get("images") or ([case["image"]] if case.get("image") else []))
    display_prompt = str(case.get("display_prompt") or "") or _display_prompt_for_task(case["task"], case["query"])
    return (
        input_image_paths,
        *_input_image_slot_updates(input_image_paths),
        gr.update(value=case["action"]),
        gr.update(value=case["task"]),
        gr.update(value=case["mode"]),
        gr.update(value=display_prompt, placeholder=display_prompt),
        *base_param_control_updates_for_action(case["action"], case["base_params"]),
        gr.update(value=preview if not is_recon3d else None, visible=not is_recon3d),
        "",
        _file_output_update(gt_downloads),
        _model3d_output_update(gt_model3d, visible_when_empty=is_recon3d),
        _safe_json(
            {
                "demo": case["label"],
                "task_action": case["action"],
                "sample_class": case["sample_class"],
                "task": case["task"],
                "mode": case["mode"],
                "query": case["query"],
                "display_prompt": display_prompt,
                "source_prompt": case["source_prompt"],
                "base_params": case["base_params"],
                "input_images": input_image_paths,
                "preview_image": case["gt"],
                "gt_model3d": case.get("gt_model3d") or "",
                "gt_text": case.get("gt_text") or "",
                "source": case["source"],
            }
        ),
    )


def _demo_gallery_value() -> List[Tuple[str, str]]:
    items = []
    for case in DEMO_CASES:
        preview_path = case.get("image") or case.get("gt")
        if preview_path:
            items.append((preview_path, str(case.get("gallery_label") or case["label"])))
    return items


def current_sample_html(case: Optional[Dict[str, Any]] = None) -> str:
    sample_name = str(case["label"]) if case else "Custom input"
    sample_class = str(case["sample_class"]) if case else "editable"
    sample_output = str(case.get("output_type") or output_type_for_action(str(case.get("action") or ""))) if case else "-"
    return f"""
    <div class="current-sample">
        <span class="current-sample-dot"></span>
        <span class="current-sample-copy">
            <span class="current-sample-main">
                <span class="current-sample-label">Current Sample</span>
                <span class="current-sample-value">{sample_name}</span>
                <span class="current-sample-class">{sample_class}</span>
            </span>
            <span class="current-sample-output">output: {sample_output}</span>
        </span>
    </div>
    """


def load_demo_case_from_gallery(evt: gr.SelectData) -> Tuple[Any, ...]:
    raw_index = None if evt is None else evt.index
    if isinstance(raw_index, (list, tuple)):
        raw_index = raw_index[0] if raw_index else None
    if raw_index is None:
        return (
            [],
            *_input_image_slot_updates([]),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            *base_param_control_updates_for_action("Open understanding"),
            "",
            gr.update(value=None, visible=True),
            "",
            _file_output_update([]),
            _model3d_output_update(""),
            _safe_json({"error": "Demo case not found."}),
            current_sample_html(),
        )
    case_index = int(raw_index)
    if case_index < 0 or case_index >= len(DEMO_CASES):
        return (
            [],
            *_input_image_slot_updates([]),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            *base_param_control_updates_for_action("Open understanding"),
            "",
            gr.update(value=None, visible=True),
            "",
            _file_output_update([]),
            _model3d_output_update(""),
            _safe_json({"error": "Demo case not found."}),
            current_sample_html(),
        )
    case = DEMO_CASES[case_index]
    return (*load_demo_case(str(case["label"])), current_sample_html(case))


def build_demo() -> gr.Blocks:
    css = """
    :root,
    body,
    body.dark,
    .dark,
    .gradio-container,
    .gradio-container.dark,
    .dark .gradio-container {
        color-scheme: light !important;
        --body-background-fill: #f6f9fc !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --block-background-fill: #ffffff !important;
        --block-border-color: #dbe3ef !important;
        --block-label-background-fill: #dfe5ff !important;
        --block-label-text-color: #6366f1 !important;
        --body-text-color: #172033 !important;
        --body-text-color-subdued: #667085 !important;
        --input-background-fill: #ffffff !important;
        --input-border-color: #d7deea !important;
        --button-secondary-background-fill: #ffffff !important;
        --button-secondary-text-color: #26324b !important;
        --checkbox-background-color: #ffffff !important;
        --checkbox-border-color: #e0e7ff !important;
        --ink: #eef4ff;
        --ink-strong: #ffffff;
        --muted: #9aa9c7;
        --line: rgba(148, 163, 184, 0.20);
        --panel: rgba(10, 18, 42, 0.72);
        --panel-strong: rgba(13, 24, 56, 0.88);
        --field: rgba(255, 255, 255, 0.96);
        --field-text: #172033;
        --accent: #6d5dfc;
        --accent-2: #22d3ee;
        --accent-soft: rgba(109, 93, 252, 0.18);
    }
    html {
        min-height: 100% !important;
        height: auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    body {
        min-height: 100% !important;
        height: auto !important;
        overflow-y: visible !important;
        overflow-x: hidden !important;
    }
    body, .gradio-container {
        background:
            linear-gradient(180deg, #fbfdff 0%, #f6f9fc 44%, #eef3f8 100%) !important;
        color: var(--ink);
    }
    .gradio-container {
        min-height: 100vh !important;
        height: auto !important;
        overflow-y: visible !important;
        overflow-x: hidden !important;
    }
    footer { display: none !important; }
    .main-shell {
        width: min(1420px, 100%) !important;
        max-width: 100vw !important;
        margin: 8px auto 40px;
        padding: 0 clamp(12px, 2vw, 20px) 24px !important;
        gap: 16px !important;
        box-sizing: border-box !important;
    }
    .app-header {
        min-height: 0;
        padding: 0 4px 16px;
        border: 0;
        border-radius: 0;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.38), rgba(244, 248, 255, 0.22)),
            radial-gradient(circle at 92% 18%, rgba(37, 99, 235, 0.10), transparent 30%);
        box-shadow: none;
        overflow: visible;
    }
    .top-nav {
        display: grid;
        grid-template-columns: minmax(0, 1fr) max-content;
        align-items: center;
        gap: 10px;
        width: 100%;
        box-sizing: border-box;
        padding: 0 0 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.24);
    }
    .nav-brand {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        min-width: 0;
        color: #1d2b4f;
        font-size: 13px;
        font-weight: 850;
    }
    .nav-logo-mark {
        width: 82px;
        height: 46px;
        border-radius: 8px;
        display: grid;
        place-items: center;
        background: transparent;
        border: 0;
        box-shadow: none;
        overflow: hidden;
        color: #1d2b4f;
        font-size: 11px;
        font-weight: 900;
        line-height: 1;
    }
    .nav-logo {
        width: 122%;
        height: 122%;
        display: block;
        object-fit: contain;
    }
    .nav-links {
        display: grid;
        grid-template-columns: repeat(3, max-content);
        gap: 8px;
        width: fit-content;
        max-width: 100%;
        min-width: 0;
        justify-content: flex-end;
        justify-self: end;
    }
    .nav-chip {
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 999px;
        padding: 7px 13px;
        color: #405071;
        background: rgba(255, 255, 255, 0.50);
        font-size: 12px;
        font-weight: 760;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 0;
        box-sizing: border-box;
    }
    .nav-chip:hover {
        color: #2563eb;
        border-color: rgba(37, 99, 235, 0.30);
        background: #ffffff;
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 0;
        padding-top: 0;
        padding-right: 0;
    }
    .brand-copy {
        min-width: 0;
    }
    .brand-title {
        margin: 0;
        font-size: 52px;
        line-height: 1.14;
        letter-spacing: 0;
        font-weight: 900;
        padding: 2px 0 4px;
        overflow-wrap: anywhere;
        color: #ffffff;
        background: linear-gradient(90deg, #7dd3fc 0%, #8b8cff 46%, #d8b4fe 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
        margin: 12px 0 0;
        color: #5e6b86;
        font-size: 15px;
        max-width: 760px;
    }
    .workbench {
        display: grid !important;
        grid-template-columns: minmax(0, 6fr) minmax(0, 5fr);
        gap: 16px !important;
        align-items: start;
        width: 100% !important;
        min-width: 0 !important;
    }
    .workbench > .panel,
    .input-pane,
    .output-pane {
        min-width: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    .panel {
        padding: 18px !important;
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        box-shadow: 0 18px 48px rgba(42, 62, 124, 0.10) !important;
        backdrop-filter: blur(18px);
        box-sizing: border-box !important;
    }
    .section-title {
        font-size: 13px;
        font-weight: 850;
        color: #344054;
        margin: 0 0 12px;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .input-config-card, .output-card {
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 12px !important;
        background: var(--field) !important;
        padding: 14px !important;
    }
    .input-media-card img,
    .result-media-card img,
    .input-media-card .image-container img,
    .result-media-card .image-container img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }
    .input-media-card .image-container,
    .result-media-card .image-container {
        background: #f8fafc !important;
    }
    .image-input-row {
        position: relative !important;
        align-items: stretch !important;
        gap: 0 !important;
        margin-bottom: 10px !important;
    }
    .image-input-row > .form {
        width: 100% !important;
    }
    .input-strip-card {
        width: 100% !important;
        height: 190px !important;
        min-height: 190px !important;
        background: #f8fafc !important;
        border-radius: 10px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-gutter: stable !important;
    }
    .input-image-board {
        width: 100% !important;
        height: 190px !important;
        min-height: 190px !important;
        border-radius: 12px !important;
        background: #f8fafc !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding: 18px 64px 18px 16px !important;
        box-sizing: border-box !important;
    }
    .input-thumb-row {
        width: max-content !important;
        min-width: 100% !important;
        height: 150px !important;
        gap: 12px !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    .input-empty-hint {
        height: 150px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #667085 !important;
        font-weight: 760 !important;
    }
    .input-thumb-slot {
        width: 170px !important;
        min-width: 170px !important;
        max-width: 170px !important;
        flex: 0 0 170px !important;
        position: relative !important;
    }
    .input-thumb-image:not(.fullscreen) {
        width: 170px !important;
        height: 140px !important;
        border: 0 !important;
        background: #ffffff !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .input-thumb-image.fullscreen {
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
        border-radius: 0 !important;
        background: #000000 !important;
        box-shadow: none !important;
        overflow: auto !important;
        z-index: 1000 !important;
    }
    .input-thumb-image.fullscreen .image-container,
    .input-thumb-image.fullscreen .image-container button,
    .input-thumb-image.fullscreen .image-frame {
        width: 100% !important;
        height: 100% !important;
    }
    .input-thumb-image.fullscreen img,
    .input-thumb-image.fullscreen .image-container img {
        width: auto !important;
        height: auto !important;
        max-width: 90vw !important;
        max-height: 90vh !important;
        object-fit: contain !important;
    }
    .input-thumb-image img,
    .input-thumb-image .image-container img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }
    .input-thumb-delete-btn,
    .input-thumb-delete-btn button {
        position: absolute !important;
        top: -8px !important;
        right: -8px !important;
        width: 24px !important;
        min-width: 24px !important;
        height: 24px !important;
        min-height: 24px !important;
        border-radius: 999px !important;
        padding: 0 !important;
        z-index: 3 !important;
        color: var(--accent) !important;
        background: #ffffff !important;
        border: 1px solid rgba(109, 93, 252, 0.24) !important;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12) !important;
        font-size: 16px !important;
        line-height: 1 !important;
    }
    .input-strip {
        height: 190px;
        display: flex;
        flex-wrap: nowrap;
        gap: 12px;
        align-items: center;
        overflow-x: auto;
        overflow-y: hidden;
        padding: 26px 64px 18px 16px;
        box-sizing: border-box;
        scrollbar-gutter: stable;
    }
    .input-strip.empty {
        justify-content: center;
        color: #667085;
        font-weight: 720;
    }
    .input-thumb {
        position: relative;
        flex: 0 0 170px;
        width: 170px;
        height: 140px;
        margin: 0;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10);
        overflow: visible;
    }
    .input-thumb img {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: contain;
        border-radius: 8px;
        background: #ffffff;
    }
    .append-image-upload {
        position: absolute !important;
        top: 12px !important;
        right: 12px !important;
        width: 44px !important;
        min-width: 44px !important;
        max-width: 44px !important;
        height: 44px !important;
        min-height: 44px !important;
        z-index: 8 !important;
        border: 0 !important;
        background: rgba(255, 255, 255, 0.96) !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12) !important;
        overflow: hidden !important;
        cursor: pointer !important;
        border-radius: 10px !important;
    }
    .append-image-upload::after {
        content: none !important;
    }
    .append-image-upload button {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        padding: 0 !important;
        border: 0 !important;
        background: #ffffff !important;
        color: var(--accent) !important;
        font-size: 26px !important;
        line-height: 1 !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.80) !important;
        cursor: pointer !important;
    }
    .append-image-upload button::after {
        content: none !important;
        font-size: 30px;
        line-height: 1;
        font-weight: 400;
    }
    .append-image-upload .wrap,
    .append-image-upload [data-testid="file"],
    .append-image-upload .upload-container,
    .append-image-upload [class*="upload"],
    .append-image-upload [class*="drop"] {
        width: 44px !important;
        min-width: 44px !important;
        max-width: 44px !important;
        height: 44px !important;
        min-height: 44px !important;
        height: 100% !important;
        border: 0 !important;
        background: rgba(255, 255, 255, 0) !important;
        box-shadow: none !important;
        cursor: pointer !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: var(--accent) !important;
        font-size: inherit !important;
        line-height: 1 !important;
    }
    .append-image-upload input[type="file"] {
        cursor: pointer !important;
    }
    .append-image-upload .icon-wrap,
    .append-image-upload svg,
    .append-image-upload span,
    .append-image-upload p {
        display: none !important;
    }
    .append-image-upload [data-testid="file"]::after,
    .append-image-upload .upload-container::after,
    .append-image-upload [class*="drop"]::after {
        content: none !important;
        color: var(--accent);
        font-size: 30px;
        line-height: 1;
        font-weight: 400;
    }
    .current-sample {
        display: flex;
        align-items: flex-start;
        gap: 9px;
        border: 1px solid rgba(37, 99, 235, 0.16);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.96), rgba(245, 243, 255, 0.92));
        padding: 11px 13px;
        color: #24324f;
        font-size: 13px;
        margin-bottom: 10px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
    }
    .current-sample-dot {
        width: 9px;
        height: 9px;
        margin-top: 6px;
        border-radius: 999px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.10);
        flex: 0 0 auto;
    }
    .current-sample-copy {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
        min-width: 0;
    }
    .current-sample-main {
        display: flex;
        align-items: baseline;
        gap: 7px;
        min-width: 0;
        flex-wrap: wrap;
    }
    .current-sample-label {
        color: #667085;
        font-weight: 760;
    }
    .current-sample-value {
        color: #172033;
        font-weight: 860;
    }
    .current-sample-class {
        color: #475467;
        border: 1px solid rgba(100, 116, 139, 0.18);
        background: rgba(255, 255, 255, 0.62);
        border-radius: 999px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 760;
    }
    .current-sample-output {
        color: #475467;
        font-size: 12px;
        font-weight: 760;
        line-height: 1.25;
    }
    .task-pills .wrap {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
        column-gap: 12px !important;
        row-gap: 12px !important;
        align-items: start !important;
        justify-content: stretch !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding-left: 124px !important;
    }
    .task-pills label {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 0 !important;
        width: 100% !important;
        border-radius: 999px !important;
        border: 1px solid rgba(109, 93, 252, 0.18) !important;
        background: #fff !important;
        color: #26324b !important;
        padding: 7px 12px !important;
        font-size: 12px !important;
        font-weight: 750 !important;
        line-height: 1.2 !important;
        box-shadow: none !important;
    }
    .task-pills label:nth-of-type(5),
    .task-pills label:nth-of-type(8),
    .task-pills label:nth-of-type(10),
    .task-pills label:nth-of-type(12) {
        grid-column: 1 !important;
    }
    .task-pills label:nth-of-type(n + 5) {
        margin-top: 14px !important;
    }
    .task-pills label:nth-of-type(1)::before,
    .task-pills label:nth-of-type(5)::before,
    .task-pills label:nth-of-type(8)::before,
    .task-pills label:nth-of-type(10)::before,
    .task-pills label:nth-of-type(12)::before {
        position: absolute !important;
        left: -124px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 114px !important;
        color: #64748b !important;
        font-size: 11px !important;
        font-weight: 820 !important;
        line-height: 1 !important;
        text-align: right !important;
        letter-spacing: 0 !important;
        pointer-events: none !important;
    }
    .task-pills label:nth-of-type(1)::before {
        content: "Detection" !important;
    }
    .task-pills label:nth-of-type(5)::before {
        content: "Segmentation" !important;
    }
    .task-pills label:nth-of-type(8)::before {
        content: "Geometry" !important;
    }
    .task-pills label:nth-of-type(10)::before {
        content: "Multi View" !important;
    }
    .task-pills label:nth-of-type(12)::before {
        content: "General" !important;
    }
    .task-pills label:nth-of-type(5)::after,
    .task-pills label:nth-of-type(8)::after,
    .task-pills label:nth-of-type(10)::after,
    .task-pills label:nth-of-type(12)::after {
        content: "" !important;
        position: absolute !important;
        left: -124px !important;
        top: -12px !important;
        width: calc(400% + 160px) !important;
        height: 1px !important;
        background: rgba(148, 163, 184, 0.22) !important;
        pointer-events: none !important;
    }
    .task-pills input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    .task-pills label span {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        color: inherit !important;
        min-width: 0 !important;
    }
    .task-pills label span::before {
        content: "" !important;
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        border-radius: 999px !important;
        border: 1px solid #cfe0ff !important;
        background: #ffffff !important;
        box-shadow: inset 0 0 0 3px #ffffff !important;
    }
    .task-pills label:has(input:checked),
    .task-pills label.selected {
        background: linear-gradient(135deg, #2563eb 0%, #6d5dfc 56%, #9333ea 100%) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(76, 93, 252, 0.22) !important;
    }
    .task-pills label:has(input:checked) span,
    .task-pills label.selected span {
        color: #ffffff !important;
        font-weight: 850 !important;
    }
    .task-pills label:has(input:checked) span::before,
    .task-pills label.selected span::before {
        border-color: rgba(255, 255, 255, 0.72) !important;
        background: radial-gradient(circle at center, #ffffff 0 3px, rgba(255, 255, 255, 0.18) 3.5px) !important;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12) !important;
    }
    .task-pills input[type="radio"]:checked + span,
    .task-pills input[type="radio"]:checked ~ span {
        color: #ffffff !important;
        font-weight: 850 !important;
    }
    .run-btn, .run-btn button {
        min-height: 42px !important;
        border-radius: 10px !important;
        font-weight: 850 !important;
        background: linear-gradient(135deg, #2563eb 0%, #6d5dfc 58%, #9333ea 100%) !important;
        border: 0 !important;
    }
    .sample-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    .sample-more {
        color: #93c5fd;
        font-size: 13px;
        font-weight: 750;
    }
    .sample-gallery {
        border: 0 !important;
        background: transparent !important;
        height: auto !important;
        min-height: 340px !important;
        overflow: visible !important;
    }
    .sample-gallery .grid-container,
    .sample-gallery .grid-wrap,
    .sample-gallery [class*="grid-container"],
    .sample-gallery [class*="grid-wrap"] {
        height: auto !important;
        min-height: 320px !important;
        max-height: none !important;
        overflow-y: hidden !important;
        overflow-x: hidden !important;
        scrollbar-width: none !important;
    }
    .sample-gallery .grid-wrap::-webkit-scrollbar,
    .sample-gallery [class*="grid-wrap"]::-webkit-scrollbar {
        display: none !important;
    }
    .sample-gallery .grid-container,
    .sample-gallery [class*="grid-container"] {
        grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
        grid-template-rows: repeat(2, minmax(150px, 150px)) !important;
        grid-auto-rows: minmax(150px, 150px) !important;
    }
    .sample-gallery .thumbnail-lg,
    .sample-gallery [class*="thumbnail"] {
        min-height: 150px !important;
    }
    .sample-gallery img {
        border-radius: 10px !important;
        object-fit: cover !important;
    }
    .sample-gallery .caption,
    .sample-gallery .thumbnail-item-caption,
    .sample-gallery .caption-label,
    .sample-gallery figcaption {
        left: auto !important;
        right: 8px !important;
        bottom: 8px !important;
        width: auto !important;
        max-width: calc(100% - 16px) !important;
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.32) !important;
        background: rgba(255, 255, 255, 0.92) !important;
        color: #172033 !important;
        padding: 4px 9px !important;
        font-size: 11px !important;
        line-height: 1.2 !important;
        font-weight: 780 !important;
        letter-spacing: 0 !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.16) !important;
        backdrop-filter: blur(8px);
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        opacity: 1 !important;
    }
    .sample-gallery .caption *,
    .sample-gallery .thumbnail-item-caption *,
    .sample-gallery .caption-label *,
    .sample-gallery figcaption * {
        color: #172033 !important;
    }
    .sample-gallery .thumbnail-lg:hover .caption-label {
        opacity: 1 !important;
    }
    .output-grid { gap: 14px !important; }
    .error-toast {
        position: fixed !important;
        top: 18px !important;
        right: 18px !important;
        z-index: 10000 !important;
        width: min(430px, calc(100vw - 36px)) !important;
        border: 1px solid rgba(220, 38, 38, 0.22) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.98) !important;
        box-shadow: 0 18px 44px rgba(127, 29, 29, 0.18) !important;
        padding: 12px 14px 14px !important;
        box-sizing: border-box !important;
    }
    .error-toast-header {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) max-content !important;
        align-items: center !important;
        gap: 10px !important;
        margin-bottom: 8px !important;
    }
    .error-toast-title {
        color: #991b1b !important;
        font-size: 13px !important;
        font-weight: 880 !important;
        line-height: 1.2 !important;
    }
    .error-toast-close,
    .error-toast-close button {
        width: 28px !important;
        min-width: 28px !important;
        height: 28px !important;
        min-height: 28px !important;
        border-radius: 999px !important;
        padding: 0 !important;
        border: 1px solid rgba(220, 38, 38, 0.18) !important;
        background: #fff7f7 !important;
        color: #991b1b !important;
        font-size: 18px !important;
        font-weight: 760 !important;
        line-height: 1 !important;
        box-shadow: none !important;
    }
    .error-toast-message {
        color: #7f1d1d !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
        overflow-wrap: anywhere !important;
    }
    .error-toast-message .error-toast-type {
        display: inline-flex !important;
        width: fit-content !important;
        margin-bottom: 6px !important;
        border-radius: 999px !important;
        background: #fee2e2 !important;
        color: #991b1b !important;
        padding: 3px 8px !important;
        font-size: 11px !important;
        font-weight: 860 !important;
    }
    .error-toast-message .error-toast-copy {
        color: #7f1d1d !important;
        font-weight: 680 !important;
        white-space: pre-line !important;
    }
    .result-media-card .wrap::before,
    .result-media-card .wrap::after,
    .result-media-card .progress-text,
    .result-media-card [class*="progress-text"],
    .result-media-card [class*="progress"] {
        opacity: 1 !important;
    }
    .result-media-card .progress-text,
    .result-media-card [class*="progress-text"] {
        right: 18px !important;
        bottom: 14px !important;
        color: #4f46e5 !important;
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(79, 70, 229, 0.18) !important;
        border-radius: 999px !important;
        padding: 4px 10px !important;
        font-weight: 850 !important;
        letter-spacing: 0 !important;
        text-shadow: none !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.14) !important;
    }
    .result-media-card .progress-bar,
    .result-media-card [class*="progress-bar"] {
        background: linear-gradient(90deg, #2563eb, #7c3aed, #06b6d4) !important;
        opacity: 1 !important;
        height: 4px !important;
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.36) !important;
    }
    .result-media-card .generating,
    .result-media-card [class*="generating"],
    .result-media-card .loading,
    .result-media-card [class*="loading"] {
        color: #4f46e5 !important;
        opacity: 1 !important;
    }
    .result-media-card svg {
        filter: drop-shadow(0 0 6px rgba(79, 70, 229, 0.45));
    }
    .raw-output-download {
        min-height: 0 !important;
        color-scheme: light !important;
        --body-background-fill: #ffffff !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --block-background-fill: #ffffff !important;
        --input-background-fill: #ffffff !important;
        --body-text-color: #172033 !important;
        --button-secondary-background-fill: #ffffff !important;
        --button-secondary-text-color: #5b5ff4 !important;
    }
    .raw-output-download [data-testid="file"],
    .raw-output-download .file-preview,
    .raw-output-download .wrap {
        max-height: 96px !important;
    }
    .raw-output-download [data-testid="file"],
    .raw-output-download .file-preview,
    .raw-output-download .wrap,
    .raw-output-download ul,
    .raw-output-download li,
    .raw-output-download a {
        background: #ffffff !important;
        color: #172033 !important;
        border-color: transparent !important;
    }
    .raw-output-download [data-testid="file"] *,
    .raw-output-download .file-preview *,
    .raw-output-download li *,
    .raw-output-download a * {
        background: transparent !important;
        background-color: transparent !important;
        color: #172033 !important;
    }
    .raw-output-download a,
    .raw-output-download button,
    .raw-output-download [role="button"] {
        color: #5b5ff4 !important;
    }
    .raw-output-download svg {
        color: #5b5ff4 !important;
        stroke: #5b5ff4 !important;
    }
    .raw-output-download [class*="upload"],
    .raw-output-download [class*="drop"] {
        min-height: 42px !important;
    }
    .raw-output-download label,
    .raw-output-download .block-info {
        max-height: none !important;
        min-height: 18px !important;
        margin-bottom: 4px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .raw-output-inline-preview {
        margin-top: 10px !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .raw-output-inline-preview .grid-container,
    .raw-output-inline-preview .grid-wrap,
    .raw-output-inline-preview [class*="grid-container"],
    .raw-output-inline-preview [class*="grid-wrap"] {
        max-height: 260px !important;
        min-height: 0 !important;
        overflow-y: auto !important;
    }
    .raw-output-inline-preview img {
        object-fit: contain !important;
        background: #f8fafc !important;
    }
    .metadata-box textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
        font-size: 12px !important;
    }
    .base-param-panel {
        border: 0 !important;
        background: transparent !important;
        padding: 0 !important;
    }
    .base-param-grid {
        gap: 14px !important;
        margin-bottom: 10px !important;
    }
    .base-param-grid > div {
        min-width: 0 !important;
    }
    .base-param-grid label > span,
    .base-param-grid .block-info {
        display: inline-flex !important;
        width: fit-content !important;
        border-radius: 6px !important;
        background: #dfe5ff !important;
        color: #6366f1 !important;
        padding: 4px 7px !important;
        font-size: 13px !important;
        font-weight: 820 !important;
        letter-spacing: 0 !important;
    }
    @media (max-width: 1180px) {
        .workbench {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 900px) {
        .main-shell {
            width: 100% !important;
            max-width: 100vw !important;
            padding-inline: 10px !important;
            margin-top: 6px;
        }
        .brand-title { font-size: clamp(28px, 8vw, 34px); }
        .top-nav {
            align-items: flex-start;
            grid-template-columns: 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }
        .brand-row {
            align-items: flex-start;
            padding-top: 0;
            padding-right: 0;
        }
        .task-pills .wrap {
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        }
        .sample-gallery .grid-container,
        .sample-gallery [class*="grid-container"] {
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        }
    }
    @media (max-width: 560px) {
        .main-shell {
            width: 100% !important;
            max-width: 100vw !important;
            padding-inline: 8px !important;
            margin-top: 4px;
        }
        .app-header {
            padding: 0 0 12px;
        }
        .nav-chip {
            padding: 7px 8px;
            font-size: 11px;
        }
        .brand-row {
            flex-direction: column;
            gap: 10px;
        }
        .brand-copy {
            width: 100%;
        }
        .brand-title {
            font-size: clamp(30px, 10vw, 38px);
            line-height: 1.12;
        }
        .brand-subtitle {
            margin-top: 6px;
            font-size: 13px;
        }
        .task-pills .wrap {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            padding-left: 0 !important;
        }
        .task-pills label:nth-of-type(1),
        .task-pills label:nth-of-type(5),
        .task-pills label:nth-of-type(8),
        .task-pills label:nth-of-type(10),
        .task-pills label:nth-of-type(12) {
            margin-left: 0 !important;
            margin-top: 18px !important;
        }
        .task-pills label:nth-of-type(1)::before,
        .task-pills label:nth-of-type(5)::before,
        .task-pills label:nth-of-type(8)::before,
        .task-pills label:nth-of-type(10)::before,
        .task-pills label:nth-of-type(12)::before {
            left: 0 !important;
            right: auto !important;
            top: -13px !important;
            transform: none !important;
            width: auto !important;
            text-align: left !important;
        }
        .task-pills label:nth-of-type(5)::after,
        .task-pills label:nth-of-type(8)::after,
        .task-pills label:nth-of-type(10)::after,
        .task-pills label:nth-of-type(12)::after {
            left: 0 !important;
            width: calc(200% + 8px) !important;
        }
        .sample-gallery .grid-container,
        .sample-gallery [class*="grid-container"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        }
    }
    """
    initial_case: Dict[str, Any] = {}
    initial_action = "Open understanding"
    initial_task = "raw_query"
    initial_mode = "understanding"
    initial_images: List[str] = []
    initial_prompt = ""
    initial_params = base_params_for_action(initial_action)
    initial_param_values = _control_values_from_params(initial_params)
    initial_param_visible = set(initial_params)
    if "cfg_interval" in initial_params:
        initial_param_visible.update({"cfg_interval_start", "cfg_interval_end"})
    logo_src = _image_data_uri(LOG_IMAGE_PATH)
    logo_html = (
        f'<img class="nav-logo" src="{logo_src}" alt="SenseNova-Vision logo" />'
        if logo_src else "SV"
    )

    with gr.Blocks(title=APP_TITLE, css=css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["main-shell"]):
            gr.HTML(f"""
            <div class="app-header">
                <div class="top-nav">
                    <div class="nav-brand"><span class="nav-logo-mark">{logo_html}</span></div>
                    <div class="nav-links">
                        <a class="nav-chip" href="https://github.com/OpenSenseNova/SenseNova-Vision" target="_blank" rel="noopener noreferrer">Project</a>
                        <a class="nav-chip" href="https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT" target="_blank" rel="noopener noreferrer">Models</a>
                        <a class="nav-chip" href="https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M" target="_blank" rel="noopener noreferrer">Dataset</a>
                    </div>
                </div>
                <div class="brand-row">
                    <div class="brand-copy">
                        <h1 class="brand-title">SenseNova-Vision</h1>
                        <p class="brand-subtitle">Vision as Unified Multimodal Generation</p>
                    </div>
                </div>
            </div>
            """)

            with gr.Row(elem_classes=["workbench"]):
                with gr.Column(scale=6, elem_classes=["panel", "input-pane"]):
                    gr.HTML('<div class="section-title">Input</div>')
                    with gr.Group(elem_classes=["input-config-card"]):
                        task_action = gr.Radio(
                            label="",
                            show_label=False,
                            choices=TASK_ACTION_CHOICES,
                            value=initial_action,
                            elem_classes=["task-pills"],
                        )
                        current_sample = gr.HTML(current_sample_html())
                        input_image_paths = gr.Gallery(
                            value=initial_images,
                            type="filepath",
                            file_types=["image", ".webp"],
                            visible=False,
                        )
                        input_slot_outputs = []
                        delete_buttons = []
                        with gr.Row(elem_classes=["image-input-row"]):
                            with gr.Column(elem_classes=["input-image-board"]):
                                empty_hint = gr.HTML(
                                    '<div class="input-empty-hint">Use + to upload images</div>',
                                    visible=True,
                                )
                                input_slot_outputs.append(empty_hint)
                                with gr.Row(elem_classes=["input-thumb-row"]):
                                    for slot_index in range(MAX_INPUT_IMAGES):
                                        with gr.Column(visible=False, elem_classes=["input-thumb-slot"]) as slot:
                                            slot_image = gr.Image(
                                                label="",
                                                show_label=False,
                                                type="filepath",
                                                interactive=False,
                                                visible=False,
                                                height=140,
                                                elem_classes=["input-thumb-image"],
                                            )
                                            delete_button = gr.Button(
                                                "×",
                                                visible=False,
                                                elem_classes=["input-thumb-delete-btn"],
                                            )
                                        input_slot_outputs.extend([slot, slot_image, delete_button])
                                        delete_buttons.append((delete_button, slot_index))
                            upload_component = getattr(gr, "UploadButton", None)
                            if upload_component is not None:
                                append_images = upload_component(
                                    label="+",
                                    file_count="multiple",
                                    file_types=["image", ".webp"],
                                    type="filepath",
                                    elem_classes=["append-image-upload"],
                                )
                            else:
                                append_images = gr.File(
                                    label="+",
                                    show_label=False,
                                    file_count="multiple",
                                    file_types=["image", ".webp"],
                                    type="filepath",
                                    elem_classes=["append-image-upload"],
                                )
                        query = gr.Textbox(
                            label="Prompt / Query",
                            lines=4,
                            max_lines=6,
                            value=initial_prompt,
                            placeholder=initial_prompt,
                            elem_classes=["prompt-box"],
                        )
                        task = gr.Textbox(value=initial_task, visible=False)
                        mode = gr.Textbox(value=initial_mode, visible=False)
                        with gr.Accordion("Advanced params (Optional)", open=False):
                            with gr.Group(elem_classes=["base-param-panel"]):
                                with gr.Row(elem_classes=["base-param-grid"]):
                                    cfg_text_scale = gr.Number(
                                        label="cfg_text_scale",
                                        value=initial_param_values["cfg_text_scale"],
                                        visible="cfg_text_scale" in initial_param_visible,
                                    )
                                    cfg_img_scale = gr.Number(
                                        label="cfg_img_scale",
                                        value=initial_param_values["cfg_img_scale"],
                                        visible="cfg_img_scale" in initial_param_visible,
                                    )
                                with gr.Row(elem_classes=["base-param-grid"]):
                                    cfg_interval_start = gr.Number(
                                        label="cfg_interval start",
                                        value=initial_param_values["cfg_interval_start"],
                                        visible="cfg_interval_start" in initial_param_visible,
                                    )
                                    cfg_interval_end = gr.Number(
                                        label="cfg_interval end",
                                        value=initial_param_values["cfg_interval_end"],
                                        visible="cfg_interval_end" in initial_param_visible,
                                    )
                                with gr.Row(elem_classes=["base-param-grid"]):
                                    timestep_shift = gr.Number(
                                        label="timestep_shift",
                                        value=initial_param_values["timestep_shift"],
                                        visible="timestep_shift" in initial_param_visible,
                                    )
                                    num_timesteps = gr.Number(
                                        label="num_timesteps",
                                        value=initial_param_values["num_timesteps"],
                                        precision=0,
                                        visible="num_timesteps" in initial_param_visible,
                                    )
                                with gr.Row(elem_classes=["base-param-grid"]):
                                    cfg_renorm_min = gr.Number(
                                        label="cfg_renorm_min",
                                        value=initial_param_values["cfg_renorm_min"],
                                        visible="cfg_renorm_min" in initial_param_visible,
                                    )
                                    max_length_token = gr.Number(
                                        label="max_length_token",
                                        value=initial_param_values["max_length_token"],
                                        precision=0,
                                        visible="max_length_token" in initial_param_visible,
                                    )
                                with gr.Row(elem_classes=["base-param-grid"]):
                                    seed = gr.Number(
                                        label="seed",
                                        value=initial_param_values["seed"],
                                        precision=0,
                                        visible="seed" in initial_param_visible,
                                    )
                            base_param_outputs = [
                                cfg_text_scale,
                                cfg_img_scale,
                                cfg_interval_start,
                                cfg_interval_end,
                                timestep_shift,
                                num_timesteps,
                                cfg_renorm_min,
                                max_length_token,
                                seed,
                            ]
                        run_btn = gr.Button("Run", variant="primary", elem_classes=["run-btn"])

                with gr.Column(scale=5, elem_classes=["panel", "output-pane"]):
                    gr.HTML('<div class="section-title">Output</div>')
                    with gr.Column(elem_classes=["output-grid"]):
                        if Version(gr.__version__) >= Version("6"):
                            textbox_copy_kwargs = {"buttons": ["copy"]}
                        else:
                            textbox_copy_kwargs = {"show_copy_button": True}
                        with gr.Column(elem_classes=["output-card"]):
                            result_image = gr.Image(
                                label="Visualization",
                                type="pil",
                                height=360,
                                elem_classes=["result-media-card"],
                            )
                            result_model3d = gr.Model3D(
                                label="3D Reconstruction Preview",
                                height=360,
                                visible=False,
                            )
                        with gr.Column(elem_classes=["output-card"]):
                            result_text = gr.Textbox(
                                label="Text Output",
                                lines=18,
                                interactive=False,
                                **textbox_copy_kwargs,
                            )
                            result_files = gr.File(
                                label="Download raw output",
                                file_count="multiple",
                                type="filepath",
                                visible=True,
                                elem_classes=["raw-output-download"],
                            )
                            result_raw_images = gr.Gallery(
                                label="",
                                show_label=False,
                                columns=2,
                                rows=1,
                                object_fit="contain",
                                visible=False,
                                elem_classes=["raw-image-output", "raw-output-inline-preview"],
                            )
                            metadata = gr.Textbox(
                                label="Metadata",
                                lines=8,
                                interactive=False,
                                visible=False,
                                elem_classes=["metadata-box"],
                                **textbox_copy_kwargs,
                            )

            with gr.Column(elem_classes=["panel", "sample-panel"]):
                gr.HTML("""
                <div class="sample-title-row">
                    <div class="section-title">Sample Library</div>
                </div>
                """)
                demo_gallery = gr.Gallery(
                    label="",
                    value=_demo_gallery_value(),
                    columns=5,
                    rows=2,
                    object_fit="cover",
                    allow_preview=False,
                    show_label=False,
                    elem_classes=["sample-gallery"],
                )

            demo_select_event = demo_gallery.select(
                fn=load_demo_case_from_gallery,
                inputs=[],
                outputs=[
                    input_image_paths,
                    *input_slot_outputs,
                    task_action,
                    task,
                    mode,
                    query,
                    *base_param_outputs,
                    result_image,
                    result_text,
                    result_files,
                    result_model3d,
                    metadata,
                    current_sample,
                ],
            )
            demo_select_event.then(
                fn=clear_raw_image_output,
                inputs=[],
                outputs=result_raw_images,
            )
            task_change_event = task_action.change(
                fn=load_task_action_sample,
                inputs=task_action,
                outputs=[
                    input_image_paths,
                    *input_slot_outputs,
                    task_action,
                    task,
                    mode,
                    query,
                    *base_param_outputs,
                    result_image,
                    result_text,
                    result_files,
                    result_model3d,
                    metadata,
                    current_sample,
                ],
            )
            task_change_event.then(
                fn=clear_raw_image_output,
                inputs=[],
                outputs=result_raw_images,
            )
            upload_event = append_images.upload if hasattr(append_images, "upload") else append_images.change
            upload_event(
                fn=append_input_images,
                inputs=[input_image_paths, append_images],
                outputs=[input_image_paths, *input_slot_outputs, append_images],
            )
            for param_control in base_param_outputs:
                param_control.change(
                    fn=validate_params_for_error,
                    inputs=[task_action, *base_param_outputs],
                )
            for delete_button, delete_index in delete_buttons:
                delete_button.click(
                    fn=lambda paths, idx=delete_index: delete_input_image(paths, idx),
                    inputs=input_image_paths,
                    outputs=[input_image_paths, *input_slot_outputs],
                )
            run_event = run_btn.click(
                fn=run_with_validation,
                inputs=[
                    input_image_paths,
                    task_action,
                    task,
                    mode,
                    query,
                    *base_param_outputs,
                ],
                outputs=[
                    result_image,
                    result_text,
                    result_files,
                    result_model3d,
                    metadata,
                ],
                api_name="predict",
            )
            run_event.success(
                fn=update_raw_output_display,
                inputs=metadata,
                outputs=[result_files, result_raw_images],
            )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseNova-Vision Gradio demo.")
    parser.add_argument("--host", type=str, default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "9001")))
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes", "on"},
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[DEMO_EXAMPLE_ROOT, DEMO_INFERENCE_EXAMPLE_ROOT],
    )


if __name__ == "__main__":
    main()
