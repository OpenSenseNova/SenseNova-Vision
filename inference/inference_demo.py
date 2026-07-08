import argparse
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data.prompts import (
    CAPTION_QUESTION_LIST,
    COCO_CATEGORIES,
    GCG_QUESTION_LIST,
    MASK_QUESTION_LIST,
)
from inference.sensenova_vision import (
    BASE_PARAMS,
    CAMERA_POSE_PROMPT,
    RECON3D_PROMPT,
    SenseNovaVisionModel,
    set_seed,
)
from inference.utils_3d import resolve_pose_string
from utils.visualize import (
    VisualizationConfig,
    load_font,
    visualize_binary_segmentation,
    visualize_concat_col,
    visualize_detection,
    visualize_gcg_segmentation,
    visualize_panoptic_segmentation,
)

try:
    import readline
except ImportError:
    readline = None


DEFAULT_MODEL_PATH = "sensenova/SenseNova-Vision-7B-MoT"
DEFAULT_OUTPUT_DIR = "examples/output/demo/"
DEPTH_PROMPT = (
    "Estimate relative depth for each pixel in the image, with closer objects "
    "appearing brighter and distant objects appearing darker. Output is a "
    "grayscale image with pixel values ranging from 0-255."
)
NORMAL_PROMPT = (
    "Estimate surface normals and encode as an RGB image. Each channel "
    "corresponds to a direction component (X, Y, Z) with continuous value "
    "variations, creating smooth color gradients distinct from other task outputs."
)
OCR_PROMPT = (
    "Perform word-level text detection and recognition on the entire image. "
    "Output a structured text list containing every detected word, its bounding "
    "box coordinates with <bbox> format, and the recognized text content."
)
BBOX_DETECTION_PROMPT_TEMPLATE = (
    "Detect all instances of {categories} in the image. Output the results "
    "as a structured text list with each detection including category and "
    "bounding box coordinates in <bbox> format."
)
POINT_DETECTION_PROMPT_TEMPLATE = (
    "Locate and identify {categories} within the scene. Output detection "
    "results as text entries, each containing the object class and pixel "
    "coordinates defining the object point location."
)
KEYPOINT_PROMPT_TEMPLATE = {
    "human": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "nose, left eye, right eye, left ear, right ear, left shoulder, right "
        "shoulder, left elbow, right elbow, left wrist, right wrist, left hip, "
        "right hip, left knee, right knee, left ankle, right ankle in "
        "<kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "animal": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "left eye, right eye, nose, neck, root of tail, left shoulder, left "
        "elbow, left front paw, right shoulder, right elbow, right front paw, "
        "left hip, left knee, left back paw, right hip, right knee, right back "
        "paw in <kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "mixed": (
        "Detect all instances of {categories} in the image. For human "
        "instances, output a bounding box in <bbox> format and the COCO human "
        "keypoints: nose, left eye, right eye, left ear, right ear, left "
        "shoulder, right shoulder, left elbow, right elbow, left wrist, right "
        "wrist, left hip, right hip, left knee, right knee, left ankle, right "
        "ankle. For animal instances, output a bounding box in <bbox> format "
        "and the AP-10K animal keypoints: left eye, right eye, nose, neck, "
        "root of tail, left shoulder, left elbow, left front paw, right "
        "shoulder, right elbow, right front paw, left hip, left knee, left back "
        "paw, right hip, right knee, right back paw. Use <kpt>[x,y]</kpt> "
        "format for every keypoint and return results as a structured list."
    ),
}
HUMAN_KEYPOINT_CATEGORIES = {"human", "person", "people", "pedestrian"}


TaskPromptBuilder = Callable[[str], str]
NO_QUERY_TASKS = {"depth", "normal", "gcg_seg", "ocr", "recon3d", "camera_pose"}
DETECTION_VIS_TASKS = {"bbox_detection", "point_detection", "keypoint", "ocr"}


@contextmanager
def timer_context(name: str):
    start = time.time()
    try:
        yield
    finally:
        print(f"[Time] {name}: {time.time() - start:.2f}s")


# Prompt builders

def _fixed_prompt(prompt: str) -> TaskPromptBuilder:
    def build(_: str) -> str:
        return prompt

    return build


def _category_prompt(template: str) -> TaskPromptBuilder:
    def build(query: str) -> str:
        return template.format(categories=_format_categories(query))

    return build


def _keypoint_prompt(query: str) -> str:
    categories = _format_categories(query)
    category_names = _extract_category_names(categories)
    has_human = any(name in HUMAN_KEYPOINT_CATEGORIES for name in category_names)
    has_animal = any(name not in HUMAN_KEYPOINT_CATEGORIES for name in category_names)
    if has_human and has_animal:
        keypoint_type = "mixed"
    elif has_human:
        keypoint_type = "human"
    else:
        keypoint_type = "animal"
    return KEYPOINT_PROMPT_TEMPLATE[keypoint_type].format(categories=categories)


def _prompt_category_name(name: str) -> str:
    for suffix in ("-stuff", "-other", "-merged"):
        name = name.replace(suffix, "")
    return name


def _extract_category_names(categories: str) -> List[str]:
    names = []
    for match in re.finditer(r"<p>(.*?)</p>", categories, flags=re.IGNORECASE | re.DOTALL):
        name = " ".join(match.group(1).split()).lower()
        if name:
            names.append(name)
    return names


def _format_categories(query: str) -> str:
    text = query.strip()
    if not text:
        raise ValueError("category tasks require explicit categories in --query.")
    if "<p>" in text and "</p>" in text:
        return text
    categories = [item.strip() for item in text.split(",") if item.strip()]
    return ", ".join(f"<p>{category}</p>" for category in categories)


def _format_coco_categories() -> str:
    return ", ".join(
        f"<p>{_prompt_category_name(category['name'])}</p>"
        for category in COCO_CATEGORIES
    )


def _binary_seg_prompt(query: str) -> str:
    return MASK_QUESTION_LIST[0].format(
        categories=_format_categories(query),
        task_type="binary",
    )


def _panoptic_prompt(query: str) -> str:
    if query.strip():
        categories = _format_categories(query)
    else:
        categories = _format_coco_categories()
    category_prompt = MASK_QUESTION_LIST[0].format(categories=categories, task_type="panoptic")
    caption_prompt = CAPTION_QUESTION_LIST[0].format(task_type="panoptic")
    return f"{category_prompt} {caption_prompt}"


# Task routing

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


TASK_TO_PROMPT_BUILDER: Dict[str, TaskPromptBuilder] = {
    "raw_query": str.strip,
    "depth": _fixed_prompt(DEPTH_PROMPT),
    "normal": _fixed_prompt(NORMAL_PROMPT),
    "binary_seg": _binary_seg_prompt,
    "pan_seg": _panoptic_prompt,
    "gcg_seg": _fixed_prompt(GCG_QUESTION_LIST[0]),
    "bbox_detection": _category_prompt(BBOX_DETECTION_PROMPT_TEMPLATE),
    "point_detection": _category_prompt(POINT_DETECTION_PROMPT_TEMPLATE),
    "keypoint": _keypoint_prompt,
    "ocr": _fixed_prompt(OCR_PROMPT),
    "recon3d": _fixed_prompt(RECON3D_PROMPT),
    "camera_pose": _fixed_prompt(CAMERA_POSE_PROMPT),
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
    "recon3d": (
        "multi-view 3D reconstruction, multi-view images needed, "
        "no further query needed"
    ),
    "camera_pose": (
        "relative camera pose estimation, multi-view images needed, "
        "no further query needed"
    ),
}


assert set(TASK_ORDER) == set(TASK_TO_PROMPT_BUILDER) == set(TASK_TO_MODE) == set(TASK_HELP)


# Request parsing and prompt resolution

@dataclass
class InferenceRequest:
    image_paths: List[str]
    task: str
    mode: str
    query: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputContext:
    task_dir: str
    vis_dir: str
    save_prefix: str


def resolve_path(path_text: str) -> str:
    path = os.path.expanduser(path_text.strip().strip("'\""))
    if not path:
        raise ValueError("Empty image path.")
    if os.path.isabs(path):
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            return resolved
        raise FileNotFoundError(f"Image not found: {resolved}")

    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Image not found: {os.path.abspath(path)}")


def parse_image_paths_arg(image_path_arg: str) -> List[str]:
    image_paths = [item.strip() for item in image_path_arg.split(",") if item.strip()]
    if not image_paths:
        image_paths = [image_path_arg]
    return [resolve_path(path) for path in image_paths]


def build_prompt(task: str, query: str) -> str:
    if task not in TASK_TO_PROMPT_BUILDER:
        raise ValueError(
            f"Unsupported task: {task}. Available tasks: {', '.join(TASK_TO_PROMPT_BUILDER)}"
        )
    prompt = TASK_TO_PROMPT_BUILDER[task](query)
    if not prompt:
        raise ValueError("query must be non-empty when --task raw_query is used.")
    return prompt


def resolve_task_mode(task: str) -> str:
    if task not in TASK_TO_MODE:
        raise ValueError(
            f"Unsupported task: {task}. Available tasks: {', '.join(TASK_TO_MODE)}"
        )
    return TASK_TO_MODE[task]


def resolve_request_mode(task: str, requested_mode: str) -> str:
    if task != "raw_query":
        return resolve_task_mode(task)
    mode = requested_mode.strip() or resolve_task_mode(task)
    if mode not in BASE_PARAMS:
        raise ValueError(f"Unsupported mode: {mode}. Available modes: {', '.join(BASE_PARAMS)}")
    return mode


# Output helpers

def make_save_prefix(image_path: str, task: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_") or "image"
    safe_task = re.sub(r"[^0-9A-Za-z._-]+", "_", task).strip("_") or "task"
    return f"{safe_stem}_{safe_task}_{time.strftime('%Y%m%d_%H%M%S')}"


def ensure_output_dirs(output_dir: str, task: str) -> Tuple[str, str]:
    task_dir = os.path.join(output_dir, task)
    vis_dir = os.path.join(task_dir, "vis")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    return task_dir, vis_dir


def make_output_context(output_dir: str, request: InferenceRequest) -> OutputContext:
    task_dir, vis_dir = ensure_output_dirs(output_dir, request.task)
    save_prefix = make_save_prefix(request.image_paths[0], request.task)
    if len(request.image_paths) > 1:
        save_prefix = f"{save_prefix}_{len(request.image_paths)}img"
    return OutputContext(task_dir=task_dir, vis_dir=vis_dir, save_prefix=save_prefix)


def model_question(prompt: str, image_count: int) -> str:
    if image_count <= 0:
        return prompt
    return f'{("<image>" * image_count)} {prompt}'.strip()


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: Optional[int] = None,
) -> List[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textbbox((0, 0), f"{lines[-1]}...", font=font)[2] > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = f"{lines[-1]}..."
    return lines or [""]


def make_text_panel(width: int, prompt: str, text_output: str) -> Image.Image:
    margin = 14
    gap = 10
    font = load_font(15)
    title_font = load_font(16)
    probe = Image.new("RGB", (1, 1), (255, 255, 255))
    draw = ImageDraw.Draw(probe)
    max_width = max(80, width - margin * 2)
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 6
    title_h = draw.textbbox((0, 0), "Ag", font=title_font)[3] + 6

    sections = [("Prompt", prompt)]
    if text_output:
        sections.append(("Text Output", text_output))

    section_lines = []
    panel_h = margin
    for title, body in sections:
        max_lines = 5 if title == "Prompt" else None
        lines = wrap_text(draw, body, font, max_width, max_lines=max_lines)
        section_lines.append((title, lines))
        panel_h += title_h + line_h * len(lines) + gap
    panel_h += margin - gap

    panel = Image.new("RGB", (width, panel_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    y = margin
    for title, lines in section_lines:
        draw.text((margin, y), f"{title}:", fill=(20, 20, 20), font=title_font)
        y += title_h
        for line in lines:
            draw.text((margin, y), line, fill=(0, 0, 0), font=font)
            y += line_h
        y += gap
    return panel


def blank_prediction_panel(size: Tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (255, 255, 255))
    ImageDraw.Draw(panel).text(
        (20, 20),
        "No image output",
        fill=(0, 0, 0),
        font=load_font(14),
    )
    return panel


def binary_label_from_prompt(prompt: str) -> str:
    category_names = _extract_category_names(prompt)
    return ", ".join(category_names) if category_names else "binary segmentation"


def make_prediction_panel(
    task: str,
    input_image: Image.Image,
    output_image: Optional[Image.Image],
    prompt: str,
    text_output: str,
) -> Image.Image:
    if output_image is None:
        if task in DETECTION_VIS_TASKS and text_output:
            try:
                return visualize_detection(
                    input_image,
                    text_output,
                    task_name=task,
                    prompt=prompt,
                    config=VisualizationConfig(),
                    include_prompt=True,
                )
            except Exception as exc:
                print(f"[Warn] detection visualization failed: {exc}")
        return blank_prediction_panel(input_image.size)

    try:
        if task == "binary_seg":
            return visualize_binary_segmentation(
                input_image,
                output_image,
                label=binary_label_from_prompt(prompt),
                config=VisualizationConfig(),
            )
        if task == "gcg_seg" and text_output:
            return visualize_gcg_segmentation(
                input_image,
                output_image,
                text_output,
                config=VisualizationConfig(),
            )
        if task == "pan_seg" and text_output:
            return visualize_panoptic_segmentation(
                input_image,
                output_image,
                text_output,
                question=prompt,
                config=VisualizationConfig(),
            )
    except Exception as exc:
        print(f"[Warn] task visualization failed: {exc}")

    return output_image.convert("RGB")


def make_visualization(
    task: str,
    input_image: Image.Image,
    output_image: Optional[Image.Image],
    prompt: str,
    text_output: str,
) -> Image.Image:
    source = input_image.convert("RGB")
    pred = make_prediction_panel(task, source, output_image, prompt, text_output)
    body = visualize_concat_col(source, pred, concat_col=2)
    text_panel = make_text_panel(body.width, prompt, text_output)

    canvas = Image.new(
        "RGB",
        (body.width, text_panel.height + body.height),
        (255, 255, 255),
    )
    canvas.paste(text_panel, (0, 0))
    canvas.paste(body, (0, text_panel.height))
    return canvas


# Interactive shell

def print_task_list() -> None:
    print("\n===== Tasks =====")
    for task in TASK_ORDER:
        print(f"{task}: {TASK_HELP[task]}")


def print_current_state(request: InferenceRequest) -> None:
    print("\n===== Current =====")
    print(f"task: {request.task}")
    print(f"mode: {request.mode}")
    print("image_path:")
    for idx, path in enumerate(request.image_paths, start=1):
        print(f"  [{idx}] {path}")


def parse_command(raw: str) -> Tuple[str, str]:
    command, _, value = raw.partition(" ")
    return command.strip(), value.strip()


def query_prompt(task: str) -> str:
    if task in NO_QUERY_TASKS:
        return "\nQuery (no further query needed): "
    return "\nQuery: "


def read_query(task: str, prefill_panoptic_categories: bool = True) -> str:
    if task != "pan_seg" or not prefill_panoptic_categories:
        return input(query_prompt(task)).strip()

    default_query = _format_coco_categories()
    if readline is None:
        return input(f"\nQuery: {default_query}").strip()

    readline.set_startup_hook(lambda: readline.insert_text(default_query))
    try:
        return input("\nQuery: ").strip()
    finally:
        readline.set_startup_hook()


class TaskRunner:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        output_dir: str,
    ):
        self.output_dir = output_dir
        with timer_context("Init SenseNovaVisionModel"):
            self.model = SenseNovaVisionModel(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )

    def run(self, request: InferenceRequest) -> None:
        mode = request.mode
        if mode not in BASE_PARAMS:
            raise ValueError(f"Unsupported mode: {mode}. Available modes: {', '.join(BASE_PARAMS)}")
        prompt = build_prompt(request.task, request.query)
        input_image = ImageOps.exif_transpose(Image.open(request.image_paths[0])).convert("RGB")
        output_context = make_output_context(self.output_dir, request)

        print("\n===== Inference Input =====")
        print(f"task: {request.task}")
        print(f"mode: {mode}")
        print("image_path:")
        for idx, path in enumerate(request.image_paths, start=1):
            print(f"  [{idx}] {path}")
        print("prompt:")
        print(prompt)

        with timer_context("Inference"):
            if request.task == "recon3d":
                output = self.run_recon3d(request, prompt, output_context)
            elif request.task == "camera_pose":
                output = self.run_camera_pose(request, prompt)
            else:
                output = self.run_generate(request, prompt)

        self.save_outputs(request, prompt, output, input_image, output_context)

    def run_generate(self, request: InferenceRequest, prompt: str) -> Dict:
        params = dict(request.params or {})
        seed = params.pop("seed", None)
        if seed is not None:
            set_seed(int(seed))
        result = self.model.generate(
            question=model_question(prompt, len(request.image_paths)),
            images=request.image_paths,
            mode=request.mode,
            return_intermediate_outputs=True,
            **params,
        )
        if isinstance(result, dict):
            return result
        if isinstance(result, Image.Image):
            return {"image": result, "text": None}
        return {"image": None, "text": str(result) if result is not None else None}

    def run_recon3d(
        self,
        request: InferenceRequest,
        prompt: str,
        output_context: OutputContext,
    ) -> Dict:
        raw_output = os.path.join(
            output_context.task_dir,
            f"{output_context.save_prefix}_pts3d.npy",
        )
        glb_output = os.path.join(output_context.task_dir, f"{output_context.save_prefix}.glb")
        params = dict(request.params or {})
        seed = params.pop("seed", None)
        noise_seed = int(seed if seed is not None else params.pop("noise_seed", 123456))
        result = self.model.reconstruct_3d(
            images=request.image_paths,
            prompt=prompt,
            raw_output=raw_output,
            glb_output=glb_output,
            noise_seed=noise_seed,
            postprocess_predictions=True,
            **params,
        )
        pts3d = result.get("pts3d")
        summary = [
            f"pts3d shape: {getattr(pts3d, 'shape', None)}",
            f"raw_output: {raw_output}",
            f"glb_output: {glb_output}",
        ]
        if result.get("text"):
            summary.extend(["", "Model Text:", str(result["text"])])
        result.update(
            {
                "text": "\n".join(summary),
                "raw_output": raw_output,
                "glb_output": glb_output,
            }
        )
        return result

    def run_camera_pose(self, request: InferenceRequest, prompt: str) -> Dict:
        if len(request.image_paths) < 2:
            raise ValueError("camera_pose requires at least two multi-view images.")
        if len(request.image_paths) > 10:
            print("[Warn] camera_pose limits input to 10 frames.")
            request = InferenceRequest(
                image_paths=request.image_paths[:10],
                task=request.task,
                mode=request.mode,
                query=request.query,
                params=request.params,
            )

        params = dict(request.params or {})
        seed = params.pop("seed", None)
        if seed is not None:
            set_seed(int(seed))
        raw_text = self.model.generate(
            question=model_question(prompt, len(request.image_paths)),
            images=request.image_paths,
            mode=request.mode,
            vit_transform=self.model.camera_vit_transform,
            **params,
        )
        if raw_text is None:
            raise RuntimeError("camera_pose did not return text output.")
        raw_text = str(raw_text)
        parsed_pose = resolve_pose_string(raw_text)
        parsed_text = json.dumps(parsed_pose, indent=2) if parsed_pose is not None else "null"
        return {
            "image": None,
            "text": f"{raw_text}\n\nParsed Pose:\n{parsed_text}",
            "raw_text": raw_text,
            "parsed_pose": parsed_pose,
        }

    def save_outputs(
        self,
        request: InferenceRequest,
        prompt: str,
        output: Dict,
        input_image: Image.Image,
        output_context: OutputContext,
    ) -> None:
        prompt_path = os.path.join(
            output_context.task_dir,
            f"{output_context.save_prefix}_prompt.txt",
        )
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"[Saved] prompt: {prompt_path}")

        text_output = ""
        if output.get("text"):
            text_output = str(output["text"])
            text_path = os.path.join(output_context.task_dir, f"{output_context.save_prefix}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_output)
            print(f"[Saved] text result: {text_path}")
            print("===== Text Output =====")
            print(text_output)

        if output.get("parsed_pose") is not None:
            pose_path = os.path.join(
                output_context.task_dir,
                f"{output_context.save_prefix}_pose.json",
            )
            with open(pose_path, "w", encoding="utf-8") as f:
                json.dump(output["parsed_pose"], f, indent=2)
            print(f"[Saved] parsed pose: {pose_path}")

        for key, label in (("raw_output", "raw 3D points"), ("glb_output", "3D scene")):
            if output.get(key):
                print(f"[Saved] {label}: {output[key]}")

        out_image = None
        if output.get("image"):
            out_images = output["image"] if isinstance(output["image"], list) else [output["image"]]
            out_image = out_images[0]
            for idx, image in enumerate(out_images, start=1):
                suffix = "" if len(out_images) == 1 else f"_{idx:02d}"
                image_path = os.path.join(
                    output_context.task_dir,
                    f"{output_context.save_prefix}{suffix}.png",
                )
                image.save(image_path)
                print(f"[Saved] image result{suffix}: {image_path}")

        viz = make_visualization(request.task, input_image, out_image, prompt, text_output)
        viz_path = os.path.join(output_context.vis_dir, f"{output_context.save_prefix}.png")
        viz.save(viz_path)
        print(f"[Saved] visualization: {viz_path}")


class WebTaskRunner(TaskRunner):
    def run_web(
        self,
        request: InferenceRequest,
        prompt: str,
        input_image: Image.Image,
        output_context: OutputContext,
    ) -> Tuple[Dict, Image.Image, List[str]]:
        mode = request.mode
        if mode not in BASE_PARAMS:
            raise ValueError(f"Unsupported mode: {mode}. Available modes: {', '.join(BASE_PARAMS)}")

        print("\n===== Inference Input =====")
        print(f"task: {request.task}")
        print(f"mode: {mode}")
        print("image_path:")
        for idx, path in enumerate(request.image_paths, start=1):
            print(f"  [{idx}] {path}")
        print("prompt:")
        print(prompt)

        with timer_context("Inference"):
            if request.task == "recon3d":
                output = self.run_recon3d(request, prompt, output_context)
            elif request.task == "camera_pose":
                output = self.run_camera_pose(request, prompt)
            else:
                output = self.run_generate(request, prompt)

        saved_files: List[str] = []
        raw_output_files: List[str] = []
        prompt_path = os.path.join(
            output_context.task_dir,
            f"{output_context.save_prefix}_prompt.txt",
        )
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        saved_files.append(prompt_path)

        text_output = str(output.get("text") or "")
        if text_output:
            text_path = os.path.join(output_context.task_dir, f"{output_context.save_prefix}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_output)
            saved_files.append(text_path)
            raw_output_files.append(text_path)

        if output.get("parsed_pose") is not None:
            pose_path = os.path.join(
                output_context.task_dir,
                f"{output_context.save_prefix}_pose.json",
            )
            with open(pose_path, "w", encoding="utf-8") as f:
                json.dump(output["parsed_pose"], f, indent=2)
            saved_files.append(pose_path)
            raw_output_files.append(pose_path)

        for key in ("raw_output", "glb_output"):
            if output.get(key):
                output_path = str(output[key])
                saved_files.append(output_path)
                raw_output_files.append(output_path)

        image_output = output.get("image")
        out_image = None
        if image_output:
            out_images = image_output if isinstance(image_output, list) else [image_output]
            out_image = out_images[0]
            for idx, image in enumerate(out_images, start=1):
                suffix = "" if len(out_images) == 1 else f"_{idx:02d}"
                image_path = os.path.join(
                    output_context.task_dir,
                    f"{output_context.save_prefix}{suffix}.png",
                )
                image.save(image_path)
                saved_files.append(image_path)
                raw_output_files.append(image_path)

        visualization = make_prediction_panel(
            request.task,
            input_image,
            out_image,
            prompt,
            text_output,
        )
        viz_path = os.path.join(output_context.vis_dir, f"{output_context.save_prefix}.png")
        visualization.save(viz_path)
        saved_files.append(viz_path)

        output["visualization_path"] = viz_path
        output["saved_files"] = saved_files
        output["raw_output_files"] = raw_output_files
        return output, visualization, saved_files


def run_interactive_loop(runner: TaskRunner, request: InferenceRequest) -> None:
    print("\nInteractive mode enabled.")
    print(
        "Commands: /task TASK, /mode MODE, /image PATH[,PATH], "
        "/status, /tasks, /modes, /help, q"
    )
    print(
        "Note: /task switches both task and mode. "
        "/mode switches to raw_query with the selected mode."
    )
    current = request
    prefill_panoptic_categories = current.task == "pan_seg"
    while True:
        print_current_state(current)
        raw = read_query(current.task, prefill_panoptic_categories)
        if raw.lower() == "q":
            break
        if raw.startswith("/"):
            command, value = parse_command(raw)
            if command == "/task":
                if not value:
                    print_task_list()
                    continue
                if value not in TASK_TO_PROMPT_BUILDER:
                    print(f"[Error] Unsupported task: {value}")
                    continue
                current.task = value
                current.mode = resolve_task_mode(value)
                current.query = ""
                prefill_panoptic_categories = value == "pan_seg"
                print(f"[Switched] task={current.task}, mode={current.mode}")
                continue
            if command == "/mode":
                if not value:
                    print(f"Available raw_query modes: {', '.join(BASE_PARAMS)}")
                    continue
                if value not in BASE_PARAMS:
                    print(f"[Error] Unsupported mode: {value}")
                    continue
                current.task = "raw_query"
                current.mode = value
                prefill_panoptic_categories = False
                print(f"[Switched] task=raw_query, mode={current.mode}")
                continue
            if command == "/image":
                try:
                    current.image_paths = parse_image_paths_arg(value)
                    print(f"[Switched] image count={len(current.image_paths)}")
                except (FileNotFoundError, ValueError) as exc:
                    print(f"[Error] {exc}")
                continue
            if command == "/status":
                print_current_state(current)
                continue
            if command == "/tasks":
                print_task_list()
                continue
            if command == "/modes":
                print(f"Available raw_query modes: {', '.join(BASE_PARAMS)}")
                continue
            if command == "/help":
                print(
                    "Commands: /task TASK, /mode MODE, /image PATH[,PATH], "
                    "/status, /tasks, /modes, /help, q"
                )
                print(
                    "/task switches both task and mode. "
                    "/mode switches to raw_query with the selected mode."
                )
                continue
            print(f"[Error] Unknown command: {command}")
            continue

        if current.task == "pan_seg":
            current.query = raw
        elif raw:
            current.query = raw
        try:
            runner.run(current)
            if current.task == "pan_seg":
                prefill_panoptic_categories = False
        except Exception as exc:
            print(f"[Error] {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseNova-Vision CLI & interactive demo.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--task", type=str, choices=TASK_ORDER, default="raw_query")
    parser.add_argument(
        "--mode",
        type=str,
        default=TASK_TO_MODE["raw_query"],
        help="Inference parameter mode for raw_query. Other tasks use TASK_TO_MODE.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Input image path. Use commas for multiple images.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Task query, raw prompt, or segmentation categories.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the model loaded and enter a query loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = resolve_request_mode(args.task, args.mode)
    request = InferenceRequest(
        image_paths=parse_image_paths_arg(args.image_path),
        task=args.task,
        mode=mode,
        query=args.query.strip(),
    )
    runner = TaskRunner(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )
    runner.run(request)
    if args.interactive:
        run_interactive_loop(runner, request)


if __name__ == "__main__":
    main()
