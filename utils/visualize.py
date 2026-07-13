import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colormap import build_palette, color_for_label, color_for_segment
from .mask import decode_rle, mask_boundary, rgb2id, thicken_mask
from .parsing_output import (
    gcg_prediction_from_raw_output,
    normalize_categories,
    normalize_semantic_categories,
    panoptic_prediction_from_raw_output,
    parse_detection_output,
    parse_visual_prompt,
)

# Shared drawing


@dataclass
class VisualizationConfig:
    alpha: float = 0.55
    min_label_area: int = 80
    max_labels: int = 120
    font_size: int = 0
    draw_width: int = 0
    point_radius: int = 0
    keypoint_radius: int = 0
    max_label_chars: int = 96
    prefer_unwrapped_labels: bool = False


def ensure_config(config: Optional[VisualizationConfig] = None, **kwargs):
    if config is None:
        config = VisualizationConfig()
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config


def load_font(size: int = 14, bold: bool = False):
    size = max(8, int(size))
    regular_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    bold_candidates = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    candidates = bold_candidates + regular_candidates if bold else regular_candidates
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def xsam_default_font_size(width: int, height: int) -> int:
    return max(9, int(round(min(width, height) / 55)))


def xsam_label_font_size(width: int, height: int, bbox=None) -> int:
    default_size = xsam_default_font_size(width, height)
    if bbox is None:
        return default_size
    y0, y1 = bbox[1], bbox[3]
    height_ratio = max(1, y1 - y0) / max(1.0, np.sqrt(width * height))
    scale = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2.0) * 0.5
    return max(8, int(round(scale * default_size)))


def wrap_label_text(text: Any, font, max_width: int, max_lines: int = 3) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    normalized = " ".join(words)
    if probe.textlength(normalized, font=font) <= max_width:
        return normalized

    lines = []
    current = words[0]
    consumed = 1
    for word in words[1:]:
        trial = f"{current} {word}"
        if probe.textlength(trial, font=font) <= max_width:
            current = trial
            consumed += 1
            continue
        lines.append(current)
        current = word
        consumed += 1
        if len(lines) >= max_lines - 1:
            break
    lines.append(current)
    if len(lines) == max_lines and consumed < len(words):
        lines[-1] = textwrap.shorten(
            lines[-1],
            width=max(8, len(lines[-1]) - 1),
            placeholder="...",
        )
    return "\n".join(lines[:max_lines])


def wrap_label_text_no_ellipsis(
    text: Any, font, max_width: int, max_lines: int = 4
) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_width = max(40, int(max_width))

    words = []
    for word in text.split():
        if probe.textlength(word, font=font) <= max_width:
            words.append(word)
            continue
        current = ""
        for char in word:
            trial = current + char
            if current and probe.textlength(trial, font=font) > max_width:
                words.append(current)
                current = char
            else:
                current = trial
        if current:
            words.append(current)

    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}" if current else word
        if probe.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])])


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy,
    text: Any,
    color,
    font,
    max_label_chars: int = 96,
    max_width: Optional[int] = None,
    anchor: str = "top_left",
    prefer_unwrapped: bool = False,
):
    """Draw a detection-family label using the reference overlay style."""
    text = " ".join(str(text or "").split())
    if not text:
        return
    if max_label_chars > 0 and len(text) > max_label_chars:
        if max_label_chars <= 3:
            text = text[:max_label_chars]
        else:
            text = text[: max_label_chars - 3].rstrip() + "..."
    wrap_width = max_width or max(80, int(draw.im.size[0] * 0.42))
    if prefer_unwrapped:
        wrap_width = max(wrap_width, draw.im.size[0] - 8)
    text = wrap_label_text_no_ellipsis(text, font, wrap_width, max_lines=4)

    bbox = draw.multiline_textbbox(
        (0, 0), text, font=font, stroke_width=1, spacing=2
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(2, int(getattr(font, "size", 12) * 0.18))
    if anchor == "bottom_left":
        raw_x = xy[0] + pad
        raw_y = xy[1] - text_h - pad
        if raw_y - pad < 0:
            raw_y = xy[1] + pad
    else:
        raw_x, raw_y = xy
    x = int(max(2, min(raw_x, draw.im.size[0] - text_w - pad * 2 - 2)))
    y = int(max(2, min(raw_y, draw.im.size[1] - text_h - pad * 2 - 2)))
    box = (x - pad, y - pad, x + text_w + pad, y + text_h + pad)
    draw.rounded_rectangle(
        box,
        radius=max(2, pad),
        fill=tuple(int(c) for c in color) + (210,),
    )
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
        spacing=2,
    )


# Panel composition


def concat_horizontally(*images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    out = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for image in images:
        out.paste(image, (x_offset, 0))
        x_offset += image.size[0]
    return out


def add_header(
    img: Image.Image, tag: str, tag_height: Optional[int] = None
) -> Image.Image:
    img = img.convert("RGB")
    tag_height = tag_height or max(28, int(round(img.height * 0.055)))
    out = Image.new("RGB", (img.width, img.height + tag_height), (255, 255, 255))
    out.paste(img, (0, tag_height))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, img.width, tag_height), fill=(18, 22, 28))
    font = load_font(max(10, int(tag_height * 0.42)))
    bbox = draw.textbbox((0, 0), tag, font=font)
    draw.text(
        (
            (img.width - (bbox[2] - bbox[0])) // 2,
            (tag_height - (bbox[3] - bbox[1])) // 2,
        ),
        tag,
        font=font,
        fill=(255, 255, 255),
    )
    return out


def visualize_concat_col(
    source: Image.Image,
    pred: Image.Image,
    concat_col: int = 1,
    gt: Optional[Image.Image] = None,
    prompt: Optional[Image.Image] = None,
    source_label: str = "Image",
    gt_label: str = "GT",
    pred_label: str = "Prediction",
    prompt_label: str = "Prompt",
) -> Image.Image:
    """Compose final panels. Case-level visualize_* functions only draw pred."""
    if concat_col <= 1:
        return pred
    first = prompt if prompt is not None else source
    first_label = prompt_label if prompt is not None else source_label
    panels = [add_header(first, first_label)]
    if concat_col >= 3 and gt is not None:
        panels.append(add_header(gt, gt_label))
    panels.append(add_header(pred, pred_label))
    return concat_horizontally(*panels)


# Segmentation


def label_position(mask: np.ndarray):
    num_components, component_map, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if num_components <= 1 or stats[1:, -1].size == 0:
        return None
    component_id = int(np.argmax(stats[1:, -1]) + 1)
    ys, xs = np.nonzero(component_map == component_id)
    if len(xs) == 0:
        return None
    x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
    y0 = int(stats[component_id, cv2.CC_STAT_TOP])
    width = int(stats[component_id, cv2.CC_STAT_WIDTH])
    height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    return (
        int(np.median(xs)),
        int(np.median(ys)),
        (x0, y0, x0 + width - 1, y0 + height - 1),
    )


def draw_segmentation_label(
    draw: ImageDraw.ImageDraw,
    xy,
    text: Any,
    color,
    font,
) -> None:
    """Draw a segmentation label using the reference PIL overlay style."""
    x = int(round(xy[0]))
    y = int(round(xy[1]))
    max_text_width = max(80, int(draw.im.size[0] * 0.46))
    text = wrap_label_text(text, font, max_text_width)
    if not text:
        return

    bbox = draw.multiline_textbbox(
        (x, y), text, font=font, stroke_width=1, spacing=2
    )
    pad = max(3, int(getattr(font, "size", 12) * 0.22))
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max(2, min(x, draw.im.size[0] - text_width - pad * 2 - 2))
    y = max(2, min(y, draw.im.size[1] - text_height - pad * 2 - 2))
    bbox = draw.multiline_textbbox(
        (x, y), text, font=font, stroke_width=1, spacing=2
    )
    box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)

    label_color = tuple(int(c) for c in color)
    if sum(label_color) > 650:
        label_color = (72, 80, 92)
    draw.rounded_rectangle(
        box,
        radius=max(3, pad),
        fill=label_color + (220,),
        outline=(255, 255, 255, 210),
        width=1,
    )
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
        spacing=2,
        align="center",
    )


def overlay_segments(
    image: Image.Image,
    segments: list[dict],
    alpha: float = 0.55,
    min_label_area: int = 80,
    max_labels: int = 120,
    font_size: int = 0,
) -> Image.Image:
    overlay = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    draw_items = []

    for seg in sorted(segments, key=lambda x: x.get("area", 0), reverse=True):
        mask = np.asarray(seg["mask"]).astype(bool)
        if mask.shape[:2] != overlay.shape[:2]:
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize(
                (overlay.shape[1], overlay.shape[0]), Image.NEAREST
            )
            mask = np.asarray(mask_img) > 0
        color = np.asarray(seg["color"], dtype=np.float32)
        overlay[mask] = np.clip(
            overlay[mask].astype(np.float32) * (1.0 - alpha) + color * alpha,
            0,
            255,
        ).astype(np.uint8)
        edge = mask_boundary(mask)
        overlay[edge] = np.clip(
            color * 0.25 + np.array([255, 255, 255], dtype=np.float32) * 0.75,
            0,
            255,
        ).astype(np.uint8)
        area = int(seg.get("area", int(mask.sum())))
        if area >= min_label_area:
            pos = label_position(mask)
            if pos is not None:
                x, y, bbox = pos
                draw_items.append(
                    (
                        area,
                        (x, y),
                        bbox,
                        seg.get("label", ""),
                        tuple(int(c) for c in seg["color"]),
                    )
                )

    out = Image.fromarray(overlay, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font_cache = {}
    for _, pos, bbox, label, color in sorted(
        draw_items, key=lambda x: x[0], reverse=True
    )[:max_labels]:
        resolved_font_size = (
            font_size
            if font_size and font_size > 0
            else xsam_label_font_size(out.width, out.height, bbox)
        )
        font_cache.setdefault(
            resolved_font_size, load_font(resolved_font_size, bold=True)
        )
        draw_segmentation_label(
            draw,
            pos,
            label,
            color,
            font_cache[resolved_font_size],
        )
    return out.convert("RGB")


def overlay_segments_with_config(
    image: Image.Image,
    segments: list[dict],
    config: VisualizationConfig,
) -> Image.Image:
    return overlay_segments(
        image,
        segments,
        config.alpha,
        config.min_label_area,
        config.max_labels,
        config.font_size,
    )


def overlay_rgb_mask(
    image: Image.Image, rgb_mask: Image.Image, alpha: float = 0.55
) -> Image.Image:
    image = image.convert("RGB")
    if rgb_mask.size != image.size:
        rgb_mask = rgb_mask.resize(image.size, resample=Image.NEAREST)
    base = np.asarray(image, dtype=np.float32)
    colors = np.asarray(rgb_mask.convert("RGB"), dtype=np.float32)
    mask = np.any(colors > 0, axis=2)
    out = base.copy()
    out[mask] = out[mask] * (1.0 - alpha) + colors[mask] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def draw_visual_prompt(
    image: Image.Image,
    visual_prompt,
    prompt_style: str = "fill",
    color=(255, 0, 0),
) -> Image.Image:
    """Draw an interactive segmentation prompt on the source image."""
    if visual_prompt is None:
        return image.copy()
    if isinstance(visual_prompt, (str, Path)):
        visual_prompt = Image.open(visual_prompt)
    out = image.convert("RGB")
    prompt = visual_prompt.convert("L")
    if prompt.size != out.size:
        prompt = prompt.resize(out.size, resample=Image.NEAREST)
    mask = np.asarray(prompt) > 127
    if prompt_style == "boundary":
        mask = thicken_mask(mask_boundary(mask), radius=1)
    arr = np.asarray(out, dtype=np.uint8).copy()
    arr[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _category_is_thing(category: dict) -> bool:
    for key in ("isthing", "is_thing", "thing"):
        if key in category:
            return bool(category[key])
    category_type = str(category.get("type", "")).lower()
    if category_type in {"stuff", "semantic"}:
        return False
    return True


def _category_color(_category: dict, palette, fallback_id: int):
    return palette[int(fallback_id) % len(palette)]


def visualize_gcg_prediction(
    image: Image.Image,
    prediction: dict,
    palette=None,
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    """Visualize a structured GCG prediction."""
    config = ensure_config(config)
    palette = palette or build_palette()
    phrases = prediction.get("gcg_phrases", [])
    segments = []
    for idx, segmentation in enumerate(prediction.get("segmentation", [])):
        mask = decode_rle(segmentation)
        if mask.shape[:2] != (image.height, image.width):
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize(image.size, resample=Image.NEAREST)
            mask = np.asarray(mask_img) > 0
        area = int(mask.sum())
        if area == 0:
            continue
        label = phrases[idx] if idx < len(phrases) else f"mask-{idx}"
        color = color_for_segment(
            palette,
            None,
            idx,
            f"{prediction.get('image_id')}-{idx}",
        )
        segments.append({"mask": mask, "area": area, "label": label, "color": color})
    return overlay_segments_with_config(image, segments, config)


def visualize_gcg_segmentation(
    image: Image.Image,
    raw_mask: Image.Image,
    caption: str,
    palette=None,
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    """Visualize raw GCG output: original image + raw RGB mask + raw caption."""
    prediction = gcg_prediction_from_raw_output(raw_mask, image, caption)
    return visualize_gcg_prediction(image, prediction, palette=palette, config=config)


def visualize_panoptic_prediction(
    image: Image.Image,
    prediction: dict,
    categories=None,
    palette=None,
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    config = ensure_config(config)
    palette = palette or build_palette()
    id_source = prediction.get("id_map")
    if id_source is None:
        id_source = prediction.get("panoptic_map")
    if id_source is None:
        id_source = prediction.get("png")
    if id_source is None:
        raise ValueError(
            "Panoptic prediction requires `id_map`, `panoptic_map`, or `png`."
        )
    if isinstance(id_source, (str, Path)):
        id_source = Image.open(id_source)
    id_map = rgb2id(id_source)
    ann = prediction.get("annotation", prediction)
    category_map = normalize_categories(categories or prediction.get("categories"))
    phrases = ann.get("gcg_phrases", [])
    category_counts = {}
    segments = []
    for idx, segment_info in enumerate(ann.get("segments_info", [])):
        segment_id = int(segment_info["id"])
        mask = id_map == segment_id
        area = int(mask.sum())
        if area == 0:
            continue
        category_id = int(segment_info.get("category_id", -1))
        category = category_map.get(
            category_id,
            {"id": category_id, "name": str(category_id), "isthing": 1},
        )
        is_thing = _category_is_thing(category)
        if idx < len(phrases):
            label = phrases[idx]
        elif is_thing:
            label = f"{category['name']}-{category_counts.get(category_id, 0)}"
        else:
            label = category["name"]
        category_counts[category_id] = category_counts.get(category_id, 0) + 1
        color = (
            color_for_segment(palette, category_id, idx, segment_id)
            if is_thing
            else _category_color(category, palette, category_id)
        )
        segments.append({"mask": mask, "area": area, "label": label, "color": color})
    return overlay_segments_with_config(image, segments, config)


def visualize_panoptic_segmentation(
    image: Image.Image,
    raw_mask: Image.Image,
    caption: str,
    categories=None,
    question: str = "",
    palette=None,
    config: Optional[VisualizationConfig] = None,
    strict_categories: bool = False,
) -> Image.Image:
    prediction = panoptic_prediction_from_raw_output(
        image,
        raw_mask,
        caption,
        categories,
        question=question,
        strict_categories=strict_categories,
    )
    return visualize_panoptic_prediction(
        image,
        prediction,
        categories=prediction["categories"],
        palette=palette,
        config=config,
    )


def visualize_semantic_segmentation(
    image: Image.Image,
    semantic_mask,
    categories,
    palette=None,
    ignore_ids=(255,),
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    config = ensure_config(config)
    palette = palette or build_palette()
    if isinstance(semantic_mask, (str, Path)):
        semantic_mask = Image.open(semantic_mask)
    label_mask = np.asarray(semantic_mask)
    if label_mask.ndim == 3:
        label_mask = label_mask[:, :, 0]
    label_mask = label_mask.astype(np.int32)
    category_map = normalize_semantic_categories(categories)
    segments = []
    for contiguous_id in sorted(int(value) for value in np.unique(label_mask)):
        if contiguous_id in ignore_ids:
            continue
        category = category_map.get(contiguous_id) or {
            "id": contiguous_id,
            "name": str(contiguous_id),
        }
        mask = label_mask == contiguous_id
        area = int(mask.sum())
        if area == 0:
            continue
        category_id = int(category.get("id", contiguous_id))
        segments.append(
            {
                "mask": mask,
                "area": area,
                "label": category.get("name", str(category_id)),
                "color": _category_color(category, palette, category_id),
            }
        )
    return overlay_segments_with_config(image, segments, config)


def visualize_binary_segmentation(
    image: Image.Image,
    pred_mask,
    label: str = "prediction",
    palette=None,
    color=None,
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    config = ensure_config(config)
    palette = palette or build_palette()
    if isinstance(pred_mask, dict) and "segments" in pred_mask:
        segments = pred_mask["segments"]
    else:
        if isinstance(pred_mask, (str, Path)):
            pred_mask = Image.open(pred_mask)
        if (
            isinstance(pred_mask, dict)
            and "size" in pred_mask
            and "counts" in pred_mask
        ):
            mask = decode_rle(pred_mask)
        else:
            mask_array = np.asarray(
                pred_mask.convert("L")
                if isinstance(pred_mask, Image.Image)
                else pred_mask
            )
            mask = (
                mask_array > 127
                if mask_array.dtype == np.uint8
                else mask_array.astype(bool)
            )
        area = int(mask.sum())
        segments = (
            [{"mask": mask, "area": area, "label": label, "color": color or palette[0]}]
            if area
            else []
        )
    return overlay_segments_with_config(image, segments, config)


# Detection

_DETECTION_LABEL_BOX_WIDTH_RATIO = 1.2

PERSON_SKELETON_CONNECTIONS = [
    ("nose", "left eye"),
    ("nose", "right eye"),
    ("left eye", "left ear"),
    ("right eye", "right ear"),
    ("left shoulder", "right shoulder"),
    ("left shoulder", "left elbow"),
    ("right shoulder", "right elbow"),
    ("left elbow", "left wrist"),
    ("right elbow", "right wrist"),
    ("left shoulder", "left hip"),
    ("right shoulder", "right hip"),
    ("left hip", "right hip"),
    ("left hip", "left knee"),
    ("right hip", "right knee"),
    ("left knee", "left ankle"),
    ("right knee", "right ankle"),
]

HAND_SKELETON_CONNECTIONS = [
    ("wrist", "thumb root"),
    ("thumb root", "thumb's third knuckle"),
    ("thumb's third knuckle", "thumb's second knuckle"),
    ("thumb's second knuckle", "thumb's first knuckle"),
    ("wrist", "forefinger's root"),
    ("forefinger's root", "forefinger's third knuckle"),
    ("forefinger's third knuckle", "forefinger's second knuckle"),
    ("forefinger's second knuckle", "forefinger's first knuckle"),
    ("wrist", "middle finger's root"),
    ("middle finger's root", "middle finger's third knuckle"),
    ("middle finger's third knuckle", "middle finger's second knuckle"),
    ("middle finger's second knuckle", "middle finger's first knuckle"),
    ("wrist", "ring finger's root"),
    ("ring finger's root", "ring finger's third knuckle"),
    ("ring finger's third knuckle", "ring finger's second knuckle"),
    ("ring finger's second knuckle", "ring finger's first knuckle"),
    ("wrist", "pinky finger's root"),
    ("pinky finger's root", "pinky finger's third knuckle"),
    ("pinky finger's third knuckle", "pinky finger's second knuckle"),
    ("pinky finger's second knuckle", "pinky finger's first knuckle"),
]

ANIMAL_SKELETON_CONNECTIONS = [
    ("left eye", "right eye"),
    ("left eye", "nose"),
    ("right eye", "nose"),
    ("nose", "neck"),
    ("neck", "left shoulder"),
    ("neck", "right shoulder"),
    ("neck", "root of tail"),
    ("root of tail", "left hip"),
    ("root of tail", "right hip"),
    ("left shoulder", "left elbow"),
    ("right shoulder", "right elbow"),
    ("left elbow", "left front paw"),
    ("right elbow", "right front paw"),
    ("left hip", "left knee"),
    ("right hip", "right knee"),
    ("left knee", "left back paw"),
    ("right knee", "right back paw"),
]


def detection_font_size(width: int, height: int, config: VisualizationConfig) -> int:
    if config.font_size and config.font_size > 0:
        return config.font_size
    return max(9, int(round(min(width, height) / 48)))


def draw_bbox(draw, bbox, color, width: int):
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    for offset in range(width):
        draw.rectangle(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            outline=tuple(color) + (235,),
            width=1,
        )


def draw_point(draw, point, color, radius: int):
    x, y = [int(round(value)) for value in point]
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=tuple(color) + (170,),
        outline=tuple(color) + (255,),
        width=max(1, radius // 3),
    )


def draw_polygon(draw, polygon, color, width: int):
    points = [(int(round(x)), int(round(y))) for x, y in polygon]
    if len(points) < 3:
        return
    draw.polygon(points, fill=tuple(color) + (45,))
    draw.line(
        points + [points[0]],
        fill=tuple(color) + (245,),
        width=width,
        joint="curve",
    )


def primitive_area(primitive):
    kind = primitive.get("kind")
    if kind == "bbox":
        x0, y0, x1, y1 = primitive["bbox"]
        return max(1.0, (x1 - x0) * (y1 - y0))
    if kind == "point":
        return 1.0
    if kind == "mask":
        return float(primitive.get("area", 0))
    if kind == "polygon":
        points = list(primitive["polygon"])
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
            area += x0 * y1 - x1 * y0
        return max(1.0, abs(area) / 2.0)
    if kind == "keypoint":
        bbox = primitive.get("bbox")
        if bbox:
            x0, y0, x1, y1 = bbox
            return max(1.0, (x1 - x0) * (y1 - y0))
    return 1.0


def canonical_keypoint_name(name: Any) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def select_keypoint_connections(primitive):
    keypoint_type = str(primitive.get("keypoint_type", "")).lower()
    context = " ".join(
        str(primitive.get(key, "")).lower()
        for key in ("keypoint_type", "keypoint_context", "label")
    )
    keypoints = {
        canonical_keypoint_name(name) for name in primitive.get("keypoints", {})
    }
    if keypoint_type == "hand" or "wrist" in keypoints:
        return HAND_SKELETON_CONNECTIONS
    if "ap-10k" in context or "ap10k" in context:
        return ANIMAL_SKELETON_CONNECTIONS
    if "coco" in context:
        return PERSON_SKELETON_CONNECTIONS
    if keypoint_type in {"animal", "quadruped"} or {
        "root of tail",
        "left front paw",
    } & keypoints:
        return ANIMAL_SKELETON_CONNECTIONS
    if keypoint_type == "person" or {"left wrist", "right ankle"} & keypoints:
        return PERSON_SKELETON_CONNECTIONS
    return ANIMAL_SKELETON_CONNECTIONS


def draw_keypoint_skeleton(draw, primitive, width: int):
    keypoints = {
        canonical_keypoint_name(name): point
        for name, point in primitive.get("keypoints", {}).items()
    }
    color = tuple(int(c) for c in primitive.get("color", (255, 0, 0)))
    for a, b in select_keypoint_connections(primitive):
        start = keypoints.get(canonical_keypoint_name(a))
        end = keypoints.get(canonical_keypoint_name(b))
        if start is not None and end is not None:
            draw.line(
                [tuple(start), tuple(end)],
                fill=color + (230,),
                width=max(2, width),
            )


def keypoint_radius_for_primitive(
    primitive, image_size, config: VisualizationConfig, fallback_radius: int
) -> int:
    if config.keypoint_radius and config.keypoint_radius > 0:
        return config.keypoint_radius
    bbox = primitive.get("bbox")
    if bbox:
        x0, y0, x1, y1 = bbox
        bbox_area = max(1.0, (x1 - x0) * (y1 - y0))
        bbox_radius = max(2, min(5, int((bbox_area / 10000) ** 0.5 * 4)))
        image_floor = max(3, int(round(min(image_size) / 260)))
        return max(bbox_radius, image_floor)
    return max(3, fallback_radius // 2)


def overlay_primitives(
    image: Image.Image,
    primitives: list[dict],
    config: Optional[VisualizationConfig] = None,
) -> Image.Image:
    config = ensure_config(config)
    if not primitives:
        return image.copy()

    mask_segments = []
    draw_primitives = []
    for primitive in primitives:
        if primitive.get("kind") == "mask":
            mask_segments.append(
                {
                    "mask": primitive["mask"],
                    "area": primitive.get(
                        "area", int(np.asarray(primitive["mask"]).sum())
                    ),
                    "label": primitive.get("label", ""),
                    "color": primitive.get("color", (255, 0, 0)),
                }
            )
        else:
            draw_primitives.append(primitive)

    out = image.convert("RGB")
    if mask_segments:
        out = overlay_segments_with_config(out, mask_segments, config)

    out = out.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font = load_font(detection_font_size(out.width, out.height, config), bold=True)
    line_width = (
        config.draw_width
        if config.draw_width and config.draw_width > 0
        else max(2, int(round(min(out.size) / 360)))
    )
    point_radius = (
        config.point_radius
        if config.point_radius and config.point_radius > 0
        else max(4, int(round(min(out.size) / 160)))
    )

    labeled = 0
    for primitive in sorted(
        draw_primitives,
        key=lambda x: (1 if x.get("is_prompt") else 0, -primitive_area(x)),
    ):
        color = tuple(int(c) for c in primitive.get("color", (255, 0, 0)))
        label_xy = None
        if primitive["kind"] == "bbox":
            bbox = primitive["bbox"]
            draw_bbox(draw, bbox, color, line_width)
            label_xy = (bbox[0], bbox[1])
        elif primitive["kind"] == "point":
            point = primitive["point"]
            draw_point(draw, point, color, point_radius)
            label_xy = (point[0] + point_radius + 3, point[1] + point_radius + 3)
        elif primitive["kind"] == "polygon":
            polygon = primitive["polygon"]
            draw_polygon(draw, polygon, color, line_width)
            label_xy = (min(p[0] for p in polygon), min(p[1] for p in polygon))
        elif primitive["kind"] == "keypoint":
            bbox = primitive.get("bbox")
            if bbox:
                draw_bbox(draw, bbox, color, line_width)
                label_xy = (bbox[0], bbox[1])
            keypoint_radius = keypoint_radius_for_primitive(
                primitive, out.size, config, point_radius
            )
            draw_keypoint_skeleton(draw, primitive, line_width)
            for point in primitive.get("keypoints", {}).values():
                draw_point(draw, point, color, keypoint_radius)

        if label_xy is not None and labeled < config.max_labels:
            label_max_width = None
            bbox = primitive.get("bbox")
            if bbox:
                label_max_width = max(
                    80,
                    int(
                        (bbox[2] - bbox[0])
                        * _DETECTION_LABEL_BOX_WIDTH_RATIO
                    ),
                )
            draw_label(
                draw,
                label_xy,
                primitive.get("label", ""),
                color,
                font,
                config.max_label_chars,
                label_max_width,
                "top_left" if primitive["kind"] == "point" else "bottom_left",
                config.prefer_unwrapped_labels,
            )
            labeled += 1
    return out.convert("RGB")


def visualize_detection(
    image: Image.Image,
    predictions,
    task_name: Optional[str] = None,
    prompt: Optional[str] = None,
    palette=None,
    config: Optional[VisualizationConfig] = None,
    include_prompt: bool = True,
) -> Image.Image:
    config = ensure_config(config)
    palette = palette or build_palette()
    keypoint_context = task_name or ""
    primitives = parse_detection_output(
        predictions, image.size, keypoint_context=keypoint_context
    )
    if include_prompt:
        primitives = parse_visual_prompt(prompt or "", image.size) + primitives
    for primitive in primitives:
        primitive["color"] = (
            (255, 0, 0)
            if primitive.get("is_prompt")
            else color_for_label(
                palette,
                primitive.get("label", "object"),
                primitive.get("color_index", 0),
            )
        )
    return overlay_primitives(image, primitives, config)
