"""Parsers for SenseNova-Vision structured text and mask outputs.

This module converts model-facing formats such as tagged captions,
`<bbox>[x0,y0,x1,y1]</bbox>`, `<point>[x,y]</point>`, `<kpt>...</kpt>`, RLE
masks, and panoptic color masks into evaluation/visualization-friendly Python
structures. Coordinates may arrive either normalized to `[0, 1]` or already in
pixel space; normalization helpers convert both forms to clipped pixel values.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import (
    extract_ordered_caption_instances,
    get_gcg_caption,
    load_json_or_jsonl,
    normalize_category,
)
from .mask import decode_rle

_TO_MULTI_MASK_IMPL = None


def to_multi_mask(*args, **kwargs):
    """Dispatch to the fastest available RGB-mask-to-class-index converter.

    The torch implementation is preferred when importable because benchmark
    scripts already depend on torch and this path is much faster for dense
    masks. Environments that only need lightweight visualization can fall back
    to the numpy implementation in `utils.mask`.
    """
    global _TO_MULTI_MASK_IMPL
    if _TO_MULTI_MASK_IMPL is None:
        try:
            from .torch_image import to_multi_mask as _impl
        except (ImportError, ModuleNotFoundError):
            from .mask import to_multi_mask as _impl
        _TO_MULTI_MASK_IMPL = _impl
    return _TO_MULTI_MASK_IMPL(*args, **kwargs)


# Category metadata


def normalize_categories(categories) -> dict:
    """Normalize panoptic/category metadata into `{category_id: category}`.

    Accepted inputs:
    - `None`, returning an empty dict.
    - A dict keyed by category id, with either dict values or plain names.
    - A list of category dicts, as found in COCO-style JSON files.
    - A list of plain category names.

    The returned category dicts always contain at least `id` and `name`.
    """
    if categories is None:
        return {}
    if isinstance(categories, dict):
        out = {}
        for key, value in categories.items():
            if isinstance(value, dict):
                cid = int(value.get("id", key))
                out[cid] = dict(value)
                out[cid].setdefault("id", cid)
                out[cid].setdefault("name", str(cid))
            else:
                out[int(key)] = {"id": int(key), "name": str(value), "isthing": 1}
        return out
    out = {}
    for idx, cat in enumerate(categories):
        if isinstance(cat, dict):
            cid = int(cat.get("id", idx))
            out[cid] = dict(cat)
            out[cid].setdefault("id", cid)
            out[cid].setdefault("name", str(cid))
        else:
            out[idx] = {"id": idx, "name": str(cat), "isthing": 1}
    return out


def normalize_semantic_categories(categories) -> dict:
    """Normalize semantic label metadata into `{label_id: category}`.

    Unlike panoptic category ids, semantic masks often use contiguous label ids
    as pixel values. This helper preserves those contiguous label ids as keys
    while still keeping the dataset category id in the nested `id` field.
    """
    if categories is None:
        return {}
    if isinstance(categories, dict):
        out = {}
        for key, value in categories.items():
            label_id = int(key)
            if isinstance(value, dict):
                out[label_id] = dict(value)
                out[label_id].setdefault("id", int(value.get("id", label_id)))
                out[label_id].setdefault("name", str(out[label_id]["id"]))
            else:
                out[label_id] = {"id": label_id, "name": str(value), "isthing": 1}
        return out
    out = {}
    for label_id, category in enumerate(categories):
        if isinstance(category, dict):
            out[label_id] = dict(category)
            out[label_id].setdefault("id", int(category.get("id", label_id)))
            out[label_id].setdefault("name", str(out[label_id]["id"]))
        else:
            out[label_id] = {"id": label_id, "name": str(category), "isthing": 1}
    return out


def semantic_category_maps_from_panoptic_json(path: str) -> tuple[dict, dict]:
    """Load a panoptic annotation JSON and build semantic category maps.

    Returns two maps:
    - `contiguous_categories`: keyed by the category order in the JSON file,
      useful for semantic masks whose pixel values are contiguous labels.
    - `category_id_categories`: keyed by dataset category id, useful for
      panoptic `segments_info`.
    """
    data = load_json_or_jsonl(path)
    contiguous_categories = {}
    category_id_categories = {}
    for contiguous_id, category in enumerate(data.get("categories", [])):
        info = {
            "id": int(category["id"]),
            "name": str(category.get("name", category["id"]))
            .replace("-merged", "")
            .replace("-other", ""),
            "color": category.get("color"),
            "isthing": category.get(
                "isthing", category.get("is_thing", category.get("thing"))
            ),
        }
        contiguous_categories[contiguous_id] = dict(info)
        category_id_categories[int(category["id"])] = dict(info)
    return contiguous_categories, category_id_categories


def semantic_categories_from_panoptic_json(
    path: str, key_mode: str = "contiguous"
) -> dict:
    """Return one semantic category map from a panoptic annotation JSON.

    `key_mode="contiguous"` returns labels keyed by contiguous semantic ids.
    `key_mode="id"` returns labels keyed by dataset category ids.
    """
    contiguous_categories, category_id_categories = (
        semantic_category_maps_from_panoptic_json(path)
    )
    if key_mode == "id":
        return category_id_categories
    if key_mode != "contiguous":
        raise ValueError(f"Unsupported semantic category key mode: {key_mode}")
    return contiguous_categories


# Tagged segmentation captions


def clean_tagged_phrase(text) -> str:
    """Remove markup from a tagged phrase and normalize whitespace."""
    text = re.sub(
        r"<color>.*?</color>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    return " ".join(text.split())


def extract_tagged_phrases(text) -> list[str]:
    """Extract clean phrases from `<p>...</p>` spans.

    Color tags and any other nested tags are stripped. For example,
    `<p>person-0<color>(255,0,0)</color></p>` becomes `person-0`.
    """
    pattern = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
    return [
        clean_tagged_phrase(match.group(1))
        for match in pattern.finditer(str(text or ""))
    ]


def rich_caption_color_labels(text) -> dict[tuple[int, int, int], str]:
    """Map RGB colors in rich segmentation captions back to text labels.

    Expected caption format:
    `<p>person-0<color>(255,0,0)</color></p>`.
    The result is `{(255, 0, 0): "person-0"}`.
    """
    pattern = re.compile(
        r"<p>\s*(?P<label>.*?)\s*<color>\(\s*(?P<r>\d+)\s*,\s*(?P<g>\d+)\s*,\s*(?P<b>\d+)\s*\)</color>\s*</p>",
        re.IGNORECASE | re.DOTALL,
    )
    labels = {}
    for match in pattern.finditer(str(text or "")):
        label = clean_tagged_phrase(match.group("label"))
        if not label:
            continue
        color = (int(match.group("r")), int(match.group("g")), int(match.group("b")))
        labels[color] = label
    return labels


# GCG


def class_mask_from_raw_output(raw_mask, caption: str) -> dict:
    """Parse a generated color mask and caption into class-index masks.

    The model returns a raw RGB mask image and text containing ordered color
    definitions, for example:
    `<p>person-0<color>(255,0,0)</color></p>`.

    `class_mask` uses the same index order as `gcg_phrases`: index 0 is the
    first parsed phrase/color, index 1 is the second, and so on. Black is always
    appended to `class_define` as background for `to_multi_mask`.

    If no color tags are present, this intentionally keeps `gcg_phrases=[]` but
    uses a white dummy color before appending black. That preserves the legacy
    benchmark behavior: mask conversion still runs, but no fake phrase/segment
    is introduced by this helper.

    The caller is responsible for resizing or validating `raw_mask` before
    calling this function.
    """
    if isinstance(raw_mask, (str, Path)):
        raw_mask = Image.open(raw_mask)
    mask_image = raw_mask.convert("RGB")

    instances = extract_ordered_caption_instances(caption)
    phrases = [phrase for phrase, _ in instances]
    if len(instances) == 0:
        class_define = np.array([(255, 255, 255)], dtype=np.float32)
    else:
        class_define = np.array([rgb for _, rgb in instances], dtype=np.float32)
    if not np.any(np.all(class_define == np.array([0, 0, 0]), axis=1)):
        class_define = np.vstack(
            [class_define, np.array([[0, 0, 0]], dtype=np.float32)]
        )
    class_mask = to_multi_mask(mask_image, class_define)

    return {
        "mask_image": mask_image,
        "class_mask": class_mask,
        "class_define": class_define,
        "gcg_phrases": phrases,
        "gcg_caption": get_gcg_caption(caption),
        "raw_caption": caption or "",
    }


def gcg_prediction_from_raw_output(raw_mask, image: Image.Image, caption: str) -> dict:
    """Build GCG segmentation predictions from a raw mask and rich caption.

    This is the high-level variant used by visualization tools: it resizes the
    raw mask to the input image size, parses caption colors, and returns only
    non-empty binary masks together with their phrases and colors.
    """
    if isinstance(raw_mask, (str, Path)):
        raw_mask = Image.open(raw_mask)
    mask_image = raw_mask.convert("RGB")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, resample=Image.NEAREST)
    parsed = class_mask_from_raw_output(mask_image, caption)
    segmentation = []
    phrases = []
    colors = []
    for idx, phrase in enumerate(parsed["gcg_phrases"]):
        mask = parsed["class_mask"] == idx
        if not np.any(mask):
            continue
        segmentation.append(mask)
        phrases.append(phrase)
        colors.append(tuple(int(c) for c in parsed["class_define"][idx][:3]))
    return {
        "segmentation": segmentation,
        "gcg_phrases": phrases,
        "gcg_caption": parsed["gcg_caption"],
        "raw_caption": parsed["raw_caption"],
        "colors": colors,
    }


# Panoptic segmentation


def parse_panoptic_phrase(phrase, occurrence_by_base, used_ids, category_name_to_id):
    """
    Convert one generated panoptic phrase into category and segment ids.

    The model is expected to emit phrases such as `person-0`, `person-1`, or
    `traffic light-0`, where the optional numeric suffix is the local instance
    index for that category. The category name is normalized before lookup in
    `category_name_to_id`.

    Examples:
        `person-0` with `{"person": 1}` returns category id 1 and panoptic id
        `1000`.
        `person-3` with `{"person": 1}` returns panoptic id `1003`.
        `person` has no explicit suffix, so the next local index is taken from
        `occurrence_by_base["person"]`.
        If `person-0` already produced panoptic id `1000`, another conflicting
        `person-0` is shifted to the next free id, such as `1001`.
        Unknown categories return `(None, None, None, None)` so callers can skip
        the segment.
    """
    base_cat = phrase
    local_idx = None
    if "-" in phrase:
        maybe_base, idx_str = phrase.rsplit("-", 1)
        if idx_str.isdigit():
            base_cat = maybe_base
            local_idx = int(idx_str)

    norm = normalize_category(base_cat)
    if norm not in category_name_to_id:
        return None, None, None, None

    if local_idx is None:
        local_idx = occurrence_by_base[norm]

    category_id = int(category_name_to_id[norm])
    panoptic_id = int(category_id * 1000 + local_idx)
    while panoptic_id in used_ids:
        local_idx += 1
        panoptic_id = int(category_id * 1000 + local_idx)
    occurrence_by_base[norm] = max(occurrence_by_base[norm], local_idx + 1)
    used_ids.add(panoptic_id)
    return norm, category_id, panoptic_id, local_idx


def panoptic_prediction_from_raw_output(
    image: Image.Image,
    raw_mask,
    caption: str,
    categories=None,
    question: str = "",
    strict_categories: bool = False,
) -> dict:
    """Build a panoptic prediction object from raw image/caption outputs.

    The returned dict contains:
    - `id_map`: a 2D panoptic id map where each segment id is
      `category_id * 1000 + local_instance_index`.
    - `annotation`: COCO-style metadata with `segments_info`, `gcg_phrases`,
      and `gcg_caption`.
    - `categories`: normalized category metadata, including categories inferred
      from the prompt when `strict_categories=False`.
    - `raw`: the intermediate output from `class_mask_from_raw_output`.

    `question` is used only to pre-register prompt categories that may not
    appear in the dataset category list. When `strict_categories=True`, unknown
    generated phrases are skipped instead of creating new categories.
    """
    if isinstance(raw_mask, (str, Path)):
        raw_mask = Image.open(raw_mask)
    mask_image = raw_mask.convert("RGB")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, resample=Image.NEAREST)
    parsed = class_mask_from_raw_output(mask_image, caption)
    normalized_categories = list(normalize_categories(categories).values())
    name_to_category = {}
    max_category_id = max(
        (int(category["id"]) for category in normalized_categories), default=0
    )
    for item in normalized_categories:
        category_id = int(item["id"])
        name_to_category[normalize_category(str(item["name"]))] = item

    def register_category(name):
        nonlocal max_category_id
        normalized_name = normalize_category(str(name))
        if normalized_name in name_to_category:
            return name_to_category[normalized_name]
        if strict_categories:
            return None
        max_category_id += 1
        category = {"id": max_category_id, "name": str(name), "isthing": 1}
        normalized_categories.append(category)
        name_to_category[normalized_name] = category
        return category

    prompt_phrases = [
        clean_tagged_phrase(match.group(1))
        for match in re.finditer(
            r"<p>(.*?)</p>", str(question or ""), flags=re.IGNORECASE | re.DOTALL
        )
        if "<color>" not in match.group(1).lower()
    ]
    for prompt_phrase in prompt_phrases:
        base_name, separator, suffix = prompt_phrase.rpartition("-")
        register_category(
            base_name if separator and suffix.isdigit() else prompt_phrase
        )

    id_map = np.zeros_like(parsed["class_mask"], dtype=np.int32)
    segments_info = []
    segment_phrases = []
    occurrence_by_category = defaultdict(int)
    used_ids = set()
    for instance_id, phrase in enumerate(parsed["gcg_phrases"]):
        base_name, separator, suffix = phrase.rpartition("-")
        if separator and suffix.isdigit():
            local_idx = int(suffix)
        else:
            base_name = phrase
            local_idx = None

        normalized_name = normalize_category(base_name)
        category = name_to_category.get(normalized_name)
        if category is None:
            category = register_category(base_name)
        if category is None:
            continue

        mask = parsed["class_mask"] == instance_id
        if not np.any(mask):
            continue
        category_id = int(category["id"])
        if local_idx is None:
            local_idx = occurrence_by_category[category_id]
        panoptic_id = int(category_id * 1000 + local_idx)
        while panoptic_id in used_ids:
            local_idx += 1
            panoptic_id = int(category_id * 1000 + local_idx)
        occurrence_by_category[category_id] = max(
            occurrence_by_category[category_id], local_idx + 1
        )
        used_ids.add(panoptic_id)
        id_map[mask] = panoptic_id
        segments_info.append(
            {"id": panoptic_id, "category_id": category_id, "score": 1.0}
        )
        segment_phrases.append(phrase)

    return {
        "id_map": id_map,
        "annotation": {
            "segments_info": segments_info,
            "gcg_phrases": segment_phrases,
            "gcg_caption": parsed["gcg_caption"],
        },
        "categories": normalized_categories,
        "raw": parsed,
    }


# Detection


def is_number_list(value) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(x, (int, float)) for x in value
    )


def parse_coord_list(text: str) -> list[float]:
    """Parse a comma-separated coordinate list into floats.

    Invalid numeric tokens return an empty list so callers can skip malformed
    model fragments without raising.
    """
    coords = []
    for part in str(text).split(","):
        try:
            coords.append(float(part.strip()))
        except ValueError:
            return []
    return coords


def clip_normalized_coords(coords) -> list[float]:
    """Clip normalized coordinate values into the half-open visual range."""
    return [max(0.0, min(0.999, float(coord))) for coord in coords]


def parse_bbox_output(text_output: str, normalize_labels: bool = True) -> dict:
    """Parse detection text in `<p>label</p><bbox>[x0,y0,x1,y1]</bbox>` form.

    Returns `{label: [[x0, y0, x1, y1], ...]}` with coordinates kept in
    normalized space and clipped to `[0, 0.999]`. By default labels are
    normalized for category matching; OCR-style benchmarks can disable that to
    preserve the original case in `raw_response`.
    """
    results = {}
    s = str(text_output or "").strip().rstrip(" .\n")
    for part in s.split("<p>")[1:]:
        if "</p>" not in part:
            continue
        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :]

        bboxes = []
        while "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<bbox>[") : end]
            rest = rest[end + len("]</bbox>") :]
            try:
                coords = [x.strip() for x in coord_str.split(",")]
                if len(coords) == 4:
                    bboxes.append(clip_normalized_coords(coords))
            except (ValueError, AttributeError):
                continue

        if bboxes:
            label = normalize_category(category) if normalize_labels else category
            results[label] = bboxes
    return results


def parse_point_output(text_output: str) -> dict:
    """Parse point detection text in `<p>label</p><point>[x,y]</point>` form."""
    results = {}
    s = str(text_output or "").strip().rstrip(" .\n")
    for part in s.split("<p>")[1:]:
        if "</p>" not in part:
            continue
        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :]

        points = []
        while "<point>[" in rest:
            start = rest.find("<point>[")
            end = rest.find("]</point>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<point>[") : end]
            rest = rest[end + len("]</point>") :]
            try:
                coords = [x.strip() for x in coord_str.split(",")]
                if len(coords) == 2:
                    points.append(clip_normalized_coords(coords))
            except (ValueError, AttributeError):
                continue

        if points:
            results[normalize_category(category)] = points
    return results


def parse_keypoint_output(text_output: str) -> dict:
    """Parse keypoint benchmark output grouped by category and instance.

    Expected structure:
    `<p>person</p><bbox>[...]</bbox>left shoulder<kpt>[x,y]</kpt>...`.
    Optional `<ins>...</ins>` tags are ignored here; this parser returns a
    category-keyed dict where each value is a list of instances containing
    optional `bbox` and `keypoints`.

    Invisible keypoints encoded as `<kpt>unvisible</kpt>` are stored as
    `[-1, -1]` and converted to `None` by `convert_keypoints_to_pixel()`.
    """
    results = {}
    s = str(text_output or "").strip().rstrip(" .\n")
    for part in s.split("<p>")[1:]:
        if "</p>" not in part:
            continue

        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :].replace("<ins>", "").replace("</ins>", "")

        bbox = None
        if "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start != -1 and end != -1:
                coord_str = rest[start + len("<bbox>[") : end]
                try:
                    coords = [x.strip() for x in coord_str.split(",")]
                    if len(coords) == 4:
                        bbox = clip_normalized_coords(coords)
                except (ValueError, AttributeError):
                    pass
                rest = rest[:start] + rest[end + len("]</bbox>") :]

        keypoints = {}
        while "<kpt>" in rest:
            start = rest.find("<kpt>")
            end = rest.find("</kpt>", start)
            if start == -1 or end == -1:
                break

            keypoint_name = rest[:start].strip()
            content = rest[start + len("<kpt>") : end].strip()
            rest = rest[end + len("</kpt>") :]

            if not keypoint_name:
                continue
            if content == "unvisible":
                keypoints[keypoint_name] = [-1, -1]
                continue
            try:
                coords = [x.strip() for x in content.strip("[]").split(",")]
                if len(coords) == 2:
                    keypoints[keypoint_name] = clip_normalized_coords(coords)
            except (ValueError, AttributeError):
                continue

        if category:
            cat_clean = normalize_category(category)
            results.setdefault(cat_clean, [])
            instance = {}
            if bbox is not None:
                instance["bbox"] = bbox
            instance["keypoints"] = keypoints
            results[cat_clean].append(instance)
    return results


def convert_bbox_to_pixel(bbox_list, img_width, img_height):
    """Convert normalized bounding boxes to pixel coordinates."""
    pixel_bboxes = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        pixel_bboxes.append(
            [
                round(x1 * img_width, 2),
                round(y1 * img_height, 2),
                round(x2 * img_width, 2),
                round(y2 * img_height, 2),
            ]
        )
    return pixel_bboxes


def convert_point_to_pixel(points_list, img_width, img_height):
    """Convert normalized points to pixel coordinates."""
    pixel_points = []
    for point in points_list:
        x, y = [float(x) for x in point]
        pixel_points.append([x * img_width, y * img_height])
    return pixel_points


def convert_keypoints_to_pixel(keypoints_dict, img_width, img_height):
    """Convert normalized keypoints to pixels, preserving invisibility as None."""
    pixel_keypoints = {}
    for name, coords in keypoints_dict.items():
        x_norm, y_norm = coords
        if x_norm == -1 and y_norm == -1:
            pixel_keypoints[name] = None
        else:
            pixel_keypoints[name] = [
                round(x_norm * img_width, 2),
                round(y_norm * img_height, 2),
            ]
    return pixel_keypoints


def parse_detection_text_output(
    text_output: str, output_type: str, normalize_labels: bool = True
) -> dict:
    """Dispatch benchmark detection text parsing by output type.

    `output_type` must be one of `bbox`, `point`, or `keypoint`. The returned
    coordinates are still normalized; use `convert_detection_output_to_pixel()`
    when writing pixel-space predictions.
    """
    if output_type == "bbox":
        return parse_bbox_output(text_output, normalize_labels=normalize_labels)
    if output_type == "point":
        return parse_point_output(text_output)
    if output_type == "keypoint":
        return parse_keypoint_output(text_output)
    raise ValueError(f"Unsupported detection output type: {output_type}")


def convert_detection_output_to_pixel(
    parsed_output: dict,
    img_width: int,
    img_height: int,
    output_type: str,
) -> dict:
    """Convert parsed benchmark detection outputs from normalized to pixel space.

    The input should come from `parse_detection_text_output()`. The output shape
    mirrors the parser output so benchmark scripts can preserve their
    task-specific JSON format while changing only the coordinate system.
    """
    if output_type == "bbox":
        return {
            category: convert_bbox_to_pixel(items, img_width, img_height)
            for category, items in parsed_output.items()
        }
    if output_type == "point":
        return {
            category: convert_point_to_pixel(items, img_width, img_height)
            for category, items in parsed_output.items()
        }
    if output_type == "keypoint":
        converted = {}
        for category, instances in parsed_output.items():
            converted_instances = []
            for instance in instances:
                converted_instance = {
                    "keypoints": convert_keypoints_to_pixel(
                        instance.get("keypoints", {}), img_width, img_height
                    )
                }
                if instance.get("bbox"):
                    converted_instance["bbox"] = convert_bbox_to_pixel(
                        [instance["bbox"]], img_width, img_height
                    )[0]
                converted_instances.append(converted_instance)
            converted[category] = converted_instances
        return converted
    raise ValueError(f"Unsupported detection output type: {output_type}")


def normalize_bbox(coords, width: int, height: int):
    """Return a clipped pixel-space bbox from normalized or pixel coordinates.

    Values whose absolute maximum is <= 1.5 are treated as normalized. Larger
    values are treated as already being in pixel space. Invalid or degenerate
    boxes return `None`.
    """
    if not is_number_list(coords) or len(coords) != 4:
        return None
    vals = [float(x) for x in coords]
    if max(abs(x) for x in vals) <= 1.5:
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x0, y0, x1, y1 = vals
    x0, x1 = sorted(
        (max(0.0, min(float(width - 1), x0)), max(0.0, min(float(width - 1), x1)))
    )
    y0, y1 = sorted(
        (max(0.0, min(float(height - 1), y0)), max(0.0, min(float(height - 1), y1)))
    )
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_point(coords, width: int, height: int):
    """Return a clipped pixel-space point from normalized or pixel coordinates."""
    if not is_number_list(coords) or len(coords) != 2:
        return None
    vals = [float(x) for x in coords]
    if max(abs(x) for x in vals) <= 1.5:
        vals = [vals[0] * width, vals[1] * height]
    x = max(0.0, min(float(width - 1), vals[0]))
    y = max(0.0, min(float(height - 1), vals[1]))
    return [x, y]


def normalize_polygon(coords, width: int, height: int):
    """Return clipped polygon vertices from normalized or pixel coordinates."""
    if not is_number_list(coords) or len(coords) < 6 or len(coords) % 2 != 0:
        return None
    vals = [float(x) for x in coords]
    if max(abs(x) for x in vals) <= 1.5:
        vals = [x * (width if i % 2 == 0 else height) for i, x in enumerate(vals)]
    points = []
    for x, y in zip(vals[::2], vals[1::2]):
        points.append(
            (max(0.0, min(float(width - 1), x)), max(0.0, min(float(height - 1), y)))
        )
    return points if len(points) >= 3 else None


def normalize_keypoints(keypoints, width: int, height: int):
    """Normalize a keypoint-name-to-coordinate mapping into pixel space."""
    out = {}
    if not isinstance(keypoints, dict):
        return out
    for name, coords in keypoints.items():
        if coords is None:
            continue
        point = normalize_point(coords, width, height)
        if point is not None:
            out[str(name).replace("_", " ")] = point
    return out


def parse_detection_text(text: str) -> dict:
    """Parse generic detection text containing `<bbox>` and `<point>` tags.

    This parser is intentionally loose and is used by visualization/fallback
    paths. It keeps the label text as written in `<p>...</p>` and returns raw
    coordinate lists; `_detection_primitives_from_value()` later normalizes them
    according to the image size.
    """
    results = defaultdict(list)
    bbox_pattern = re.compile(r"<bbox>\s*\[([^\]]+)\]\s*</bbox>", re.IGNORECASE)
    point_pattern = re.compile(r"<point>\s*\[([^\]]+)\]\s*</point>", re.IGNORECASE)
    for part in str(text or "").split("<p>")[1:]:
        if "</p>" not in part:
            continue
        label, rest = part.split("</p>", 1)
        label = " ".join(label.split())
        for match in bbox_pattern.finditer(rest):
            coords = parse_coord_list(match.group(1))
            if len(coords) == 4:
                results[label].append(coords)
        for match in point_pattern.finditer(rest):
            coords = parse_coord_list(match.group(1))
            if len(coords) == 2:
                results[label].append(coords)
    return dict(results)


HUMAN_KEYPOINT_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]
ANIMAL_KEYPOINT_NAMES = [
    "left eye",
    "right eye",
    "nose",
    "neck",
    "root of tail",
    "left shoulder",
    "left elbow",
    "left front paw",
    "right shoulder",
    "right elbow",
    "right front paw",
    "left hip",
    "left knee",
    "left back paw",
    "right hip",
    "right knee",
    "right back paw",
]
KEYPOINT_NAMES = sorted(
    set(HUMAN_KEYPOINT_NAMES + ANIMAL_KEYPOINT_NAMES),
    key=len,
    reverse=True,
)


def _clean_keypoint_name(text: str) -> str:
    """Normalize free-form keypoint labels before matching known skeleton names."""
    text = re.sub(r"<.*?>", " ", str(text), flags=re.DOTALL)
    text = text.replace(":", " ").replace("=", " ")
    text = " ".join(text.split()).strip(" ,;.-")
    return text.lower()


def _infer_keypoint_type(keypoints: dict, keypoint_context: str = "") -> str:
    """Infer whether parsed keypoints follow the human or animal skeleton.

    Explicit context wins first (`human`, `person`, `coco`, `animal`, `ap-10k`).
    If no context is available, keypoint names unique to one skeleton are used
    as weak evidence.
    """
    context = str(keypoint_context or "").lower()
    if "animal" in context or "ap-10k" in context or "ap10k" in context:
        return "animal"
    if "human" in context or "person" in context or "coco" in context:
        return "human"
    names = {str(name).lower() for name in keypoints}
    animal_only_names = {
        "root of tail",
        "left front paw",
        "right front paw",
        "left back paw",
        "right back paw",
    }
    if names & animal_only_names:
        return "animal"
    if names & {
        "left wrist",
        "right wrist",
        "left ankle",
        "right ankle",
        "left ear",
        "right ear",
    }:
        return "human"
    return context


def _parse_keypoints_from_segment(segment: str) -> dict:
    """Extract known keypoint names and coordinate pairs from one instance span.

    The model may write either `left wrist<kpt>[x,y]</kpt>` or looser text such
    as `left wrist: [x,y]`. This helper accepts both forms and only keeps names
    that match the known human/animal keypoint vocabulary.
    """
    keypoints = {}
    without_bboxes = re.sub(
        r"<bbox>\s*\[[^\]]+\]\s*</bbox>",
        " ",
        str(segment),
        flags=re.IGNORECASE,
    )
    kpt_pattern = re.compile(
        r"(?P<name>[^<>]*?)<kpt>\s*\[([^\]]+)\]\s*</kpt>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in kpt_pattern.finditer(without_bboxes):
        name = _clean_keypoint_name(match.group("name"))
        if not name:
            continue
        for known_name in KEYPOINT_NAMES:
            if name.endswith(known_name):
                coords = parse_coord_list(match.group(2))
                if len(coords) == 2:
                    keypoints[known_name] = coords
                break

    lower_segment = without_bboxes.lower()
    for known_name in KEYPOINT_NAMES:
        if known_name in keypoints:
            continue
        pattern = re.compile(
            rf"\b{re.escape(known_name)}\b\s*[:=]?\s*\[([^\]]+)\]",
            re.IGNORECASE,
        )
        match = pattern.search(lower_segment)
        if match:
            coords = parse_coord_list(match.group(1))
            if len(coords) == 2:
                keypoints[known_name] = coords
    return keypoints


def parse_keypoint_text(
    text: str, image_size, keypoint_context: str = ""
) -> list[dict]:
    """Parse keypoint-rich text into visualization primitives.

    Each `<p>label</p>` block may contain one or more `<ins>...</ins>` instance
    spans. If no `<ins>` tags are present, the whole block is treated as one
    instance when it contains `<bbox>` or `<kpt>`. Returned primitives include
    optional bbox, normalized keypoints in pixel space, inferred skeleton type,
    and the original context string.
    """
    width, height = image_size
    primitives = []
    bbox_pattern = re.compile(r"<bbox>\s*\[([^\]]+)\]\s*</bbox>", re.IGNORECASE)
    for part in str(text or "").split("<p>")[1:]:
        if "</p>" not in part:
            continue
        label, rest = part.split("</p>", 1)
        label = " ".join(label.split())
        instance_segments = re.findall(
            r"<ins>(.*?)</ins>",
            rest,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not instance_segments and (
            "<kpt>" in rest.lower() or "<bbox>" in rest.lower()
        ):
            instance_segments = [rest]
        for segment in instance_segments:
            bbox_match = bbox_pattern.search(segment)
            bbox = None
            if bbox_match:
                bbox = normalize_bbox(
                    parse_coord_list(bbox_match.group(1)),
                    width,
                    height,
                )
            keypoints = normalize_keypoints(
                _parse_keypoints_from_segment(segment), width, height
            )
            if not bbox and not keypoints:
                continue
            keypoint_type = _infer_keypoint_type(keypoints, keypoint_context)
            primitives.append(
                {
                    "kind": "keypoint",
                    "label": label,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "keypoint_type": keypoint_type,
                    "keypoint_context": str(keypoint_context or "").lower(),
                }
            )
    return primitives


def parse_visual_prompt(question: str, image_size) -> list[dict]:
    """Extract visual prompt primitives embedded in the user question.

    Supported prompt tags are `<bbox>[...]</bbox>` and `<point>[...]</point>`.
    Returned primitives are marked with `is_prompt=True` so visualization code
    can distinguish prompts from model predictions.
    """
    if not question:
        return []
    width, height = image_size
    primitives = []
    pattern = re.compile(
        r"<(?P<kind>bbox|point)>\s*\[([^\]]+)\]\s*</(?P=kind)>", re.IGNORECASE
    )
    for match in pattern.finditer(str(question)):
        kind = match.group("kind").lower()
        coords = parse_coord_list(match.group(2))
        if kind == "bbox":
            bbox = normalize_bbox(coords, width, height)
            if bbox is not None:
                primitives.append(
                    {
                        "kind": "bbox",
                        "label": "bbox prompt",
                        "bbox": bbox,
                        "is_prompt": True,
                    }
                )
        elif kind == "point":
            point = normalize_point(coords, width, height)
            if point is not None:
                primitives.append(
                    {
                        "kind": "point",
                        "label": "point prompt",
                        "point": point,
                        "is_prompt": True,
                    }
                )
    return primitives


def _decode_mask_primitive(rle, image_size, label, color_index):
    """Decode one COCO RLE mask into a visualization primitive."""
    if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
        return None
    width, height = image_size
    mask = decode_rle(rle)
    if mask.shape[:2] != (height, width):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
        mask = np.asarray(mask_img) > 0
    area = int(mask.sum())
    if area == 0:
        return None
    return {
        "kind": "mask",
        "label": label,
        "mask": mask,
        "area": area,
        "color_index": color_index,
    }


def _detection_primitives_from_value(
    value, label, image_size, color_index=0, keypoint_context=""
) -> list[dict]:
    """Normalize heterogeneous detection outputs into visualization primitives.

    Accepted value shapes include:
    - Text with `<bbox>`/`<point>` tags.
    - Dicts such as `{"bbox": ..., "keypoints": ...}`.
    - COCO RLE mask dicts with `size` and `counts`.
    - Number lists representing points, boxes, or polygons.
    - Nested dict/list structures keyed by labels.

    The returned primitives use a common schema with `kind` set to one of
    `point`, `bbox`, `polygon`, `mask`, or `keypoint`.
    """
    width, height = image_size
    primitives = []
    if value is None:
        return primitives

    if isinstance(value, str):
        return _detection_primitives_from_value(
            parse_detection_text(value),
            label,
            image_size,
            color_index,
            keypoint_context,
        )

    if isinstance(value, dict):
        if "type" in value and "coords" in value:
            vtype = str(value.get("type", "")).lower()
            return _detection_primitives_from_value(
                value.get("coords"),
                label if label else vtype,
                image_size,
                color_index,
                keypoint_context,
            )
        if "bbox" in value or "keypoints" in value:
            bbox = normalize_bbox(value.get("bbox"), width, height)
            kpts = normalize_keypoints(value.get("keypoints"), width, height)
            phrase = value.get("phrase") or label
            keypoint_type = (
                value.get("keypoint_type")
                or value.get("skeleton_type")
                or keypoint_context
            )
            if bbox or kpts:
                primitives.append(
                    {
                        "kind": "keypoint",
                        "label": str(phrase),
                        "bbox": bbox,
                        "keypoints": kpts,
                        "keypoint_type": (
                            str(keypoint_type).lower() if keypoint_type else ""
                        ),
                        "keypoint_context": (
                            str(keypoint_context).lower() if keypoint_context else ""
                        ),
                        "color_index": color_index,
                    }
                )
            return primitives
        if "size" in value and "counts" in value:
            mask = _decode_mask_primitive(value, image_size, str(label), color_index)
            return [mask] if mask is not None else []
        group = []
        for idx, (sub_label, sub_value) in enumerate(value.items()):
            group.extend(
                _detection_primitives_from_value(
                    sub_value, sub_label, image_size, idx, keypoint_context
                )
            )
        return group

    if is_number_list(value):
        if len(value) == 2:
            point = normalize_point(value, width, height)
            if point is not None:
                primitives.append(
                    {
                        "kind": "point",
                        "label": str(label),
                        "point": point,
                        "color_index": color_index,
                    }
                )
        elif len(value) == 4:
            bbox = normalize_bbox(value, width, height)
            if bbox is not None:
                primitives.append(
                    {
                        "kind": "bbox",
                        "label": str(label),
                        "bbox": bbox,
                        "color_index": color_index,
                    }
                )
        else:
            polygon = normalize_polygon(value, width, height)
            if polygon is not None:
                primitives.append(
                    {
                        "kind": "polygon",
                        "label": str(label),
                        "polygon": polygon,
                        "color_index": color_index,
                    }
                )
        return primitives

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            primitives.extend(
                _detection_primitives_from_value(
                    item, label, image_size, idx, keypoint_context
                )
            )
    return primitives


def parse_detection_output(
    output, image_size, fallback_label="object", keypoint_context=""
) -> list[dict]:
    """Public entry point for visualization-oriented detection parsing.

    This function accepts either raw model text or already-structured prediction
    objects and returns a flat list of primitives suitable for overlay drawing.
    It first tries the keypoint parser when the context or text indicates
    keypoints, then falls back to generic bbox/point parsing and finally to
    recursive structured-value normalization.
    """
    if isinstance(output, str):
        if "keypoint" in str(keypoint_context).lower() or "<kpt>" in output.lower():
            primitives = parse_keypoint_text(output, image_size, keypoint_context)
            if primitives:
                return primitives
        output = parse_detection_text(output)
    if isinstance(output, dict):
        primitives = []
        for idx, (label, value) in enumerate(output.items()):
            primitives.extend(
                _detection_primitives_from_value(
                    value, label, image_size, idx, keypoint_context
                )
            )
        return primitives
    return _detection_primitives_from_value(
        output, fallback_label, image_size, 0, keypoint_context
    )
