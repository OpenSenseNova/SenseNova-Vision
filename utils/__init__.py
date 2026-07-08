import hashlib
import json
import os
import re
import string
from typing import Any, Optional


def safe_stem(text: Any) -> str:
    valid = f"-_.{string.ascii_letters}{string.digits}"
    return "".join(ch if ch in valid else "_" for ch in str(text))[:180] or "sample"


def safe_filename_part(value, max_len=80):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    safe = safe or "item"
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
    return f"{safe[: max_len - 11]}_{digest}"


def ensure_dir(path: str | os.PathLike | None):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json_or_jsonl(path: str) -> Any:
    if str(path).endswith(".jsonl"):
        return load_jsonl(path)
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl_split(input_jsonl, total_test_length=None, total_split=1, split_num=0):
    with open(input_jsonl, "r") as f:
        lines = f.readlines()

    total_len = total_test_length or len(lines)
    lines = lines[:total_len]

    split_size = (total_len + total_split - 1) // total_split
    start_idx = split_num * split_size
    end_idx = min(start_idx + split_size, total_len)
    split_lines = lines[start_idx:end_idx]

    print(
        f"Processing split {split_num}/{total_split}, "
        f"range [{start_idx}, {end_idx}), total {len(split_lines)} samples"
    )
    return split_lines, start_idx, end_idx


def image_path_for_item(
    item: dict, image_root: Optional[str] = None, data_path: Optional[str] = None
) -> Optional[str]:
    image = item.get("image") or item.get("image_path") or item.get("file_name")
    if isinstance(image, list):
        image = image[0]
    if not image:
        return None
    if image_root and not os.path.isabs(str(image)):
        candidate = os.path.join(image_root, os.path.basename(str(image)))
        if os.path.exists(candidate):
            return candidate
    return resolve_path(str(image), data_path)


def resolve_path(path: str | None, base_dir: str | None) -> str | None:
    if base_dir is None or path is None:
        return path

    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def safe_get_single_image_from_output(output_image):
    """Normalize a SenseNova-Vision image output to a single image object."""
    if isinstance(output_image, list):
        return output_image[0]
    return output_image


def normalize_category(category: str) -> str:
    """Normalize category names for matching."""
    return (
        category.replace("-merged", "")
        .replace("-other", "")
        .replace("-stuff", "")
        .replace("-negative", "")
        .replace("-", " ")
        .lower()
        .strip()
    )


def get_gcg_caption(caption):
    caption = re.sub(r"<.*?>", "", caption)
    caption = re.sub(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def extract_ordered_caption_instances(caption):
    """
    Return ordered (phrase, rgb) entries from colored caption tags.

    This preserves repeated phrases such as multiple instances of the same
    object name, unlike a phrase->color dict.
    """
    if not caption:
        text = ""
    elif isinstance(caption, (list, tuple)):
        text = "\n".join(str(item) for item in caption if item is not None)
    elif isinstance(caption, dict):
        text = json.dumps(caption, ensure_ascii=False)
    else:
        text = str(caption)

    pattern = re.compile(
        r"<p>\s*(.*?)\s*<color>\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*</color>\s*</p>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    instances = []
    for match in pattern.finditer(text):
        phrase = re.sub(r"<.*?>", "", match.group(1)).strip()
        rgb = tuple(int(match.group(i)) for i in range(2, 5))
        instances.append((phrase, rgb))
    return instances
