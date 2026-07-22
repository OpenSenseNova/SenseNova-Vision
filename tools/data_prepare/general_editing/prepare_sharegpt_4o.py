# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Convert the public ShareGPT-4o image-edit metadata to training JSONL."""

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from PIL import Image

from _json_array import iter_json_array, write_jsonl_record


INPUT_JSON = Path("datas/train_data/sharegpt_4o/text_and_image_to_image.json")
IMAGE_ROOT = Path("datas/train_data/sharegpt_4o")
OUTPUT_JSONL = Path("jsonl_generate/train_jsonls/editing/sharegpt_4o_edit.jsonl")
EXPECTED_COUNT = 46539


def safe_media_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "://" in value:
        raise ValueError(f"{field} is not a safe relative path: {value}")
    return value


def convert_record(record: Any) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record is not a JSON object")
    input_images = record.get("input_image")
    if not isinstance(input_images, list) or len(input_images) != 1:
        raise ValueError("input_image must contain exactly one image")
    input_image = safe_media_path(input_images[0], "input_image[0]")
    output_image = safe_media_path(record.get("output_image"), "output_image")
    prompt = record.get("input_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input_prompt must be a non-empty string")
    return {
        "image": [input_image, output_image],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": "<gen_image>"},
        ],
    }


def decode_first_case(record: Dict[str, Any], image_root: Path) -> List[Dict[str, Any]]:
    decoded = []
    for media_path in record["image"]:
        path = image_root.joinpath(*PurePosixPath(media_path).parts)
        if not path.is_file():
            raise FileNotFoundError(f"first-case image does not exist: {path}")
        with Image.open(path) as image:
            image.load()
            decoded.append(
                {"image": media_path, "size": list(image.size), "mode": image.mode}
            )
    return decoded


def convert(
    input_json: Path,
    image_root: Path,
    output_jsonl: Path,
    expected_count: int,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    count = 0
    first_case: List[Dict[str, Any]] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_jsonl.parent,
            prefix=f".{output_jsonl.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for count, source_record in enumerate(iter_json_array(input_json), start=1):
                try:
                    record = convert_record(source_record)
                except ValueError as exc:
                    raise ValueError(f"record {count}: {exc}") from exc
                if first_case is None:
                    first_case = decode_first_case(record, image_root)
                write_jsonl_record(output, record)

        if count != expected_count:
            raise ValueError(
                f"expected {expected_count} records, but converted {count}"
            )
        os.replace(temporary, output_jsonl)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "records": count,
        "output_jsonl": str(output_jsonl),
        "first_case": first_case,
    }


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    parse_args()
    report = convert(
        INPUT_JSON,
        IMAGE_ROOT,
        OUTPUT_JSONL,
        EXPECTED_COUNT,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
