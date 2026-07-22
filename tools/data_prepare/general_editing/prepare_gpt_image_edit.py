# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Prepare GPT-Image-Edit-1.5M JSONL files from official training JSON."""

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, NamedTuple, Sequence

from PIL import Image

from _json_array import iter_json_array, write_jsonl_record


ANNOTATION_DIR = Path("datas/train_data/gpt_image_edit/annotations/training_json")
IMAGE_ROOT = Path("datas/train_data/gpt_image_edit/gpt-edit")
OUTPUT_DIR = Path("jsonl_generate/train_jsonls/editing")


class Component(NamedTuple):
    filename: str
    prefix: str


class Family(NamedTuple):
    components: Sequence[Component]
    output_name: str
    expected_count: int


FAMILIES = {
    "hqedit": Family(
        components=(
            Component("hqedit_gpt_edit.json", "hqedit/edit"),
            Component("hqedit_gpt_generate.json", "hqedit/generate"),
        ),
        output_name="gpt_image_edit_hqedit.jsonl",
        expected_count=183182,
    ),
    "omniedit": Family(
        components=(
            Component("omniedit_gpt.json", "omniedit"),
            Component("omniedit_gpt_rewrite.json", "omniedit"),
            Component("complexedit_gpt.json", "omniedit/complex-edit"),
        ),
        output_name="gpt_image_edit_omniedit.jsonl",
        expected_count=1270385,
    ),
    "ultraedit": Family(
        components=(Component("ultraedit_gpt.json", "ultraedit"),),
        output_name="gpt_image_edit_ultraedit.jsonl",
        expected_count=100008,
    ),
}


def prefix_media_path(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("image path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "://" in value:
        raise ValueError(f"image path is not repository-relative: {value}")
    return str(PurePosixPath(prefix) / path)


def convert_record(record: Any, prefix: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record is not a JSON object")
    images = record.get("image")
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError("image must contain exactly an input and output path")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    converted = dict(record)
    converted["image"] = [prefix_media_path(value, prefix) for value in images]
    return converted


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


def convert_family(
    name: str,
    family: Family,
    annotation_dir: Path,
    image_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_path = output_dir / family.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    total = 0
    first_cases = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for component in family.components:
                input_path = annotation_dir / component.filename
                component_count = 0
                for component_count, source_record in enumerate(
                    iter_json_array(input_path), start=1
                ):
                    try:
                        record = convert_record(source_record, component.prefix)
                    except ValueError as exc:
                        raise ValueError(
                            f"{component.filename} record {component_count}: {exc}"
                        ) from exc
                    if component_count == 1:
                        first_cases.append(
                            {
                                "component": component.filename,
                                "media": decode_first_case(record, image_root),
                            }
                        )
                    write_jsonl_record(output, record)
                total += component_count

        if total != family.expected_count:
            raise ValueError(
                f"{name}: expected {family.expected_count} records, converted {total}"
            )
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "dataset": name,
        "records": total,
        "output_jsonl": str(output_path),
        "first_cases": first_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(FAMILIES),
        default=tuple(FAMILIES),
        help="Dataset families to prepare (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.datasets
    reports = [
        convert_family(
            name,
            FAMILIES[name],
            ANNOTATION_DIR,
            IMAGE_ROOT,
            OUTPUT_DIR,
        )
        for name in names
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
