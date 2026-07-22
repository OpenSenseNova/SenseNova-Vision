# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Convert the public LLaVA-v1.5 mixture to a media-validated JSONL."""

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


SOURCE_PREFIXES = ("coco", "gqa", "ocr_vqa", "textvqa", "vg")
EXPECTED_COUNT = 624254
IMAGE_ROOT = Path("datas/train_data/llava_images")
OUTPUT_JSONL = Path(
    "jsonl_generate/train_jsonls/understanding/llava_v1_5_mix665k.jsonl"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_media_paths(record: Any) -> Tuple[List[str], Optional[str]]:
    if not isinstance(record, dict):
        return [], "record is not a JSON object"

    image = record.get("image")
    if isinstance(image, str):
        media_paths = [image]
    elif isinstance(image, list) and all(isinstance(item, str) for item in image):
        media_paths = image
    else:
        return [], "image must be a string or a list of strings"

    if not media_paths or any(not item for item in media_paths):
        return [], "image is empty"
    return media_paths, None


def resolve_media_path(image_root: Path, media_path: str) -> Path:
    posix_path = PurePosixPath(media_path)
    if posix_path.is_absolute() or ".." in posix_path.parts or "://" in media_path:
        raise ValueError(f"image path is not repository-relative: {media_path}")
    return image_root / Path(*posix_path.parts)


def index_media_errors(
    records: List[Any],
    image_root: Path,
    workers: int,
) -> Dict[str, Optional[str]]:
    """Resolve and stat each distinct media path once."""
    resolved_paths: Dict[str, Path] = {}
    media_errors: Dict[str, Optional[str]] = {}

    for record in tqdm(records, desc="Indexing LLaVA media", unit="row"):
        media_paths, error = get_media_paths(record)
        if error is not None:
            continue
        for media_path in media_paths:
            if media_path in resolved_paths or media_path in media_errors:
                continue
            try:
                resolved_paths[media_path] = resolve_media_path(image_root, media_path)
            except ValueError as exc:
                media_errors[media_path] = str(exc)

    def check_path(item: Tuple[str, Path]) -> Tuple[str, Optional[str]]:
        media_path, resolved_path = item
        if resolved_path.is_file():
            return media_path, None
        return media_path, f"image does not exist: {media_path}"

    items = iter(resolved_paths.items())
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(
            total=len(resolved_paths),
            desc="Checking distinct LLaVA media",
            unit="image",
        ) as progress,
    ):
        while batch := list(islice(items, 4096)):
            for media_path, error in executor.map(check_path, batch):
                media_errors[media_path] = error
                progress.update()
    return media_errors


def validate_indexed_record(
    record: Any,
    media_errors: Dict[str, Optional[str]],
) -> Optional[str]:
    media_paths, error = get_media_paths(record)
    if error is not None:
        return error
    for media_path in media_paths:
        error = media_errors[media_path]
        if error is not None:
            return error
    return None


def validate_first_cases(
    jsonl_path: Path,
    image_root: Path,
) -> List[Dict[str, Any]]:
    first_cases: Dict[str, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(first_cases) == len(SOURCE_PREFIXES):
                break
            if not line.strip():
                continue

            record = json.loads(line)
            media_paths, error = get_media_paths(record)
            if error is not None:
                raise ValueError(f"line {line_number}: {error}")

            source = media_paths[0].split("/", 1)[0]
            if source not in SOURCE_PREFIXES or source in first_cases:
                continue

            decoded_media = []
            for media_path in media_paths:
                resolved_path = resolve_media_path(image_root, media_path)
                if not resolved_path.is_file():
                    raise FileNotFoundError(
                        f"{source} first case line {line_number}: {resolved_path}"
                    )
                with Image.open(resolved_path) as image:
                    image.load()
                    decoded_media.append(
                        {
                            "image": media_path,
                            "size": list(image.size),
                            "mode": image.mode,
                        }
                    )

            first_cases[source] = {
                "source": source,
                "line": line_number,
                "media": decoded_media,
            }

    missing_sources = [
        source for source in SOURCE_PREFIXES if source not in first_cases
    ]
    if missing_sources:
        raise ValueError(
            f"missing first-case records for sources: {', '.join(missing_sources)}"
        )
    return [first_cases[source] for source in SOURCE_PREFIXES]


def convert_json_to_jsonl(
    input_json: Path,
    image_root: Path,
    output_jsonl: Path,
    reject_jsonl: Path,
    workers: int,
    expected_count: int,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    records = load_json(input_json)
    if not isinstance(records, list):
        raise ValueError("LLaVA annotation input must be a JSON array")
    if output_jsonl.resolve() == reject_jsonl.resolve():
        raise ValueError("output JSONL and reject JSONL must be different files")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    media_errors = index_media_errors(records, image_root, workers)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reject_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_temporary: Optional[Path] = None
    reject_temporary: Optional[Path] = None
    accepted = 0
    rejected = 0
    first_cases: List[Dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_jsonl.parent,
            prefix=f".{output_jsonl.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            output_temporary = Path(output_handle.name)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=reject_jsonl.parent,
                prefix=f".{reject_jsonl.name}.",
                suffix=".tmp",
                delete=False,
            ) as reject_handle:
                reject_temporary = Path(reject_handle.name)
                for record in tqdm(
                    records,
                    desc="Converting LLaVA annotations",
                    unit="row",
                ):
                    error = validate_indexed_record(record, media_errors)
                    serialized = json.dumps(record, ensure_ascii=False) + "\n"
                    if error is None:
                        output_handle.write(serialized)
                        accepted += 1
                    else:
                        reject_handle.write(serialized)
                        rejected += 1

        os.replace(reject_temporary, reject_jsonl)
        reject_temporary = None
        if accepted == expected_count:
            assert output_temporary is not None
            first_cases = validate_first_cases(output_temporary, image_root)
            os.replace(output_temporary, output_jsonl)
            output_temporary = None
    finally:
        if output_temporary is not None:
            output_temporary.unlink(missing_ok=True)
        if reject_temporary is not None:
            reject_temporary.unlink(missing_ok=True)

    return (
        {
            "source_records": len(records),
            "accepted_records": accepted,
            "rejected_records": rejected,
        },
        first_cases,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Threads used to check distinct media paths (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reject_jsonl = args.input_json.with_name(f"{args.input_json.stem}_rejected.jsonl")
    summary, first_cases = convert_json_to_jsonl(
        input_json=args.input_json,
        image_root=IMAGE_ROOT,
        output_jsonl=OUTPUT_JSONL,
        reject_jsonl=reject_jsonl,
        workers=args.num_workers,
        expected_count=EXPECTED_COUNT,
    )

    report = {
        **summary,
        "output_jsonl": str(OUTPUT_JSONL),
        "output_updated": summary["accepted_records"] == EXPECTED_COUNT,
        "reject_jsonl": str(reject_jsonl),
        "first_cases": first_cases,
    }
    print(json.dumps(report, indent=2))

    if summary["accepted_records"] != EXPECTED_COUNT:
        print(
            f"Expected {EXPECTED_COUNT} accepted records, but produced "
            f"{summary['accepted_records']}. Inspect {reject_jsonl}."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
