# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Download OCR-VQA images with the filenames expected by LLaVA-v1.5."""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


METADATA_JSON = Path("datas/train_data/llava_images/ocr_vqa/dataset.json")
IMAGE_DIR = Path("datas/train_data/llava_images/ocr_vqa/images")
FAILURE_JSONL = Path("datas/train_data/llava_images/ocr_vqa/download_errors.jsonl")
REQUEST_TIMEOUT = 120.0
DOWNLOAD_RETRIES = 3


def validate_downloaded_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("downloaded file is empty")
    with Image.open(path) as image:
        image.verify()


def download_one_image(
    image_id: str,
    url: str,
    image_dir: Path,
    timeout: float,
    retries: int,
) -> Tuple[str, Optional[str]]:
    if Path(image_id).name != image_id:
        return "failed", f"unsafe image id: {image_id!r}"

    output_path = image_dir / f"{image_id}.jpg"
    if output_path.exists():
        try:
            validate_downloaded_image(output_path)
            return "skipped", None
        except Exception:
            output_path.unlink()

    last_error = "unknown download error"
    for _ in range(retries):
        temporary_path: Optional[Path] = None
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SenseNova-Vision data preparation"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(
                    dir=image_dir,
                    prefix=f".{image_id}.",
                    suffix=".part",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    shutil.copyfileobj(response, temporary_file)
            validate_downloaded_image(temporary_path)
            os.replace(temporary_path, output_path)
            return "downloaded", None
        except Exception as exc:
            last_error = str(exc)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return "failed", last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Concurrent image downloads (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")

    with METADATA_JSON.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("OCR-VQA metadata must be a JSON object keyed by image id")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[str, str]] = []
    failures: List[Dict[str, str]] = []
    for image_id, item in metadata.items():
        if not isinstance(item, dict) or not isinstance(item.get("imageURL"), str):
            failures.append(
                {
                    "image_id": str(image_id),
                    "url": "",
                    "error": "missing imageURL",
                }
            )
            continue
        tasks.append((str(image_id), item["imageURL"]))

    counts = {"downloaded": 0, "skipped": 0, "failed": len(failures)}
    batch_size = max(256, args.num_workers * 16)
    with (
        ThreadPoolExecutor(max_workers=args.num_workers) as executor,
        tqdm(
            total=len(tasks),
            desc="Downloading OCR-VQA images",
            unit="image",
        ) as progress,
    ):
        for offset in range(0, len(tasks), batch_size):
            future_to_task = {
                executor.submit(
                    download_one_image,
                    image_id,
                    url,
                    IMAGE_DIR,
                    REQUEST_TIMEOUT,
                    DOWNLOAD_RETRIES,
                ): (image_id, url)
                for image_id, url in tasks[offset : offset + batch_size]
            }
            for future in as_completed(future_to_task):
                image_id, url = future_to_task[future]
                status, error = future.result()
                counts[status] += 1
                if error is not None:
                    failures.append({"image_id": image_id, "url": url, "error": error})
                progress.update()

    FAILURE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FAILURE_JSONL.open("w", encoding="utf-8") as handle:
        for item in sorted(failures, key=lambda value: value["image_id"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({**counts, "failure_jsonl": str(FAILURE_JSONL)}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
