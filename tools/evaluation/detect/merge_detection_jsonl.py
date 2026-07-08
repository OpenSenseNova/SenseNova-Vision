#!/usr/bin/env python3
"""Merge sharded detection benchmark JSONL outputs.

The detection inference script writes shard files like ``COCO_000.jsonl`` or
``point.COCO_003.jsonl``.  The evaluator consumes dataset-level files like
``COCO.jsonl`` and ``point.COCO.jsonl``.  This helper merges all shard groups in
one directory into the evaluator-friendly filenames.  By default, shard files
are removed after their merged file is written successfully.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path


SHARD_RE = re.compile(r"^(.+)_(\d+)\.jsonl$")


def iter_jsonl_lines(path: Path):
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if line:
                yield line


def find_shard_groups(folder: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in folder.glob("*.jsonl"):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        prefix, index = match.groups()
        groups.setdefault(prefix, []).append((int(index), path))
    return groups


def merge_group(prefix: str, shards: list[tuple[int, Path]], folder: Path) -> int:
    output_path = folder / f"{prefix}.jsonl"
    shards = sorted(shards, key=lambda item: (item[0], item[1].name))
    line_count = 0

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(folder)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for _, shard_path in shards:
                for line in iter_jsonl_lines(shard_path):
                    out.write(line)
                    out.write("\n")
                    line_count += 1
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    print(
        f"Merged {len(shards)} shard(s), {line_count} line(s): "
        f"{output_path.name}"
    )
    return line_count


def delete_shards(shards: list[tuple[int, Path]]) -> None:
    for _, shard_path in shards:
        shard_path.unlink()
        print(f"Removed shard: {shard_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge detection JSONL shards named <prefix>_<number>.jsonl."
    )
    parser.add_argument(
        "folder",
        help="Directory containing detection JSONL files.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep shard files after their merged file is written successfully.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        return 2

    groups = find_shard_groups(folder)
    if not groups:
        print(f"No detection JSONL shards found in: {folder}")
        return 0

    total_lines = 0
    for prefix in sorted(groups):
        shards = groups[prefix]
        total_lines += merge_group(prefix, shards, folder)
        if not args.keep_shards:
            delete_shards(shards)

    print(
        f"Merge complete: {len(groups)} group(s), {total_lines} total line(s), "
        f"delete_shards={not args.keep_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
