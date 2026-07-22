# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for streaming public JSON arrays into atomic JSONL files."""

import json
from pathlib import Path
from typing import Any, Iterator, TextIO


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Yield values from a top-level JSON array without loading it into RAM."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            chunk = handle.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = chunk == ""

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        refill()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"{path}: expected a top-level JSON array")
        position += 1
        first = True

        while True:
            skip_whitespace()
            if position >= len(buffer):
                raise ValueError(f"{path}: unterminated JSON array")
            if buffer[position] == "]":
                position += 1
                break

            if not first:
                if buffer[position] != ",":
                    raise ValueError(f"{path}: expected ',' between array values")
                position += 1
                skip_whitespace()

            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    skip_whitespace()

            yield value
            first = False

        skip_whitespace()
        if position != len(buffer):
            raise ValueError(f"{path}: unexpected data after the JSON array")


def write_jsonl_record(handle: TextIO, record: Any) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
