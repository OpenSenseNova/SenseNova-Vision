# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild image-generation training JSONL from canonical BLIP3o keys."""

import argparse
import gzip
import io
import json
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, NamedTuple, TextIO

from PIL import Image


DEFAULT_KEY_LIST = Path(
    "jsonl_generate/train_jsonls/image_generation/keys/"
    "BLIP3o-Pretrain-Long-Caption.keys.txt.gz"
)
DEFAULT_JSONL_OUT = Path(
    "jsonl_generate/train_jsonls/image_generation/BLIP3o-Pretrain-Long-Caption.jsonl"
)
DATASETS = {
    "BLIP3o-Pretrain-Long-Caption",
    "BLIP3o-Pretrain-Short-Caption",
}


class CanonicalKey(NamedTuple):
    value: str
    dataset: str
    archive: str
    member: str

    @property
    def image_path(self) -> str:
        return str(PurePosixPath(self.archive) / self.member)


def parse_canonical_key(value: str) -> CanonicalKey:
    """Parse BLIP3o/<dataset>/<tar stem>/<image member>."""
    path = PurePosixPath(value)
    parts = path.parts
    if len(parts) != 4 or parts[0] != "BLIP3o":
        raise ValueError("key must be BLIP3o/<dataset>/<tar stem>/<image member>")
    _, dataset, archive, member = parts
    if dataset not in DATASETS:
        raise ValueError(f"unsupported BLIP3o dataset: {dataset}")
    if not archive or archive in {".", ".."}:
        raise ValueError(f"invalid archive name: {archive!r}")
    member_path = PurePosixPath(member)
    if (
        member_path.is_absolute()
        or len(member_path.parts) != 1
        or member in {".", ".."}
        or member_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
    ):
        raise ValueError(f"invalid image member: {member!r}")
    return CanonicalKey(value, dataset, archive, member)


def open_key_list(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def resolve_dataset_root(blip3o_root: Path, dataset: str) -> Path:
    candidates = (
        blip3o_root / dataset,
        blip3o_root / "BLIP3o" / dataset,
        blip3o_root,
    )
    for candidate in candidates:
        if candidate.is_dir() and (
            candidate.name == dataset
            or (candidate / dataset).is_dir()
            or any(candidate.glob("*.tar"))
        ):
            if candidate.name == dataset or any(candidate.glob("*.tar")):
                return candidate
            return candidate / dataset
    raise FileNotFoundError(f"cannot find {dataset} below BLIP3o root: {blip3o_root}")


def caption_member_name(image_member: str) -> str:
    return str(PurePosixPath(image_member).with_suffix(".txt"))


def image_metadata(image_bytes: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        width, height = image.size
        image_format = image.format or PurePosixPath(image.filename or "").suffix[1:]
    return width, height, image_format.upper()


class ArchiveReader:
    """Read image/text pairs from one extracted shard or tar archive."""

    def __init__(self, dataset_root: Path, archive: str):
        self.archive = archive
        self.directory = dataset_root / archive
        self.tar_path = dataset_root / f"{archive}.tar"
        self.tar: tarfile.TarFile | None = None
        self.members: dict[str, tarfile.TarInfo] = {}

        if self.directory.is_dir():
            return
        if not self.tar_path.is_file():
            raise FileNotFoundError(
                f"missing extracted shard or tar: {self.directory} / {self.tar_path}"
            )
        self.tar = tarfile.open(self.tar_path, mode="r:*")
        self.members = {
            member.name.removeprefix("./"): member
            for member in self.tar.getmembers()
            if member.isfile()
        }

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read(self, member: str) -> bytes:
        if self.directory.is_dir():
            path = self.directory / member
            if not path.is_file():
                raise FileNotFoundError(f"missing shard member: {path}")
            return path.read_bytes()

        assert self.tar is not None
        info = self.members.get(member)
        if info is None:
            raise FileNotFoundError(f"missing tar member: {self.tar_path}:{member}")
        extracted: BinaryIO | None = self.tar.extractfile(info)
        if extracted is None:
            raise FileNotFoundError(f"cannot read tar member: {self.tar_path}:{member}")
        with extracted:
            return extracted.read()

    def read_pair(self, image_member: str) -> tuple[bytes, str]:
        image_bytes = self.read(image_member)
        caption_bytes = self.read(caption_member_name(image_member))
        return image_bytes, caption_bytes.decode("utf-8").strip()


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        CREATE TABLE entries (
            position INTEGER PRIMARY KEY,
            canonical_key TEXT NOT NULL UNIQUE,
            dataset TEXT NOT NULL,
            archive TEXT NOT NULL,
            member TEXT NOT NULL,
            caption TEXT,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            file_format TEXT
        );
        """
    )
    return connection


def ingest_keys(connection: sqlite3.Connection, key_list: Path) -> int:
    batch = []
    count = 0
    with open_key_list(key_list) as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.rstrip("\r\n")
            if not value:
                raise ValueError(f"{key_list}:{line_number}: empty key")
            try:
                key = parse_canonical_key(value)
            except ValueError as exc:
                raise ValueError(f"{key_list}:{line_number}: {exc}") from exc
            batch.append((count, key.value, key.dataset, key.archive, key.member))
            count += 1
            if len(batch) >= 10_000:
                try:
                    connection.executemany(
                        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                        batch,
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"duplicate key at or before line {line_number}"
                    ) from exc
                batch.clear()
    if batch:
        try:
            connection.executemany(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                batch,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate key at or before line {count}") from exc
    if count == 0:
        raise ValueError(f"key list is empty: {key_list}")
    connection.execute(
        "CREATE INDEX entries_by_archive ON entries(dataset, archive, position)"
    )
    connection.commit()
    return count


def iter_archives(connection: sqlite3.Connection) -> Iterator[tuple[str, str]]:
    yield from connection.execute(
        "SELECT DISTINCT dataset, archive FROM entries ORDER BY dataset, archive"
    )


def populate_records(
    connection: sqlite3.Connection, blip3o_root: Path
) -> tuple[int, int]:
    dataset_roots: dict[str, Path] = {}
    archive_count = 0
    record_count = 0
    for dataset, archive in iter_archives(connection):
        dataset_root = dataset_roots.setdefault(
            dataset, resolve_dataset_root(blip3o_root, dataset)
        )
        updates = []
        with ArchiveReader(dataset_root, archive) as reader:
            rows = connection.execute(
                "SELECT position, member FROM entries "
                "WHERE dataset = ? AND archive = ? ORDER BY position",
                (dataset, archive),
            )
            for position, member in rows:
                image_bytes, caption = reader.read_pair(member)
                width, height, image_format = image_metadata(image_bytes)
                updates.append(
                    (
                        caption,
                        width,
                        height,
                        len(image_bytes),
                        image_format,
                        position,
                    )
                )
                record_count += 1
                if len(updates) >= 1_000:
                    connection.executemany(
                        "UPDATE entries SET caption=?, width=?, height=?, "
                        "file_size=?, file_format=? WHERE position=?",
                        updates,
                    )
                    updates.clear()
        if updates:
            connection.executemany(
                "UPDATE entries SET caption=?, width=?, height=?, "
                "file_size=?, file_format=? WHERE position=?",
                updates,
            )
        connection.commit()
        archive_count += 1
        print(
            f"processed archives={archive_count} records={record_count} "
            f"current={dataset}/{archive}",
            flush=True,
        )
    return archive_count, record_count


def write_jsonl(connection: sqlite3.Connection, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    count = 0
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
            rows = connection.execute(
                "SELECT position, archive, member, caption, width, height, "
                "file_size, file_format FROM entries ORDER BY position"
            )
            for (
                position,
                archive,
                member,
                caption,
                width,
                height,
                file_size,
                file_format,
            ) in rows:
                if caption is None:
                    raise RuntimeError(f"record {position} was not populated")
                record = {
                    "id": position,
                    "image": str(PurePosixPath(archive) / member),
                    "conversations": [
                        {"from": "human", "value": caption},
                        {"from": "gpt", "value": "<image>"},
                    ],
                    "index": position,
                    "width": width,
                    "height": height,
                    "file_size": file_size,
                    "file_format": file_format,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def convert(
    blip3o_root: Path,
    key_list: Path,
    output_path: Path,
    work_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
    if not key_list.is_file():
        raise FileNotFoundError(f"key list does not exist: {key_list}")
    work_dir = work_dir or output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=work_dir,
            prefix=f".{output_path.name}.",
            suffix=".sqlite3",
            delete=False,
        ) as temporary_database:
            database_path = Path(temporary_database.name)
        with create_database(database_path) as connection:
            key_count = ingest_keys(connection, key_list)
            archive_count, populated_count = populate_records(connection, blip3o_root)
            written_count = write_jsonl(connection, output_path)
        if not (key_count == populated_count == written_count):
            raise RuntimeError(
                "record count mismatch: "
                f"keys={key_count}, populated={populated_count}, written={written_count}"
            )
    finally:
        if database_path is not None:
            database_path.unlink(missing_ok=True)

    return {
        "key_list": str(key_list),
        "output_jsonl": str(output_path),
        "records": written_count,
        "archives": archive_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blip3o-root",
        type=Path,
        required=True,
        help="Root containing the two official BLIP3o dataset directories",
    )
    parser.add_argument(
        "--key-list",
        type=Path,
        default=DEFAULT_KEY_LIST,
        help=f"Canonical key list (default: {DEFAULT_KEY_LIST})",
    )
    parser.add_argument(
        "--jsonl-out-path",
        type=Path,
        default=DEFAULT_JSONL_OUT,
        help=f"Output training JSONL (default: {DEFAULT_JSONL_OUT})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for the temporary SQLite index (default: output directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSONL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = convert(
        blip3o_root=args.blip3o_root,
        key_list=args.key_list,
        output_path=args.jsonl_out_path,
        work_dir=args.work_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
