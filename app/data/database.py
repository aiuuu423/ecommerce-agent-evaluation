import io
import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.data.manifest import (
    TABLE_NAMES,
    dataset_id_for_tables,
    logical_table_sha256,
    validate_manifest,
)

_QUALITY_KEYS = {
    "status",
    "source_label",
    "failed_checks",
    "schema_errors",
    "row_counts",
}


def _load_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _read_regular_file(path: Path, missing_message: str) -> bytes:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(missing_message) from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"snapshot artifact must not be a symbolic link: {path.name}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"snapshot artifact must be a regular file: {path.name}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"snapshot artifact could not be opened safely: {path.name}") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"snapshot artifact must be a regular file: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def _snapshot_directory(dataset_dir: Path | str) -> Path:
    path = Path(dataset_dir).absolute()
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"dataset directory does not exist: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"dataset directory must not be a symbolic link: {path}")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"dataset path must be a directory: {path}")
    for artifact in path.iterdir():
        if artifact.is_symlink():
            raise ValueError(
                f"snapshot artifact must not be a symbolic link: {artifact.name}"
            )
    return path


def _load_verified_tables(
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> dict[str, pd.DataFrame]:
    expected_files = {f"{name}.parquet" for name in TABLE_NAMES}
    actual_files = {
        path.name
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix == ".parquet"
    }
    if actual_files - expected_files:
        raise ValueError("snapshot contains unexpected parquet files")

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        metadata = manifest["tables"][name]
        data = _read_regular_file(
            dataset_dir / metadata["file"],
            f"missing table file: {name}",
        )
        if sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError(f"table hash mismatch: {name}")
        try:
            frame = pd.read_parquet(io.BytesIO(data))
        except Exception as exc:
            raise ValueError(f"table parquet is invalid: {name}") from exc
        if len(frame) != metadata["rows"]:
            raise ValueError(f"table row count mismatch: {name}")
        if logical_table_sha256(frame) != metadata["logical_sha256"]:
            raise ValueError(f"table logical hash mismatch: {name}")
        tables[name] = frame

    if dataset_id_for_tables(tables) != manifest["dataset_id"]:
        raise ValueError("manifest dataset identity mismatch")
    return tables


def _validate_quality_report(
    manifest: dict[str, Any],
    quality_data: bytes,
    tables: dict[str, pd.DataFrame],
) -> None:
    metadata = manifest["data_quality_report"]
    if sha256(quality_data).hexdigest() != metadata["sha256"]:
        raise ValueError("quality report hash mismatch")

    quality = _load_json_object(quality_data, "quality report")
    expected_rows = {name: len(tables[name]) for name in TABLE_NAMES}
    if not (
        set(quality) == _QUALITY_KEYS
        and quality.get("status") == "pass"
        and quality.get("source_label") == manifest["source_label"]
        and quality.get("failed_checks") == []
        and quality.get("schema_errors") == []
        and quality.get("row_counts") == expected_rows
    ):
        raise ValueError("quality report is invalid")


class Catalog:
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        tables: dict[str, pd.DataFrame],
    ) -> None:
        self.__connection = connection
        self.__tables = tables
        self.__closed = False

    def execute(
        self,
        query: str,
        parameters: object | None = None,
    ) -> "Catalog":
        if self.__closed:
            raise ValueError("catalog is closed")
        statements = self.__connection.extract_statements(query)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise PermissionError("catalog is read-only; only one SELECT is allowed")
        if parameters is None:
            self.__connection.execute(query)
        else:
            self.__connection.execute(query, parameters)
        return self

    @property
    def description(self) -> tuple[tuple[Any, ...], ...] | None:
        if self.__closed:
            raise ValueError("catalog is closed")
        description = self.__connection.description
        if description is None:
            return None
        return tuple(tuple(column) for column in description)

    def fetchone(self) -> tuple[Any, ...] | None:
        if self.__closed:
            raise ValueError("catalog is closed")
        return self.__connection.fetchone()

    def fetchall(self) -> list[tuple[Any, ...]]:
        if self.__closed:
            raise ValueError("catalog is closed")
        return self.__connection.fetchall()

    def close(self) -> None:
        if not self.__closed:
            self.__closed = True
            self.__connection.close()
            self.__tables.clear()

    def __enter__(self) -> "Catalog":
        if self.__closed:
            raise ValueError("catalog is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_dataset(dataset_dir: Path | str) -> Catalog:
    directory = _snapshot_directory(dataset_dir)
    manifest_data = _read_regular_file(
        directory / "manifest.json",
        "missing manifest",
    )
    manifest = _load_json_object(manifest_data, "manifest")
    validate_manifest(manifest)
    tables = _load_verified_tables(manifest, directory)
    quality_data = _read_regular_file(
        directory / manifest["data_quality_report"]["file"],
        "missing quality report",
    )
    _validate_quality_report(manifest, quality_data, tables)

    connection = duckdb.connect(
        database=":memory:",
        config={"enable_external_access": "false"},
    )
    try:
        for name, frame in tables.items():
            connection.register(name, frame)
        return Catalog(connection, tables)
    except BaseException:
        connection.close()
        raise
