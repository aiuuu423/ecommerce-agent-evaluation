import json
import re
from pathlib import Path
from typing import Any

import duckdb

from app.data.manifest import contained_path, file_sha256, manifest_id
from app.data.validation import TABLE_NAMES

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID = re.compile(r"^[0-9a-f]{16}$")
_MANIFEST_KEYS = {
    "dataset_id",
    "dataset_version",
    "schema_version",
    "source_label",
    "seed",
    "config_sha256",
    "writer",
    "tables",
    "data_quality_report",
}
_TABLE_METADATA_KEYS = {"file", "rows", "logical_sha256", "sha256"}
_QUALITY_KEYS = {
    "status",
    "source_label",
    "failed_checks",
    "schema_errors",
    "row_counts",
}


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _validate_table_set(
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> dict[str, Path]:
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("manifest tables must be an object")
    for name in tables:
        if not isinstance(name, str) or _SAFE_IDENTIFIER.fullmatch(name) is None:
            raise ValueError(f"unsafe table identifier: {name!r}")
    if set(tables) != set(TABLE_NAMES):
        raise ValueError("manifest must contain exactly the five required tables")

    expected_files = {f"{name}.parquet" for name in TABLE_NAMES}
    actual_files = {path.name for path in dataset_dir.glob("*.parquet")}
    if actual_files - expected_files:
        raise ValueError("snapshot contains unexpected parquet files")

    table_paths: dict[str, Path] = {}
    for name in TABLE_NAMES:
        metadata = tables[name]
        valid_metadata = (
            isinstance(metadata, dict)
            and set(metadata) == _TABLE_METADATA_KEYS
            and metadata.get("file") == f"{name}.parquet"
            and type(metadata.get("rows")) is int
            and metadata["rows"] >= 0
            and _valid_digest(metadata.get("logical_sha256"))
            and _valid_digest(metadata.get("sha256"))
        )
        if not valid_metadata:
            raise ValueError(f"table metadata is invalid: {name}")
        try:
            path = contained_path(dataset_dir, metadata["file"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"table metadata is invalid: {name}") from exc
        if not path.is_file():
            raise ValueError(f"missing table file: {name}")
        table_paths[name] = path
    return table_paths


def _validate_manifest(
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> dict[str, Path]:
    scalar_fields_are_valid = (
        set(manifest) == _MANIFEST_KEYS
        and isinstance(manifest.get("dataset_id"), str)
        and _DATASET_ID.fullmatch(manifest["dataset_id"]) is not None
        and all(
            isinstance(manifest.get(field), str) and bool(manifest[field])
            for field in ("dataset_version", "schema_version", "source_label")
        )
        and type(manifest.get("seed")) is int
        and _valid_digest(manifest.get("config_sha256"))
        and isinstance(manifest.get("writer"), dict)
        and bool(manifest["writer"])
    )
    if not scalar_fields_are_valid:
        raise ValueError("manifest is invalid")

    quality_metadata = manifest.get("data_quality_report")
    if not (
        isinstance(quality_metadata, dict)
        and set(quality_metadata) == {"file", "sha256"}
        and quality_metadata.get("file") == "data_quality_report.json"
        and _valid_digest(quality_metadata.get("sha256"))
    ):
        raise ValueError("manifest is invalid")
    table_paths = _validate_table_set(manifest, dataset_dir)
    identity = {
        name: {
            "rows": manifest["tables"][name]["rows"],
            "logical_sha256": manifest["tables"][name]["logical_sha256"],
        }
        for name in TABLE_NAMES
    }
    if manifest_id(identity) != manifest["dataset_id"]:
        raise ValueError("manifest dataset identity mismatch")
    return table_paths


def _validate_quality_report(
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> None:
    metadata = manifest["data_quality_report"]
    quality_path = dataset_dir / metadata["file"]
    if not quality_path.is_file():
        raise ValueError("missing quality report")
    if file_sha256(quality_path) != metadata["sha256"]:
        raise ValueError("quality report hash mismatch")

    quality = _load_json_object(quality_path, "quality report")
    expected_rows = {
        name: manifest["tables"][name]["rows"] for name in TABLE_NAMES
    }
    if not (
        set(quality) == _QUALITY_KEYS
        and quality.get("status") == "pass"
        and quality.get("source_label") == manifest["source_label"]
        and quality.get("failed_checks") == []
        and quality.get("schema_errors") == []
        and quality.get("row_counts") == expected_rows
    ):
        raise ValueError("quality report is invalid")


def open_dataset(dataset_dir: Path | str) -> duckdb.DuckDBPyConnection:
    dataset_dir = Path(dataset_dir).resolve()
    manifest = _load_json_object(dataset_dir / "manifest.json", "manifest")
    table_paths = _validate_manifest(manifest, dataset_dir)

    for name in TABLE_NAMES:
        if file_sha256(table_paths[name]) != manifest["tables"][name]["sha256"]:
            raise ValueError(f"table hash mismatch: {name}")
    _validate_quality_report(manifest, dataset_dir)

    connection = duckdb.connect(database=":memory:")
    for name, path in table_paths.items():
        connection.read_parquet(str(path)).create_view(name)
    return connection
