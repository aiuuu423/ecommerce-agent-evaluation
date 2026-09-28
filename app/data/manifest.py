import json
import math
import re
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE_NAMES = ("products", "customers", "traffic", "marketing", "orders")
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
    "generator_source_sha256",
    "writer",
    "tables",
    "data_quality_report",
}
_TABLE_METADATA_KEYS = {"file", "rows", "logical_sha256", "sha256"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()[:16]


def _normalized_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _normalized_value(value.item())
    if isinstance(value, (datetime, date)):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("logical table values must be finite")
        return {"type": "decimal", "value": format(value.normalize(), "f")}
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("logical table values must be finite")
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported logical table value: {type(value).__name__}")


def logical_table_sha256(frame: pd.DataFrame) -> str:
    digest = sha256()
    header = {"columns": [str(column) for column in frame.columns]}
    digest.update(
        json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    for row in frame.itertuples(index=False, name=None):
        normalized = [_normalized_value(value) for value in row]
        digest.update(
            json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_id_for_tables(tables: dict[str, pd.DataFrame]) -> str:
    identity = {
        name: {
            "rows": len(frame),
            "logical_sha256": logical_table_sha256(frame),
        }
        for name, frame in sorted(tables.items())
    }
    return manifest_id(identity)


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def validate_manifest(manifest: dict[str, Any]) -> None:
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
        and _valid_digest(manifest.get("generator_source_sha256"))
        and isinstance(manifest.get("writer"), dict)
        and bool(manifest["writer"])
    )
    if not scalar_fields_are_valid:
        raise ValueError("manifest is invalid")

    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("manifest tables must be an object")
    for name in tables:
        if not isinstance(name, str) or _SAFE_IDENTIFIER.fullmatch(name) is None:
            raise ValueError(f"unsafe table identifier: {name!r}")
    if set(tables) != set(TABLE_NAMES):
        raise ValueError("manifest must contain exactly the five required tables")
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

    quality_metadata = manifest.get("data_quality_report")
    if not (
        isinstance(quality_metadata, dict)
        and set(quality_metadata) == {"file", "sha256"}
        and quality_metadata.get("file") == "data_quality_report.json"
        and _valid_digest(quality_metadata.get("sha256"))
    ):
        raise ValueError("manifest is invalid")


def contained_path(output_dir: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"output path must be contained in snapshot: {relative}")

    root = output_dir.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output path must be contained in snapshot: {relative}"
        ) from exc
    return target


def json_sha256(payload: dict[str, Any]) -> str:
    return sha256(_json_bytes(payload)).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))
