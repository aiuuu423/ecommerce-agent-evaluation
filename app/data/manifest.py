import json
import math
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
