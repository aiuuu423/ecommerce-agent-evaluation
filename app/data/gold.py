import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.data.database import Catalog, open_dataset

TASK_SQL = {
    "gmv_diagnosis": "gmv_change.sql",
    "product_anomaly": "product_anomalies.sql",
    "conversion_decline": "conversion_decline.sql",
    "products_to_watch": "products_to_watch.sql",
    "next_week_priority": "next_week_priorities.sql",
}
SQL_DIRECTORY = Path(__file__).parents[2] / "sql/gold"


def _json_value(value: object) -> int | float | str | bool | None:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("gold query returned a non-finite number")
        return float(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("gold query returned a non-finite number")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"gold query returned unsupported type: {type(value).__name__}")


def _records(catalog: Catalog, sql_path: Path) -> list[dict[str, Any]]:
    relation = catalog.execute(sql_path.read_text(encoding="utf-8"))
    description = relation.description
    if description is None:
        raise ValueError(f"gold query returned no result columns: {sql_path.name}")
    columns = [column[0] for column in description]
    return [
        {
            column: _json_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in relation.fetchall()
    ]


def build_gold_bundle(dataset_dir: Path | str) -> dict[str, Any]:
    with open_dataset(dataset_dir) as catalog:
        manifest = catalog.manifest
        tasks: dict[str, dict[str, Any]] = {}
        for task_name, filename in TASK_SQL.items():
            records = _records(catalog, SQL_DIRECTORY / filename)
            evidence = [
                {
                    "evidence_id": f"EV_{task_name.upper()}_{row_number:03d}",
                    "source": filename,
                    **record,
                }
                for row_number, record in enumerate(records, start=1)
            ]
            tasks[task_name] = {"sql": filename, "evidence": evidence}

    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "config_sha256": manifest["config_sha256"],
        "tasks": tasks,
    }
