import json
from pathlib import Path

import pytest

from app.data import gold
from app.data.gold import build_gold_bundle

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "data/synthetic/v1"
GOLD_SQL = ROOT / "sql/gold"

EXPECTED_SQL = {
    "gmv_change.sql",
    "product_anomalies.sql",
    "conversion_decline.sql",
    "products_to_watch.sql",
    "next_week_priorities.sql",
}
EXPECTED_TASKS = {
    "gmv_diagnosis",
    "product_anomaly",
    "conversion_decline",
    "products_to_watch",
    "next_week_priority",
}


@pytest.fixture(scope="module")
def gold_bundle() -> dict:
    return build_gold_bundle(SNAPSHOT)


def test_gold_bundle_covers_five_business_tasks_and_manifest_identity(
    gold_bundle: dict,
) -> None:
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))

    assert gold_bundle["dataset_id"] == manifest["dataset_id"]
    assert gold_bundle["dataset_version"] == manifest["dataset_version"]
    assert gold_bundle["config_sha256"] == manifest["config_sha256"]
    assert set(gold_bundle["tasks"]) == EXPECTED_TASKS
    assert all(task["evidence"] for task in gold_bundle["tasks"].values())


def test_gold_sql_is_independent_and_uses_only_catalog_tables() -> None:
    assert {path.name for path in GOLD_SQL.glob("*.sql")} == EXPECTED_SQL

    forbidden = (
        "read_csv",
        "read_json",
        "read_parquet",
        "yaml",
        "synthetic_v1",
        "anomalies",
        "config_sha256",
    )
    for sql_path in GOLD_SQL.glob("*.sql"):
        query = sql_path.read_text(encoding="utf-8").lower()
        assert not any(token in query for token in forbidden), sql_path.name


def test_evidence_ids_are_stable_task_level_ids(gold_bundle: dict) -> None:
    rebuilt = build_gold_bundle(SNAPSHOT)

    for task_name, task in gold_bundle["tasks"].items():
        expected_ids = [
            f"EV_{task_name.upper()}_{row_number:03d}"
            for row_number in range(1, len(task["evidence"]) + 1)
        ]
        assert [row["evidence_id"] for row in task["evidence"]] == expected_ids
        assert rebuilt["tasks"][task_name]["evidence"] == task["evidence"]


def test_gold_bundle_contains_only_json_safe_values(gold_bundle: dict) -> None:
    encoded = json.dumps(gold_bundle, allow_nan=False, sort_keys=True)

    assert json.loads(encoded) == gold_bundle


def test_gold_detects_p003_conversion_decline(gold_bundle: dict) -> None:
    rows = gold_bundle["tasks"]["conversion_decline"]["evidence"]
    p003 = next(row for row in rows if row["product_id"] == "P003")

    assert p003["current_cvr"] < p003["previous_cvr"]
    assert p003["cvr_change"] < 0


def test_rule_queries_expose_their_result_columns(gold_bundle: dict) -> None:
    assert {"product_id", "anomaly_type"} <= set(
        gold_bundle["tasks"]["product_anomaly"]["evidence"][0]
    )
    assert {"product_id", "watch_reason"} <= set(
        gold_bundle["tasks"]["products_to_watch"]["evidence"][0]
    )
    assert {
        "product_id",
        "priority_reason",
        "evidence_metric",
        "evidence_value",
    } <= set(gold_bundle["tasks"]["next_week_priority"]["evidence"][0])


def test_gold_builder_closes_catalog_when_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = gold.open_dataset(SNAPSHOT)
    monkeypatch.setattr(gold, "open_dataset", lambda _: catalog)
    monkeypatch.setitem(gold.TASK_SQL, "gmv_diagnosis", "missing.sql")

    with pytest.raises(FileNotFoundError):
        build_gold_bundle(SNAPSHOT)

    with pytest.raises(ValueError, match="catalog is closed"):
        catalog.execute("select 1")
