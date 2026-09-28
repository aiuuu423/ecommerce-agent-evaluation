import json
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

import duckdb
import pandas as pd
import pytest

from app.data import gold
from app.data.generator import build_snapshot
from app.data.gold import build_gold_bundle

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
GOLD_SQL = ROOT / "sql/gold"
HAND_CHECKED_GOLD = ROOT / "tests/fixtures/hand_checked_gold_snapshot.json"

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


@pytest.fixture
def generated_snapshot(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    dataset_dir = tmp_path / "v1"
    manifest = build_snapshot(CONFIG, dataset_dir)
    return dataset_dir, manifest


@pytest.fixture
def gold_bundle(generated_snapshot: tuple[Path, dict[str, object]]) -> dict:
    dataset_dir, _ = generated_snapshot
    return build_gold_bundle(dataset_dir)


def _hand_checked_connection() -> duckdb.DuckDBPyConnection:
    fixture = json.loads(HAND_CHECKED_GOLD.read_text(encoding="utf-8"))
    as_of_date = date.fromisoformat(fixture["as_of_date"])
    previous_start = as_of_date - timedelta(days=59)
    current_start = as_of_date - timedelta(days=29)
    traffic_rows = []
    order_rows = []

    for product in fixture["products"]:
        product_id = product["product_id"]
        for period, start in (("previous", previous_start), ("current", current_start)):
            missing_days = product.get(f"{period}_missing_days", 0)
            total_visits = product[f"{period}_visits"]
            observed_days = 30 - missing_days
            daily_visits, remainder = divmod(total_visits, max(observed_days, 1))
            for day_offset in range(30):
                is_missing = day_offset >= observed_days
                traffic_rows.append(
                    {
                        "date": start + timedelta(days=day_offset),
                        "product_id": product_id,
                        "visits": None
                        if is_missing
                        else daily_visits + (day_offset < remainder),
                        "is_missing": is_missing,
                    }
                )

            order_count = product[f"{period}_orders"]
            total_revenue = product[f"{period}_revenue"]
            refund_count = product.get(f"{period}_refunds", 0)
            for order_number in range(order_count):
                order_rows.append(
                    {
                        "order_id": f"{product_id}-{period}-{order_number:03d}",
                        "product_id": product_id,
                        "order_date": start + timedelta(days=order_number % 30),
                        "revenue": total_revenue / order_count,
                        "is_refund": order_number < refund_count,
                    }
                )

    connection = duckdb.connect(database=":memory:")
    connection.register(
        "products",
        pd.DataFrame({"product_id": [row["product_id"] for row in fixture["products"]]}),
    )
    connection.register("traffic", pd.DataFrame(traffic_rows))
    connection.register("orders", pd.DataFrame(order_rows))
    return connection


def _query_hand_checked(sql_name: str) -> list[dict]:
    with _hand_checked_connection() as connection:
        relation = connection.execute((GOLD_SQL / sql_name).read_text(encoding="utf-8"))
        columns = [column[0] for column in relation.description]
        return [
            {
                column: gold._json_value(value)
                for column, value in zip(columns, row, strict=True)
            }
            for row in relation.fetchall()
        ]


def test_gold_bundle_covers_five_business_tasks_and_manifest_identity(
    gold_bundle: dict,
    generated_snapshot: tuple[Path, dict[str, object]],
) -> None:
    _, manifest = generated_snapshot

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
        "anomaly_id",
        "multiplier",
        "config_sha256",
    )
    for sql_path in GOLD_SQL.glob("*.sql"):
        query = sql_path.read_text(encoding="utf-8").lower()
        assert not any(token in query for token in forbidden), sql_path.name


def test_all_gold_queries_use_traffic_as_of_date_and_expose_windows() -> None:
    required_columns = {
        "as_of_date",
        "current_start",
        "current_end",
        "previous_start",
        "previous_end",
    }

    for sql_path in GOLD_SQL.glob("*.sql"):
        query = sql_path.read_text(encoding="utf-8").lower()
        assert "max(date) as as_of_date" in query, sql_path.name
        with _hand_checked_connection() as connection:
            relation = connection.execute(query)
            assert required_columns <= {
                column[0] for column in relation.description
            }, sql_path.name


def test_evidence_ids_are_stable_task_level_ids(
    gold_bundle: dict,
    generated_snapshot: tuple[Path, dict[str, object]],
) -> None:
    dataset_dir, _ = generated_snapshot
    rebuilt = build_gold_bundle(dataset_dir)

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


def test_gold_uses_catalog_verified_manifest_without_rereading_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = MappingProxyType(
        {
            "dataset_id": "0123456789abcdef",
            "dataset_version": "test-v1",
            "config_sha256": "a" * 64,
        }
    )

    class FakeCatalog:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        @property
        def manifest(self):
            return manifest

    monkeypatch.setattr(gold, "open_dataset", lambda _: FakeCatalog())
    monkeypatch.setattr(gold, "_records", lambda *_: [{"value": 1}])

    bundle = build_gold_bundle(Path("/does/not/need/a/manifest/file"))

    assert bundle["dataset_id"] == manifest["dataset_id"]
    assert bundle["dataset_version"] == manifest["dataset_version"]
    assert bundle["config_sha256"] == manifest["config_sha256"]


def test_gold_detects_p003_conversion_decline(gold_bundle: dict) -> None:
    rows = gold_bundle["tasks"]["conversion_decline"]["evidence"]
    p003 = next(row for row in rows if row["product_id"] == "P003")

    assert p003["current_cvr"] < p003["previous_cvr"]
    assert p003["cvr_change"] < 0


@pytest.mark.parametrize("sql_name", sorted(EXPECTED_SQL))
def test_hand_checked_snapshot_asserts_complete_ordered_sql_results(
    sql_name: str,
) -> None:
    fixture = json.loads(HAND_CHECKED_GOLD.read_text(encoding="utf-8"))

    assert _query_hand_checked(sql_name) == fixture["expected_results"][sql_name]


def test_conversion_uses_full_product_period_grid_and_only_returns_declines() -> None:
    rows = _query_hand_checked("conversion_decline.sql")

    assert [row["product_id"] for row in rows] == ["P001", "P005", "P002"]
    zero_order_row = rows[-1]
    assert zero_order_row["current_orders"] == 0
    assert zero_order_row["current_cvr"] == 0
    assert all(row["cvr_change"] < 0 for row in rows)


def test_conversion_nulls_incomplete_or_zero_denominator_and_reports_coverage() -> None:
    conversion_rows = _query_hand_checked("conversion_decline.sql")
    anomaly_rows = {
        row["product_id"]: row for row in _query_hand_checked("product_anomalies.sql")
    }

    assert "P003" not in {row["product_id"] for row in conversion_rows}
    assert anomaly_rows["P003"]["current_cvr"] is None
    assert anomaly_rows["P003"]["current_observed_days"] == 29
    assert "P006" not in {row["product_id"] for row in conversion_rows}
    assert anomaly_rows["P006"]["current_cvr"] is None
    assert anomaly_rows["P006"]["previous_cvr"] is None


def test_thresholds_and_ties_are_deterministic_on_hand_checked_snapshot() -> None:
    anomaly_rows = {
        row["product_id"]: row for row in _query_hand_checked("product_anomalies.sql")
    }
    conversion_rows = _query_hand_checked("conversion_decline.sql")

    assert anomaly_rows["P004"]["current_refund_rate"] == pytest.approx(0.12)
    assert anomaly_rows["P004"]["anomaly_type"] == "high_refund"
    assert anomaly_rows["P001"]["current_cvr"] - anomaly_rows["P001"][
        "previous_cvr"
    ] == pytest.approx(-0.01)
    assert anomaly_rows["P001"]["anomaly_type"] == "conversion_drop"
    assert [row["product_id"] for row in conversion_rows[:2]] == ["P001", "P005"]


def test_sales_drop_is_identified_by_independent_gmv_and_aov_changes() -> None:
    anomaly_rows = {
        row["product_id"]: row for row in _query_hand_checked("product_anomalies.sql")
    }

    assert anomaly_rows["P007"]["anomaly_type"] == "sales_drop"
    assert anomaly_rows["P007"]["gmv_change_rate"] == pytest.approx(-0.5)
    assert anomaly_rows["P007"]["aov_change_rate"] == pytest.approx(-0.5)
    assert anomaly_rows["P007"]["current_cvr"] == anomaly_rows["P007"]["previous_cvr"]


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
    generated_snapshot: tuple[Path, dict[str, object]],
) -> None:
    dataset_dir, _ = generated_snapshot
    catalog = gold.open_dataset(dataset_dir)
    monkeypatch.setattr(gold, "open_dataset", lambda _: catalog)
    monkeypatch.setitem(gold.TASK_SQL, "gmv_diagnosis", "missing.sql")

    with pytest.raises(FileNotFoundError):
        build_gold_bundle(dataset_dir)

    with pytest.raises(ValueError, match="catalog is closed"):
        catalog.execute("select 1")
