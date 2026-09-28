import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from app.data.config import SyntheticDataConfig, load_data_config
from app.data.generator import generate_dataset
from app.data.metrics import safe_divide
from app.data.validation import calculate_hand_checked_metrics, validate_dataset

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
HAND_CHECKED_METRICS = ROOT / "tests/fixtures/hand_checked_metrics.json"
REPORT_KEYS = {"status", "source_label", "failed_checks", "row_counts"}


@pytest.fixture(scope="module")
def data_config() -> SyntheticDataConfig:
    return load_data_config(CONFIG)


@pytest.fixture(scope="module")
def generated_tables(
    data_config: SyntheticDataConfig,
) -> dict[str, pd.DataFrame]:
    return generate_dataset(data_config)


def changed_tables(
    tables: dict[str, pd.DataFrame],
    change: Callable[[dict[str, pd.DataFrame]], None],
) -> dict[str, pd.DataFrame]:
    changed = {name: frame.copy(deep=True) for name, frame in tables.items()}
    change(changed)
    return changed


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(10, 0), (None, 10), (10, None), (None, None)],
)
def test_safe_divide_preserves_unavailable_results(
    numerator: float | None, denominator: float | None
) -> None:
    assert safe_divide(numerator, denominator) is None


def test_safe_divide_calculates_available_ratio() -> None:
    assert safe_divide(3, 4) == pytest.approx(0.75)


def test_metrics_match_independent_hand_calculation() -> None:
    fixture = json.loads(HAND_CHECKED_METRICS.read_text(encoding="utf-8"))

    assert calculate_hand_checked_metrics(fixture) == fixture["expected"]


def test_hand_checked_metrics_do_not_turn_missing_traffic_into_zero() -> None:
    fixture = json.loads(HAND_CHECKED_METRICS.read_text(encoding="utf-8"))
    fixture["traffic"] = {"impressions": None, "clicks": None, "visits": None}

    metrics = calculate_hand_checked_metrics(fixture)

    assert metrics["ctr"] is None
    assert metrics["cvr"] is None
    assert metrics["gmv"] == 300.0


def test_generated_dataset_passes_all_quality_checks(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    report = validate_dataset(generated_tables, data_config)

    assert set(report) == REPORT_KEYS
    assert report["status"] == "pass"
    assert report["source_label"] == "Synthetic E-commerce Data"
    assert report["failed_checks"] == []
    assert report["row_counts"] == {
        name: len(generated_tables[name])
        for name in ("products", "customers", "traffic", "marketing", "orders")
    }


def test_report_shape_is_stable_when_a_required_table_is_missing(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    incomplete = {
        name: frame for name, frame in generated_tables.items() if name != "orders"
    }

    report = validate_dataset(incomplete, data_config)

    assert set(report) == REPORT_KEYS
    assert report["status"] == "fail"
    assert report["failed_checks"] == ["required_tables"]
    assert report["row_counts"] == {
        "products": len(incomplete["products"]),
        "customers": len(incomplete["customers"]),
        "traffic": len(incomplete["traffic"]),
        "marketing": len(incomplete["marketing"]),
        "orders": None,
    }


@pytest.mark.parametrize(
    ("table_name", "column", "invalid_value", "expected_check"),
    [
        ("products", "price", -1, "products_schema"),
        ("customers", "region", "", "customers_schema"),
        ("traffic", "impressions", -1, "traffic_schema"),
        ("marketing", "spend", -1, "marketing_schema"),
        ("orders", "quantity", 0, "orders_schema"),
    ],
)
def test_each_table_is_checked_against_its_schema(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    table_name: str,
    column: str,
    invalid_value: object,
    expected_check: str,
) -> None:
    def corrupt(tables: dict[str, pd.DataFrame]) -> None:
        tables[table_name].loc[tables[table_name].index[0], column] = invalid_value

    report = validate_dataset(changed_tables(generated_tables, corrupt), data_config)

    assert expected_check in report["failed_checks"]


@pytest.mark.parametrize(
    ("table_name", "key_columns", "expected_check"),
    [
        ("products", ["product_id"], "product_id_unique"),
        ("customers", ["customer_id"], "customer_id_unique"),
        ("traffic", ["date", "product_id"], "traffic_primary_key"),
        (
            "marketing",
            ["date", "product_id", "campaign_id"],
            "marketing_primary_key",
        ),
        ("orders", ["order_id"], "order_id_unique"),
    ],
)
def test_each_table_primary_key_is_checked(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    table_name: str,
    key_columns: list[str],
    expected_check: str,
) -> None:
    def duplicate_key(tables: dict[str, pd.DataFrame]) -> None:
        duplicate = tables[table_name].iloc[[0]].copy()
        tables[table_name] = pd.concat(
            [tables[table_name], duplicate], ignore_index=True
        )
        assert tables[table_name].duplicated(key_columns).any()

    report = validate_dataset(
        changed_tables(generated_tables, duplicate_key), data_config
    )

    assert expected_check in report["failed_checks"]


@pytest.mark.parametrize(
    ("table_name", "column", "invalid_value", "expected_check"),
    [
        ("traffic", "product_id", "P999", "traffic_product_fk"),
        ("marketing", "product_id", "P999", "marketing_product_fk"),
        ("orders", "product_id", "P999", "order_product_fk"),
        ("orders", "customer_id", "C9999", "order_customer_fk"),
    ],
)
def test_all_fact_foreign_keys_are_checked(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    table_name: str,
    column: str,
    invalid_value: str,
    expected_check: str,
) -> None:
    def break_fk(tables: dict[str, pd.DataFrame]) -> None:
        tables[table_name].loc[tables[table_name].index[0], column] = invalid_value

    report = validate_dataset(changed_tables(generated_tables, break_fk), data_config)

    assert expected_check in report["failed_checks"]


@pytest.mark.parametrize("table_name", ["traffic", "marketing"])
def test_daily_fact_grain_requires_the_complete_configured_grid(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    table_name: str,
) -> None:
    def remove_day(tables: dict[str, pd.DataFrame]) -> None:
        tables[table_name] = tables[table_name].iloc[1:].reset_index(drop=True)

    report = validate_dataset(changed_tables(generated_tables, remove_day), data_config)

    assert f"{table_name}_daily_grain" in report["failed_checks"]


@pytest.mark.parametrize(
    ("column", "value", "expected_check"),
    [
        ("clicks", 10_000, "traffic_funnel_impressions_clicks"),
        ("visits", 0, "traffic_funnel_clicks_visits"),
    ],
)
def test_traffic_funnel_consistency_is_checked(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    column: str,
    value: int,
    expected_check: str,
) -> None:
    def break_funnel(tables: dict[str, pd.DataFrame]) -> None:
        tables["traffic"].loc[tables["traffic"].index[0], column] = value

    report = validate_dataset(
        changed_tables(generated_tables, break_funnel), data_config
    )

    assert expected_check in report["failed_checks"]


def test_order_amount_identity_is_checked(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    def break_amount(tables: dict[str, pd.DataFrame]) -> None:
        index = tables["orders"].index[0]
        tables["orders"].loc[index, "revenue"] += 1

    report = validate_dataset(
        changed_tables(generated_tables, break_amount), data_config
    )

    assert "order_revenue_identity" in report["failed_checks"]
