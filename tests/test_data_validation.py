import json
from collections.abc import Callable
from decimal import Decimal
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
REPORT_KEYS = {
    "status",
    "source_label",
    "failed_checks",
    "row_counts",
    "schema_errors",
}


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


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (Decimal("1.00"), Decimal("4.00"), Decimal("0.25")),
        (Decimal("1.00"), 4.0, Decimal("0.25")),
        (1.0, Decimal("4.00"), Decimal("0.25")),
    ],
)
def test_safe_divide_supports_decimal_and_float_combinations(
    numerator: float | Decimal,
    denominator: float | Decimal,
    expected: Decimal,
) -> None:
    assert safe_divide(numerator, denominator) == expected


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (float("nan"), 1),
        (1, float("nan")),
        (float("inf"), 1),
        (1, float("-inf")),
        (Decimal("NaN"), Decimal("1")),
        (Decimal("1"), Decimal("Infinity")),
    ],
)
def test_safe_divide_returns_none_for_non_finite_values(
    numerator: float | Decimal,
    denominator: float | Decimal,
) -> None:
    assert safe_divide(numerator, denominator) is None


def test_metrics_match_independent_hand_calculation() -> None:
    fixture = json.loads(HAND_CHECKED_METRICS.read_text(encoding="utf-8"))

    assert calculate_hand_checked_metrics(fixture) == fixture["expected"]


def test_metrics_consistently_exclude_non_final_orders() -> None:
    payload = {
        "orders": [
            {
                "order_id": "paid",
                "revenue": 100.0,
                "is_refund": False,
                "status": "paid",
            },
            {
                "order_id": "refunded",
                "revenue": 50.0,
                "is_refund": True,
                "status": "refunded",
            },
            {
                "order_id": "cancelled",
                "revenue": 900.0,
                "is_refund": False,
                "status": "cancelled",
            },
        ],
        "traffic": {"impressions": 100, "clicks": 20, "visits": 10},
        "marketing_spend": 50.0,
    }

    assert calculate_hand_checked_metrics(payload) == {
        "gmv": 150.0,
        "orders": 2,
        "aov": 75.0,
        "ctr": 0.2,
        "cvr": 0.2,
        "refund_rate": 0.5,
        "roas": 3.0,
    }


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
    assert report["schema_errors"] == []
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


def test_schema_validation_reports_every_invalid_row_with_location(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    def corrupt(tables: dict[str, pd.DataFrame]) -> None:
        tables["products"].loc[tables["products"].index[0], "price"] = -1
        tables["products"].loc[tables["products"].index[1], "cost"] = 0

    report = validate_dataset(changed_tables(generated_tables, corrupt), data_config)

    assert "products_schema" in report["failed_checks"]
    assert [
        (error["table"], error["row_index"], error["location"])
        for error in report["schema_errors"]
        if error["table"] == "products"
    ] == [
        ("products", 0, ["price"]),
        ("products", 1, ["cost"]),
    ]


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


def test_marketing_daily_grain_allows_multiple_campaigns_per_product_day(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    def add_campaign(tables: dict[str, pd.DataFrame]) -> None:
        extra = tables["marketing"].iloc[[0]].copy()
        extra.loc[extra.index[0], "campaign_id"] = "M999"
        tables["marketing"] = pd.concat(
            [tables["marketing"], extra],
            ignore_index=True,
        )

    report = validate_dataset(changed_tables(generated_tables, add_campaign), data_config)

    assert "marketing_primary_key" not in report["failed_checks"]
    assert "marketing_daily_grain" not in report["failed_checks"]


@pytest.mark.parametrize(
    ("table_name", "id_column", "replacement", "expected_check"),
    [
        ("products", "product_id", "P999", "product_dimension_ids"),
        ("customers", "customer_id", "C9999", "customer_dimension_ids"),
    ],
)
def test_dimensions_require_exact_configured_count_and_id_set(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
    table_name: str,
    id_column: str,
    replacement: str,
    expected_check: str,
) -> None:
    def replace_id(tables: dict[str, pd.DataFrame]) -> None:
        tables[table_name].loc[tables[table_name].index[-1], id_column] = replacement

    report = validate_dataset(
        changed_tables(generated_tables, replace_id),
        data_config,
    )

    assert expected_check in report["failed_checks"]


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


def test_order_amount_identity_compares_money_at_two_decimal_places(
    data_config: SyntheticDataConfig,
    generated_tables: dict[str, pd.DataFrame],
) -> None:
    def use_float_values(tables: dict[str, pd.DataFrame]) -> None:
        index = tables["orders"].index[0]
        tables["orders"].loc[index, "quantity"] = 3
        tables["orders"].loc[index, "unit_price"] = 0.1
        tables["orders"].loc[index, "revenue"] = 0.3

    report = validate_dataset(
        changed_tables(generated_tables, use_float_values),
        data_config,
    )

    assert "order_revenue_identity" not in report["failed_checks"]
