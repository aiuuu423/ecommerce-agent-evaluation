from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from app.data.config import SyntheticDataConfig
from app.data.metrics import safe_divide
from app.data.schemas import CustomerRow, MarketingRow, OrderRow, ProductRow, TrafficRow

TABLE_NAMES = ("products", "customers", "traffic", "marketing", "orders")
TABLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "products": ProductRow,
    "customers": CustomerRow,
    "traffic": TrafficRow,
    "marketing": MarketingRow,
    "orders": OrderRow,
}


def _normalized_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def calculate_hand_checked_metrics(
    payload: Mapping[str, Any],
) -> dict[str, float | int | None]:
    orders = payload["orders"]
    gmv = sum(
        row["revenue"]
        for row in orders
        if row["status"] in {"paid", "refunded"}
    )
    order_count = len({row["order_id"] for row in orders})
    refund_count = sum(row["is_refund"] for row in orders)
    traffic = payload["traffic"]
    spend = payload["marketing_spend"]
    return {
        "gmv": gmv,
        "orders": order_count,
        "aov": safe_divide(gmv, order_count),
        "ctr": safe_divide(traffic["clicks"], traffic["impressions"]),
        "cvr": safe_divide(order_count, traffic["visits"]),
        "refund_rate": safe_divide(refund_count, order_count),
        "roas": safe_divide(gmv, spend),
    }


def _has_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return set(columns).issubset(frame.columns)


def _has_duplicate_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return not _has_columns(frame, columns) or frame.duplicated(list(columns)).any()


def _has_broken_fk(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
) -> bool:
    if child_column not in child or parent_column not in parent:
        return True
    return bool(set(child[child_column].dropna()) - set(parent[parent_column].dropna()))


def _has_exact_daily_grain(
    frame: pd.DataFrame,
    config: SyntheticDataConfig,
) -> bool:
    key_columns = ("date", "product_id")
    if not _has_columns(frame, key_columns):
        return False
    expected = {
        (
            config.start_date + timedelta(days=day_index),
            f"P{product_number:03d}",
        )
        for day_index in range(config.days)
        for product_number in range(1, config.product_count + 1)
    }
    actual = set(frame.loc[:, list(key_columns)].itertuples(index=False, name=None))
    return len(frame) == len(expected) and actual == expected


def _schema_is_valid(frame: pd.DataFrame, model: type[BaseModel]) -> bool:
    try:
        for row in _normalized_records(frame):
            model.model_validate(row)
    except (TypeError, ValidationError):
        return False
    return True


def _funnel_is_valid(
    traffic: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> bool:
    required = ("is_missing", left_column, right_column)
    if not _has_columns(traffic, required):
        return False
    observed = traffic.loc[traffic["is_missing"].eq(False)]
    try:
        return bool((observed[left_column] >= observed[right_column]).all())
    except TypeError:
        return False


def _revenue_identity_is_valid(orders: pd.DataFrame) -> bool:
    columns = ("revenue", "quantity", "unit_price")
    if not _has_columns(orders, columns):
        return False
    try:
        expected = orders["quantity"] * orders["unit_price"]
        return bool(orders["revenue"].eq(expected).all())
    except TypeError:
        return False


def _report(
    tables: Mapping[str, pd.DataFrame],
    config: SyntheticDataConfig,
    failed_checks: list[str],
) -> dict[str, Any]:
    return {
        "status": "pass" if not failed_checks else "fail",
        "source_label": config.source_label,
        "failed_checks": failed_checks,
        "row_counts": {
            name: len(tables[name]) if name in tables else None for name in TABLE_NAMES
        },
    }


def validate_dataset(
    tables: Mapping[str, pd.DataFrame],
    config: SyntheticDataConfig,
) -> dict[str, Any]:
    failed: list[str] = []
    if set(tables) != set(TABLE_NAMES):
        failed.append("required_tables")
        return _report(tables, config, failed)

    for table_name in TABLE_NAMES:
        if not _schema_is_valid(tables[table_name], TABLE_SCHEMAS[table_name]):
            failed.append(f"{table_name}_schema")

    primary_keys = (
        ("products", ("product_id",), "product_id_unique"),
        ("customers", ("customer_id",), "customer_id_unique"),
        ("traffic", ("date", "product_id"), "traffic_primary_key"),
        (
            "marketing",
            ("date", "product_id", "campaign_id"),
            "marketing_primary_key",
        ),
        ("orders", ("order_id",), "order_id_unique"),
    )
    for table_name, columns, check_name in primary_keys:
        if _has_duplicate_key(tables[table_name], columns):
            failed.append(check_name)

    foreign_keys = (
        ("traffic", "product_id", "products", "product_id", "traffic_product_fk"),
        (
            "marketing",
            "product_id",
            "products",
            "product_id",
            "marketing_product_fk",
        ),
        ("orders", "product_id", "products", "product_id", "order_product_fk"),
        ("orders", "customer_id", "customers", "customer_id", "order_customer_fk"),
    )
    for child, child_column, parent, parent_column, check_name in foreign_keys:
        if _has_broken_fk(
            tables[child],
            child_column,
            tables[parent],
            parent_column,
        ):
            failed.append(check_name)

    for table_name in ("traffic", "marketing"):
        if not _has_exact_daily_grain(tables[table_name], config):
            failed.append(f"{table_name}_daily_grain")

    if not _funnel_is_valid(tables["traffic"], "impressions", "clicks"):
        failed.append("traffic_funnel_impressions_clicks")
    if not _funnel_is_valid(tables["traffic"], "visits", "clicks"):
        failed.append("traffic_funnel_clicks_visits")
    if not _revenue_identity_is_valid(tables["orders"]):
        failed.append("order_revenue_identity")

    return _report(tables, config, failed)
