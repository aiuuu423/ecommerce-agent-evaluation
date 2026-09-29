from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from app.data.database import Catalog, open_dataset
from app.data.generator import build_snapshot
from app.tools import ToolContext, ToolRegistry, build_default_registry
from app.tools.queries import _python_value

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


@pytest.fixture(scope="session")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Catalog]:
    dataset_dir = tmp_path_factory.mktemp("tool-queries") / "v1"
    build_snapshot(CONFIG, dataset_dir)
    with open_dataset(dataset_dir) as opened:
        yield opened


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def tool_context(catalog: Catalog) -> ToolContext:
    return ToolContext(
        catalog=catalog,
        prior_executions=(),
        next_result_id=lambda: "result_0007",
    )


@pytest.fixture
def catalog_with_one_traffic_day_omitted() -> Iterator[Catalog]:
    products = pd.DataFrame(
        [
            {
                "product_id": "P001",
                "product_name": "测试商品",
                "category": "test",
                "price": Decimal("10.00"),
                "cost": Decimal("5.00"),
                "launch_date": date(2025, 1, 1),
            }
        ]
    )
    traffic = pd.DataFrame(
        [
            {
                "date": day,
                "product_id": "P001",
                "impressions": 100,
                "clicks": 10,
                "visits": 12,
                "is_missing": False,
            }
            for day in (
                date(2026, 3, 29),
                date(2026, 3, 30),
                date(2026, 3, 31),
                date(2026, 4, 1),
                date(2026, 4, 2),
            )
        ]
    )
    tables = {
        "products": products,
        "customers": pd.DataFrame(),
        "traffic": traffic,
        "marketing": pd.DataFrame(),
        "orders": pd.DataFrame(),
    }
    manifest = {
        "dataset_id": "0123456789abcdef",
        "dataset_version": "test",
        "source_label": "Synthetic E-commerce Data",
        "tables": {
            name: {"rows": len(frame)}
            for name, frame in tables.items()
        },
    }
    connection = duckdb.connect(database=":memory:")
    for name, frame in tables.items():
        if len(frame.columns):
            connection.register(name, frame)
    with Catalog(connection, tables, manifest) as opened:
        yield opened


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2026, 4, 1), "2026-04-01"),
        (datetime(2026, 4, 1, 12, 30), "2026-04-01T12:30:00"),
        (Decimal("12.50"), 12.5),
        (np.int64(7), 7),
        (pd.NA, None),
        (pd.NaT, None),
        (float("nan"), None),
    ],
)
def test_query_result_values_are_json_safe(value: object, expected: object) -> None:
    assert _python_value(value) == expected


def test_query_product_returns_only_requested_products(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_product",
        {"product_ids": ["P002", "P001"]},
        tool_context,
    )

    assert result.result_id == "result_0007"
    assert result.tool_name == "query_product"
    assert result.dataset_id == catalog.verified_summary["dataset_id"]
    assert result.source_label == catalog.verified_summary["source_label"]
    assert result.columns == [
        "product_id",
        "product_name",
        "category",
        "price",
        "cost",
        "launch_date",
    ]
    assert [row.product_id for row in result.rows] == ["P001", "P002"]
    assert result.row_count == 2
    assert all(
        "anomaly" not in key
        for row in result.rows
        for key in row.model_dump().keys()
    )


def test_query_product_empty_match_is_not_an_error(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_product",
        {"product_ids": ["P999"]},
        tool_context,
    )

    assert result.rows == []
    assert result.row_count == 0
    assert result.warnings == ["no rows matched the request"]


def test_query_sales_matches_independent_catalog_aggregates(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    arguments = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
        "include_refunds": True,
    }

    result = registry.invoke("query_sales", arguments, tool_context)

    assert result.columns == [
        "period",
        "product_id",
        "category",
        "region",
        "channel",
        "gmv",
        "orders",
        "units",
        "refund_orders",
    ]
    assert result.rows == sorted(
        result.rows,
        key=lambda row: (row.period, row.product_id, row.region, row.channel),
    )
    for period, start_date, end_date in (
        ("current", "2026-04-01", "2026-04-30"),
        ("previous", "2026-03-02", "2026-03-31"),
    ):
        expected = catalog.execute(
            """
            select
                sum(revenue),
                count(distinct order_id),
                sum(quantity),
                count(distinct order_id) filter (where is_refund)
            from orders
            where order_date between ? and ?
            """,
            [start_date, end_date],
        ).fetchone()
        assert expected is not None
        rows = [row for row in result.rows if row.period == period]
        assert sum(row.gmv for row in rows) == pytest.approx(float(expected[0]))
        assert sum(row.orders for row in rows) == expected[1]
        assert sum(row.units for row in rows) == expected[2]
        assert sum(row.refund_orders for row in rows) == expected[3]


def test_query_sales_product_filter_and_refund_switch(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    base = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P004"],
    }

    included = registry.invoke(
        "query_sales", {**base, "include_refunds": True}, tool_context
    )
    excluded = registry.invoke(
        "query_sales", {**base, "include_refunds": False}, tool_context
    )

    assert {row.product_id for row in included.rows} == {"P004"}
    assert {row.product_id for row in excluded.rows} == {"P004"}
    assert sum(row.refund_orders for row in included.rows) > 0
    assert sum(row.refund_orders for row in excluded.rows) == 0
    for period, start_date, end_date in (
        ("current", "2026-04-01", "2026-04-30"),
        ("previous", "2026-03-02", "2026-03-31"),
    ):
        expected = catalog.execute(
            """
            select sum(revenue), count(distinct order_id), sum(quantity)
            from orders
            where order_date between ? and ?
              and product_id = ?
              and not is_refund
            """,
            [start_date, end_date, "P004"],
        ).fetchone()
        assert expected is not None
        rows = [row for row in excluded.rows if row.period == period]
        assert sum(row.gmv for row in rows) == pytest.approx(float(expected[0]))
        assert sum(row.orders for row in rows) == expected[1]
        assert sum(row.units for row in rows) == expected[2]


def test_query_traffic_preserves_explicit_missingness_and_include_switch(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    arguments = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P005"],
        "include_missing": True,
    }

    included = registry.invoke("query_traffic", arguments, tool_context)

    assert included.columns == [
        "period",
        "product_id",
        "category",
        "impressions",
        "clicks",
        "visits",
        "observed_days",
        "missing_days",
    ]
    assert [(row.period, row.product_id) for row in included.rows] == [
        ("current", "P005"),
        ("previous", "P005"),
    ]
    current = included.rows[0]
    previous = included.rows[1]
    assert (current.observed_days, current.missing_days) == (24, 6)
    assert (previous.observed_days, previous.missing_days) == (30, 0)
    assert current.observed_days + current.missing_days == 30
    assert previous.observed_days + previous.missing_days == 30

    excluded = registry.invoke(
        "query_traffic",
        {**arguments, "include_missing": False},
        tool_context,
    )
    assert [(row.period, row.product_id) for row in excluded.rows] == [
        ("previous", "P005")
    ]


def test_query_traffic_counts_implicit_missing_product_day(
    catalog_with_one_traffic_day_omitted: Catalog,
) -> None:
    context = ToolContext(
        catalog=catalog_with_one_traffic_day_omitted,
        prior_executions=(),
        next_result_id=lambda: "result_0007",
    )
    arguments = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "comparison_start_date": "2026-03-29",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P001"],
        "include_missing": True,
    }

    included = build_default_registry().invoke(
        "query_traffic",
        arguments,
        context,
    )

    current = next(row for row in included.rows if row.period == "current")
    previous = next(row for row in included.rows if row.period == "previous")
    assert (current.observed_days, current.missing_days) == (2, 1)
    assert (previous.observed_days, previous.missing_days) == (3, 0)
    assert current.observed_days + current.missing_days == 3
    assert previous.observed_days + previous.missing_days == 3
    assert (current.impressions, current.clicks, current.visits) == (200, 20, 24)

    excluded = build_default_registry().invoke(
        "query_traffic",
        {**arguments, "include_missing": False},
        context,
    )
    assert [(row.period, row.product_id) for row in excluded.rows] == [
        ("previous", "P001")
    ]


def test_query_traffic_returns_stably_sorted_period_product_rows(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_traffic",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-03",
            "comparison_start_date": "2026-03-29",
            "comparison_end_date": "2026-03-31",
            "product_ids": ["P003", "P001", "P002"],
            "include_missing": True,
        },
        tool_context,
    )

    assert result.rows == sorted(
        result.rows,
        key=lambda row: (row.period, row.product_id),
    )


def test_query_traffic_empty_product_filter_covers_existing_catalog_scale(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    product_count = catalog.execute("select count(*) from products").fetchone()
    assert product_count is not None
    assert product_count[0] == 40

    result = registry.invoke(
        "query_traffic",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "comparison_start_date": "2026-03-02",
            "comparison_end_date": "2026-03-31",
            "product_ids": [],
            "include_missing": True,
        },
        tool_context,
    )

    assert result.row_count == 2 * product_count[0]
    assert {row.product_id for row in result.rows} == {
        f"P{product_id:03d}" for product_id in range(1, product_count[0] + 1)
    }
    assert all(row.observed_days + row.missing_days == 30 for row in result.rows)


def test_query_marketing_matches_catalog_spend_at_campaign_grain(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_marketing",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "comparison_start_date": "2026-03-02",
            "comparison_end_date": "2026-03-31",
            "product_ids": ["P001"],
        },
        tool_context,
    )

    assert result.columns == [
        "period",
        "product_id",
        "category",
        "campaign_id",
        "spend",
    ]
    assert {row.product_id for row in result.rows} == {"P001"}
    for period, start_date, end_date in (
        ("current", "2026-04-01", "2026-04-30"),
        ("previous", "2026-03-02", "2026-03-31"),
    ):
        expected = catalog.execute(
            """
            select product_id, campaign_id, sum(spend)
            from marketing
            where product_id = ? and date between ? and ?
            group by product_id, campaign_id
            order by product_id, campaign_id
            """,
            ["P001", start_date, end_date],
        ).fetchall()
        actual = [
            (row.product_id, row.campaign_id, row.spend)
            for row in result.rows
            if row.period == period
        ]
        assert actual == [
            (product_id, campaign_id, pytest.approx(float(spend)))
            for product_id, campaign_id, spend in expected
        ]


def test_query_marketing_empty_product_filter_and_stable_campaign_sort(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_marketing",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-03",
            "comparison_start_date": "2026-03-29",
            "comparison_end_date": "2026-03-31",
            "product_ids": [],
        },
        tool_context,
    )

    assert len({row.product_id for row in result.rows}) > 1
    assert result.rows == sorted(
        result.rows,
        key=lambda row: (row.period, row.product_id, row.category, row.campaign_id),
    )


def test_query_marketing_accepts_366_day_window_at_execution_boundary(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_marketing",
        {
            "start_date": "2025-05-01",
            "end_date": "2026-05-01",
            "comparison_start_date": "2024-05-01",
            "comparison_end_date": "2025-05-01",
            "product_ids": ["P001"],
        },
        tool_context,
    )

    assert result.tool_name == "query_marketing"
