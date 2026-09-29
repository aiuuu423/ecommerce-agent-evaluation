from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

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
