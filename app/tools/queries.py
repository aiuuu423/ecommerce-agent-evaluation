from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any

import pandas as pd

from app.tools.base import ToolContext
from app.tools.schemas import (
    PRODUCT_COLUMNS,
    QueryProductInput,
    QueryProductResult,
)


def _python_value(value: Any) -> Any:
    if value is None or bool(pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _frame_result(
    tool_name: str,
    frame: pd.DataFrame,
    context: ToolContext,
) -> QueryProductResult:
    columns = list(frame.columns)
    rows = [
        {
            column: _python_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in frame.itertuples(index=False, name=None)
    ]
    summary = context.catalog.verified_summary
    return QueryProductResult.model_validate(
        {
            "result_id": context.next_result_id(),
            "tool_name": tool_name,
            "dataset_id": summary["dataset_id"],
            "source_label": summary["source_label"],
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "warnings": [] if rows else ["no rows matched the request"],
        }
    )


def query_product(
    args: QueryProductInput,
    context: ToolContext,
) -> QueryProductResult:
    placeholders = ", ".join("?" for _ in args.product_ids)
    frame = context.catalog.execute(
        f"""
        select product_id, product_name, category, price, cost, launch_date
        from products
        where product_id in ({placeholders})
        order by product_id
        """,
        list(args.product_ids),
    ).fetch_df()
    frame = frame.loc[:, PRODUCT_COLUMNS]
    return _frame_result("query_product", frame, context)
