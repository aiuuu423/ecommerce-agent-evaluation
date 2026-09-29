from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any, TypeVar

import pandas as pd

from app.tools.base import ToolContext
from app.tools.schemas import (
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    QueryProductInput,
    QueryProductResult,
    QuerySalesInput,
    QuerySalesResult,
)

QueryResultT = TypeVar("QueryResultT", QueryProductResult, QuerySalesResult)


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
    result_model: type[QueryResultT],
) -> QueryResultT:
    columns = list(frame.columns)
    rows = [
        {
            column: _python_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in frame.itertuples(index=False, name=None)
    ]
    summary = context.catalog.verified_summary
    return result_model.model_validate(
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
    return _frame_result("query_product", frame, context, QueryProductResult)


def query_sales(
    args: QuerySalesInput,
    context: ToolContext,
) -> QuerySalesResult:
    product_clause = ""
    product_parameters: list[str] = []
    if args.product_ids:
        placeholders = ", ".join("?" for _ in args.product_ids)
        product_clause = f"and orders.product_id in ({placeholders})"
        product_parameters = list(args.product_ids)

    frame = context.catalog.execute(
        f"""
        with periods(period, period_start, period_end) as (
          values
            ('current', cast(? as date), cast(? as date)),
            ('previous', cast(? as date), cast(? as date))
        )
        select
          periods.period,
          orders.product_id,
          products.category,
          customers.region,
          customers.channel,
          sum(orders.revenue) as gmv,
          count(distinct orders.order_id) as orders,
          sum(orders.quantity) as units,
          count(distinct orders.order_id)
            filter (where orders.is_refund) as refund_orders
        from periods
        join orders on orders.order_date between period_start and period_end
        join products using (product_id)
        join customers using (customer_id)
        where (? or not orders.is_refund)
          {product_clause}
        group by all
        order by period, product_id, region, channel
        """,
        [
            args.start_date,
            args.end_date,
            args.comparison_start_date,
            args.comparison_end_date,
            args.include_refunds,
            *product_parameters,
        ],
    ).fetch_df()
    frame = frame.loc[:, SALES_COLUMNS]
    return _frame_result("query_sales", frame, context, QuerySalesResult)
