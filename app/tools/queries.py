from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any, TypeVar

import pandas as pd

from app.tools.base import ToolContext
from app.tools.schemas import (
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    TRAFFIC_COLUMNS,
    QueryProductInput,
    QueryProductResult,
    QuerySalesInput,
    QuerySalesResult,
    QueryTrafficInput,
    QueryTrafficResult,
)

QueryResultT = TypeVar(
    "QueryResultT",
    QueryProductResult,
    QuerySalesResult,
    QueryTrafficResult,
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


def query_traffic(
    args: QueryTrafficInput,
    context: ToolContext,
) -> QueryTrafficResult:
    product_clause = ""
    product_parameters: list[str] = []
    if args.product_ids:
        placeholders = ", ".join("?" for _ in args.product_ids)
        product_clause = f"where product_id in ({placeholders})"
        product_parameters = list(args.product_ids)

    frame = context.catalog.execute(
        f"""
        with periods(period, period_start, period_end) as (
          values
            ('current', cast(? as date), cast(? as date)),
            ('previous', cast(? as date), cast(? as date))
        ),
        selected_products as (
          select product_id, category
          from products
          {product_clause}
        ),
        traffic_grid as (
          select
            periods.period,
            periods.period_start,
            periods.period_end,
            selected_products.product_id,
            selected_products.category,
            cast(calendar.day as date) as day
          from periods
          cross join selected_products
          cross join lateral generate_series(
            periods.period_start,
            periods.period_end,
            interval '1 day'
          ) as calendar(day)
        ),
        aggregates as (
          select
            traffic_grid.period,
            traffic_grid.product_id,
            traffic_grid.category,
            coalesce(
              sum(traffic.impressions) filter (
                where traffic.date is not null and not traffic.is_missing
              ),
              0
            ) as impressions,
            coalesce(
              sum(traffic.clicks) filter (
                where traffic.date is not null and not traffic.is_missing
              ),
              0
            ) as clicks,
            coalesce(
              sum(traffic.visits) filter (
                where traffic.date is not null and not traffic.is_missing
              ),
              0
            ) as visits,
            count(*) filter (
              where traffic.date is not null and not traffic.is_missing
            ) as observed_days,
            count(*) filter (
              where traffic.date is null or traffic.is_missing
            ) as missing_days,
            date_diff(
              'day',
              traffic_grid.period_start,
              traffic_grid.period_end
            ) + 1 as expected_days
          from traffic_grid
          left join traffic
            on traffic.product_id = traffic_grid.product_id
           and traffic.date = traffic_grid.day
          group by
            traffic_grid.period,
            traffic_grid.period_start,
            traffic_grid.period_end,
            traffic_grid.product_id,
            traffic_grid.category
        )
        select
          period,
          product_id,
          category,
          impressions,
          clicks,
          visits,
          observed_days,
          missing_days
        from aggregates
        where observed_days + missing_days = expected_days
          and (? or missing_days = 0)
        order by period, product_id
        """,
        [
            args.start_date,
            args.end_date,
            args.comparison_start_date,
            args.comparison_end_date,
            *product_parameters,
            args.include_missing,
        ],
    ).fetch_df()
    frame = frame.loc[:, TRAFFIC_COLUMNS]
    return _frame_result("query_traffic", frame, context, QueryTrafficResult)
