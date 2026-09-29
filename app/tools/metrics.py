from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.data.metrics import safe_divide
from app.tools.base import ToolContext
from app.tools.schemas import (
    CalculateMetricsInput,
    CalculateMetricsResult,
    CalculateMetricsRow,
    MetricName,
    PriorToolExecution,
    QueryMarketingInput,
    QuerySalesInput,
    QueryTrafficInput,
)

METRIC_REQUIREMENTS: dict[str, frozenset[str]] = {
    "current_gmv": frozenset({"query_sales"}),
    "previous_gmv": frozenset({"query_sales"}),
    "gmv_change_rate": frozenset({"query_sales"}),
    "current_orders": frozenset({"query_sales"}),
    "previous_orders": frozenset({"query_sales"}),
    "current_aov": frozenset({"query_sales"}),
    "previous_aov": frozenset({"query_sales"}),
    "aov_change_rate": frozenset({"query_sales"}),
    "current_impressions": frozenset({"query_traffic"}),
    "previous_impressions": frozenset({"query_traffic"}),
    "current_clicks": frozenset({"query_traffic"}),
    "previous_clicks": frozenset({"query_traffic"}),
    "current_visits": frozenset({"query_traffic"}),
    "previous_visits": frozenset({"query_traffic"}),
    "current_ctr": frozenset({"query_traffic"}),
    "previous_ctr": frozenset({"query_traffic"}),
    "current_cvr": frozenset({"query_sales", "query_traffic"}),
    "previous_cvr": frozenset({"query_sales", "query_traffic"}),
    "cvr_change": frozenset({"query_sales", "query_traffic"}),
    "cvr_change_rate": frozenset({"query_sales", "query_traffic"}),
    "traffic_change_rate": frozenset({"query_traffic"}),
    "current_observed_days": frozenset({"query_traffic"}),
    "previous_observed_days": frozenset({"query_traffic"}),
    "current_spend": frozenset({"query_marketing"}),
    "previous_spend": frozenset({"query_marketing"}),
    "current_roas": frozenset({"query_sales", "query_marketing"}),
    "previous_roas": frozenset({"query_sales", "query_marketing"}),
    "current_refund_rate": frozenset({"query_sales"}),
    "refund_rate": frozenset({"query_sales"}),
    "evidence_value": frozenset({"query_sales", "query_traffic"}),
}

_INPUT_MODELS = {
    "query_sales": QuerySalesInput,
    "query_traffic": QueryTrafficInput,
    "query_marketing": QueryMarketingInput,
}
_WINDOW_FIELDS = (
    "start_date",
    "end_date",
    "comparison_start_date",
    "comparison_end_date",
)
_NATURAL_KEYS = {
    "query_sales": ("period", "product_id", "category", "region", "channel"),
    "query_traffic": ("period", "product_id", "category"),
    "query_marketing": ("period", "product_id", "category", "campaign_id"),
}
_ADDITIVE_FIELDS = {
    "query_sales": ("gmv", "orders", "units", "refund_orders"),
    "query_traffic": (
        "impressions",
        "clicks",
        "visits",
        "observed_days",
        "missing_days",
    ),
    "query_marketing": ("spend",),
}


def _required_tool_names(metrics: Iterable[MetricName]) -> set[str]:
    return {
        tool_name
        for metric in metrics
        for tool_name in METRIC_REQUIREMENTS[metric.value]
    }


def _resolve_sources(
    executions: tuple[PriorToolExecution, ...],
    metrics: list[MetricName],
) -> dict[str, PriorToolExecution]:
    required = _required_tool_names(metrics)
    matches: dict[str, list[PriorToolExecution]] = {
        tool_name: [] for tool_name in required
    }
    for execution in executions:
        tool_name = execution.result.tool_name
        if tool_name in matches:
            matches[tool_name].append(execution)

    for tool_name in sorted(required):
        count = len(matches[tool_name])
        if count == 0:
            raise ValueError(f"requested metrics requires {tool_name}")
        if count > 1:
            raise ValueError(f"duplicate {tool_name} source execution")
    return {tool_name: items[0] for tool_name, items in matches.items()}


def _normalize_arguments(
    sources: dict[str, PriorToolExecution],
) -> dict[str, QuerySalesInput | QueryTrafficInput | QueryMarketingInput]:
    return {
        tool_name: _INPUT_MODELS[tool_name].model_validate(execution.arguments)
        for tool_name, execution in sources.items()
    }


def _validate_source_compatibility(
    sources: dict[str, PriorToolExecution],
    arguments: dict[
        str,
        QuerySalesInput | QueryTrafficInput | QueryMarketingInput,
    ],
    context: ToolContext,
) -> None:
    summary = context.catalog.verified_summary
    expected_dataset = summary["dataset_id"]
    expected_label = summary["source_label"]
    for execution in sources.values():
        if execution.result.dataset_id != expected_dataset:
            raise ValueError("dataset_id does not match the current Catalog")
        if execution.result.source_label != expected_label:
            raise ValueError("source_label does not match the current Catalog")

    normalized = list(arguments.items())
    if normalized:
        _, first = normalized[0]
        first_window = tuple(getattr(first, field) for field in _WINDOW_FIELDS)
        first_products = tuple(sorted(first.product_ids))
        for _, current in normalized[1:]:
            current_window = tuple(
                getattr(current, field) for field in _WINDOW_FIELDS
            )
            if current_window != first_window:
                raise ValueError("source query window boundaries are incompatible")
            if tuple(sorted(current.product_ids)) != first_products:
                raise ValueError("source product_ids are incompatible")

    sales = arguments.get("query_sales")
    if isinstance(sales, QuerySalesInput) and sales.include_refunds is not True:
        raise ValueError("query_sales include_refunds must be true")
    traffic = arguments.get("query_traffic")
    if isinstance(traffic, QueryTrafficInput) and traffic.include_missing is not True:
        raise ValueError("query_traffic include_missing must be true")


def _validate_grouping(
    sources: dict[str, PriorToolExecution],
    group_by: list[str],
) -> None:
    for tool_name, execution in sources.items():
        missing = [dimension for dimension in group_by if dimension not in execution.result.columns]
        if missing:
            raise ValueError(
                f"{tool_name} does not provide group_by dimension {missing[0]}"
            )


def _row_value(row: Any, field: str) -> Any:
    return getattr(row, field)


def _aggregate_sources(
    sources: dict[str, PriorToolExecution],
    group_by: list[str],
) -> tuple[
    dict[tuple[Any, ...], dict[str, dict[str, dict[str, float | int]]]],
    set[tuple[Any, ...]],
    dict[tuple[Any, ...], dict[str, dict[str, int]]],
]:
    aggregates: dict[
        tuple[Any, ...],
        dict[str, dict[str, dict[str, float | int]]],
    ] = {}
    group_keys: set[tuple[Any, ...]] = {()} if not group_by else set()
    traffic_observed_days: dict[
        tuple[Any, ...],
        dict[str, dict[str, int]],
    ] = {}

    for tool_name, execution in sources.items():
        seen_natural_keys: set[tuple[Any, ...]] = set()
        for row in execution.result.rows:
            natural_key = tuple(
                _row_value(row, field) for field in _NATURAL_KEYS[tool_name]
            )
            if natural_key in seen_natural_keys:
                raise ValueError(f"duplicate {tool_name} result key")
            seen_natural_keys.add(natural_key)

            group_key = tuple(_row_value(row, field) for field in group_by)
            group_keys.add(group_key)
            period = _row_value(row, "period")
            if tool_name == "query_traffic":
                traffic_observed_days.setdefault(group_key, {}).setdefault(
                    period, {}
                )[_row_value(row, "product_id")] = _row_value(
                    row, "observed_days"
                )
            values = (
                aggregates
                .setdefault(group_key, {})
                .setdefault(tool_name, {})
                .setdefault(period, {})
            )
            for field in _ADDITIVE_FIELDS[tool_name]:
                values[field] = values.get(field, 0) + _row_value(row, field)

    return aggregates, group_keys, traffic_observed_days


def _period_values(
    aggregate: dict[str, dict[str, dict[str, float | int]]],
    tool_name: str,
    period: str,
) -> dict[str, float | int]:
    return aggregate.get(tool_name, {}).get(period, {})


def _number(
    aggregate: dict[str, dict[str, dict[str, float | int]]],
    tool_name: str,
    period: str,
    field: str,
) -> float | int:
    return _period_values(aggregate, tool_name, period).get(field, 0)


def _difference_rate(current: float | int | None, previous: float | int | None):
    if current is None or previous is None:
        return None
    return safe_divide(current - previous, previous)


def _calculate_values(
    aggregate: dict[str, dict[str, dict[str, float | int]]],
    expected_days: dict[str, int],
    observed_days_by_product: dict[str, dict[str, int]],
) -> dict[str, float | int | None]:
    current_gmv = _number(aggregate, "query_sales", "current", "gmv")
    previous_gmv = _number(aggregate, "query_sales", "previous", "gmv")
    current_orders = _number(aggregate, "query_sales", "current", "orders")
    previous_orders = _number(aggregate, "query_sales", "previous", "orders")
    current_aov = safe_divide(current_gmv, current_orders)
    previous_aov = safe_divide(previous_gmv, previous_orders)

    current_impressions = _number(
        aggregate, "query_traffic", "current", "impressions"
    )
    previous_impressions = _number(
        aggregate, "query_traffic", "previous", "impressions"
    )
    current_clicks = _number(aggregate, "query_traffic", "current", "clicks")
    previous_clicks = _number(aggregate, "query_traffic", "previous", "clicks")
    current_visits = _number(aggregate, "query_traffic", "current", "visits")
    previous_visits = _number(aggregate, "query_traffic", "previous", "visits")
    current_observed_days = _number(
        aggregate, "query_traffic", "current", "observed_days"
    )
    previous_observed_days = _number(
        aggregate, "query_traffic", "previous", "observed_days"
    )

    product_ids = {
        product_id
        for period_values in observed_days_by_product.values()
        for product_id in period_values
    }
    complete_periods = {
        period: bool(product_ids)
        and all(
            observed_days_by_product.get(period, {}).get(product_id)
            == period_expected_days
            for product_id in product_ids
        )
        for period, period_expected_days in expected_days.items()
    }
    current_cvr = None
    if complete_periods["current"]:
        current_cvr = safe_divide(current_orders, current_visits)
    previous_cvr = None
    if complete_periods["previous"]:
        previous_cvr = safe_divide(previous_orders, previous_visits)
    cvr_change = None
    if current_cvr is not None and previous_cvr is not None:
        cvr_change = current_cvr - previous_cvr

    current_spend = _number(aggregate, "query_marketing", "current", "spend")
    previous_spend = _number(aggregate, "query_marketing", "previous", "spend")
    refund_orders = _number(
        aggregate, "query_sales", "current", "refund_orders"
    )
    refund_rate = safe_divide(refund_orders, current_orders)
    traffic_change_rate = _difference_rate(current_visits, previous_visits)

    evidence_value = None
    if not complete_periods["current"] or not complete_periods["previous"]:
        evidence_value = float(
            min(current_observed_days, previous_observed_days)
        )
    elif refund_rate is not None and refund_rate >= 0.12:
        evidence_value = refund_rate
    elif cvr_change is not None and cvr_change <= -0.01:
        evidence_value = cvr_change
    elif traffic_change_rate is not None and traffic_change_rate <= -0.20:
        evidence_value = traffic_change_rate

    return {
        "current_gmv": current_gmv,
        "previous_gmv": previous_gmv,
        "gmv_change_rate": _difference_rate(current_gmv, previous_gmv),
        "current_orders": current_orders,
        "previous_orders": previous_orders,
        "current_aov": current_aov,
        "previous_aov": previous_aov,
        "aov_change_rate": _difference_rate(current_aov, previous_aov),
        "current_impressions": current_impressions,
        "previous_impressions": previous_impressions,
        "current_clicks": current_clicks,
        "previous_clicks": previous_clicks,
        "current_visits": current_visits,
        "previous_visits": previous_visits,
        "current_cvr": current_cvr,
        "previous_cvr": previous_cvr,
        "cvr_change": cvr_change,
        "cvr_change_rate": _difference_rate(current_cvr, previous_cvr),
        "current_ctr": safe_divide(current_clicks, current_impressions),
        "previous_ctr": safe_divide(previous_clicks, previous_impressions),
        "traffic_change_rate": traffic_change_rate,
        "current_observed_days": current_observed_days,
        "previous_observed_days": previous_observed_days,
        "current_spend": current_spend,
        "previous_spend": previous_spend,
        "current_roas": safe_divide(current_gmv, current_spend),
        "previous_roas": safe_divide(previous_gmv, previous_spend),
        "current_refund_rate": refund_rate,
        "refund_rate": refund_rate,
        "evidence_value": evidence_value,
    }


def _expected_days(
    arguments: dict[
        str,
        QuerySalesInput | QueryTrafficInput | QueryMarketingInput,
    ],
) -> dict[str, int]:
    first = next(iter(arguments.values()))
    return {
        "current": (first.end_date - first.start_date).days + 1,
        "previous": (
            first.comparison_end_date - first.comparison_start_date
        ).days
        + 1,
    }


def _group_payload(group_by: list[str], group_key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(group_by, group_key, strict=True))


def calculate_metrics(
    args: CalculateMetricsInput,
    context: ToolContext,
) -> CalculateMetricsResult:
    if MetricName.EVIDENCE_VALUE in args.metrics and args.group_by != ["product_id"]:
        raise ValueError("evidence_value requires group_by=['product_id']")

    sources = _resolve_sources(context.prior_executions, args.metrics)
    normalized_arguments = _normalize_arguments(sources)
    _validate_source_compatibility(sources, normalized_arguments, context)
    group_by = [dimension.value for dimension in args.group_by]
    _validate_grouping(sources, group_by)
    aggregates, group_keys, traffic_observed_days = _aggregate_sources(
        sources, group_by
    )
    expected_days = _expected_days(normalized_arguments)

    rows: list[CalculateMetricsRow] = []
    for group_key in sorted(group_keys):
        values = _calculate_values(
            aggregates.get(group_key, {}),
            expected_days,
            traffic_observed_days.get(group_key, {}),
        )
        payload = _group_payload(group_by, group_key)
        payload.update({metric.value: values[metric.value] for metric in args.metrics})
        rows.append(CalculateMetricsRow.model_validate(payload))

    summary = context.catalog.verified_summary
    columns = [*group_by, *(metric.value for metric in args.metrics)]
    return CalculateMetricsResult.model_validate(
        {
            "result_id": context.next_result_id(),
            "tool_name": "calculate_metrics",
            "dataset_id": summary["dataset_id"],
            "source_label": summary["source_label"],
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "warnings": [] if rows else ["no rows matched the prior results"],
        }
    )
