from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from functools import cmp_to_key

from app.tools.schemas import (
    AnyToolResult,
    CalculateMetricsResult,
    CalculateMetricsRow,
    PriorToolExecution,
    QueryProductResult,
)

from .config import DEFAULT_TOP_K, STABLE_SORT_FIELDS
from .schemas import ParsedRequest, TaskKind

_INSUFFICIENT = "数据不足"


class AnswerEvidenceError(ValueError):
    pass


def _decimal(value: int | float) -> Decimal:
    return Decimal(str(value))


def format_money(value: int | float | None) -> str:
    if value is None:
        return _INSUFFICIENT
    rounded = _decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"¥{rounded:,.2f}"


def format_count(value: int | None) -> str:
    if value is None:
        return _INSUFFICIENT
    return f"{value:,d}"


def format_rate(value: int | float | None) -> str:
    if value is None:
        return _INSUFFICIENT
    percentage = (_decimal(value) * 100).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    if percentage == 0:
        percentage = abs(percentage)
    sign = "+" if percentage > 0 else ""
    return f"{sign}{percentage:.2f}%"


def tool_results_by_name(
    executions: Sequence[PriorToolExecution],
) -> dict[str, AnyToolResult]:
    results: dict[str, AnyToolResult] = {}
    for execution in executions:
        name = execution.result.tool_name
        if name in results:
            raise AnswerEvidenceError(f"duplicate tool result: {name}")
        results[name] = execution.result
    return results


def _source_line(results: dict[str, AnyToolResult]) -> str:
    labels = tuple(dict.fromkeys(result.source_label for result in results.values()))
    if not labels:
        return "数据来源：数据不足。"
    return f"数据来源：{'、'.join(labels)}。"


def _window_line(parsed: ParsedRequest) -> str:
    windows = parsed.windows
    return (
        f"时间窗口：当前期 {windows.start_date.isoformat()} 至 "
        f"{windows.end_date.isoformat()}；对照期 "
        f"{windows.comparison_start_date.isoformat()} 至 "
        f"{windows.comparison_end_date.isoformat()}。"
    )


def _answer_preamble(
    parsed: ParsedRequest,
    results: dict[str, AnyToolResult],
) -> list[str]:
    return [_source_line(results), _window_line(parsed)]


def _has_warning(results: dict[str, AnyToolResult]) -> bool:
    return any(result.warnings for result in results.values())


def _metrics_result(
    results: dict[str, AnyToolResult],
) -> CalculateMetricsResult | None:
    result = results.get("calculate_metrics")
    if isinstance(result, CalculateMetricsResult):
        return result
    return None


def _insufficient_answer(
    parsed: ParsedRequest,
    results: dict[str, AnyToolResult],
    reason: str,
) -> str:
    return "\n".join(
        [
            *_answer_preamble(parsed, results),
            f"数据不足：{reason}，无法形成结论。",
        ]
    )


def _contains_missing_value(
    result: CalculateMetricsResult,
    rows: Sequence[CalculateMetricsRow],
) -> bool:
    return any(getattr(row, column) is None for row in rows for column in result.columns)


def _answer_heading(conclusion: str, has_missing_value: bool) -> str:
    if has_missing_value:
        return "限制：实际输出行存在数据不足；以下仅展示工具返回值，不形成肯定判断。"
    return conclusion


def _format_row(
    row: CalculateMetricsRow,
    fields: Sequence[tuple[str, str, Callable[[object], str]]],
) -> str:
    values = []
    for field, label, formatter in fields:
        values.append(f"{label} {formatter(getattr(row, field))}")
    return "；".join(values)


def _money(value: object) -> str:
    return format_money(value)  # type: ignore[arg-type]


def _count(value: object) -> str:
    return format_count(value)  # type: ignore[arg-type]


def _rate(value: object) -> str:
    return format_rate(value)  # type: ignore[arg-type]


def _render_gmv(
    row: CalculateMetricsRow,
    *,
    has_missing_value: bool,
) -> list[str]:
    return [
        _answer_heading(
            "结论：当前 GMV、订单量与客单价相对对照期的变化如下。",
            has_missing_value,
        ),
        "- "
        + _format_row(
            row,
            (
                ("current_gmv", "当前 GMV", _money),
                ("previous_gmv", "对照 GMV", _money),
                ("gmv_change_rate", "GMV 变化", _rate),
            ),
        ),
        "- "
        + _format_row(
            row,
            (
                ("current_orders", "当前订单", _count),
                ("previous_orders", "对照订单", _count),
            ),
        ),
        "- "
        + _format_row(
            row,
            (
                ("current_aov", "当前客单价", _money),
                ("previous_aov", "对照客单价", _money),
                ("aov_change_rate", "客单价变化", _rate),
            ),
        ),
    ]


_CONVERSION_FIELDS: tuple[tuple[str, str, Callable[[object], str]], ...] = (
    ("current_cvr", "当前转化率", _rate),
    ("previous_cvr", "对照转化率", _rate),
    ("cvr_change", "转化率变化", _rate),
    ("cvr_change_rate", "转化率相对变化", _rate),
    ("current_visits", "当前访问", _count),
    ("previous_visits", "对照访问", _count),
    ("current_orders", "当前订单", _count),
    ("previous_orders", "对照订单", _count),
    ("current_observed_days", "当前观测天数", _count),
    ("previous_observed_days", "对照观测天数", _count),
)


def _render_conversion(
    rows: Sequence[CalculateMetricsRow],
    *,
    grouped: bool,
    has_missing_value: bool,
) -> list[str]:
    lines = [
        _answer_heading(
            "结论：转化、访问与订单的同期变化如下，不据此作因果判断。",
            has_missing_value,
        )
    ]
    if grouped:
        for row in rows:
            label = row.product_id if row.product_id is not None else _INSUFFICIENT
            lines.append(f"- {label}：{_format_row(row, _CONVERSION_FIELDS)}")
        return lines

    row = rows[0]
    lines.extend(
        [
            "- " + _format_row(row, _CONVERSION_FIELDS[:4]),
            "- " + _format_row(row, _CONVERSION_FIELDS[4:8]),
            "- " + _format_row(row, _CONVERSION_FIELDS[8:]),
        ]
    )
    return lines


_PRODUCT_FIELDS: dict[
    TaskKind,
    tuple[tuple[str, str, Callable[[object], str]], ...],
] = {
    TaskKind.PRODUCT_ANOMALY: (
        ("current_gmv", "当前 GMV", _money),
        ("previous_gmv", "对照 GMV", _money),
        ("gmv_change_rate", "GMV 变化", _rate),
        ("current_orders", "当前订单", _count),
        ("previous_orders", "对照订单", _count),
        ("current_refund_rate", "退款率", _rate),
        ("current_visits", "当前访问", _count),
        ("previous_visits", "对照访问", _count),
        ("traffic_change_rate", "访问变化", _rate),
        ("current_ctr", "当前点击率", _rate),
        ("previous_ctr", "对照点击率", _rate),
        ("current_cvr", "当前转化率", _rate),
        ("previous_cvr", "对照转化率", _rate),
        ("cvr_change", "转化率变化", _rate),
    ),
    TaskKind.NEXT_WEEK_PRIORITY: (
        ("current_gmv", "当前 GMV", _money),
        ("gmv_change_rate", "GMV 变化", _rate),
        ("current_visits", "当前访问", _count),
        ("traffic_change_rate", "访问变化", _rate),
        ("current_cvr", "当前转化率", _rate),
        ("cvr_change", "转化率变化", _rate),
    ),
    TaskKind.PRODUCTS_TO_WATCH: (
        ("evidence_value", "证据值", _rate),
        ("gmv_change_rate", "GMV 变化", _rate),
        ("traffic_change_rate", "访问变化", _rate),
        ("cvr_change", "转化率变化", _rate),
        ("current_refund_rate", "退款率", _rate),
    ),
}

_PRODUCT_CONCLUSIONS = {
    TaskKind.PRODUCT_ANOMALY: "结论：以下商品指标出现同步变化，仅陈述工具证据。",
    TaskKind.NEXT_WEEK_PRIORITY: "结论：以下商品可按冻结指标顺序进入下周关注队列。",
    TaskKind.PRODUCTS_TO_WATCH: "结论：以下商品具有工具结果返回的关注信号。",
}


def _compare_rows(
    left: CalculateMetricsRow,
    right: CalculateMetricsRow,
    sort_fields: Sequence[str],
) -> int:
    for specification in sort_fields:
        field, direction = specification.rsplit(":", 1)
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            if left_value is right_value:
                continue
            return 1 if left_value is None else -1
        if left_value == right_value:
            continue
        comparison = -1 if left_value < right_value else 1
        return comparison if direction == "asc" else -comparison
    return 0


def _product_names(results: dict[str, AnyToolResult]) -> dict[str, str]:
    result = results.get("query_product")
    if not isinstance(result, QueryProductResult):
        return {}
    return {row.product_id: row.product_name for row in result.rows}


def _render_products(
    task: TaskKind,
    rows: Sequence[CalculateMetricsRow],
    results: dict[str, AnyToolResult],
    *,
    has_missing_value: bool,
) -> list[str]:
    names = _product_names(results)
    lines = [
        _answer_heading(
            _PRODUCT_CONCLUSIONS[task],
            has_missing_value,
        )
    ]
    for row in rows:
        product_id = row.product_id
        if product_id is None:
            label = _INSUFFICIENT
        elif product_id in names:
            label = f"{names[product_id]}（{product_id}）"
        else:
            label = product_id
        lines.append(f"- {label}：{_format_row(row, _PRODUCT_FIELDS[task])}")
    return lines


def _sorted_top_rows(
    task: TaskKind,
    rows: Sequence[CalculateMetricsRow],
) -> tuple[CalculateMetricsRow, ...]:
    return tuple(
        sorted(
            rows,
            key=cmp_to_key(
                lambda left, right: _compare_rows(
                    left,
                    right,
                    STABLE_SORT_FIELDS[task],
                )
            ),
        )[:DEFAULT_TOP_K]
    )


def _select_output_rows(
    parsed: ParsedRequest,
    result: CalculateMetricsResult,
) -> tuple[CalculateMetricsRow, ...]:
    is_grouped_conversion = (
        parsed.task is TaskKind.CONVERSION_DECLINE
        and "product_id" in result.columns
    )
    if parsed.task in _PRODUCT_FIELDS or is_grouped_conversion:
        return _sorted_top_rows(parsed.task, result.rows)
    return tuple(result.rows[:1])


def render_final_answer(
    parsed: ParsedRequest,
    executions: Sequence[PriorToolExecution],
) -> str:
    results = tool_results_by_name(executions)
    if _has_warning(results):
        return _insufficient_answer(parsed, results, "工具结果包含警告")

    metrics = _metrics_result(results)
    if metrics is None:
        return _insufficient_answer(parsed, results, "缺少指标结果")
    if not metrics.rows:
        return _insufficient_answer(parsed, results, "指标结果为空")

    rows = _select_output_rows(parsed, metrics)
    has_missing_value = _contains_missing_value(metrics, rows)
    if parsed.task is TaskKind.GMV_DIAGNOSIS:
        lines = _render_gmv(rows[0], has_missing_value=has_missing_value)
    elif parsed.task is TaskKind.CONVERSION_DECLINE:
        lines = _render_conversion(
            rows,
            grouped="product_id" in metrics.columns,
            has_missing_value=has_missing_value,
        )
    elif parsed.task in _PRODUCT_FIELDS:
        lines = _render_products(
            parsed.task,
            rows,
            results,
            has_missing_value=has_missing_value,
        )
    else:
        return _insufficient_answer(parsed, results, "任务类型不受支持")

    return "\n".join([*_answer_preamble(parsed, results), *lines])
