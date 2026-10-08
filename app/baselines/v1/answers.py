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
    results: dict[str, AnyToolResult],
    reason: str,
) -> str:
    return f"{_source_line(results)}\n数据不足：{reason}，无法形成结论。"


def _contains_missing_value(
    result: CalculateMetricsResult,
    rows: Sequence[CalculateMetricsRow],
) -> bool:
    metric_columns = [column for column in result.columns if column != "product_id"]
    return any(getattr(row, column) is None for row in rows for column in metric_columns)


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


def _render_gmv(row: CalculateMetricsRow) -> list[str]:
    return [
        "结论：当前 GMV、订单量与客单价相对对照期的变化如下。",
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


def _render_conversion(row: CalculateMetricsRow) -> list[str]:
    return [
        "结论：转化、访问与订单的同期变化如下，不据此作因果判断。",
        "- "
        + _format_row(
            row,
            (
                ("current_cvr", "当前转化率", _rate),
                ("previous_cvr", "对照转化率", _rate),
                ("cvr_change", "转化率变化", _rate),
                ("cvr_change_rate", "转化率相对变化", _rate),
            ),
        ),
        "- "
        + _format_row(
            row,
            (
                ("current_visits", "当前访问", _count),
                ("previous_visits", "对照访问", _count),
                ("current_orders", "当前订单", _count),
                ("previous_orders", "对照订单", _count),
            ),
        ),
        "- "
        + _format_row(
            row,
            (
                ("current_observed_days", "当前观测天数", _count),
                ("previous_observed_days", "对照观测天数", _count),
            ),
        ),
    ]


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
    result: CalculateMetricsResult,
    results: dict[str, AnyToolResult],
) -> list[str]:
    rows = sorted(
        result.rows,
        key=cmp_to_key(
            lambda left, right: _compare_rows(
                left,
                right,
                STABLE_SORT_FIELDS[task],
            )
        ),
    )[:DEFAULT_TOP_K]
    names = _product_names(results)
    lines = [_PRODUCT_CONCLUSIONS[task]]
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


def render_final_answer(
    parsed: ParsedRequest,
    executions: Sequence[PriorToolExecution],
) -> str:
    results = tool_results_by_name(executions)
    if _has_warning(results):
        return _insufficient_answer(results, "工具结果包含警告")

    metrics = _metrics_result(results)
    if metrics is None:
        return _insufficient_answer(results, "缺少指标结果")
    if not metrics.rows:
        return _insufficient_answer(results, "指标结果为空")

    rows = metrics.rows
    if parsed.task is TaskKind.GMV_DIAGNOSIS:
        lines = _render_gmv(rows[0])
    elif parsed.task is TaskKind.CONVERSION_DECLINE:
        lines = _render_conversion(rows[0])
    elif parsed.task in _PRODUCT_FIELDS:
        lines = _render_products(parsed.task, metrics, results)
    else:
        return _insufficient_answer(results, "任务类型不受支持")

    if _contains_missing_value(metrics, rows[:DEFAULT_TOP_K]):
        lines.append("限制：部分指标数据不足，无法计算对应值。")
    return "\n".join([_source_line(results), *lines])
