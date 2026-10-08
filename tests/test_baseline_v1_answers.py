from datetime import date

import pytest

from app.baselines.v1.answers import (
    AnswerEvidenceError,
    format_count,
    format_money,
    format_rate,
    render_final_answer,
    tool_results_by_name,
)
from app.baselines.v1.schemas import DateWindows, ParsedRequest, TaskKind
from app.llm import AdapterRequest
from app.tools.schemas import (
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    CalculateMetricsResult,
    PriorToolExecution,
    QueryProductResult,
    QuerySalesResult,
)

DATASET_ID = "e1e81533c25e03e5"
SOURCE_LABEL = "Synthetic E-commerce Data"
WINDOWS = DateWindows(
    start_date=date(2026, 4, 1),
    end_date=date(2026, 4, 30),
    comparison_start_date=date(2026, 3, 2),
    comparison_end_date=date(2026, 3, 31),
)


def parsed(
    task: TaskKind,
    product_ids: tuple[str, ...] = (),
    windows: DateWindows = WINDOWS,
) -> ParsedRequest:
    return ParsedRequest(task=task, product_ids=product_ids, windows=windows)


def execution(result: object, number: int = 1) -> PriorToolExecution:
    return PriorToolExecution(
        call_id=f"call_{number}",
        arguments={},
        result=result,
    )


def metrics_execution(
    task: TaskKind,
    rows: list[dict[str, object]],
    *,
    warnings: list[str] | None = None,
) -> PriorToolExecution:
    metric_columns = {
        TaskKind.GMV_DIAGNOSIS: [
            "current_gmv",
            "previous_gmv",
            "gmv_change_rate",
            "current_orders",
            "previous_orders",
            "current_aov",
            "previous_aov",
            "aov_change_rate",
        ],
        TaskKind.CONVERSION_DECLINE: [
            "current_visits",
            "previous_visits",
            "current_orders",
            "previous_orders",
            "current_cvr",
            "previous_cvr",
            "cvr_change",
            "cvr_change_rate",
            "current_observed_days",
            "previous_observed_days",
        ],
        TaskKind.PRODUCT_ANOMALY: [
            "product_id",
            "current_gmv",
            "previous_gmv",
            "gmv_change_rate",
            "current_orders",
            "previous_orders",
            "current_refund_rate",
            "current_visits",
            "previous_visits",
            "traffic_change_rate",
            "current_ctr",
            "previous_ctr",
            "current_cvr",
            "previous_cvr",
            "cvr_change",
        ],
        TaskKind.NEXT_WEEK_PRIORITY: [
            "product_id",
            "current_gmv",
            "gmv_change_rate",
            "current_visits",
            "traffic_change_rate",
            "current_cvr",
            "cvr_change",
        ],
        TaskKind.PRODUCTS_TO_WATCH: [
            "product_id",
            "evidence_value",
            "gmv_change_rate",
            "traffic_change_rate",
            "cvr_change",
            "current_refund_rate",
        ],
    }[task]
    if rows and "product_id" in rows[0] and "product_id" not in metric_columns:
        metric_columns.insert(0, "product_id")
    result = CalculateMetricsResult(
        result_id="result_0001",
        tool_name="calculate_metrics",
        dataset_id=DATASET_ID,
        source_label=SOURCE_LABEL,
        columns=metric_columns,
        rows=rows,
        row_count=len(rows),
        warnings=warnings or [],
    )
    return execution(result)


def product_execution(names: dict[str, str]) -> PriorToolExecution:
    rows = [
        {
            "product_id": product_id,
            "product_name": name,
            "category": "测试类目",
            "price": 10.0,
            "cost": 5.0,
            "launch_date": date(2026, 1, 1),
        }
        for product_id, name in names.items()
    ]
    return execution(
        QueryProductResult(
            result_id="result_0002",
            tool_name="query_product",
            dataset_id=DATASET_ID,
            source_label=SOURCE_LABEL,
            columns=PRODUCT_COLUMNS,
            rows=rows,
            row_count=len(rows),
        ),
        2,
    )


def product_row(product_id: str, value: float) -> dict[str, object]:
    return {
        "product_id": product_id,
        "current_gmv": 1000.0 + value,
        "previous_gmv": 1200.0,
        "gmv_change_rate": value,
        "current_orders": 80,
        "previous_orders": 100,
        "current_refund_rate": 0.125,
        "current_visits": 400,
        "previous_visits": 500,
        "traffic_change_rate": -0.2,
        "current_ctr": 0.25,
        "previous_ctr": 0.3,
        "current_cvr": 0.2,
        "previous_cvr": 0.2,
        "cvr_change": 0.0,
    }


def conversion_row(
    product_id: str,
    cvr_change: float | None,
    *,
    current_visits: int | None = 900,
) -> dict[str, object]:
    return {
        "product_id": product_id,
        "current_visits": current_visits,
        "previous_visits": 1000,
        "current_orders": 90,
        "previous_orders": 120,
        "current_cvr": 0.1,
        "previous_cvr": 0.12,
        "cvr_change": cvr_change,
        "cvr_change_rate": -0.1667,
        "current_observed_days": 30,
        "previous_observed_days": 30,
    }


def test_number_formatters_are_stable() -> None:
    assert format_money(1234.5) == "¥1,234.50"
    assert format_money(-0.004) == "¥0.00"
    assert format_count(1234) == "1,234"
    assert format_rate(0.12345) == "+12.35%"
    assert format_rate(-0.2) == "-20.00%"
    assert format_money(None) == "数据不足"
    assert format_count(None) == "数据不足"
    assert format_rate(None) == "数据不足"


def test_tool_results_are_indexed_by_name_and_duplicates_are_rejected() -> None:
    result = metrics_execution(TaskKind.GMV_DIAGNOSIS, [])

    assert tool_results_by_name([result]) == {
        "calculate_metrics": result.result
    }
    with pytest.raises(
        AnswerEvidenceError,
        match="^duplicate tool result: calculate_metrics$",
    ):
        tool_results_by_name([result, result])


@pytest.mark.parametrize(
    ("task", "row", "expected_fragments"),
    [
        (
            TaskKind.GMV_DIAGNOSIS,
            {
                "current_gmv": 1234.5,
                "previous_gmv": 1000.0,
                "gmv_change_rate": 0.2345,
                "current_orders": 125,
                "previous_orders": 100,
                "current_aov": 9.876,
                "previous_aov": 10.0,
                "aov_change_rate": -0.0124,
            },
            ("¥1,234.50", "+23.45%", "125", "¥9.88"),
        ),
        (
            TaskKind.CONVERSION_DECLINE,
            {
                "current_visits": 900,
                "previous_visits": 1000,
                "current_orders": 90,
                "previous_orders": 120,
                "current_cvr": 0.1,
                "previous_cvr": 0.12,
                "cvr_change": -0.02,
                "cvr_change_rate": -0.1667,
                "current_observed_days": 30,
                "previous_observed_days": 30,
            },
            ("900", "1,000", "10.00%", "12.00%", "-2.00%", "30"),
        ),
        (
            TaskKind.PRODUCT_ANOMALY,
            product_row("P003", -0.25),
            ("P003", "¥999.75", "-25.00%", "12.50%"),
        ),
        (
            TaskKind.NEXT_WEEK_PRIORITY,
            {
                "product_id": "P004",
                "current_gmv": 800.0,
                "gmv_change_rate": -0.3,
                "current_visits": 500,
                "traffic_change_rate": -0.2,
                "current_cvr": 0.16,
                "cvr_change": -0.04,
            },
            ("P004", "¥800.00", "-30.00%", "500", "16.00%", "-4.00%"),
        ),
        (
            TaskKind.PRODUCTS_TO_WATCH,
            {
                "product_id": "P005",
                "evidence_value": 0.15,
                "gmv_change_rate": -0.1,
                "traffic_change_rate": -0.2,
                "cvr_change": -0.03,
                "current_refund_rate": 0.15,
            },
            ("P005", "+15.00%", "-10.00%", "-20.00%", "-3.00%"),
        ),
    ],
)
def test_five_answer_templates_only_render_tool_evidence(
    task: TaskKind,
    row: dict[str, object],
    expected_fragments: tuple[str, ...],
) -> None:
    answer = render_final_answer(parsed(task), [metrics_execution(task, [row])])

    assert SOURCE_LABEL in answer
    assert (
        "时间窗口：当前期 2026-04-01 至 2026-04-30；"
        "对照期 2026-03-02 至 2026-03-31。"
    ) in answer
    for fragment in expected_fragments:
        assert fragment in answer
    lowered = answer.lower()
    for forbidden in ("gold", "expected", "生成配置", "因为", "导致", "造成"):
        assert forbidden not in lowered


def test_product_name_only_comes_from_query_product_and_otherwise_uses_id() -> None:
    metrics = metrics_execution(
        TaskKind.PRODUCT_ANOMALY,
        [product_row("P003", -0.25)],
    )

    named = render_final_answer(
        parsed(TaskKind.PRODUCT_ANOMALY),
        [product_execution({"P003": "工具商品名"}), metrics],
    )
    unnamed = render_final_answer(parsed(TaskKind.PRODUCT_ANOMALY), [metrics])

    assert "工具商品名（P003）" in named
    assert "工具商品名" not in unnamed
    assert "P003" in unnamed


def test_product_answers_use_frozen_sort_and_top_five() -> None:
    rows = [
        product_row(product_id, value)
        for product_id, value in (
            ("P006", -0.1),
            ("P005", -0.4),
            ("P004", -0.4),
            ("P003", -0.2),
            ("P002", -0.3),
            ("P001", -0.5),
        )
    ]

    answer = render_final_answer(
        parsed(TaskKind.PRODUCT_ANOMALY),
        [metrics_execution(TaskKind.PRODUCT_ANOMALY, rows)],
    )

    assert "P006" not in answer
    product_ids = ("P001", "P004", "P005", "P002", "P003")
    positions = [answer.index(product_id) for product_id in product_ids]
    assert positions == sorted(positions)


def test_grouped_conversion_uses_frozen_sort_and_top_five() -> None:
    rows = [
        conversion_row(product_id, value)
        for product_id, value in (
            ("P006", -0.01),
            ("P005", -0.04),
            ("P004", -0.04),
            ("P003", -0.02),
            ("P002", -0.03),
            ("P001", -0.05),
        )
    ]

    answer = render_final_answer(
        parsed(TaskKind.CONVERSION_DECLINE, ("P001", "P002")),
        [metrics_execution(TaskKind.CONVERSION_DECLINE, rows)],
    )

    assert "P006" not in answer
    product_ids = ("P001", "P004", "P005", "P002", "P003")
    positions = [answer.index(product_id) for product_id in product_ids]
    assert positions == sorted(positions)


def test_ungrouped_conversion_only_renders_the_single_overall_row() -> None:
    answer = render_final_answer(
        parsed(TaskKind.CONVERSION_DECLINE),
        [
            metrics_execution(
                TaskKind.CONVERSION_DECLINE,
                [
                    {
                        key: value
                        for key, value in conversion_row("P001", -0.02).items()
                        if key != "product_id"
                    },
                    {
                        key: value
                        for key, value in conversion_row("P002", -0.03).items()
                        if key != "product_id"
                    },
                ],
            )
        ],
    )

    assert answer.count("当前转化率") == 1


def test_none_outside_selected_top_five_does_not_limit_answer() -> None:
    rows = [
        conversion_row(f"P{index:03d}", -0.01 * index)
        for index in range(1, 6)
    ]
    rows.insert(0, conversion_row("P999", None, current_visits=None))

    answer = render_final_answer(
        parsed(TaskKind.CONVERSION_DECLINE, ("P001",)),
        [metrics_execution(TaskKind.CONVERSION_DECLINE, rows)],
    )

    assert "P999" not in answer
    assert "结论：" in answer
    assert "限制：" not in answer


def test_none_inside_selected_top_five_uses_limitation_before_any_claim() -> None:
    rows = [
        conversion_row("P001", -0.05, current_visits=None),
        conversion_row("P002", -0.04),
        conversion_row("P003", -0.03),
        conversion_row("P004", -0.02),
        conversion_row("P005", -0.01),
        conversion_row("P006", 0.0),
    ]

    answer = render_final_answer(
        parsed(TaskKind.CONVERSION_DECLINE, ("P001",)),
        [metrics_execution(TaskKind.CONVERSION_DECLINE, rows)],
    )

    assert "P001" in answer
    assert "限制：" in answer
    assert "结论：" not in answer


@pytest.mark.parametrize(
    ("task", "row"),
    [
        (
            TaskKind.GMV_DIAGNOSIS,
            {
                "current_gmv": None,
                "previous_gmv": 1000.0,
                "gmv_change_rate": None,
                "current_orders": 100,
                "previous_orders": 100,
                "current_aov": None,
                "previous_aov": 10.0,
                "aov_change_rate": None,
            },
        ),
        (
            TaskKind.CONVERSION_DECLINE,
            conversion_row("P001", -0.02, current_visits=None),
        ),
        (
            TaskKind.PRODUCT_ANOMALY,
            {**product_row("P003", -0.25), "current_gmv": None},
        ),
        (
            TaskKind.NEXT_WEEK_PRIORITY,
            {
                "product_id": "P004",
                "current_gmv": None,
                "gmv_change_rate": -0.3,
                "current_visits": 500,
                "traffic_change_rate": -0.2,
                "current_cvr": 0.16,
                "cvr_change": -0.04,
            },
        ),
        (
            TaskKind.PRODUCTS_TO_WATCH,
            {
                "product_id": "P005",
                "evidence_value": None,
                "gmv_change_rate": -0.1,
                "traffic_change_rate": -0.2,
                "cvr_change": -0.03,
                "current_refund_rate": 0.15,
            },
        ),
    ],
)
def test_five_limited_answer_templates_render_parsed_windows(
    task: TaskKind,
    row: dict[str, object],
) -> None:
    answer = render_final_answer(parsed(task), [metrics_execution(task, [row])])

    assert answer.startswith(f"数据来源：{SOURCE_LABEL}。\n时间窗口：")
    assert (
        "当前期 2026-04-01 至 2026-04-30；"
        "对照期 2026-03-02 至 2026-03-31。"
    ) in answer
    assert "限制：" in answer


def test_changing_only_parsed_windows_changes_answer() -> None:
    metrics = metrics_execution(
        TaskKind.GMV_DIAGNOSIS,
        [
            {
                "current_gmv": 1234.5,
                "previous_gmv": 1000.0,
                "gmv_change_rate": 0.2345,
                "current_orders": 125,
                "previous_orders": 100,
                "current_aov": 9.876,
                "previous_aov": 10.0,
                "aov_change_rate": -0.0124,
            }
        ],
    )
    changed_windows = DateWindows(
        start_date=date(2026, 4, 8),
        end_date=date(2026, 4, 14),
        comparison_start_date=date(2026, 4, 1),
        comparison_end_date=date(2026, 4, 7),
    )

    original = render_final_answer(parsed(TaskKind.GMV_DIAGNOSIS), [metrics])
    changed = render_final_answer(
        parsed(TaskKind.GMV_DIAGNOSIS, windows=changed_windows),
        [metrics],
    )

    assert original != changed
    assert (
        "时间窗口：当前期 2026-04-08 至 2026-04-14；"
        "对照期 2026-04-01 至 2026-04-07。"
    ) in changed
    assert "2026-03-02" not in changed


@pytest.mark.parametrize("mode", ["none", "empty", "warnings"])
def test_incomplete_tool_results_report_insufficient_data(mode: str) -> None:
    rows: list[dict[str, object]] = [
        {
            "current_gmv": None,
            "previous_gmv": 1000.0,
            "gmv_change_rate": None,
            "current_orders": 0,
            "previous_orders": 100,
            "current_aov": None,
            "previous_aov": 10.0,
            "aov_change_rate": None,
        }
    ]
    warnings: list[str] = []
    if mode == "empty":
        rows = []
    if mode == "warnings":
        warnings = ["incomplete source rows"]

    answer = render_final_answer(
        parsed(TaskKind.GMV_DIAGNOSIS),
        [metrics_execution(TaskKind.GMV_DIAGNOSIS, rows, warnings=warnings)],
    )

    assert "数据不足" in answer
    assert "确定" not in answer


def test_changing_tool_result_changes_answer_and_policy_uses_renderer() -> None:
    from app.baselines.v1 import BaselinePolicyV1, PolicyContext

    context = PolicyContext(
        dataset_id=DATASET_ID,
        dataset_version="v1",
        as_of_date=date(2026, 4, 30),
    )
    policy = BaselinePolicyV1(context)
    user_input = "诊断最近30天 GMV 变化"
    first = policy(
        AdapterRequest(
            user_input=user_input,
            tools=[],
            prior_tool_executions=[],
        )
    )
    sales = execution(
        QuerySalesResult(
            result_id="result_0003",
            tool_name="query_sales",
            dataset_id=DATASET_ID,
            source_label=SOURCE_LABEL,
            columns=SALES_COLUMNS,
            rows=[],
            row_count=0,
        ),
        3,
    ).model_copy(
        update={
            "call_id": first.tool_calls[0].call_id,
            "arguments": first.tool_calls[0].arguments,
        }
    )
    second = policy(
        AdapterRequest(
            user_input=user_input,
            tools=[],
            prior_tool_executions=[sales],
        )
    )

    def final_answer(current_gmv: float) -> str:
        metrics = metrics_execution(
            TaskKind.GMV_DIAGNOSIS,
            [
                {
                    "current_gmv": current_gmv,
                    "previous_gmv": 1000.0,
                    "gmv_change_rate": 0.25,
                    "current_orders": 125,
                    "previous_orders": 100,
                    "current_aov": 10.0,
                    "previous_aov": 10.0,
                    "aov_change_rate": 0.0,
                }
            ],
        ).model_copy(
            update={
                "call_id": second.tool_calls[0].call_id,
                "arguments": second.tool_calls[0].arguments,
            }
        )
        return policy(
            AdapterRequest(
                user_input=user_input,
                tools=[],
                prior_tool_executions=[sales, metrics],
            )
        ).final_answer or ""

    first_answer = final_answer(1234.5)
    changed_answer = final_answer(9876.5)

    assert "¥1,234.50" in first_answer
    assert "¥9,876.50" in changed_answer
    assert first_answer != changed_answer
