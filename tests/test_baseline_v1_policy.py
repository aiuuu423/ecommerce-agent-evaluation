from datetime import date

import pytest
from pydantic import ValidationError

from app.baselines.v1 import (
    BaselinePolicyProtocolError,
    BaselinePolicyV1,
    DateWindows,
    ExecutionPlan,
    ParsedRequest,
    PolicyContext,
    TaskKind,
    ToolStep,
    canonical_policy_bytes,
    policy_snapshot,
)
from app.baselines.v1.config import (
    DEFAULT_TOP_K,
    DEFAULT_WINDOW_DAYS,
    METRICS,
    ROUTING_KEYWORDS,
    TOOL_PATHS,
)
from app.llm import AdapterRequest
from app.tools.metrics import METRIC_REQUIREMENTS
from app.tools.registry import build_default_registry
from app.tools.schemas import (
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    TRAFFIC_COLUMNS,
    CalculateMetricsResult,
    MetricName,
    PriorToolExecution,
    QueryProductResult,
    QuerySalesResult,
    QueryTrafficResult,
)

CONTEXT = PolicyContext(
    dataset_id="e1e81533c25e03e5",
    dataset_version="v1",
    as_of_date=date(2026, 4, 30),
)
TOOLS = [{"type": "function", "function": {"name": "query_marketing"}}]


def request(
    user_input: str,
    prior: list[PriorToolExecution] | None = None,
) -> AdapterRequest:
    return AdapterRequest(
        user_input=user_input,
        tools=TOOLS,
        prior_tool_executions=prior or [],
    )


def empty_execution(action: object, result_number: int) -> PriorToolExecution:
    call = action.tool_calls[0]  # type: ignore[attr-defined]
    common = {
        "result_id": f"result_{result_number:04d}",
        "tool_name": call.name,
        "dataset_id": CONTEXT.dataset_id,
        "source_label": "Synthetic E-commerce Data",
        "rows": [],
        "row_count": 0,
    }
    if call.name == "query_product":
        result = QueryProductResult(columns=PRODUCT_COLUMNS, **common)
    elif call.name == "query_sales":
        result = QuerySalesResult(columns=SALES_COLUMNS, **common)
    elif call.name == "query_traffic":
        result = QueryTrafficResult(columns=TRAFFIC_COLUMNS, **common)
    else:
        result = CalculateMetricsResult(
            columns=[*call.arguments["group_by"], *call.arguments["metrics"]],
            **common,
        )
    return PriorToolExecution(
        call_id=call.call_id,
        arguments=call.arguments,
        result=result,
    )


def test_policy_snapshot_freezes_v1_contract_without_evaluation_leakage() -> None:
    snapshot = policy_snapshot()

    assert snapshot["policy_name"] == "deterministic-baseline-v1"
    assert snapshot["policy_version"] == "1.0.0"
    assert snapshot["default_window_days"] == DEFAULT_WINDOW_DAYS == 30
    assert snapshot["default_top_k"] == DEFAULT_TOP_K == 5
    assert snapshot["tool_paths"]["gmv_diagnosis"] == [
        "query_sales",
        "calculate_metrics",
    ]
    assert set(snapshot["tool_paths"]) == {
        task.value for task in TaskKind if task is not TaskKind.UNSUPPORTED
    }
    assert all(
        "query_marketing" not in tool_path
        for tool_path in snapshot["tool_paths"].values()
    )
    assert all(
        MetricName(metric).value == metric
        for metrics in snapshot["metrics"].values()
        for metric in metrics
    )
    assert all(
        len(keyword_groups) == 2
        for keyword_groups in snapshot["routing_keywords"].values()
    )
    assert all(
        field == "product_id:asc"
        or MetricName(field.removesuffix(":asc").removesuffix(":desc"))
        for fields in snapshot["stable_sort_fields"].values()
        for field in fields
    )

    serialized = canonical_policy_bytes(snapshot).decode("utf-8").lower()
    for forbidden in (
        "case_",
        "gold",
        "expected",
        "success_criteria",
    ):
        assert forbidden not in serialized


def test_policy_snapshot_is_canonical_and_returns_a_deep_copy() -> None:
    first = policy_snapshot()
    expected = canonical_policy_bytes(first)
    first["tool_paths"]["gmv_diagnosis"].append("query_marketing")

    second = policy_snapshot()
    assert "query_marketing" not in second["tool_paths"]["gmv_diagnosis"]
    assert canonical_policy_bytes(second) == expected
    assert canonical_policy_bytes(second).endswith(b"\n")


def test_public_policy_models_are_strict_and_serializable() -> None:
    windows = DateWindows(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        comparison_start_date=date(2026, 3, 2),
        comparison_end_date=date(2026, 3, 31),
    )
    context = PolicyContext(
        dataset_id="e1e81533c25e03e5",
        dataset_version="v1",
        as_of_date=date(2026, 4, 30),
    )
    parsed = ParsedRequest(
        task=TaskKind.GMV_DIAGNOSIS,
        product_ids=("P003",),
        windows=windows,
    )
    plan = ExecutionPlan(
        task=parsed.task,
        steps=(
            ToolStep(tool_name="query_sales"),
            ToolStep(
                tool_name="calculate_metrics",
                metrics=(MetricName.CURRENT_GMV,),
            ),
        ),
    )

    assert context.dataset_version == "v1"
    assert parsed.model_dump(mode="json")["windows"]["end_date"] == "2026-04-30"
    assert plan.steps[1].metrics == (MetricName.CURRENT_GMV,)


def test_policy_config_containers_are_deeply_immutable() -> None:
    with pytest.raises(TypeError):
        TOOL_PATHS[TaskKind.GMV_DIAGNOSIS] = ("query_sales",)  # type: ignore[index]
    with pytest.raises(TypeError):
        TOOL_PATHS[TaskKind.GMV_DIAGNOSIS][0] = "query_traffic"  # type: ignore[index]
    with pytest.raises(TypeError):
        ROUTING_KEYWORDS[TaskKind.GMV_DIAGNOSIS][0][0] = "revenue"  # type: ignore[index]


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            PolicyContext,
            {
                "dataset_id": "e1e81533c25e03e5",
                "dataset_version": 1,
                "as_of_date": "2026-04-30",
            },
        ),
        (
            DateWindows,
            {
                "start_date": "2026-04-01",
                "end_date": "2026-04-30",
                "comparison_start_date": "2026-03-02",
                "comparison_end_date": "2026-03-31",
            },
        ),
    ],
)
def test_policy_models_reject_implicit_type_conversion(
    model: type[PolicyContext] | type[DateWindows],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_policy_model_containers_are_deeply_immutable() -> None:
    windows = DateWindows(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        comparison_start_date=date(2026, 3, 2),
        comparison_end_date=date(2026, 3, 31),
    )
    parsed = ParsedRequest(
        task=TaskKind.GMV_DIAGNOSIS,
        product_ids=("P003",),
        windows=windows,
    )

    assert parsed.product_ids == ("P003",)
    with pytest.raises(TypeError):
        parsed.product_ids[0] = "P004"  # type: ignore[index]


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        (TaskKind.UNSUPPORTED, None),
        (TaskKind.UNSUPPORTED, ""),
        (TaskKind.GMV_DIAGNOSIS, "not supported"),
    ],
)
def test_parsed_request_rejects_inconsistent_unsupported_reason(
    task: TaskKind,
    reason: str | None,
) -> None:
    windows = DateWindows(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        comparison_start_date=date(2026, 3, 2),
        comparison_end_date=date(2026, 3, 31),
    )
    with pytest.raises(ValidationError):
        ParsedRequest(
            task=task,
            product_ids=(),
            windows=windows,
            unsupported_reason=reason,
        )


@pytest.mark.parametrize(
    "step",
    [
        {"tool_name": "calculate_metrics"},
        {"tool_name": "query_sales", "metrics": (MetricName.CURRENT_GMV,)},
        {"tool_name": "query_sales", "group_by": ("product_id",)},
        {
            "tool_name": "calculate_metrics",
            "metrics": (MetricName.CURRENT_GMV, MetricName.CURRENT_GMV),
        },
        {
            "tool_name": "calculate_metrics",
            "metrics": (MetricName.CURRENT_GMV,),
            "group_by": ("product_id", "product_id"),
        },
    ],
)
def test_tool_step_rejects_invalid_metric_and_grouping_states(
    step: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ToolStep.model_validate(step)


def test_execution_plan_rejects_empty_or_unsupported_plans() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan(task=TaskKind.GMV_DIAGNOSIS, steps=())
    with pytest.raises(ValidationError):
        ExecutionPlan(
            task=TaskKind.UNSUPPORTED,
            steps=(ToolStep(tool_name="query_sales"),),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"outer": {1: "value"}},
        {"outer": {"nested": object()}},
        {"outer": {"nested": float("nan")}},
        {"outer": {"nested": float("inf")}},
    ],
)
def test_canonical_policy_bytes_recursively_rejects_non_json_values(
    payload: dict[object, object],
) -> None:
    with pytest.raises(ValueError):
        canonical_policy_bytes(payload)  # type: ignore[arg-type]


def test_policy_tools_exist_and_metric_dependencies_are_in_each_tool_path() -> None:
    registered_names = set(build_default_registry().names())

    for task, tool_path in TOOL_PATHS.items():
        assert set(tool_path) <= registered_names
        required_tools = {
            tool_name
            for metric in METRICS[task]
            for tool_name in METRIC_REQUIREMENTS[metric.value]
        }
        assert required_tools <= set(tool_path)


@pytest.mark.parametrize(
    ("user_input", "first_tool"),
    [
        ("诊断最近30天 GMV 变化", "query_sales"),
        ("分析最近30天 P003 商品异常", "query_product"),
        ("分析最近30天商品异常", "query_sales"),
        ("分析最近30天转化下降", "query_traffic"),
        ("列出最近30天需要关注的商品", "query_sales"),
        ("给出最近30天下周重点动作", "query_sales"),
    ],
)
def test_policy_selects_the_frozen_first_action(
    user_input: str,
    first_tool: str,
) -> None:
    action = BaselinePolicyV1(CONTEXT)(request(user_input))

    assert action.tool_calls[0].name == first_tool


@pytest.mark.parametrize(
    ("user_input", "expected_path"),
    [
        ("诊断最近30天 GMV 变化", ("query_sales", "calculate_metrics")),
        (
            "分析最近30天 P003 商品异常",
            ("query_product", "query_sales", "query_traffic", "calculate_metrics"),
        ),
        (
            "分析最近30天商品异常",
            ("query_sales", "query_traffic", "calculate_metrics"),
        ),
        (
            "分析最近30天转化下降",
            ("query_traffic", "query_sales", "calculate_metrics"),
        ),
        (
            "列出最近30天需要关注的商品",
            ("query_sales", "query_traffic", "calculate_metrics"),
        ),
        (
            "给出最近30天下周重点动作",
            ("query_sales", "query_traffic", "calculate_metrics"),
        ),
    ],
)
def test_policy_runs_complete_fixed_paths_with_stable_unique_call_ids(
    user_input: str,
    expected_path: tuple[str, ...],
) -> None:
    policy = BaselinePolicyV1(CONTEXT)
    prior: list[PriorToolExecution] = []
    names: list[str] = []
    call_ids: list[str] = []

    for index, expected_name in enumerate(expected_path, start=1):
        action = policy(request(user_input, prior))
        repeated = policy(request(user_input, prior))
        assert action == repeated
        call = action.tool_calls[0]
        assert call.name == expected_name
        names.append(call.name)
        call_ids.append(call.call_id)
        prior.append(empty_execution(action, index))

    final = policy(request(user_input, prior))
    assert final.tool_calls == []
    assert final.final_answer
    assert names == list(expected_path)
    assert len(call_ids) == len(set(call_ids))


def test_policy_builds_exact_window_query_arguments() -> None:
    expected_window = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P003"],
    }

    sales = BaselinePolicyV1(CONTEXT)(request("诊断最近30天 P003 GMV 变化"))
    traffic = BaselinePolicyV1(CONTEXT)(request("分析最近30天 P003 转化下降"))

    assert sales.tool_calls[0].arguments == {
        **expected_window,
        "include_refunds": True,
    }
    assert traffic.tool_calls[0].arguments == {
        **expected_window,
        "include_missing": True,
    }


@pytest.mark.parametrize(
    ("user_input", "task", "group_by"),
    [
        ("诊断 GMV 变化", TaskKind.GMV_DIAGNOSIS, []),
        ("分析 P003 商品异常", TaskKind.PRODUCT_ANOMALY, ["product_id"]),
        ("分析转化下降", TaskKind.CONVERSION_DECLINE, []),
        ("分析 P003 转化下降", TaskKind.CONVERSION_DECLINE, ["product_id"]),
        ("列出需要关注的商品", TaskKind.PRODUCTS_TO_WATCH, ["product_id"]),
        ("给出下周重点动作", TaskKind.NEXT_WEEK_PRIORITY, ["product_id"]),
    ],
)
def test_policy_uses_frozen_metrics_and_grouping(
    user_input: str,
    task: TaskKind,
    group_by: list[str],
) -> None:
    policy = BaselinePolicyV1(CONTEXT)
    prior: list[PriorToolExecution] = []

    while True:
        action = policy(request(user_input, prior))
        call = action.tool_calls[0]
        if call.name == "calculate_metrics":
            assert call.arguments == {
                "metrics": [metric.value for metric in METRICS[task]],
                "group_by": group_by,
            }
            return
        prior.append(empty_execution(action, len(prior) + 1))


def test_policy_returns_unsupported_answer_without_tools() -> None:
    action = BaselinePolicyV1(CONTEXT)(request("请介绍你的能力"))

    assert action.tool_calls == []
    assert action.final_answer == "当前仅支持商品经营、转化、关注清单、下周重点和 GMV 诊断。"


def test_policy_rejects_unexpected_prior_tool_sequence() -> None:
    wrong_first = BaselinePolicyV1(CONTEXT)(request("分析转化下降"))
    wrong_first = wrong_first.model_copy(
        update={
            "tool_calls": [
                wrong_first.tool_calls[0].model_copy(
                    update={"name": "query_sales"}
                )
            ]
        }
    )
    prior = [empty_execution(wrong_first, 1)]

    with pytest.raises(
        BaselinePolicyProtocolError,
        match="^unexpected prior tool sequence$",
    ):
        BaselinePolicyV1(CONTEXT)(request("分析转化下降", prior))


def test_each_case_uses_an_independent_policy_instance() -> None:
    first = BaselinePolicyV1(CONTEXT)
    second = BaselinePolicyV1(CONTEXT)
    first_action = first(request("诊断 GMV 变化"))
    first_prior = [empty_execution(first_action, 1)]

    assert first(request("诊断 GMV 变化", first_prior)).tool_calls[0].name == (
        "calculate_metrics"
    )
    second_action = second(request("分析 P003 商品异常"))
    assert second_action.tool_calls[0].name == "query_product"
    assert second_action.tool_calls[0].call_id == first_action.tool_calls[0].call_id
