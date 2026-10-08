from datetime import date

import pytest
from pydantic import ValidationError

from app.baselines.v1 import (
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
from app.tools.metrics import METRIC_REQUIREMENTS
from app.tools.registry import build_default_registry
from app.tools.schemas import MetricName


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
