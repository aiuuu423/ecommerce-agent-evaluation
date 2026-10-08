from datetime import date

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
from app.baselines.v1.config import DEFAULT_TOP_K, DEFAULT_WINDOW_DAYS
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
        product_ids=["P003"],
        windows=windows,
    )
    plan = ExecutionPlan(
        task=parsed.task,
        steps=(
            ToolStep(tool_name="query_sales"),
            ToolStep(
                tool_name="calculate_metrics",
                metrics=("current_gmv",),
            ),
        ),
    )

    assert context.dataset_version == "v1"
    assert parsed.model_dump(mode="json")["windows"]["end_date"] == "2026-04-30"
    assert plan.steps[1].metrics == (MetricName.CURRENT_GMV,)
