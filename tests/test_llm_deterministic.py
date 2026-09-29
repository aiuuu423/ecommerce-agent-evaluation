from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.llm import (
    AdapterRequest,
    AssistantAction,
    DeterministicAdapter,
    DeterministicPlan,
    ToolCall,
    Usage,
)
from app.tools.schemas import PriorToolExecution, QueryProductResult


def request_with(
    *executions: PriorToolExecution,
    user_input: str = "查看 P001",
) -> AdapterRequest:
    return AdapterRequest(
        user_input=user_input,
        tools=[],
        prior_tool_executions=list(executions),
    )


def product_execution(
    call: ToolCall,
    *,
    result_id: str = "result_0001",
) -> PriorToolExecution:
    result = QueryProductResult.model_validate(
        {
            "result_id": result_id,
            "tool_name": "query_product",
            "dataset_id": "e1e81533c25e03e5",
            "source_label": "Synthetic E-commerce Data",
            "columns": [
                "product_id",
                "product_name",
                "category",
                "price",
                "cost",
                "launch_date",
            ],
            "rows": [
                {
                    "product_id": "P001",
                    "product_name": "示例商品",
                    "category": "示例",
                    "price": 100.0,
                    "cost": 60.0,
                    "launch_date": "2026-01-01",
                }
            ],
            "row_count": 1,
        }
    )
    return PriorToolExecution(
        call_id=call.call_id,
        arguments=call.arguments,
        result=result,
    )


def test_deterministic_adapter_calls_policy_and_has_no_usage() -> None:
    seen: list[AdapterRequest] = []

    def policy(request: AdapterRequest) -> AssistantAction:
        seen.append(request)
        return AssistantAction(
            tool_calls=[
                ToolCall(
                    call_id="call_0001",
                    name="query_product",
                    arguments={"product_ids": ["P001"]},
                )
            ]
        )

    adapter = DeterministicAdapter(policy)
    request = request_with()
    first = adapter.complete(request)
    second = adapter.complete(request)

    assert first == second
    assert first.usage is None
    assert first.raw_response is None
    assert seen == [request, request]
    assert adapter.adapter_name == "deterministic"


def test_action_requires_exactly_one_mode_and_one_tool_call() -> None:
    call = ToolCall(call_id="c1", name="query_product", arguments={})
    with pytest.raises(ValidationError):
        AssistantAction()
    with pytest.raises(ValidationError):
        AssistantAction(tool_calls=[call], final_answer="不能同时出现")
    with pytest.raises(ValidationError, match="one tool call"):
        AssistantAction(tool_calls=[call, call.model_copy(update={"call_id": "c2"})])


@pytest.mark.parametrize("bad_value", [True, 1.5, "1", -1])
def test_usage_requires_non_negative_strict_integers(bad_value: Any) -> None:
    with pytest.raises(ValidationError):
        Usage(
            prompt_tokens=bad_value,
            completion_tokens=0,
            total_tokens=0,
        )


def test_usage_rejects_inconsistent_total() -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        Usage(prompt_tokens=10, completion_tokens=4, total_tokens=15)


def test_explicit_plan_advances_only_after_real_prior_execution() -> None:
    first_call = ToolCall(
        call_id="call_0001",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    second_call = ToolCall(
        call_id="call_0002",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    adapter = DeterministicAdapter(
        DeterministicPlan(
            tool_calls=[first_call, second_call],
            final_answer="已根据工具结果完成分析。",
        )
    )
    adapter.start_run()

    assert adapter.complete(request_with()).action.tool_calls == [first_call]
    assert adapter.complete(request_with()).action.tool_calls == [first_call]

    first_execution = product_execution(first_call)
    assert adapter.complete(request_with(first_execution)).action.tool_calls == [second_call]

    second_execution = product_execution(second_call, result_id="result_0002")
    response = adapter.complete(request_with(first_execution, second_execution))
    assert response.action.tool_calls == []
    assert response.action.final_answer == (
        "已根据工具结果完成分析。\n\n"
        "Evidence: [call_id=call_0001 result_id=result_0001]; "
        "[call_id=call_0002 result_id=result_0002]"
    )
    assert response.raw_response is None
    assert response.usage is None


def test_plan_rejects_fabricated_or_mismatched_prior_execution() -> None:
    call = ToolCall(
        call_id="call_0001",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    adapter = DeterministicAdapter(
        DeterministicPlan(tool_calls=[call], final_answer="完成。")
    )
    mismatched = product_execution(call).model_copy(
        update={"arguments": {"product_ids": ["P002"]}}
    )

    with pytest.raises(ValueError, match="arguments"):
        adapter.complete(request_with(mismatched))


def test_start_run_resets_run_binding_and_same_input_is_deterministic() -> None:
    call = ToolCall(
        call_id="call_0001",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    adapter = DeterministicAdapter(
        DeterministicPlan(tool_calls=[call], final_answer="完成。")
    )
    adapter.start_run()
    first = adapter.complete(request_with())
    with pytest.raises(ValueError, match="user_input"):
        adapter.complete(request_with(user_input="另一个请求"))

    adapter.start_run()
    second = adapter.complete(request_with())
    assert second == first


def test_deterministic_plan_uses_no_key_file_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall(
        call_id="call_0001",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("deterministic adapter must not perform external I/O")

    monkeypatch.setattr("os.getenv", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)

    adapter = DeterministicAdapter(
        DeterministicPlan(tool_calls=[call], final_answer="完成。")
    )
    assert adapter.complete(request_with()).action.tool_calls == [call]
