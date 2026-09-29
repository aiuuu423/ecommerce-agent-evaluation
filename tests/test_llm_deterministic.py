from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import app.llm as llm
from app.llm import (
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
    DeterministicAdapter,
    ToolCall,
    Usage,
)


def sample_request() -> AdapterRequest:
    return AdapterRequest(
        user_input="查看 P001",
        tools=[],
        prior_tool_executions=[],
    )


def test_deterministic_adapter_calls_policy_and_has_no_usage() -> None:
    seen: list[AdapterRequest] = []
    action = AssistantAction(
        tool_calls=[
            ToolCall(
                call_id="call_0001",
                name="query_product",
                arguments={"product_ids": ["P001"]},
            )
        ]
    )

    def policy(request: AdapterRequest) -> AssistantAction:
        seen.append(request)
        return action

    adapter = DeterministicAdapter(policy)
    request = sample_request()
    adapter.start_run()
    first = adapter.complete(request)
    second = adapter.complete(request)

    assert first == AdapterResponse(action=action, raw_response=None, usage=None)
    assert first == second
    assert first.usage is None
    assert first.raw_response is None
    assert seen == [request, request]
    assert adapter.adapter_name == "deterministic"


def test_deterministic_plan_is_not_part_of_public_api() -> None:
    assert not hasattr(llm, "DeterministicPlan")


def test_invalid_policy_action_is_rejected_by_response_schema() -> None:
    invalid_action = AssistantAction.model_construct(
        tool_calls=[],
        final_answer=None,
    )
    adapter = DeterministicAdapter(lambda _request: invalid_action)

    with pytest.raises(ValidationError, match="exactly one"):
        adapter.complete(sample_request())


def test_action_requires_exactly_one_mode() -> None:
    call = ToolCall(call_id="c1", name="query_product", arguments={})
    with pytest.raises(ValidationError):
        AssistantAction()
    with pytest.raises(ValidationError):
        AssistantAction(tool_calls=[call], final_answer="不能同时出现")
    with pytest.raises(ValidationError, match="one tool call"):
        AssistantAction(tool_calls=[call, call.model_copy(update={"call_id": "c2"})])


@pytest.mark.parametrize(
    "call_id",
    [
        "api_key=sk-secret",
        "path=/tmp/private/key",
        "sql=DROP TABLE sales",
        "x" * 129,
        " call_0001 ",
    ],
)
def test_tool_call_rejects_unsafe_call_id(call_id: str) -> None:
    with pytest.raises(ValidationError):
        ToolCall(call_id=call_id, name="query_product", arguments={})


def test_tool_call_accepts_safe_ascii_identifier_boundaries() -> None:
    assert ToolCall(call_id="A-z_0-9", name="query_product", arguments={}).call_id == "A-z_0-9"
    assert ToolCall(call_id="x" * 128, name="query_product", arguments={}).call_id == "x" * 128


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


def test_deterministic_adapter_is_offline_and_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("deterministic adapter must not perform external I/O")

    monkeypatch.setattr("os.getenv", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)

    action = AssistantAction(final_answer="完成。")
    adapter = DeterministicAdapter(lambda _request: action)
    request = sample_request()

    adapter.start_run()
    assert adapter.complete(request) == adapter.complete(request)
