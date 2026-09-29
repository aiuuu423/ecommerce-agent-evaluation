import itertools
import json
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic import ValidationError

from app.agents import AgentRunner, RunRequest
from app.agents.schemas import TraceEvent, TraceEventType
from app.data.database import Catalog, open_dataset
from app.data.generator import build_snapshot
from app.llm import (
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
    DeterministicAdapter,
    ToolCall,
)
from app.llm.openai_compatible import OpenAICompatibleAdapter
from app.tools import ToolDefinition, ToolRegistry, build_default_registry
from app.tools.schemas import (
    QueryProductInput,
    QueryProductResult,
    QuerySalesInput,
    QuerySalesResult,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Catalog]:
    dataset_dir = tmp_path_factory.mktemp("agent-runner") / "v1"
    build_snapshot(CONFIG, dataset_dir)
    with open_dataset(dataset_dir) as opened:
        yield opened


def call(call_id: str, name: str, arguments: dict[str, Any]) -> AssistantAction:
    return AssistantAction(tool_calls=[ToolCall(call_id=call_id, name=name, arguments=arguments)])


def runner(policy: Callable[[AdapterRequest], AssistantAction]) -> AgentRunner:
    return AgentRunner(
        registry=build_default_registry(),
        adapter=DeterministicAdapter(policy),
    )


def valid_sales_arguments() -> dict[str, Any]:
    return {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
        "include_refunds": True,
    }


def normalized_sales_arguments() -> dict[str, Any]:
    return {**valid_sales_arguments(), "product_ids": ()}


def sample_product_result(result_id: str) -> QueryProductResult:
    return QueryProductResult(
        result_id=result_id,
        tool_name="query_product",
        dataset_id="0123456789abcdef",
        source_label="Synthetic E-commerce Data",
        columns=[
            "product_id",
            "product_name",
            "category",
            "price",
            "cost",
            "launch_date",
        ],
        rows=[],
        row_count=0,
    )


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []

    def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert headers["Authorization"] == "Bearer secret"
        assert timeout_seconds == 30.0
        self.payloads.append(deepcopy(payload))
        return deepcopy(self.responses[len(self.payloads) - 1])


def tool_payload(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    usage: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[2],
        },
    }


def answer_payload(
    *,
    content: str,
    usage: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[2],
        },
    }


class CustomAdapter:
    adapter_name = "custom"

    def __init__(self, response: AdapterResponse) -> None:
        self.response = response

    def start_run(self) -> None:
        pass

    def complete(self, _request: AdapterRequest) -> AdapterResponse:
        return self.response


class StartRunFailingAdapter:
    adapter_name = "start-run-failing"

    def start_run(self) -> None:
        raise RuntimeError("secret=/tmp/private/key")

    def complete(self, _request: AdapterRequest) -> AdapterResponse:
        raise AssertionError("complete must not be called")


def registry_with_malformed_result() -> ToolRegistry:
    registry = ToolRegistry()

    def malformed(
        _arguments: QuerySalesInput,
        _context: object,
    ) -> QuerySalesResult:
        return QuerySalesResult.model_construct(
            result_id="result_0001",
            tool_name="query_sales",
            dataset_id="0123456789abcdef",
            source_label="Synthetic E-commerce Data",
            columns=[],
            rows=[],
            row_count=1,
            warnings=[],
        )

    registry.register(
        ToolDefinition(
            name="query_sales",
            description="test",
            input_model=QuerySalesInput,
            output_model=QuerySalesResult,
            handler=malformed,
        )
    )
    return registry


def registry_with_raising_handler(error: Exception) -> ToolRegistry:
    registry = ToolRegistry()

    def raising(_arguments: QuerySalesInput, _context: object) -> QuerySalesResult:
        raise error

    registry.register(
        ToolDefinition(
            name="query_sales",
            description="test",
            input_model=QuerySalesInput,
            output_model=QuerySalesResult,
            handler=raising,
        )
    )
    return registry


def registry_returning_fixed_result_id(
    result_id: str,
) -> tuple[ToolRegistry, Callable[[], int]]:
    registry = ToolRegistry()
    invocations = 0

    def fixed(
        _arguments: QueryProductInput,
        _context: object,
    ) -> QueryProductResult:
        nonlocal invocations
        invocations += 1
        return sample_product_result(result_id)

    registry.register(
        ToolDefinition(
            name="query_product",
            description="test",
            input_model=QueryProductInput,
            output_model=QueryProductResult,
            handler=fixed,
        )
    )
    return registry, lambda: invocations


def test_run_request_enforces_limits() -> None:
    assert RunRequest(user_input="x").max_steps == 8
    for payload in (
        {"user_input": ""},
        {"user_input": "x", "max_steps": 0},
        {"user_input": "x", "max_steps": 33},
        {"user_input": "x", "max_steps": 1.5},
    ):
        with pytest.raises(ValidationError):
            RunRequest.model_validate(payload, strict=True)


def test_runner_executes_tools_and_records_public_trace(catalog: Catalog) -> None:
    def policy(request: AdapterRequest) -> AssistantAction:
        if not request.prior_tool_executions:
            return call("call_0001", "query_product", {"product_ids": ["P001"]})
        return AssistantAction(final_answer="P001 产品信息已取得。")

    result = runner(policy).run(RunRequest(user_input="查看 P001"), catalog)

    assert result.status == "completed"
    assert result.final_answer == "P001 产品信息已取得。"
    assert [event.sequence for event in result.decision_trace] == list(
        range(1, len(result.decision_trace) + 1)
    )
    assert [event.event_type for event in result.decision_trace] == [
        "adapter_request",
        "adapter_response",
        "tool_call",
        "tool_result",
        "adapter_request",
        "adapter_response",
        "final_answer",
    ]
    assert result.decision_trace[0].payload == {
        "user_input": "查看 P001",
        "tool_names": (
            "calculate_metrics",
            "query_marketing",
            "query_product",
            "query_sales",
            "query_traffic",
        ),
        "result_ids": (),
    }
    tool_result = result.decision_trace[3].payload
    assert set(tool_result) == {
        "result_id",
        "tool_name",
        "dataset_id",
        "columns",
        "row_count",
        "warnings",
    }
    assert result.decision_trace[2].payload == {
        "call_id": "call_0001",
        "tool_name": "query_product",
        "arguments": {"product_ids": ("P001",)},
    }
    serialized = result.model_dump_json()
    for forbidden in ("thought", "sql", "/tmp/", "api_key", "gold"):
        assert forbidden not in serialized.lower()


def test_trace_is_deeply_immutable_and_json_serializable(catalog: Catalog) -> None:
    result = runner(
        lambda _: call("call_0001", "query_product", {"product_ids": ["P001"]})
    ).run(RunRequest(user_input="x", max_steps=1), catalog)

    assert isinstance(result.decision_trace, tuple)
    assert isinstance(result.decision_trace[0].payload["tool_names"], tuple)
    with pytest.raises(ValidationError):
        result.decision_trace[0].sequence = 99
    with pytest.raises(TypeError):
        result.decision_trace[0].payload["injected"] = "value"
    with pytest.raises(TypeError):
        result.decision_trace[0].payload["tool_names"][0] = "injected"

    assert json.loads(result.model_dump_json())["decision_trace"][0]["sequence"] == 1

    nested_event = TraceEvent(
        sequence=1,
        event_type=TraceEventType.TOOL_RESULT,
        payload={"nested": {"items": [{"value": 1}]}},
    )
    nested = nested_event.payload["nested"]
    assert not isinstance(nested, dict)
    with pytest.raises(TypeError):
        nested["items"][0]["value"] = 2
    assert json.loads(nested_event.model_dump_json())["payload"] == {
        "nested": {"items": [{"value": 1}]}
    }


def test_unvalidated_tool_data_and_sensitive_errors_do_not_enter_trace(
    catalog: Catalog,
) -> None:
    malicious_arguments = {
        "product_ids": ["P001"],
        "api_key": "sk-secret",
        "path": "/tmp/private/key",
        "sql": "DROP TABLE sales",
    }
    result = runner(lambda _: call("call_0001", "query_product", malicious_arguments)).run(
        RunRequest(user_input="x"),
        catalog,
    )

    assert result.status == "failed"
    assert [event.event_type for event in result.decision_trace[-2:]] == [
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-2].payload == {
        "call_id": "call_0001",
        "tool_name": "query_product",
    }
    assert result.decision_trace[-1].payload == {
        "code": "invalid_arguments",
        "summary": "Tool arguments failed validation.",
    }
    serialized = result.model_dump_json().lower()
    for forbidden in ("api_key", "sk-secret", "/tmp/private/key", "drop table"):
        assert forbidden not in serialized


def test_unknown_tool_trace_replaces_untrusted_name_and_arguments(
    catalog: Catalog,
) -> None:
    malicious_name = "unknown api_key=sk-secret /tmp/private/key DROP TABLE sales"
    malicious_arguments = {
        "api_key": "sk-secret",
        "path": "/tmp/private/key",
        "sql": "DROP TABLE sales",
    }
    result = runner(
        lambda _: call("call_unknown", malicious_name, malicious_arguments)
    ).run(RunRequest(user_input="x"), catalog)

    assert [event.event_type for event in result.decision_trace[-2:]] == [
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-2].payload == {
        "call_id": "call_unknown",
        "tool_name": "unknown",
    }
    assert result.decision_trace[-1].payload["code"] == "unknown_tool"
    serialized = result.model_dump_json().lower()
    for forbidden in ("api_key", "sk-secret", "/tmp/private/key", "drop table"):
        assert forbidden not in serialized


def test_tool_exception_trace_contains_only_safe_code_and_summary(catalog: Catalog) -> None:
    error = RuntimeError(
        "api_key=sk-secret path=/tmp/private/key sql=DROP TABLE sales"
    )
    result = AgentRunner(
        registry=registry_with_raising_handler(error),
        adapter=DeterministicAdapter(
            lambda _: call("call_0001", "query_sales", valid_sales_arguments())
        ),
    ).run(RunRequest(user_input="x"), catalog)

    assert [event.event_type for event in result.decision_trace[-2:]] == [
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-2].payload == {
        "call_id": "call_0001",
        "tool_name": "query_sales",
        "arguments": normalized_sales_arguments(),
    }
    assert result.decision_trace[-1].payload == {
        "code": "tool_execution_error",
        "summary": "Tool execution failed.",
    }
    serialized = result.model_dump_json().lower()
    for forbidden in ("api_key", "sk-secret", "/tmp/private/key", "drop table"):
        assert forbidden not in serialized


def test_runner_openai_adapter_fake_transport_full_chain(catalog: Catalog) -> None:
    transport = FakeTransport(
        [
            tool_payload(
                call_id="call_sales",
                name="query_sales",
                arguments=valid_sales_arguments(),
                usage=(10, 2, 12),
            ),
            tool_payload(
                call_id="call_metrics",
                name="calculate_metrics",
                arguments={
                    "metrics": [
                        "current_gmv",
                        "previous_gmv",
                        "gmv_change_rate",
                    ],
                    "group_by": [],
                },
                usage=(20, 3, 23),
            ),
            answer_payload(
                content="GMV 对比计算完成。",
                usage=(5, 4, 9),
            ),
        ]
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=transport,
    )
    with mock.patch.object(
        adapter,
        "start_run",
        wraps=adapter.start_run,
    ) as start_run:
        result = AgentRunner(
            registry=build_default_registry(),
            adapter=adapter,
        ).run(RunRequest(user_input="比较本期与上期 GMV"), catalog)

    start_run.assert_called_once_with()
    assert result.status == "completed"
    assert result.final_answer == "GMV 对比计算完成。"
    assert [item.call_id for item in result.prior_tool_executions] == [
        "call_sales",
        "call_metrics",
    ]
    assert [item.result.tool_name for item in result.prior_tool_executions] == [
        "query_sales",
        "calculate_metrics",
    ]
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "prompt_tokens": 35,
        "completion_tokens": 9,
        "total_tokens": 44,
    }
    assert [message["role"] for message in transport.payloads[2]["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert transport.payloads[1]["messages"][1]["tool_calls"][0]["id"] == "call_sales"
    assert transport.payloads[1]["messages"][2]["tool_call_id"] == "call_sales"
    assert transport.payloads[2]["messages"][3]["tool_calls"][0]["id"] == "call_metrics"
    assert transport.payloads[2]["messages"][4]["tool_call_id"] == "call_metrics"
    assert [
        json.loads(transport.payloads[2]["messages"][index]["content"])["result_id"]
        for index in (2, 4)
    ] == ["result_0001", "result_0002"]
    assert [
        event.payload.get("call_id")
        for event in result.decision_trace
        if event.event_type == "tool_call"
    ] == ["call_sales", "call_metrics"]


@pytest.mark.parametrize(
    ("policy", "error_code"),
    [
        (lambda _: call("c1", "unknown", {}), "unknown_tool"),
        (lambda _: call("c1", "query_sales", {}), "invalid_arguments"),
    ],
)
def test_runner_normalizes_tool_failures(
    policy: Callable[[AdapterRequest], AssistantAction],
    error_code: str,
    catalog: Catalog,
) -> None:
    result = runner(policy).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.final_answer is None
    assert [event.event_type for event in result.decision_trace] == [
        "adapter_request",
        "adapter_response",
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-1].event_type == "error"
    assert result.decision_trace[-1].payload["code"] == error_code


@pytest.mark.parametrize(
    ("registry", "error_code"),
    [
        (registry_with_malformed_result(), "invalid_tool_result"),
        (
            registry_with_raising_handler(KeyError("handler-miss")),
            "tool_execution_error",
        ),
    ],
)
def test_runner_distinguishes_invalid_result_from_execution_error(
    registry: ToolRegistry,
    error_code: str,
    catalog: Catalog,
) -> None:
    result = AgentRunner(
        registry=registry,
        adapter=DeterministicAdapter(lambda _: call("c1", "query_sales", valid_sales_arguments())),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert [event.event_type for event in result.decision_trace] == [
        "adapter_request",
        "adapter_response",
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-2].payload == {
        "call_id": "c1",
        "tool_name": "query_sales",
        "arguments": normalized_sales_arguments(),
    }
    assert result.decision_trace[-1].payload["code"] == error_code


def test_runner_rejects_duplicate_result_id_without_overwriting_execution(
    catalog: Catalog,
) -> None:
    counter = itertools.count(1)

    def policy(request: AdapterRequest) -> AssistantAction:
        if len(request.prior_tool_executions) < 2:
            return call(
                f"c{next(counter)}",
                "query_product",
                {"product_ids": ["P001"]},
            )
        return AssistantAction(final_answer="done")

    registry, invocation_count = registry_returning_fixed_result_id("result_0001")
    result = AgentRunner(
        registry=registry,
        adapter=DeterministicAdapter(policy),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.decision_trace[-1].payload["code"] == "duplicate_result_id"
    assert [event.event_type for event in result.decision_trace] == [
        "adapter_request",
        "adapter_response",
        "tool_call",
        "tool_result",
        "adapter_request",
        "adapter_response",
        "tool_call",
        "error",
    ]
    assert result.decision_trace[-2].payload == {
        "call_id": "c2",
        "tool_name": "query_product",
        "arguments": {"product_ids": ("P001",)},
    }
    assert invocation_count() == 2
    assert [item.result.result_id for item in result.prior_tool_executions] == ["result_0001"]


def test_runner_rejects_duplicate_call_id_before_second_invocation(
    catalog: Catalog,
) -> None:
    registry, invocation_count = registry_returning_fixed_result_id("result_0001")
    result = AgentRunner(
        registry=registry,
        adapter=DeterministicAdapter(
            lambda _: call("same", "query_product", {"product_ids": ["P001"]})
        ),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.decision_trace[-1].payload == {
        "code": "duplicate_call_id",
        "summary": "Tool call identifier was reused.",
    }
    assert invocation_count() == 1


def test_runner_stops_at_max_steps(catalog: Catalog) -> None:
    counter = itertools.count(1)

    def policy(_: AdapterRequest) -> AssistantAction:
        return call(
            f"c{next(counter)}",
            "query_product",
            {"product_ids": ["P001"]},
        )

    result = runner(policy).run(
        RunRequest(user_input="x", max_steps=2),
        catalog,
    )
    assert result.status == "failed"
    assert len(result.prior_tool_executions) == 2
    assert result.decision_trace[-1].payload == {
        "code": "max_steps_exceeded",
        "summary": "Agent step limit was exceeded.",
    }


def test_runner_structures_start_run_exception(catalog: Catalog) -> None:
    result = AgentRunner(
        registry=build_default_registry(),
        adapter=StartRunFailingAdapter(),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.prior_tool_executions == []
    assert result.decision_trace[-1].event_type == "error"
    assert result.decision_trace[-1].payload == {
        "code": "adapter_start_error",
        "summary": "Adapter initialization failed.",
    }
    assert "/tmp/" not in result.model_dump_json()


def test_runner_strictly_revalidates_untrusted_custom_adapter_response(
    catalog: Catalog,
) -> None:
    call_1 = ToolCall(
        call_id="c1",
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    call_2 = ToolCall(
        call_id="c2",
        name="query_product",
        arguments={"product_ids": ["P002"]},
    )
    invalid_actions = [
        AssistantAction.model_construct(
            tool_calls=[call_1, call_2],
            final_answer=None,
        ),
        AssistantAction.model_construct(
            tool_calls=[call_1],
            final_answer="不能与工具调用同时出现",
        ),
    ]
    for action in invalid_actions:
        registry = mock.Mock(wraps=build_default_registry())
        adapter = CustomAdapter(
            AdapterResponse.model_construct(
                action=action,
                raw_response=None,
                usage=None,
            )
        )
        result = AgentRunner(registry=registry, adapter=adapter).run(
            RunRequest(user_input="x"),
            catalog,
        )
        assert result.status == "failed"
        assert result.prior_tool_executions == []
        assert result.decision_trace[-1].payload["code"] == "adapter_protocol_error"
        assert (
            result.decision_trace[-1].payload["summary"]
            == "Adapter response violated the protocol."
        )
        registry.invoke.assert_not_called()


@pytest.mark.parametrize(
    "malicious_call_id",
    [
        "api_key=sk-secret",
        "path=/tmp/private/key",
        "sql=DROP TABLE sales",
        "x" * 129,
        " call_0001 ",
    ],
)
def test_runner_rejects_unsafe_call_id_without_leaking_it(
    malicious_call_id: str,
    catalog: Catalog,
) -> None:
    untrusted_call = ToolCall.model_construct(
        call_id=malicious_call_id,
        name="query_product",
        arguments={"product_ids": ["P001"]},
    )
    untrusted_action = AssistantAction.model_construct(
        tool_calls=[untrusted_call],
        final_answer=None,
    )
    adapter = CustomAdapter(
        AdapterResponse.model_construct(
            action=untrusted_action,
            raw_response=None,
            usage=None,
        )
    )
    registry = mock.Mock(wraps=build_default_registry())

    result = AgentRunner(registry=registry, adapter=adapter).run(
        RunRequest(user_input="x"),
        catalog,
    )

    assert result.status == "failed"
    assert result.prior_tool_executions == []
    assert result.decision_trace[-1].payload == {
        "code": "adapter_protocol_error",
        "summary": "Adapter response violated the protocol.",
    }
    assert malicious_call_id not in result.model_dump_json()
    registry.invoke.assert_not_called()


def test_runner_starts_each_run_and_does_not_leak_previous_state(
    catalog: Catalog,
) -> None:
    class StatefulAdapter:
        adapter_name = "stateful"

        def __init__(self) -> None:
            self.starts = 0

        def start_run(self) -> None:
            self.starts += 1
            if self.starts == 2:
                raise RuntimeError("second run")

        def complete(self, request: AdapterRequest) -> AdapterResponse:
            if request.prior_tool_executions:
                return AdapterResponse(action=AssistantAction(final_answer="done"))
            return AdapterResponse(
                action=call(
                    "c1",
                    "query_product",
                    {"product_ids": ["P001"]},
                )
            )

    adapter = StatefulAdapter()
    agent = AgentRunner(registry=build_default_registry(), adapter=adapter)
    first = agent.run(RunRequest(user_input="first"), catalog)
    second = agent.run(RunRequest(user_input="second"), catalog)

    assert first.status == "completed"
    assert len(first.prior_tool_executions) == 1
    assert adapter.starts == 2
    assert second.status == "failed"
    assert second.prior_tool_executions == []
    assert len(second.decision_trace) == 1
    assert second.decision_trace[0].payload["code"] == "adapter_start_error"
