import io
import json
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

import pytest

from app.llm.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAIProtocolError,
    OpenAITransportError,
    UrllibJsonTransport,
)
from app.llm.schemas import AdapterRequest
from app.tools.schemas import PriorToolExecution, QueryProductResult


class SequenceTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.payloads: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.urls.append(url)
        self.headers.append(deepcopy(headers))
        self.payloads.append(deepcopy(payload))
        assert timeout_seconds == 30.0
        return self.responses[len(self.payloads) - 1]


class FailingOnceTransport(SequenceTransport):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not self.payloads:
            self.urls.append(url)
            self.headers.append(deepcopy(headers))
            self.payloads.append(deepcopy(payload))
            raise OpenAITransportError("provider request failed")
        self.urls.append(url)
        self.headers.append(deepcopy(headers))
        self.payloads.append(deepcopy(payload))
        assert timeout_seconds == 30.0
        return self.responses[0]


class StubResponse(io.BytesIO):
    def __init__(self, body: bytes, content_length: str | None = None) -> None:
        super().__init__(body)
        self.headers = {} if content_length is None else {"Content-Length": content_length}
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


def patch_transport_response(
    monkeypatch: pytest.MonkeyPatch,
    response_or_error: StubResponse | BaseException,
) -> None:
    def open_response(*_args: object, **_kwargs: object) -> StubResponse:
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return response_or_error

    class StubOpener:
        def open(self, *_args: object, **_kwargs: object) -> StubResponse:
            return open_response()

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_handlers: StubOpener(),
    )
    monkeypatch.setattr("urllib.request.urlopen", open_response)


def answer_payload(
    content: str = "完成。",
    *,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "choices": [{"message": {"content": content}}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def tool_payload(
    *,
    call_id: str = "call_abc",
    arguments: str = '{"product_ids":["P001"]}',
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
                                "name": "query_product",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    }


def sample_request(
    *,
    user_input: str = "查看 P001",
    executions: list[PriorToolExecution] | None = None,
) -> AdapterRequest:
    return AdapterRequest(
        user_input=user_input,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "query_product",
                    "description": "查询商品",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["product_ids"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        prior_tool_executions=executions or [],
    )


def sample_product_result() -> QueryProductResult:
    return QueryProductResult(
        result_id="result_0001",
        tool_name="query_product",
        dataset_id="e1e81533c25e03e5",
        source_label="Synthetic E-commerce Data",
        columns=[
            "product_id",
            "product_name",
            "category",
            "price",
            "cost",
            "launch_date",
        ],
        rows=[
            {
                "product_id": "P001",
                "product_name": "商品一",
                "category": "A",
                "price": 10.0,
                "cost": 6.0,
                "launch_date": "2026-01-01",
            }
        ],
        row_count=1,
    )


def adapter_with(transport: SequenceTransport) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        base_url="https://example.test/v1/",
        api_key="secret",
        model="test-model",
        transport=transport,
    )


def execution_for(call_id: str = "call_abc") -> PriorToolExecution:
    return PriorToolExecution(
        call_id=call_id,
        arguments={"product_ids": ["P001"]},
        result=sample_product_result(),
    )


def adapter_state(adapter: OpenAICompatibleAdapter) -> tuple[Any, ...]:
    return (
        deepcopy(adapter._messages),
        adapter._user_input,
        deepcopy(adapter._tools),
        adapter._pending_call_id,
        deepcopy(adapter._pending_arguments),
        deepcopy(adapter._sent_executions),
        set(adapter._sent_call_ids),
    )


def exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"base_url": ""}, "base_url"),
        ({"base_url": "file:///tmp/provider"}, "base_url"),
        ({"base_url": "https:///v1"}, "base_url"),
        ({"base_url": "https://:443"}, "base_url"),
        ({"base_url": "https://example.test:invalid/v1"}, "base_url"),
        ({"api_key": ""}, "api_key"),
        ({"model": "  "}, "model"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"timeout_seconds": 0}, "timeout"),
    ],
)
def test_openai_adapter_requires_valid_config(
    override: dict[str, Any],
    message: str,
) -> None:
    config: dict[str, Any] = {
        "base_url": "https://example.test/v1",
        "api_key": "secret",
        "model": "test-model",
        "transport": SequenceTransport([answer_payload()]),
    }
    config.update(override)
    with pytest.raises((TypeError, ValueError), match=message):
        OpenAICompatibleAdapter(**config)


def test_openai_adapter_normalizes_tool_call_without_network() -> None:
    transport = SequenceTransport([tool_payload()])
    adapter = adapter_with(transport)

    response = adapter.complete(sample_request())

    assert transport.urls == ["https://example.test/v1/chat/completions"]
    assert transport.payloads[0] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "查看 P001"}],
        "tools": sample_request().tools,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    assert response.action.tool_calls[0].arguments == {"product_ids": ["P001"]}
    assert response.usage is not None
    assert response.usage.total_tokens == 14
    serialized = json.dumps(response.model_dump(), ensure_ascii=False)
    assert "secret" not in serialized
    assert "/tmp/" not in serialized


def test_openai_adapter_keeps_two_round_transcript_and_call_id() -> None:
    transport = SequenceTransport([tool_payload(), answer_payload("P001 产品信息已取得。")])
    adapter = adapter_with(transport)
    adapter.start_run()
    request = sample_request()
    first = adapter.complete(request)
    execution = PriorToolExecution(
        call_id=first.action.tool_calls[0].call_id,
        arguments=first.action.tool_calls[0].arguments,
        result=sample_product_result(),
    )

    final = adapter.complete(sample_request(executions=[execution]))

    assert final.action.final_answer == "P001 产品信息已取得。"
    second_messages = transport.payloads[1]["messages"]
    assert second_messages[0] == transport.payloads[0]["messages"][0]
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[1]["tool_calls"][0]["id"] == "call_abc"
    assert second_messages[2]["role"] == "tool"
    assert second_messages[2]["tool_call_id"] == "call_abc"
    tool_content = second_messages[2]["content"]
    assert tool_content == json.dumps(
        json.loads(tool_content),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert json.loads(tool_content)["result_id"] == "result_0001"
    assert "path" not in tool_content.lower()


def test_openai_start_run_clears_previous_transcript() -> None:
    transport = SequenceTransport([answer_payload("first"), answer_payload("second")])
    adapter = adapter_with(transport)
    adapter.start_run()
    adapter.complete(sample_request(user_input="first run"))
    adapter.start_run()
    adapter.complete(sample_request(user_input="second run"))
    assert transport.payloads[1]["messages"] == [
        {"role": "user", "content": "second run"}
    ]


def test_openai_adapter_rejects_unknown_missing_or_replayed_tool_result() -> None:
    unknown = adapter_with(SequenceTransport([answer_payload()]))
    with pytest.raises(OpenAIProtocolError, match="call_id"):
        unknown.complete(sample_request(executions=[execution_for("unknown_call")]))

    missing = adapter_with(SequenceTransport([tool_payload(), answer_payload()]))
    missing.complete(sample_request())
    with pytest.raises(OpenAIProtocolError, match="call_id"):
        missing.complete(sample_request())

    replay = adapter_with(
        SequenceTransport([tool_payload(), answer_payload(), answer_payload()])
    )
    replay.complete(sample_request())
    request = sample_request(executions=[execution_for()])
    replay.complete(request)
    with pytest.raises(OpenAIProtocolError, match="call_id"):
        replay.complete(request)


def test_openai_adapter_rejects_changed_run_inputs() -> None:
    adapter = adapter_with(SequenceTransport([tool_payload(), answer_payload()]))
    adapter.complete(sample_request())
    with pytest.raises(OpenAIProtocolError, match="user_input"):
        adapter.complete(
            sample_request(user_input="changed", executions=[execution_for()])
        )

    adapter.start_run()
    adapter.complete(sample_request())
    changed = sample_request(executions=[execution_for()])
    changed.tools[0]["function"]["description"] = "changed"
    with pytest.raises(OpenAIProtocolError, match="tools"):
        adapter.complete(changed)


def test_openai_adapter_rejects_tool_result_with_changed_arguments() -> None:
    adapter = adapter_with(SequenceTransport([tool_payload(), answer_payload()]))
    adapter.complete(sample_request())
    execution = execution_for().model_copy(
        update={
            "arguments": {
                "product_ids": ["P002"],
                "defaulted_field": [],
            }
        }
    )
    with pytest.raises(OpenAIProtocolError, match="arguments"):
        adapter.complete(sample_request(executions=[execution]))


def test_openai_adapter_rejects_tool_result_missing_original_argument() -> None:
    adapter = adapter_with(SequenceTransport([tool_payload(), answer_payload()]))
    adapter.complete(sample_request())
    execution = execution_for().model_copy(
        update={"arguments": {"defaulted_field": []}}
    )

    with pytest.raises(OpenAIProtocolError, match="arguments"):
        adapter.complete(sample_request(executions=[execution]))


def test_openai_adapter_compares_nested_argument_values_strictly() -> None:
    pending_arguments = {
        "product_ids": ["P001"],
        "filters": {"enabled": 1},
    }
    adapter = adapter_with(
        SequenceTransport(
            [
                tool_payload(arguments=json.dumps(pending_arguments)),
                answer_payload(),
            ]
        )
    )
    adapter.complete(sample_request())
    execution = execution_for().model_copy(
        update={
            "arguments": {
                "product_ids": ["P001"],
                "filters": {"enabled": True},
                "defaulted_field": [],
            }
        }
    )

    with pytest.raises(OpenAIProtocolError, match="arguments"):
        adapter.complete(sample_request(executions=[execution]))


def test_openai_adapter_rejects_non_object_provider_response() -> None:
    transport = SequenceTransport([[]])  # type: ignore[list-item]
    adapter = adapter_with(transport)
    with pytest.raises(OpenAIProtocolError, match="JSON object"):
        adapter.complete(sample_request())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"choices": []}, "exactly one choice"),
        ({"choices": [{}, {}]}, "exactly one choice"),
        (tool_payload(arguments="{bad json"), "arguments must be valid JSON"),
        (tool_payload(arguments="[]"), "arguments must be a JSON object"),
        (answer_payload(""), "non-empty content"),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": "answer",
                            "tool_calls": tool_payload()["choices"][0]["message"][
                                "tool_calls"
                            ],
                        }
                    }
                ]
            },
            "exactly one",
        ),
        (
            answer_payload(
                usage={
                    "prompt_tokens": True,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ),
            "integer",
        ),
        (
            answer_payload(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                }
            ),
            "total_tokens",
        ),
        (
            answer_payload(
                usage={"prompt_tokens": 10, "completion_tokens": 4}
            ),
            "usage",
        ),
    ],
)
def test_openai_adapter_rejects_malformed_provider_payload(
    payload: dict[str, Any],
    message: str,
) -> None:
    adapter = adapter_with(SequenceTransport([payload]))
    with pytest.raises(OpenAIProtocolError, match=message):
        adapter.complete(sample_request())


@pytest.mark.parametrize(
    "call_id",
    [
        "api_key=sk-secret",
        "path=/tmp/private/key",
        "sql=DROP TABLE sales",
        "x" * 129,
        " call_abc ",
    ],
)
def test_openai_adapter_rejects_unsafe_tool_call_id(call_id: str) -> None:
    adapter = adapter_with(SequenceTransport([tool_payload(call_id=call_id)]))

    with pytest.raises(OpenAIProtocolError, match="safe ASCII identifier") as captured:
        adapter.complete(sample_request())

    assert all(call_id not in repr(item) for item in exception_chain(captured.value))


def test_complete_restores_initial_state_after_transport_failure_and_can_retry() -> None:
    transport = FailingOnceTransport([answer_payload("retry succeeded")])
    adapter = adapter_with(transport)
    request = sample_request()
    before = adapter_state(adapter)

    with pytest.raises(OpenAITransportError):
        adapter.complete(request)

    assert adapter_state(adapter) == before
    response = adapter.complete(request)

    assert response.action.final_answer == "retry succeeded"
    assert transport.payloads[0]["messages"] == [{"role": "user", "content": "查看 P001"}]
    assert transport.payloads[1]["messages"] == transport.payloads[0]["messages"]


def test_complete_restores_pending_state_after_invalid_usage_and_can_retry() -> None:
    invalid_usage = answer_payload(
        "bad",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99},
    )
    transport = SequenceTransport(
        [tool_payload(), invalid_usage, answer_payload("retry succeeded")]
    )
    adapter = adapter_with(transport)
    adapter.complete(sample_request())
    request = sample_request(executions=[execution_for()])
    before = adapter_state(adapter)

    with pytest.raises(OpenAIProtocolError, match="total_tokens"):
        adapter.complete(request)

    assert adapter_state(adapter) == before
    response = adapter.complete(request)

    assert response.action.final_answer == "retry succeeded"
    assert transport.payloads[2]["messages"] == transport.payloads[1]["messages"]
    assert [message["role"] for message in transport.payloads[2]["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


def test_complete_restores_initial_state_after_invalid_action_and_can_retry() -> None:
    transport = SequenceTransport(
        [{"choices": []}, answer_payload("retry succeeded")]
    )
    adapter = adapter_with(transport)
    request = sample_request()
    before = adapter_state(adapter)

    with pytest.raises(OpenAIProtocolError, match="exactly one choice"):
        adapter.complete(request)

    assert adapter_state(adapter) == before
    assert adapter.complete(request).action.final_answer == "retry succeeded"
    assert transport.payloads[1]["messages"] == transport.payloads[0]["messages"]


def test_urllib_transport_http_error_chain_does_not_expose_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = b'{"secret":"http-body-sensitive"}'

    def fail_with_http_error(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(sensitive),
        )

    class FailingOpener:
        def open(self, *_args: object, **_kwargs: object) -> None:
            fail_with_http_error()

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_handlers: FailingOpener(),
    )
    monkeypatch.setattr("urllib.request.urlopen", FailingOpener().open)

    with pytest.raises(OpenAITransportError) as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer request-key-sensitive"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(sensitive.decode() not in repr(item) for item in exception_chain(captured.value))


def test_urllib_transport_url_error_chain_does_not_expose_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = "url-error-request-key-sensitive"

    def fail_with_url_error(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError(f"connection rejected for {sensitive}")

    patch_transport_response(
        monkeypatch,
        urllib.error.URLError(f"connection rejected for {sensitive}"),
    )

    with pytest.raises(OpenAITransportError) as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": f"Bearer {sensitive}"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(sensitive not in repr(item) for item in exception_chain(captured.value))


def test_urllib_transport_invalid_json_chain_does_not_expose_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = b"invalid-json-response-sensitive"
    patch_transport_response(monkeypatch, StubResponse(sensitive))

    with pytest.raises(OpenAIProtocolError) as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer request-key-sensitive"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(sensitive.decode() not in repr(item) for item in exception_chain(captured.value))


def test_urllib_transport_disables_redirects_and_sanitizes_3xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_key = "redirect-request-key-sensitive"
    sensitive_location = "https://redirect.test/collect?key=location-sensitive"
    second_requests: list[urllib.request.Request] = []

    class RecordingOpener:
        def __init__(self, handler: urllib.request.HTTPRedirectHandler) -> None:
            self.handler = handler

        def open(
            self,
            request: urllib.request.Request,
            *,
            timeout: float,
        ) -> StubResponse:
            redirected = self.handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {"Location": sensitive_location},
                sensitive_location,
            )
            if redirected is not None:
                second_requests.append(redirected)
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                f"redirect to {sensitive_location}",
                {"Location": sensitive_location},
                None,
            )

    def build_opener(
        handler: urllib.request.HTTPRedirectHandler,
    ) -> RecordingOpener:
        return RecordingOpener(handler)

    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        RecordingOpener(urllib.request.HTTPRedirectHandler()).open,
    )

    with pytest.raises(OpenAITransportError, match="HTTP 302") as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": f"Bearer {sensitive_key}"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert second_requests == []
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(
        sensitive_key not in repr(item) and sensitive_location not in repr(item)
        for item in exception_chain(captured.value)
    )


def test_urllib_transport_rejects_declared_oversized_response_without_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = StubResponse(b"{}", content_length=str(2 * 1024 * 1024 + 1))
    patch_transport_response(monkeypatch, response)

    with pytest.raises(OpenAITransportError, match="too large") as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer response-limit-key"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert response.read_sizes == []
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_urllib_transport_reads_limit_plus_one_and_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = StubResponse(b"x" * (2 * 1024 * 1024 + 1))
    patch_transport_response(monkeypatch, response)

    with pytest.raises(OpenAITransportError, match="too large"):
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer response-limit-key"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert response.read_sizes == [2 * 1024 * 1024 + 1]


def test_urllib_transport_sanitizes_unicode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_transport_response(
        monkeypatch,
        StubResponse(b'{"choices":[{"message":{"content":"\xff"}}]}'),
    )

    with pytest.raises(OpenAIProtocolError, match="valid JSON") as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer parse-key-sensitive"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_urllib_transport_sanitizes_json_recursion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_transport_response(monkeypatch, StubResponse(b"{}"))

    def recurse_forever(_value: object) -> Any:
        raise RecursionError("sensitive parser detail")

    monkeypatch.setattr(json, "loads", recurse_forever)

    with pytest.raises(OpenAIProtocolError, match="valid JSON") as captured:
        UrllibJsonTransport().post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer parse-key-sensitive"},
            payload={"model": "test-model"},
            timeout_seconds=30.0,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_openai_adapter_rejects_excessive_json_nesting() -> None:
    nested: Any = "leaf"
    for _ in range(65):
        nested = {"nested": nested}
    adapter = adapter_with(
        SequenceTransport(
            [{"choices": [{"message": {"content": "done"}}], "nested": nested}]
        )
    )

    with pytest.raises(OpenAIProtocolError, match="nesting"):
        adapter.complete(sample_request())


def test_fake_transport_stays_offline_and_receives_key_only_in_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fake transport must not access the network")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    transport = SequenceTransport([answer_payload()])
    response = adapter_with(transport).complete(sample_request())

    assert response.action.final_answer == "完成。"
    assert transport.headers == [
        {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        }
    ]
    assert "secret" not in json.dumps(transport.payloads)
