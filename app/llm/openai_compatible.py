import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from copy import deepcopy
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from app.llm.schemas import (
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
    ToolCall,
    Usage,
)
from app.tools.schemas import JsonValue, PriorToolExecution, canonical_tool_result_payload

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 64


class OpenAITransportError(RuntimeError):
    """A provider request failed without exposing provider response details."""


class OpenAIProtocolError(ValueError):
    """A provider payload or transcript transition violated the adapter protocol."""


class JsonTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        raise NotImplementedError


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


class UrllibJsonTransport:
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        http_status: int | None = None
        request_failed = False
        try:
            opener = urllib.request.build_opener(_RejectRedirectHandler())
            with opener.open(request, timeout=timeout_seconds) as response:
                content_length = response.headers.get("Content-Length")
                try:
                    declared_length = (
                        int(content_length) if content_length is not None else None
                    )
                except (TypeError, ValueError):
                    declared_length = None
                if declared_length is not None and declared_length > MAX_RESPONSE_BYTES:
                    raise OpenAITransportError(
                        "provider response is too large"
                    ) from None
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            exc.close()
        except (urllib.error.URLError, TimeoutError, OSError):
            request_failed = True
        if http_status is not None:
            raise OpenAITransportError(
                f"provider returned HTTP {http_status}"
            ) from None
        if request_failed:
            raise OpenAITransportError("provider request failed") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise OpenAITransportError("provider response is too large") from None

        invalid_json = False
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            invalid_json = True
        if invalid_json:
            raise OpenAIProtocolError(
                "provider response must contain valid JSON"
            ) from None
        if not isinstance(parsed, dict):
            raise OpenAIProtocolError("provider response must be a JSON object")
        _validate_json_value(parsed, "provider response")
        return parsed


class OpenAICompatibleAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        transport: JsonTransport | None = None,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._api_key = _validate_non_empty_string(api_key, "api_key")
        self._model = _validate_non_empty_string(model, "model")
        self._temperature = _validate_temperature(temperature)
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._transport = transport or UrllibJsonTransport()
        self.start_run()

    @property
    def adapter_name(self) -> str:
        return "openai-compatible"

    def start_run(self) -> None:
        self._messages: list[dict[str, JsonValue]] = []
        self._user_input: str | None = None
        self._tools: list[dict[str, JsonValue]] | None = None
        self._pending_call_id: str | None = None
        self._pending_arguments: dict[str, JsonValue] | None = None
        self._sent_executions: list[dict[str, JsonValue]] = []
        self._sent_call_ids: set[str] = set()

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        snapshot = (
            deepcopy(self._messages),
            self._user_input,
            deepcopy(self._tools),
            self._pending_call_id,
            deepcopy(self._pending_arguments),
            deepcopy(self._sent_executions),
            set(self._sent_call_ids),
        )
        try:
            self._advance_transcript(request)
            payload: dict[str, JsonValue] = {
                "model": self._model,
                "messages": deepcopy(self._messages),
                "tools": deepcopy(request.tools),
                "tool_choice": "auto",
                "temperature": self._temperature,
            }
            raw_response = self._transport.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )
            if not isinstance(raw_response, dict):
                raise OpenAIProtocolError("provider response must be a JSON object")
            _validate_json_value(raw_response, "provider response")
            normalized = deepcopy(raw_response)
            action = self._parse_action(normalized)
            usage = self._parse_usage(normalized)
            return AdapterResponse(
                action=action,
                raw_response=normalized,
                usage=usage,
            )
        except Exception:
            (
                self._messages,
                self._user_input,
                self._tools,
                self._pending_call_id,
                self._pending_arguments,
                self._sent_executions,
                self._sent_call_ids,
            ) = snapshot
            raise

    def _advance_transcript(self, request: AdapterRequest) -> None:
        if self._user_input is None:
            if request.prior_tool_executions:
                raise OpenAIProtocolError("unknown call_id in prior tool executions")
            self._user_input = request.user_input
            self._tools = deepcopy(request.tools)
            self._messages.append({"role": "user", "content": request.user_input})
            return

        if request.user_input != self._user_input:
            raise OpenAIProtocolError("user_input changed within one run")
        if request.tools != self._tools:
            raise OpenAIProtocolError("tools changed within one run")

        executions = request.prior_tool_executions
        accepted_count = len(self._sent_executions)
        if len(executions) < accepted_count:
            raise OpenAIProtocolError("prior tool executions must be monotonic")
        for index, accepted in enumerate(self._sent_executions):
            current = _execution_payload(executions[index])
            if current != accepted:
                raise OpenAIProtocolError("prior tool execution history changed")

        if self._pending_call_id is None:
            if executions:
                raise OpenAIProtocolError("replayed or unknown call_id")
            raise OpenAIProtocolError("call_id is not awaiting a tool result")
        if len(executions) != accepted_count + 1:
            raise OpenAIProtocolError(
                "exactly one pending call_id tool result is required"
            )

        execution = executions[-1]
        if execution.call_id != self._pending_call_id:
            raise OpenAIProtocolError("tool result call_id does not match pending call_id")
        if execution.call_id in self._sent_call_ids:
            raise OpenAIProtocolError("tool result call_id was already sent")
        if not _execution_arguments_match_pending(
            self._pending_arguments,
            execution.arguments,
        ):
            raise OpenAIProtocolError(
                "tool result arguments do not match pending tool call arguments"
            )

        content = json.dumps(
            canonical_tool_result_payload(execution.result),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": execution.call_id,
                "content": content,
            }
        )
        self._sent_executions.append(_execution_payload(execution))
        self._sent_call_ids.add(execution.call_id)
        self._pending_call_id = None
        self._pending_arguments = None

    def _parse_action(self, payload: dict[str, JsonValue]) -> AssistantAction:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpenAIProtocolError("provider response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OpenAIProtocolError("provider choice must be a JSON object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OpenAIProtocolError("provider choice must contain an assistant message")

        content = message.get("content")
        tool_calls = message.get("tool_calls")
        has_content = isinstance(content, str) and bool(content.strip())
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if has_content == has_tool_calls:
            raise OpenAIProtocolError(
                "assistant message must contain exactly one tool call or non-empty content"
            )

        if has_content:
            self._messages.append({"role": "assistant", "content": content})
            return AssistantAction(final_answer=content)

        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise OpenAIProtocolError(
                "assistant message must contain exactly one function tool call"
            )
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            raise OpenAIProtocolError("tool call must be a JSON object")
        call_id = tool_call.get("id")
        call_type = tool_call.get("type")
        function = tool_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise OpenAIProtocolError("tool call id must be a non-empty string")
        if call_id in self._sent_call_ids or call_id == self._pending_call_id:
            raise OpenAIProtocolError("provider reused a tool call_id")
        if call_type != "function" or not isinstance(function, dict):
            raise OpenAIProtocolError("tool call must have type function")
        name = function.get("name")
        encoded_arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise OpenAIProtocolError("function name must be a non-empty string")
        if not isinstance(encoded_arguments, str):
            raise OpenAIProtocolError("function arguments must be valid JSON")
        try:
            arguments = json.loads(encoded_arguments)
        except (json.JSONDecodeError, RecursionError):
            raise OpenAIProtocolError(
                "function arguments must be valid JSON"
            ) from None
        if not isinstance(arguments, dict):
            raise OpenAIProtocolError("function arguments must be a JSON object")
        _validate_json_value(arguments, "function arguments")
        try:
            parsed_tool_call = ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        except ValidationError:
            parsed_tool_call = None
        if parsed_tool_call is None:
            raise OpenAIProtocolError("tool call id must be a safe ASCII identifier")

        assistant_message = deepcopy(message)
        assistant_message["role"] = "assistant"
        self._messages.append(assistant_message)
        self._pending_call_id = call_id
        self._pending_arguments = deepcopy(arguments)
        return AssistantAction(tool_calls=[parsed_tool_call])

    @staticmethod
    def _parse_usage(payload: dict[str, JsonValue]) -> Usage | None:
        if "usage" not in payload:
            return None
        usage_payload = payload["usage"]
        if not isinstance(usage_payload, dict):
            raise OpenAIProtocolError("usage must be a JSON object")
        required = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(field not in usage_payload for field in required):
            raise OpenAIProtocolError("usage must contain all token fields")
        values = {field: usage_payload[field] for field in required}
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ):
            raise OpenAIProtocolError("usage token fields must be strict integers")
        try:
            return Usage.model_validate(values, strict=True)
        except ValidationError as exc:
            message = (
                "usage total_tokens must equal prompt_tokens + completion_tokens"
                if "total_tokens" in str(exc)
                else "usage token fields must be non-negative integers"
            )
            raise OpenAIProtocolError(message) from None


def _validate_base_url(value: object) -> str:
    base_url = _validate_non_empty_string(value, "base_url").rstrip("/")
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError("base_url must be an http(s) URL with a host") from None
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an http(s) URL with a host")
    return base_url


def _validate_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("temperature must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 2:
        raise ValueError("temperature must be between 0 and 2")
    return normalized


def _validate_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_seconds must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    return normalized


def _execution_payload(execution: PriorToolExecution) -> dict[str, JsonValue]:
    return execution.model_dump(mode="json")


def _execution_arguments_match_pending(
    pending: dict[str, JsonValue],
    execution: dict[str, JsonValue],
) -> bool:
    return all(
        key in execution and _json_values_equal(value, execution[key])
        for key, value in pending.items()
    )


def _json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return left.keys() == right.keys() and all(
            _json_values_equal(value, right[key]) for key, value in left.items()
        )
    return left == right


def _validate_json_value(value: object, location: str, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise OpenAIProtocolError(f"{location} exceeds maximum JSON nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise OpenAIProtocolError(f"{location} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, location, depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OpenAIProtocolError(f"{location} contains a non-string key")
            _validate_json_value(item, location, depth + 1)
        return
    raise OpenAIProtocolError(f"{location} contains a non-JSON value")
