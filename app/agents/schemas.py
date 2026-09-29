from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from app.llm import Usage
from app.tools.schemas import JsonValue, PriorToolExecution, _validate_json_value


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return cast(
            JsonValue,
            MappingProxyType({key: _freeze_json(item) for key, item in value.items()}),
        )
    if isinstance(value, list):
        return cast(JsonValue, tuple(_freeze_json(item) for item in value))
    return value


def _json_compatible(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    return cast(JsonValue, value)


class AgentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class TraceEventType(StrEnum):
    ADAPTER_REQUEST = "adapter_request"
    ADAPTER_RESPONSE = "adapter_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class TraceEvent(AgentModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
    )

    sequence: int = Field(ge=1, strict=True)
    event_type: TraceEventType
    payload: Mapping[str, JsonValue]

    @model_validator(mode="after")
    def freeze_payload(self) -> "TraceEvent":
        mutable_payload = dict(self.payload)
        _validate_json_value(mutable_payload, "payload")
        object.__setattr__(self, "payload", _freeze_json(mutable_payload))
        return self

    @field_serializer("payload")
    def serialize_payload(self, payload: Mapping[str, JsonValue]) -> JsonValue:
        return _json_compatible(payload)


class RunRequest(AgentModel):
    user_input: str = Field(min_length=1)
    max_steps: int = Field(default=8, ge=1, le=32, strict=True)


class RunResult(AgentModel):
    status: Literal["completed", "failed"]
    final_answer: str | None
    prior_tool_executions: list[PriorToolExecution]
    decision_trace: tuple[TraceEvent, ...]
    usage: Usage | None
