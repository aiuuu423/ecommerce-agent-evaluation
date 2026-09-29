from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llm import Usage
from app.tools.schemas import JsonValue, PriorToolExecution


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
    sequence: int = Field(ge=1, strict=True)
    event_type: TraceEventType
    payload: dict[str, JsonValue]


class RunRequest(AgentModel):
    user_input: str = Field(min_length=1)
    max_steps: int = Field(default=8, ge=1, le=32, strict=True)


class RunResult(AgentModel):
    status: Literal["completed", "failed"]
    final_answer: str | None
    prior_tool_executions: list[PriorToolExecution]
    decision_trace: list[TraceEvent]
    usage: Usage | None
