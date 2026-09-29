from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.schemas import JsonValue, PriorToolExecution


class AdapterModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class ToolCall(AdapterModel):
    call_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class AssistantAction(AdapterModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None

    @model_validator(mode="after")
    def exactly_one_mode(self) -> "AssistantAction":
        if bool(self.tool_calls) == (self.final_answer is not None):
            raise ValueError("provide tool_calls or final_answer, exactly one")
        if len(self.tool_calls) > 1:
            raise ValueError("phase 2 supports one tool call per action")
        return self


class AdapterRequest(AdapterModel):
    user_input: str = Field(min_length=1)
    tools: list[dict[str, JsonValue]]
    prior_tool_executions: list[PriorToolExecution]


class Usage(AdapterModel):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def valid_total(self) -> "Usage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class AdapterResponse(AdapterModel):
    action: AssistantAction
    raw_response: dict[str, JsonValue] | None = None
    usage: Usage | None = None
