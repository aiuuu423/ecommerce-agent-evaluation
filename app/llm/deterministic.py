from collections.abc import Callable

from pydantic import Field, model_validator

from app.llm.schemas import (
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
    ToolCall,
)

DeterministicPolicy = Callable[[AdapterRequest], AssistantAction]


class DeterministicPlan(AdapterModel):
    tool_calls: list[ToolCall] = Field(min_length=1)
    final_answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_call_ids(self) -> "DeterministicPlan":
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique")
        return self


class DeterministicAdapter:
    def __init__(self, policy: DeterministicPolicy | DeterministicPlan) -> None:
        if not isinstance(policy, DeterministicPlan) and not callable(policy):
            raise TypeError("policy must be callable or a DeterministicPlan")
        self._policy = policy
        self._active_user_input: str | None = None

    @property
    def adapter_name(self) -> str:
        return "deterministic"

    def start_run(self) -> None:
        self._active_user_input = None

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        if not isinstance(self._policy, DeterministicPlan):
            action = self._policy(request)
            validated = AssistantAction.model_validate(
                action.model_dump(mode="python"),
                strict=True,
            )
            return AdapterResponse(action=validated, raw_response=None, usage=None)

        self._bind_run(request)
        self._validate_prior_executions(request)
        completed_count = len(request.prior_tool_executions)
        if completed_count < len(self._policy.tool_calls):
            action = AssistantAction(
                tool_calls=[self._policy.tool_calls[completed_count]]
            )
        else:
            action = AssistantAction(
                final_answer=self._answer_with_evidence(request)
            )
        return AdapterResponse(action=action, raw_response=None, usage=None)

    def _bind_run(self, request: AdapterRequest) -> None:
        if self._active_user_input is None:
            self._active_user_input = request.user_input
        elif request.user_input != self._active_user_input:
            raise ValueError("user_input changed before start_run reset")

    def _validate_prior_executions(self, request: AdapterRequest) -> None:
        executions = request.prior_tool_executions
        if len(executions) > len(self._policy.tool_calls):
            raise ValueError("prior executions exceed deterministic plan")
        for index, execution in enumerate(executions):
            planned = self._policy.tool_calls[index]
            if execution.call_id != planned.call_id:
                raise ValueError(f"prior execution {index} has unexpected call_id")
            if execution.result.tool_name != planned.name:
                raise ValueError(f"prior execution {index} has unexpected tool_name")
            if execution.arguments != planned.arguments:
                raise ValueError(f"prior execution {index} has unexpected arguments")

    def _answer_with_evidence(self, request: AdapterRequest) -> str:
        references = "; ".join(
            (
                f"[call_id={execution.call_id} "
                f"result_id={execution.result.result_id}]"
            )
            for execution in request.prior_tool_executions
        )
        return f"{self._policy.final_answer}\n\nEvidence: {references}"
