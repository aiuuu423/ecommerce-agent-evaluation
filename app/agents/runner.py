from app.agents.schemas import RunRequest, RunResult, TraceEvent, TraceEventType
from app.data.database import Catalog
from app.llm import (
    AdapterRequest,
    AdapterResponse,
    AssistantAction,
    LLMAdapter,
    Usage,
)
from app.tools import (
    ToolContext,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    UnknownToolError,
)
from app.tools.schemas import AnyToolResult, JsonValue, PriorToolExecution

ERROR_SUMMARIES = {
    "adapter_start_error": "Adapter initialization failed.",
    "adapter_error": "Adapter request failed.",
    "duplicate_call_id": "Tool call identifier was reused.",
    "unknown_tool": "Requested tool is unavailable.",
    "invalid_arguments": "Tool arguments failed validation.",
    "invalid_tool_result": "Tool result failed validation.",
    "tool_execution_error": "Tool execution failed.",
    "duplicate_result_id": "Tool result identifier was reused.",
    "max_steps_exceeded": "Agent step limit was exceeded.",
}


class TraceBuilder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def _append(
        self,
        event_type: TraceEventType,
        payload: dict[str, JsonValue],
    ) -> None:
        self.events.append(
            TraceEvent(
                sequence=len(self.events) + 1,
                event_type=event_type,
                payload=payload,
            )
        )

    def adapter_request(self, request: AdapterRequest) -> None:
        self._append(
            TraceEventType.ADAPTER_REQUEST,
            {
                "user_input": request.user_input,
                "tool_names": [
                    tool["function"]["name"]
                    for tool in request.tools
                    if isinstance(tool.get("function"), dict)
                    and isinstance(tool["function"].get("name"), str)
                ],
                "result_ids": [
                    execution.result.result_id for execution in request.prior_tool_executions
                ],
            },
        )

    def adapter_response(
        self,
        action: AssistantAction,
        usage: Usage | None,
    ) -> None:
        if action.final_answer is not None:
            payload: dict[str, JsonValue] = {
                "mode": "final_answer",
                "usage_available": usage is not None,
            }
        else:
            payload = {
                "mode": "tool_call",
                "usage_available": usage is not None,
            }
        self._append(TraceEventType.ADAPTER_RESPONSE, payload)

    def tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue] | None = None,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "call_id": call_id,
            "tool_name": tool_name,
        }
        if arguments is not None:
            payload["arguments"] = arguments
        self._append(
            TraceEventType.TOOL_CALL,
            payload,
        )

    def tool_result(self, result: AnyToolResult) -> None:
        self._append(
            TraceEventType.TOOL_RESULT,
            {
                "result_id": result.result_id,
                "tool_name": result.tool_name,
                "dataset_id": result.dataset_id,
                "columns": result.columns,
                "row_count": result.row_count,
                "warnings": result.warnings,
            },
        )

    def final_answer(self, answer: str) -> None:
        self._append(TraceEventType.FINAL_ANSWER, {"answer": answer})

    def error(self, code: str) -> None:
        self._append(
            TraceEventType.ERROR,
            {"code": code, "summary": ERROR_SUMMARIES[code]},
        )


class UsageAccumulator:
    def __init__(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_usage = False

    def add(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self._has_usage = True
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._total_tokens += usage.total_tokens

    def result(self) -> Usage | None:
        if not self._has_usage:
            return None
        return Usage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
        )


def _result(
    *,
    status: str,
    trace: TraceBuilder,
    executions: list[PriorToolExecution],
    usage: UsageAccumulator,
    final_answer: str | None,
) -> RunResult:
    return RunResult(
        status=status,
        final_answer=final_answer,
        prior_tool_executions=list(executions),
        decision_trace=tuple(trace.events),
        usage=usage.result(),
    )


def _failed(
    trace: TraceBuilder,
    executions: list[PriorToolExecution],
    usage: UsageAccumulator,
    code: str,
) -> RunResult:
    trace.error(code)
    return _result(
        status="failed",
        trace=trace,
        executions=executions,
        usage=usage,
        final_answer=None,
    )


def _completed(
    trace: TraceBuilder,
    executions: list[PriorToolExecution],
    usage: UsageAccumulator,
    final_answer: str,
) -> RunResult:
    return _result(
        status="completed",
        trace=trace,
        executions=executions,
        usage=usage,
        final_answer=final_answer,
    )


class AgentRunner:
    def __init__(self, registry: ToolRegistry, adapter: LLMAdapter) -> None:
        self._registry = registry
        self._adapter = adapter

    def run(self, request: RunRequest, catalog: Catalog) -> RunResult:
        executions: list[PriorToolExecution] = []
        trace = TraceBuilder()
        seen_call_ids: set[str] = set()
        seen_result_ids: set[str] = set()
        usage = UsageAccumulator()

        try:
            self._adapter.start_run()
        except Exception:
            return _failed(
                trace,
                executions,
                usage,
                "adapter_start_error",
            )

        for _ in range(request.max_steps):
            adapter_request = AdapterRequest(
                user_input=request.user_input,
                tools=self._registry.openai_tools(),
                prior_tool_executions=list(executions),
            )
            trace.adapter_request(adapter_request)
            try:
                untrusted_response = self._adapter.complete(adapter_request)
                response = AdapterResponse.model_validate(
                    untrusted_response.model_dump(mode="python"),
                    strict=True,
                )
            except Exception:
                return _failed(
                    trace,
                    executions,
                    usage,
                    "adapter_error",
                )
            trace.adapter_response(response.action, response.usage)
            usage.add(response.usage)

            if response.action.final_answer is not None:
                trace.final_answer(response.action.final_answer)
                return _completed(
                    trace,
                    executions,
                    usage,
                    response.action.final_answer,
                )

            tool_call = response.action.tool_calls[0]
            if tool_call.call_id in seen_call_ids:
                return _failed(
                    trace,
                    executions,
                    usage,
                    "duplicate_call_id",
                )
            seen_call_ids.add(tool_call.call_id)
            try:
                parsed_arguments = self._registry.validate_input(
                    tool_call.name,
                    tool_call.arguments,
                )
            except UnknownToolError:
                trace.tool_call(tool_call.call_id, "unknown")
                return _failed(
                    trace,
                    executions,
                    usage,
                    "unknown_tool",
                )
            except ToolInputValidationError:
                trace.tool_call(tool_call.call_id, tool_call.name)
                return _failed(
                    trace,
                    executions,
                    usage,
                    "invalid_arguments",
                )
            normalized_arguments = parsed_arguments.model_dump(mode="json")
            trace.tool_call(
                tool_call.call_id,
                tool_call.name,
                normalized_arguments,
            )
            context = ToolContext(
                catalog=catalog,
                prior_executions=tuple(executions),
                next_result_id=lambda: f"result_{len(executions) + 1:04d}",
            )
            try:
                tool_result = self._registry.invoke(
                    tool_call.name,
                    parsed_arguments,
                    context,
                )
            except ToolOutputValidationError:
                return _failed(
                    trace,
                    executions,
                    usage,
                    "invalid_tool_result",
                )
            except Exception:
                return _failed(
                    trace,
                    executions,
                    usage,
                    "tool_execution_error",
                )

            if tool_result.result_id in seen_result_ids:
                return _failed(
                    trace,
                    executions,
                    usage,
                    "duplicate_result_id",
                )
            seen_result_ids.add(tool_result.result_id)
            executions.append(
                PriorToolExecution(
                    call_id=tool_call.call_id,
                    arguments=normalized_arguments,
                    result=tool_result,
                )
            )
            trace.tool_result(tool_result)

        return _failed(
            trace,
            executions,
            usage,
            "max_steps_exceeded",
        )
