from app.llm.schemas import AdapterRequest, AssistantAction, ToolCall
from app.tools.schemas import GroupBy, PriorToolExecution

from .config import METRICS, TOOL_PATHS
from .parsing import parse_request
from .schemas import ExecutionPlan, ParsedRequest, PolicyContext, TaskKind, ToolStep

_UNSUPPORTED_ANSWER = (
    "当前仅支持商品经营、转化、关注清单、下周重点和 GMV 诊断。"
)
_PRODUCT_GROUPED_TASKS = {
    TaskKind.PRODUCT_ANOMALY,
    TaskKind.NEXT_WEEK_PRIORITY,
    TaskKind.PRODUCTS_TO_WATCH,
}


class BaselinePolicyProtocolError(RuntimeError):
    pass


class BaselinePolicyV1:
    def __init__(self, context: PolicyContext) -> None:
        self._context = context
        self._parsed: ParsedRequest | None = None
        self._plan: ExecutionPlan | None = None

    def __call__(self, request: AdapterRequest) -> AssistantAction:
        if self._parsed is None:
            self._parsed = parse_request(request.user_input, self._context)
            if self._parsed.task is not TaskKind.UNSUPPORTED:
                self._plan = self._build_plan(self._parsed)

        actual_names = tuple(
            execution.result.tool_name
            for execution in request.prior_tool_executions
        )
        if self._plan is None:
            if actual_names:
                raise BaselinePolicyProtocolError(
                    "unexpected prior tool sequence"
                )
            return AssistantAction(final_answer=_UNSUPPORTED_ANSWER)

        expected_names = tuple(step.tool_name for step in self._plan.steps)
        if actual_names != expected_names[: len(actual_names)]:
            raise BaselinePolicyProtocolError("unexpected prior tool sequence")
        if len(actual_names) < len(expected_names):
            index = len(actual_names)
            return self._action_for(self._plan.steps[index], index)
        return AssistantAction(
            final_answer=self._render_final_answer(
                self._parsed,
                request.prior_tool_executions,
            )
        )

    @staticmethod
    def _build_plan(parsed: ParsedRequest) -> ExecutionPlan:
        group_by = ()
        if (
            parsed.task in _PRODUCT_GROUPED_TASKS
            or (
                parsed.task is TaskKind.CONVERSION_DECLINE
                and parsed.product_ids
            )
        ):
            group_by = (GroupBy.PRODUCT_ID,)

        steps = []
        for tool_name in TOOL_PATHS[parsed.task]:
            if tool_name == "query_product" and not parsed.product_ids:
                continue
            if tool_name == "calculate_metrics":
                steps.append(
                    ToolStep(
                        tool_name=tool_name,
                        metrics=METRICS[parsed.task],
                        group_by=group_by,
                    )
                )
            else:
                steps.append(ToolStep(tool_name=tool_name))
        return ExecutionPlan(task=parsed.task, steps=tuple(steps))

    def _action_for(self, step: ToolStep, index: int) -> AssistantAction:
        assert self._parsed is not None
        arguments: dict[str, object]
        if step.tool_name == "query_product":
            arguments = {"product_ids": list(self._parsed.product_ids)}
        elif step.tool_name == "calculate_metrics":
            arguments = {
                "metrics": [metric.value for metric in step.metrics],
                "group_by": [dimension.value for dimension in step.group_by],
            }
        else:
            windows = self._parsed.windows.model_dump(mode="json")
            arguments = {
                **windows,
                "product_ids": list(self._parsed.product_ids),
            }
            if step.tool_name == "query_sales":
                arguments["include_refunds"] = True
            else:
                arguments["include_missing"] = True

        return AssistantAction(
            tool_calls=[
                ToolCall(
                    call_id=f"baseline_v1_call_{index + 1:04d}",
                    name=step.tool_name,
                    arguments=arguments,
                )
            ]
        )

    @staticmethod
    def _render_final_answer(
        parsed: ParsedRequest,
        executions: list[PriorToolExecution],
    ) -> str:
        return (
            f"{parsed.task.value} 工具链执行完成，"
            f"共获得 {len(executions)} 项工具结果。"
        )
