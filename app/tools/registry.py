from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolInputValidationError,
    ToolOutputValidationError,
    UnknownToolError,
)
from app.tools.schemas import (
    AnyToolResult,
    CalculateMetricsInput,
    CalculateMetricsResult,
    QueryMarketingInput,
    QueryMarketingResult,
    QueryProductInput,
    QueryProductResult,
    QuerySalesInput,
    QuerySalesResult,
    QueryTrafficInput,
    QueryTrafficResult,
)


def _not_implemented(_arguments: BaseModel, _context: ToolContext) -> AnyToolResult:
    raise NotImplementedError("tool handler is implemented in a later Phase 2 task")


def _validate_metrics_columns(
    arguments: CalculateMetricsInput,
    result: CalculateMetricsResult,
) -> None:
    expected = [
        *(dimension.value for dimension in arguments.group_by),
        *(metric.value for metric in arguments.metrics),
    ]
    if result.columns != expected:
        raise ValueError(f"columns must exactly equal {expected!r}")


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition[Any, Any]] = {}

    def register(self, definition: ToolDefinition[Any, Any]) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition[Any, Any]:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_model.model_json_schema(),
                },
            }
            for definition in (self._definitions[name] for name in self.names())
        ]

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> AnyToolResult:
        definition = self.get(name)
        try:
            parsed = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputValidationError(f"invalid arguments for {name}") from exc

        result = definition.handler(parsed, context)
        try:
            validated = definition.output_model.model_validate(
                result.model_dump(mode="python", exclude_unset=True),
                strict=True,
            )
            if validated.tool_name != name:
                raise ValueError("handler returned mismatched tool_name")
            if definition.output_validator is not None:
                definition.output_validator(parsed, validated)
            return validated
        except Exception as exc:
            raise ToolOutputValidationError(f"invalid result from {name}") from exc


def _definition(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    output_validator: Callable[[Any, Any], None] | None = None,
) -> ToolDefinition[Any, Any]:
    return ToolDefinition(
        name=name,
        description=description,
        input_model=input_model,
        output_model=output_model,
        handler=_not_implemented,
        output_validator=output_validator,
    )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    definitions = (
        _definition(
            name="query_product",
            description="Return product attributes for the requested product IDs.",
            input_model=QueryProductInput,
            output_model=QueryProductResult,
        ),
        _definition(
            name="query_sales",
            description="Return bounded sales aggregates for current and comparison windows.",
            input_model=QuerySalesInput,
            output_model=QuerySalesResult,
        ),
        _definition(
            name="query_traffic",
            description="Return bounded traffic aggregates for current and comparison windows.",
            input_model=QueryTrafficInput,
            output_model=QueryTrafficResult,
        ),
        _definition(
            name="query_marketing",
            description="Return bounded marketing aggregates for current and comparison windows.",
            input_model=QueryMarketingInput,
            output_model=QueryMarketingResult,
        ),
        _definition(
            name="calculate_metrics",
            description="Calculate requested metrics from prior tool results in this run.",
            input_model=CalculateMetricsInput,
            output_model=CalculateMetricsResult,
            output_validator=_validate_metrics_columns,
        ),
    )
    for definition in definitions:
        registry.register(definition)
    return registry
