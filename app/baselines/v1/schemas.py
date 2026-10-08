from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.schemas import GroupBy, MetricName, ProductId


class PolicyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class TaskKind(StrEnum):
    PRODUCT_ANOMALY = "product_anomaly"
    CONVERSION_DECLINE = "conversion_decline"
    NEXT_WEEK_PRIORITY = "next_week_priority"
    PRODUCTS_TO_WATCH = "products_to_watch"
    GMV_DIAGNOSIS = "gmv_diagnosis"
    UNSUPPORTED = "unsupported"


class PolicyContext(PolicyModel):
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    as_of_date: date


class DateWindows(PolicyModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date

    @model_validator(mode="after")
    def validate_window_order(self) -> "DateWindows":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.comparison_start_date > self.comparison_end_date:
            raise ValueError(
                "comparison_start_date must be on or before comparison_end_date"
            )
        return self


class ParsedRequest(PolicyModel):
    task: TaskKind
    product_ids: tuple[ProductId, ...]
    windows: DateWindows
    unsupported_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_request_state(self) -> "ParsedRequest":
        if len(self.product_ids) != len(set(self.product_ids)):
            raise ValueError("product_ids must contain unique values")
        if self.task is TaskKind.UNSUPPORTED and self.unsupported_reason is None:
            raise ValueError("unsupported_reason is required for unsupported tasks")
        if self.task is not TaskKind.UNSUPPORTED and self.unsupported_reason is not None:
            raise ValueError("unsupported_reason is only valid for unsupported tasks")
        return self


ToolName = Literal[
    "query_product",
    "query_sales",
    "query_traffic",
    "calculate_metrics",
]


class ToolStep(PolicyModel):
    tool_name: ToolName
    metrics: tuple[MetricName, ...] = ()
    group_by: tuple[GroupBy, ...] = ()

    @model_validator(mode="after")
    def validate_step_state(self) -> "ToolStep":
        is_calculation = self.tool_name == "calculate_metrics"
        if is_calculation and not self.metrics:
            raise ValueError("calculate_metrics requires at least one metric")
        if not is_calculation and (self.metrics or self.group_by):
            raise ValueError("metrics and group_by are only valid for calculate_metrics")
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("metrics must contain unique values")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("group_by must contain unique values")
        return self


class ExecutionPlan(PolicyModel):
    task: TaskKind
    steps: tuple[ToolStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan_state(self) -> "ExecutionPlan":
        if self.task is TaskKind.UNSUPPORTED:
            raise ValueError("unsupported tasks cannot have an execution plan")
        return self
