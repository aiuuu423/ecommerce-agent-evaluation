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
    product_ids: list[ProductId]
    windows: DateWindows
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def validate_product_ids(self) -> "ParsedRequest":
        if len(self.product_ids) != len(set(self.product_ids)):
            raise ValueError("product_ids must contain unique values")
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


class ExecutionPlan(PolicyModel):
    task: TaskKind
    steps: tuple[ToolStep, ...]
