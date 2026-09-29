from datetime import date
from enum import StrEnum
from math import isfinite
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

JsonValue = TypeAliasType(
    "JsonValue",
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"],
)
ProductId = Annotated[str, Field(pattern=r"^P[0-9]{3}$")]
MAX_QUERY_WINDOW_DAYS = 366


def _validate_json_value(value: object, path: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path}: JSON number must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path}: JSON object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: value of type {type(value).__name__} is not valid JSON")


def _require_unique(values: list[object], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")


class ToolModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class Period(ToolModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def valid_order(self) -> "Period":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class QueryProductInput(ToolModel):
    product_ids: list[ProductId] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_product_ids(self) -> "QueryProductInput":
        _require_unique(self.product_ids, "product_ids")
        return self


class WindowQueryInput(ToolModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date
    product_ids: list[ProductId]

    @model_validator(mode="after")
    def valid_windows_and_products(self) -> "WindowQueryInput":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if (self.end_date - self.start_date).days + 1 > MAX_QUERY_WINDOW_DAYS:
            raise ValueError("current window must be at most 366 days")
        if self.comparison_start_date > self.comparison_end_date:
            raise ValueError(
                "comparison_start_date must be on or before comparison_end_date"
            )
        if (
            self.comparison_end_date - self.comparison_start_date
        ).days + 1 > MAX_QUERY_WINDOW_DAYS:
            raise ValueError("comparison window must be at most 366 days")
        _require_unique(self.product_ids, "product_ids")
        return self


class QuerySalesInput(WindowQueryInput):
    include_refunds: bool


class QueryTrafficInput(WindowQueryInput):
    include_missing: bool


class QueryMarketingInput(WindowQueryInput):
    product_ids: list[ProductId] = Field(default_factory=list)


class MetricName(StrEnum):
    CURRENT_GMV = "current_gmv"
    PREVIOUS_GMV = "previous_gmv"
    GMV_CHANGE_RATE = "gmv_change_rate"
    CURRENT_ORDERS = "current_orders"
    PREVIOUS_ORDERS = "previous_orders"
    CURRENT_AOV = "current_aov"
    PREVIOUS_AOV = "previous_aov"
    AOV_CHANGE_RATE = "aov_change_rate"
    CURRENT_IMPRESSIONS = "current_impressions"
    PREVIOUS_IMPRESSIONS = "previous_impressions"
    CURRENT_CLICKS = "current_clicks"
    PREVIOUS_CLICKS = "previous_clicks"
    CURRENT_VISITS = "current_visits"
    PREVIOUS_VISITS = "previous_visits"
    CURRENT_CVR = "current_cvr"
    PREVIOUS_CVR = "previous_cvr"
    CVR_CHANGE = "cvr_change"
    CVR_CHANGE_RATE = "cvr_change_rate"
    CURRENT_CTR = "current_ctr"
    PREVIOUS_CTR = "previous_ctr"
    TRAFFIC_CHANGE_RATE = "traffic_change_rate"
    CURRENT_OBSERVED_DAYS = "current_observed_days"
    PREVIOUS_OBSERVED_DAYS = "previous_observed_days"
    CURRENT_SPEND = "current_spend"
    PREVIOUS_SPEND = "previous_spend"
    CURRENT_ROAS = "current_roas"
    PREVIOUS_ROAS = "previous_roas"
    CURRENT_REFUND_RATE = "current_refund_rate"
    REFUND_RATE = "refund_rate"
    EVIDENCE_VALUE = "evidence_value"


class GroupBy(StrEnum):
    PRODUCT_ID = "product_id"
    CATEGORY = "category"
    REGION = "region"
    CHANNEL = "channel"


class CalculateMetricsInput(ToolModel):
    metrics: list[MetricName] = Field(min_length=1)
    group_by: list[GroupBy]

    @model_validator(mode="after")
    def unique_projection(self) -> "CalculateMetricsInput":
        _require_unique(self.metrics, "metrics")
        _require_unique(self.group_by, "group_by")
        return self


RowT = TypeVar("RowT", bound=ToolModel)


class ToolResult(ToolModel, Generic[RowT]):
    result_id: str = Field(pattern=r"^result_[0-9]{4}$")
    tool_name: str
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    source_label: Literal["Synthetic E-commerce Data"]
    columns: list[str]
    rows: list[RowT]
    row_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> "ToolResult[RowT]":
        if self.row_count != len(self.rows):
            raise ValueError("row_count must equal len(rows)")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("columns must be unique")
        column_set = set(self.columns)
        for row in self.rows:
            row_keys = set(row.model_dump(exclude_unset=True))
            if row_keys != column_set:
                raise ValueError("every row must match columns exactly")
        return self


class QueryProductRow(ToolModel):
    product_id: ProductId
    product_name: str
    category: str
    price: float
    cost: float
    launch_date: date


class QuerySalesRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    region: str
    channel: str
    gmv: float
    orders: int = Field(ge=0)
    units: int = Field(ge=0)
    refund_orders: int = Field(ge=0)


class QueryTrafficRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    visits: int = Field(ge=0)
    observed_days: int = Field(ge=0)
    missing_days: int = Field(ge=0)


class QueryMarketingRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    campaign_id: str
    spend: float = Field(ge=0)


class CalculateMetricsRow(ToolModel):
    product_id: ProductId | None = None
    category: str | None = None
    region: str | None = None
    channel: str | None = None
    current_gmv: float | None = None
    previous_gmv: float | None = None
    gmv_change_rate: float | None = None
    current_orders: int | None = Field(default=None, ge=0)
    previous_orders: int | None = Field(default=None, ge=0)
    current_aov: float | None = None
    previous_aov: float | None = None
    aov_change_rate: float | None = None
    current_impressions: int | None = Field(default=None, ge=0)
    previous_impressions: int | None = Field(default=None, ge=0)
    current_clicks: int | None = Field(default=None, ge=0)
    previous_clicks: int | None = Field(default=None, ge=0)
    current_visits: int | None = Field(default=None, ge=0)
    previous_visits: int | None = Field(default=None, ge=0)
    current_cvr: float | None = None
    previous_cvr: float | None = None
    cvr_change: float | None = None
    cvr_change_rate: float | None = None
    current_ctr: float | None = None
    previous_ctr: float | None = None
    traffic_change_rate: float | None = None
    current_observed_days: int | None = Field(default=None, ge=0)
    previous_observed_days: int | None = Field(default=None, ge=0)
    current_spend: float | None = Field(default=None, ge=0)
    previous_spend: float | None = Field(default=None, ge=0)
    current_roas: float | None = None
    previous_roas: float | None = None
    current_refund_rate: float | None = None
    refund_rate: float | None = None
    evidence_value: float | None = None


PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "price",
    "cost",
    "launch_date",
]
SALES_COLUMNS = [
    "period",
    "product_id",
    "category",
    "region",
    "channel",
    "gmv",
    "orders",
    "units",
    "refund_orders",
]
TRAFFIC_COLUMNS = [
    "period",
    "product_id",
    "category",
    "impressions",
    "clicks",
    "visits",
    "observed_days",
    "missing_days",
]
MARKETING_COLUMNS = [
    "period",
    "product_id",
    "category",
    "campaign_id",
    "spend",
]


def _validate_fixed_columns(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise ValueError(f"columns must exactly equal {expected!r}")


class QueryProductResult(ToolResult[QueryProductRow]):
    tool_name: Literal["query_product"]

    @model_validator(mode="after")
    def fixed_columns(self) -> "QueryProductResult":
        _validate_fixed_columns(self.columns, PRODUCT_COLUMNS)
        return self


class QuerySalesResult(ToolResult[QuerySalesRow]):
    tool_name: Literal["query_sales"]

    @model_validator(mode="after")
    def fixed_columns(self) -> "QuerySalesResult":
        _validate_fixed_columns(self.columns, SALES_COLUMNS)
        return self


class QueryTrafficResult(ToolResult[QueryTrafficRow]):
    tool_name: Literal["query_traffic"]

    @model_validator(mode="after")
    def fixed_columns(self) -> "QueryTrafficResult":
        _validate_fixed_columns(self.columns, TRAFFIC_COLUMNS)
        return self


class QueryMarketingResult(ToolResult[QueryMarketingRow]):
    tool_name: Literal["query_marketing"]

    @model_validator(mode="after")
    def fixed_columns(self) -> "QueryMarketingResult":
        _validate_fixed_columns(self.columns, MARKETING_COLUMNS)
        return self


class CalculateMetricsResult(ToolResult[CalculateMetricsRow]):
    tool_name: Literal["calculate_metrics"]


AnyToolResult = (
    QueryProductResult
    | QuerySalesResult
    | QueryTrafficResult
    | QueryMarketingResult
    | CalculateMetricsResult
)


def canonical_tool_result_payload(result: AnyToolResult) -> dict[str, JsonValue]:
    row_payloads = [
        row.model_dump(mode="json", exclude_unset=True)
        for row in result.rows
    ]
    return {
        "result_id": result.result_id,
        "tool_name": result.tool_name,
        "dataset_id": result.dataset_id,
        "source_label": result.source_label,
        "columns": list(result.columns),
        "rows": [
            {column: row[column] for column in result.columns}
            for row in row_payloads
        ],
        "row_count": result.row_count,
        "warnings": list(result.warnings),
    }


class PriorToolExecution(ToolModel):
    call_id: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    result: AnyToolResult

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments_json(cls, value: object) -> object:
        _validate_json_value(value, "arguments")
        return value
