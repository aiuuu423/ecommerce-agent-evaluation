from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Money = Annotated[Decimal, Field(decimal_places=2)]
NumericTolerance = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ProductId = Annotated[str, Field(pattern=r"^P[0-9]{3}$")]
CustomerId = Annotated[str, Field(pattern=r"^C[0-9]{4}$")]
OrderId = Annotated[
    str,
    Field(pattern=r"^O[0-9]{8}-P[0-9]{3}-[0-9]{6}$"),
]
CampaignId = Annotated[str, Field(pattern=r"^M[0-9]{3}$")]


class DatasetRowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductRow(DatasetRowModel):
    product_id: ProductId
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: Money = Field(gt=0)
    cost: Money = Field(gt=0)
    launch_date: date

    @model_validator(mode="after")
    def validate_margin(self) -> "ProductRow":
        if self.cost >= self.price:
            raise ValueError("cost must be lower than price")
        return self


class CustomerRow(DatasetRowModel):
    customer_id: CustomerId
    is_new_customer: bool
    region: str = Field(min_length=1)
    channel: str = Field(min_length=1)


class TrafficRow(DatasetRowModel):
    date: date
    product_id: ProductId
    impressions: int | None = Field(default=None, ge=0)
    clicks: int | None = Field(default=None, ge=0)
    visits: int | None = Field(default=None, ge=0)
    is_missing: bool

    @model_validator(mode="after")
    def validate_funnel(self) -> "TrafficRow":
        values = (self.impressions, self.clicks, self.visits)
        if self.is_missing:
            if any(value is not None for value in values):
                raise ValueError("missing traffic rows must use null metrics")
            return self
        if any(value is None for value in values):
            raise ValueError("non-missing traffic rows require all metrics")
        if not self.impressions >= self.clicks:
            raise ValueError("clicks cannot exceed impressions")
        if self.visits < self.clicks:
            raise ValueError("visits cannot be lower than clicks")
        return self


class MarketingRow(DatasetRowModel):
    date: date
    product_id: ProductId
    campaign_id: CampaignId
    spend: Money = Field(ge=0)


class OrderRow(DatasetRowModel):
    order_id: OrderId
    product_id: ProductId
    customer_id: CustomerId
    order_date: date
    quantity: int = Field(ge=1, le=20)
    unit_price: Money = Field(gt=0)
    revenue: Money = Field(gt=0)
    is_refund: bool
    status: Literal["paid", "refunded"]

    @model_validator(mode="after")
    def validate_order(self) -> "OrderRow":
        if self.revenue != self.unit_price * self.quantity:
            raise ValueError("revenue must equal unit_price * quantity")
        if self.is_refund != (self.status == "refunded"):
            raise ValueError("refund flag and status must agree")
        return self


class BusinessTask(StrEnum):
    GMV_DIAGNOSIS = "gmv_diagnosis"
    PRODUCT_ANOMALY = "product_anomaly"
    CONVERSION_DECLINE = "conversion_decline"
    PRODUCTS_TO_WATCH = "products_to_watch"
    NEXT_WEEK_PRIORITY = "next_week_priority"


class Capability(StrEnum):
    BASIC_QUERY = "basic_query"
    METRIC_CALCULATION = "metric_calculation"
    TOOL_SELECTION = "tool_selection"
    PARAMETER_SELECTION = "parameter_selection"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    RECOMMENDATION = "recommendation"
    DATA_INSUFFICIENCY = "data_insufficiency"
    ADVERSARIAL_DISTRACTOR = "adversarial_distractor"


REQUIRED_SUCCESS_CRITERIA = frozenset(
    {
        "correct_tool",
        "valid_parameters",
        "correct_core_facts",
        "no_critical_unsupported_claim",
        "required_output_complete",
    }
)
GoldValue = float | int | str | None


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedToolCall(EvaluationModel):
    name: str = Field(min_length=1)
    parameters: dict[str, Any]


class GoldEvidence(EvaluationModel):
    evidence_id: str = Field(pattern=r"^EV_CASE_[0-9]{3}_[0-9]{2}$")
    source: str = Field(min_length=1)
    dimensions: dict[str, GoldValue]
    metrics: dict[str, GoldValue] = Field(min_length=1)


class EvaluationCase(EvaluationModel):
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    case_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    business_task: BusinessTask
    primary_capability: Capability
    capability_tags: list[Capability] = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["development", "holdout"]
    user_input: str = Field(min_length=5)
    expected_tool_calls: list[ExpectedToolCall] = Field(min_length=1)
    allowed_alternatives: list[list[ExpectedToolCall]]
    gold_metrics: dict[str, GoldValue]
    gold_evidence: list[GoldEvidence] = Field(min_length=1)
    reference_answer: str = Field(min_length=5)
    expected_behavior: list[str] = Field(min_length=1)
    success_criteria: list[str]
    numeric_tolerances: dict[str, NumericTolerance]
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_case_contract(self) -> "EvaluationCase":
        tags = self.capability_tags
        if len(tags) != len(set(tags)) or self.primary_capability not in tags:
            raise ValueError(
                "capability tags must be unique and include primary capability"
            )

        criteria = self.success_criteria
        if len(criteria) != len(set(criteria)) or set(criteria) != REQUIRED_SUCCESS_CRITERIA:
            raise ValueError(
                "success criteria must contain each required success gate exactly once"
            )

        expected_evidence_prefix = f"EV_{self.case_id}_"
        evidence_ids = [evidence.evidence_id for evidence in self.gold_evidence]
        if any(
            not evidence_id.startswith(expected_evidence_prefix)
            for evidence_id in evidence_ids
        ):
            raise ValueError("evidence_id must belong to case_id")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("gold evidence ids must be unique")

        evidence_metrics = {
            (key, value)
            for evidence in self.gold_evidence
            for key, value in evidence.metrics.items()
        }
        unsupported_metrics = {
            key
            for key, value in self.gold_metrics.items()
            if (key, value) not in evidence_metrics
        }
        if unsupported_metrics:
            raise ValueError("gold metrics must be backed by evidence")

        numeric_metrics = {
            key
            for key, value in self.gold_metrics.items()
            if type(value) in {int, float}
        }
        unknown_tolerances = set(self.numeric_tolerances) - numeric_metrics
        if unknown_tolerances:
            raise ValueError("tolerances must reference numeric gold metrics")
        float_metrics = {
            key for key, value in self.gold_metrics.items() if type(value) is float
        }
        if not float_metrics.issubset(self.numeric_tolerances):
            raise ValueError("float gold metrics require tolerances")
        return self
