import json
from datetime import date
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

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
JsonValue = TypeAliasType(
    "JsonValue",
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"],
)
GoldValue = int | float | str | None


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
                raise ValueError(f"{path}: JSON object key {key!r} must be a string")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: value of type {type(value).__name__} is not valid JSON")


def _validate_gold_values(value: object, field_name: str) -> object:
    if type(value) is not dict:
        return value
    for key, item in value.items():
        if type(item) is bool:
            raise ValueError(f"{field_name}.{key}: bool is not a valid gold value")
        if type(item) is float and not isfinite(item):
            raise ValueError(f"{field_name}.{key}: numeric gold value must be finite")
        if type(item) is str:
            try:
                float(item)
            except ValueError:
                continue
            raise ValueError(
                f"{field_name}.{key}: numeric strings are not valid gold values"
            )
        if item is not None and type(item) not in {int, float, str}:
            raise ValueError(
                f"{field_name}.{key}: unsupported gold value type "
                f"{type(item).__name__}"
            )
    return value


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedToolCall(EvaluationModel):
    name: str = Field(min_length=1)
    parameters: dict[str, JsonValue]

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters_json(cls, value: object) -> object:
        _validate_json_value(value, "parameters")
        return value


class GoldEvidence(EvaluationModel):
    evidence_id: str = Field(pattern=r"^EV_CASE_[0-9]{3}_[0-9]{2}$")
    source: str = Field(min_length=1)
    dimensions: dict[str, JsonValue]
    metrics: dict[str, GoldValue] = Field(min_length=1)

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions_json(cls, value: object) -> object:
        _validate_json_value(value, "dimensions")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metric_values(cls, value: object) -> object:
        return _validate_gold_values(value, "metrics")


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
    gold_metrics: dict[str, GoldValue] = Field(min_length=1)
    gold_metric_evidence: dict[str, str]
    gold_evidence: list[GoldEvidence] = Field(min_length=1)
    reference_answer: str = Field(min_length=5)
    expected_behavior: list[str] = Field(min_length=1)
    success_criteria: list[str]
    numeric_tolerances: dict[str, NumericTolerance]
    metadata: dict[str, JsonValue]

    @field_validator("gold_metrics", mode="before")
    @classmethod
    def validate_gold_metric_values(cls, value: object) -> object:
        return _validate_gold_values(value, "gold_metrics")

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata_json(cls, value: object) -> object:
        _validate_json_value(value, "metadata")
        return value

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

        primary_path = self._tool_path_signature(self.expected_tool_calls)
        alternative_paths: set[str] = set()
        for index, alternative in enumerate(self.allowed_alternatives):
            if not alternative:
                raise ValueError(f"allowed_alternatives[{index}] must not be empty")
            signature = self._tool_path_signature(alternative)
            if signature == primary_path:
                raise ValueError(
                    f"allowed_alternatives[{index}] must differ from expected_tool_calls"
                )
            if signature in alternative_paths:
                raise ValueError(
                    f"allowed_alternatives[{index}] duplicates an earlier alternative"
                )
            alternative_paths.add(signature)

        expected_evidence_prefix = f"EV_{self.case_id}_"
        evidence_ids = [evidence.evidence_id for evidence in self.gold_evidence]
        if any(
            not evidence_id.startswith(expected_evidence_prefix)
            for evidence_id in evidence_ids
        ):
            wrong_ids = sorted(
                evidence_id
                for evidence_id in evidence_ids
                if not evidence_id.startswith(expected_evidence_prefix)
            )
            raise ValueError(
                f"evidence_id must belong to case_id: {', '.join(wrong_ids)}"
            )
        if len(evidence_ids) != len(set(evidence_ids)):
            duplicate_ids = sorted(
                {
                    evidence_id
                    for evidence_id in evidence_ids
                    if evidence_ids.count(evidence_id) > 1
                }
            )
            raise ValueError(
                f"gold evidence ids must be unique: {', '.join(duplicate_ids)}"
            )

        dimension_rows: dict[tuple[str, str], GoldEvidence] = {}
        for evidence in self.gold_evidence:
            dimension_signature = json.dumps(
                evidence.dimensions,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            row_key = (evidence.source, dimension_signature)
            if previous := dimension_rows.get(row_key):
                dimension_keys = ", ".join(sorted(evidence.dimensions)) or "<none>"
                raise ValueError(
                    "gold evidence dimension rows must be unique: "
                    f"{previous.evidence_id}, {evidence.evidence_id}; "
                    f"dimensions={dimension_keys}"
                )
            dimension_rows[row_key] = evidence

        gold_metric_names = set(self.gold_metrics)
        mapped_metric_names = set(self.gold_metric_evidence)
        missing_mappings = gold_metric_names - mapped_metric_names
        if missing_mappings:
            raise ValueError(
                "gold metric evidence mapping missing keys: "
                f"{', '.join(sorted(missing_mappings))}"
            )
        unknown_mappings = mapped_metric_names - gold_metric_names
        if unknown_mappings:
            raise ValueError(
                "gold metric evidence mapping has unknown keys: "
                f"{', '.join(sorted(unknown_mappings))}"
            )

        evidence_by_id = {
            evidence.evidence_id: evidence for evidence in self.gold_evidence
        }
        for metric_name, evidence_id in self.gold_metric_evidence.items():
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError(
                    f"{metric_name}: mapped evidence_id {evidence_id} does not exist"
                )
            if metric_name not in evidence.metrics:
                raise ValueError(
                    f"{metric_name}: mapped evidence_id {evidence_id} "
                    "does not contain the metric"
                )
            gold_value = self.gold_metrics[metric_name]
            evidence_value = evidence.metrics[metric_name]
            if type(gold_value) is not type(evidence_value) or gold_value != evidence_value:
                raise ValueError(
                    f"{metric_name}: value does not match mapped evidence_id {evidence_id}"
                )

        numeric_metrics = {
            key
            for key, value in self.gold_metrics.items()
            if type(value) in {int, float}
        }
        unknown_tolerances = set(self.numeric_tolerances) - numeric_metrics
        if unknown_tolerances:
            raise ValueError(
                "tolerances must reference numeric gold metrics: "
                f"{', '.join(sorted(unknown_tolerances))}"
            )
        float_metrics = {
            key for key, value in self.gold_metrics.items() if type(value) is float
        }
        if not float_metrics.issubset(self.numeric_tolerances):
            missing_tolerances = float_metrics - set(self.numeric_tolerances)
            raise ValueError(
                "float gold metrics require tolerances: "
                f"{', '.join(sorted(missing_tolerances))}"
            )
        return self

    @staticmethod
    def _tool_path_signature(tool_calls: list[ExpectedToolCall]) -> str:
        return json.dumps(
            [tool_call.model_dump(mode="json") for tool_call in tool_calls],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
