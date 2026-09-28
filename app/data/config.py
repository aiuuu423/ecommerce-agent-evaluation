from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.data.schemas import ProductId

AnomalyKind = Literal[
    "sales_drop",
    "traffic_drop",
    "conversion_drop",
    "high_refund",
    "missing_traffic",
    "extreme_traffic_spike",
    "multi_factor_drop",
]

DROP_ANOMALY_KINDS = {
    "sales_drop",
    "traffic_drop",
    "conversion_drop",
    "multi_factor_drop",
}
INCREASE_ANOMALY_KINDS = {"high_refund", "extreme_traffic_spike"}
NonBlankString = Annotated[str, Field(pattern=r"\S")]


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str
    kind: AnomalyKind
    product_id: ProductId
    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    multiplier: float = Field(gt=0, le=10, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_multiplier_for_kind(self) -> "AnomalyConfig":
        if self.kind in DROP_ANOMALY_KINDS and self.multiplier >= 1:
            raise ValueError(f"{self.kind}: multiplier must be lower than 1")
        if self.kind in INCREASE_ANOMALY_KINDS and self.multiplier <= 1:
            raise ValueError(f"{self.kind}: multiplier must be greater than 1")
        if self.kind == "missing_traffic" and self.multiplier != 1:
            raise ValueError("missing_traffic: multiplier must equal 1")
        return self


class SyntheticDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    schema_version: Literal["1.0"] = "1.0"
    source_label: Literal["Synthetic E-commerce Data"] = "Synthetic E-commerce Data"
    seed: int = Field(ge=0, le=2**32 - 1)
    start_date: date
    days: int = Field(ge=60, le=366)
    product_count: int = Field(ge=7, le=999)
    customer_count: int = Field(ge=20, le=9999)
    categories: list[NonBlankString] = Field(min_length=1)
    regions: list[NonBlankString] = Field(min_length=1)
    channels: list[NonBlankString] = Field(min_length=1)
    anomalies: list[AnomalyConfig]

    @field_validator("categories", "regions", "channels")
    @classmethod
    def validate_unique_dimensions(
        cls, values: list[str], info: ValidationInfo
    ) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError(f"{info.field_name} values must be unique")
        return values

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "SyntheticDataConfig":
        dimension_limits = (
            ("categories", self.categories, "product_count", self.product_count),
            ("regions", self.regions, "customer_count", self.customer_count),
            ("channels", self.channels, "customer_count", self.customer_count),
        )
        for field, values, count_field, entity_count in dimension_limits:
            if len(values) > entity_count:
                raise ValueError(f"{field} has more values than {count_field}")

        anomaly_ids = [anomaly.anomaly_id for anomaly in self.anomalies]
        if len(anomaly_ids) != len(set(anomaly_ids)):
            raise ValueError("anomaly_id values must be unique")

        for anomaly in self.anomalies:
            if anomaly.start_day > anomaly.end_day:
                raise ValueError(f"{anomaly.anomaly_id}: start_day exceeds end_day")
            if anomaly.end_day >= self.days:
                raise ValueError(f"{anomaly.anomaly_id}: anomaly outside dataset window")
            product_number = int(anomaly.product_id.removeprefix("P"))
            if not 1 <= product_number <= self.product_count:
                raise ValueError(
                    f"{anomaly.anomaly_id}: product_id is outside configured product range"
                )
        return self


def load_data_config(path: Path | str) -> SyntheticDataConfig:
    config, _ = load_data_config_with_sha256(path)
    return config


def load_data_config_with_sha256(
    path: Path | str,
) -> tuple[SyntheticDataConfig, str]:
    contents = Path(path).read_bytes()
    payload = yaml.safe_load(contents.decode("utf-8"))
    config = SyntheticDataConfig.model_validate(payload)
    return config, sha256(contents).hexdigest()


def config_sha256(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
