from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str
    kind: AnomalyKind
    product_id: ProductId
    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    multiplier: float = Field(gt=0)


class SyntheticDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    schema_version: str = "1.0"
    source_label: str = "Synthetic E-commerce Data"
    seed: int
    start_date: date
    days: int = Field(ge=60)
    product_count: int = Field(ge=7)
    customer_count: int = Field(ge=20)
    categories: list[str] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    channels: list[str] = Field(min_length=1)
    anomalies: list[AnomalyConfig]

    @model_validator(mode="after")
    def validate_anomaly_ranges(self) -> "SyntheticDataConfig":
        for anomaly in self.anomalies:
            if anomaly.start_day > anomaly.end_day:
                raise ValueError(f"{anomaly.anomaly_id}: start_day exceeds end_day")
            if anomaly.end_day >= self.days:
                raise ValueError(f"{anomaly.anomaly_id}: anomaly outside dataset window")
        return self


def load_data_config(path: Path | str) -> SyntheticDataConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SyntheticDataConfig.model_validate(payload)


def config_sha256(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
