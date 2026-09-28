from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data.config import SyntheticDataConfig, config_sha256, load_data_config

CONFIG = Path("configs/data/synthetic_v1.yaml")


def minimal_config(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_version": "v1",
        "seed": 1,
        "start_date": "2026-01-01",
        "days": 60,
        "product_count": 7,
        "customer_count": 20,
        "categories": ["home"],
        "regions": ["north"],
        "channels": ["organic"],
        "anomalies": [],
    }
    payload.update(overrides)
    return payload


def test_loads_versioned_config() -> None:
    config = load_data_config(CONFIG)

    assert config.dataset_version == "v1"
    assert config.seed == 20260928
    assert config.days == 120
    assert config.product_count == 40
    assert len(config.anomalies) == 7


@pytest.mark.parametrize(
    "anomaly",
    [
        {
            "anomaly_id": "A1",
            "kind": "traffic_drop",
            "product_id": "P001",
            "start_day": 40,
            "end_day": 45,
            "multiplier": 0.5,
        },
        {
            "anomaly_id": "A1",
            "kind": "traffic_drop",
            "product_id": "P001",
            "start_day": 20,
            "end_day": 10,
            "multiplier": 0.5,
        },
    ],
)
def test_rejects_invalid_anomaly_range(anomaly: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SyntheticDataConfig.model_validate(
            minimal_config(days=30, anomalies=[anomaly])
        )


def test_anomaly_product_id_uses_dataset_schema_contract() -> None:
    anomaly = {
        "anomaly_id": "A1",
        "kind": "traffic_drop",
        "product_id": "P１２３",
        "start_day": 0,
        "end_day": 1,
        "multiplier": 0.5,
    }

    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(minimal_config(anomalies=[anomaly]))

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("anomalies", 0, "product_id")
    assert error["type"] == "string_pattern_mismatch"


def test_config_sha256_hashes_the_versioned_yaml_bytes() -> None:
    assert config_sha256(CONFIG) == sha256(CONFIG.read_bytes()).hexdigest()
