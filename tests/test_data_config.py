from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data.config import (
    SyntheticDataConfig,
    config_sha256,
    load_data_config,
    load_data_config_with_sha256,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


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
    ("anomaly", "detail"),
    [
        (
            {
                "anomaly_id": "A1",
                "kind": "traffic_drop",
                "product_id": "P001",
                "start_day": 40,
                "end_day": 60,
                "multiplier": 0.5,
            },
            "A1: anomaly outside dataset window",
        ),
        (
            {
                "anomaly_id": "A1",
                "kind": "traffic_drop",
                "product_id": "P001",
                "start_day": 20,
                "end_day": 10,
                "multiplier": 0.5,
            },
            "A1: start_day exceeds end_day",
        ),
    ],
)
def test_rejects_invalid_anomaly_range(
    anomaly: dict[str, object], detail: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(
            minimal_config(days=60, anomalies=[anomaly])
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1, errors
    error = errors[0]
    assert error["loc"] == ()
    assert error["type"] == "value_error"
    assert str(error["ctx"]["error"]) == detail


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("source_label", "External E-commerce Data"),
    ],
)
def test_schema_metadata_only_accepts_the_declared_literals(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(minimal_config(**{field: value}))

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == "literal_error"


@pytest.mark.parametrize("field", ["categories", "regions", "channels"])
@pytest.mark.parametrize("blank", ["", " \t"])
def test_dimension_values_reject_blank_strings(field: str, blank: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(minimal_config(**{field: [blank]}))

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field, 0)
    assert error["type"] == "string_pattern_mismatch"


@pytest.mark.parametrize("field", ["categories", "regions", "channels"])
def test_dimension_values_must_be_unique(field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(
            minimal_config(**{field: ["duplicate", "duplicate"]})
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == "value_error"
    assert str(error["ctx"]["error"]) == f"{field} values must be unique"


@pytest.mark.parametrize(
    "dataset_version",
    ["../v1", "v1/next", "v1.next", "V1", "版本1", "-v1", "v1-"],
)
def test_dataset_version_must_be_a_safe_slug(dataset_version: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(
            minimal_config(dataset_version=dataset_version)
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("dataset_version",)
    assert error["type"] == "string_pattern_mismatch"


@pytest.mark.parametrize("dataset_version", ["v1", "release-2026", "phase-1-v2"])
def test_dataset_version_accepts_safe_slugs(dataset_version: str) -> None:
    config = SyntheticDataConfig.model_validate(
        minimal_config(dataset_version=dataset_version)
    )
    assert config.dataset_version == dataset_version


def anomaly_data(**overrides: object) -> dict[str, object]:
    anomaly: dict[str, object] = {
        "anomaly_id": "A1",
        "kind": "traffic_drop",
        "product_id": "P001",
        "start_day": 0,
        "end_day": 1,
        "multiplier": 0.5,
    }
    anomaly.update(overrides)
    return anomaly


@pytest.mark.parametrize(
    ("anomalies", "detail"),
    [
        (
            [anomaly_data(), anomaly_data(product_id="P002")],
            "anomaly_id values must be unique",
        ),
        (
            [anomaly_data(product_id="P008")],
            "A1: product_id is outside configured product range",
        ),
        (
            [anomaly_data(product_id="P000")],
            "A1: product_id is outside configured product range",
        ),
    ],
)
def test_rejects_ambiguous_anomaly_targets(
    anomalies: list[dict[str, object]], detail: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(minimal_config(anomalies=anomalies))

    error = exc_info.value.errors()[0]
    assert error["loc"] == ()
    assert error["type"] == "value_error"
    assert str(error["ctx"]["error"]) == detail


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf"), float("-inf")])
def test_anomaly_multiplier_must_be_finite(multiplier: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(
            minimal_config(anomalies=[anomaly_data(multiplier=multiplier)])
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("anomalies", 0, "multiplier")
    assert error["type"] == "finite_number"


@pytest.mark.parametrize(
    ("kind", "multiplier"),
    [
        ("sales_drop", 1.0),
        ("traffic_drop", 1.1),
        ("conversion_drop", 1.0),
        ("multi_factor_drop", 2.0),
        ("high_refund", 1.0),
        ("extreme_traffic_spike", 0.5),
        ("missing_traffic", 0.5),
    ],
)
def test_anomaly_multiplier_must_match_kind(kind: str, multiplier: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(
            minimal_config(
                anomalies=[anomaly_data(kind=kind, multiplier=multiplier)]
            )
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("anomalies", 0)
    assert error["type"] == "value_error"


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("seed", -1, "greater_than_equal"),
        ("seed", 2**32, "less_than_equal"),
        ("days", 59, "greater_than_equal"),
        ("days", 367, "less_than_equal"),
        ("product_count", 6, "greater_than_equal"),
        ("product_count", 1000, "less_than_equal"),
        ("customer_count", 19, "greater_than_equal"),
        ("customer_count", 10000, "less_than_equal"),
    ],
)
def test_generation_counts_and_seed_enforce_supported_bounds(
    field: str, value: int, error_type: str
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SyntheticDataConfig.model_validate(minimal_config(**{field: value}))

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == error_type


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 0),
        ("seed", 2**32 - 1),
        ("days", 60),
        ("days", 366),
        ("product_count", 7),
        ("product_count", 999),
        ("customer_count", 20),
        ("customer_count", 9999),
    ],
)
def test_generation_counts_and_seed_accept_supported_boundaries(
    field: str, value: int
) -> None:
    config = SyntheticDataConfig.model_validate(minimal_config(**{field: value}))
    assert getattr(config, field) == value


def test_load_with_sha256_reads_config_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    contents = CONFIG.read_bytes()
    config_path.write_bytes(contents)
    original_read_bytes = Path.read_bytes
    reads = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    config, digest = load_data_config_with_sha256(config_path)

    assert reads == 1
    assert config.dataset_version == "v1"
    assert digest == sha256(contents).hexdigest()
