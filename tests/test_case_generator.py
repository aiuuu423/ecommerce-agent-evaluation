from copy import deepcopy
from math import inf, nan

import pytest
from pydantic import ValidationError

from app.data.schemas import (
    BusinessTask,
    Capability,
    EvaluationCase,
    ExpectedToolCall,
    GoldEvidence,
)

REQUIRED_SUCCESS_CRITERIA = {
    "correct_tool",
    "valid_parameters",
    "correct_core_facts",
    "no_critical_unsupported_claim",
    "required_output_complete",
}


def valid_case() -> dict[str, object]:
    return {
        "case_id": "CASE_001",
        "case_version": "1.0",
        "dataset_version": "v1",
        "dataset_id": "0123456789abcdef",
        "generator_config_hash": "a" * 64,
        "business_task": "gmv_diagnosis",
        "primary_capability": "metric_calculation",
        "capability_tags": ["metric_calculation", "root_cause_analysis"],
        "difficulty": "medium",
        "split": "development",
        "user_input": "最近 30 天 GMV 为什么下降？",
        "expected_tool_calls": [
            {"name": "query_sales", "parameters": {"window_days": 30}}
        ],
        "allowed_alternatives": [],
        "gold_metrics": {"gmv_change_rate": -0.12},
        "gold_metric_evidence": {"gmv_change_rate": "EV_GMV_DIAGNOSIS_001"},
        "gold_evidence": [
            {
                "evidence_id": "EV_GMV_DIAGNOSIS_001",
                "source": "gmv_change.sql",
                "dimensions": {"period": "current_vs_previous"},
                "metrics": {"gmv_change_rate": -0.12},
            }
        ],
        "reference_answer": "基于模拟数据，最近 30 天 GMV 下降。",
        "expected_behavior": ["引用 GMV 变化", "说明主要贡献商品"],
        "success_criteria": sorted(REQUIRED_SUCCESS_CRITERIA),
        "numeric_tolerances": {"gmv_change_rate": 0.001},
        "metadata": {"source_label": "Synthetic E-commerce Data"},
    }


def test_business_task_and_capability_enums_have_the_frozen_taxonomy() -> None:
    assert {member.value for member in BusinessTask} == {
        "gmv_diagnosis",
        "product_anomaly",
        "conversion_decline",
        "products_to_watch",
        "next_week_priority",
    }
    assert {member.value for member in Capability} == {
        "basic_query",
        "metric_calculation",
        "tool_selection",
        "parameter_selection",
        "multi_step_reasoning",
        "anomaly_detection",
        "root_cause_analysis",
        "recommendation",
        "data_insufficiency",
        "adversarial_distractor",
    }


def test_component_models_preserve_recursive_json_parameters_and_evidence_structure() -> None:
    tool_call = ExpectedToolCall.model_validate(
        {
            "name": "query_sales",
            "parameters": {
                "window_days": 30,
                "filters": {
                    "product_ids": ["P001", "P002"],
                    "include_refunds": False,
                    "minimum_gmv": 10.5,
                    "optional": None,
                },
            },
        }
    )
    evidence = GoldEvidence.model_validate(
        {
            "evidence_id": "EV_GMV_DIAGNOSIS_001",
            "source": "gmv_change.sql",
            "dimensions": {"product_id": "P001"},
            "metrics": {"gmv": 100.0},
        }
    )

    assert tool_call.parameters == {
        "window_days": 30,
        "filters": {
            "product_ids": ["P001", "P002"],
            "include_refunds": False,
            "minimum_gmv": 10.5,
            "optional": None,
        },
    }
    assert evidence.dimensions == {"product_id": "P001"}
    assert evidence.metrics == {"gmv": 100.0}


def test_case_schema_accepts_a_traceable_case() -> None:
    case = EvaluationCase.model_validate(valid_case())

    assert case.business_task is BusinessTask.GMV_DIAGNOSIS
    assert case.primary_capability is Capability.METRIC_CALCULATION
    assert case.split == "development"
    assert case.generator_config_hash == "a" * 64
    assert case.gold_evidence[0].evidence_id == "EV_GMV_DIAGNOSIS_001"
    assert case.model_dump(mode="json")["gold_evidence"][0] == {
        "evidence_id": "EV_GMV_DIAGNOSIS_001",
        "source": "gmv_change.sql",
        "dimensions": {"period": "current_vs_previous"},
        "metrics": {"gmv_change_rate": -0.12},
    }


def test_case_schema_roundtrips_all_recursive_json_values() -> None:
    payload = valid_case()
    payload["metadata"] = {
        "source_label": "Synthetic E-commerce Data",
        "generation": {
            "seed": 20260928,
            "flags": [True, False, None],
            "weights": [1, 0.5],
        },
    }
    payload["expected_tool_calls"][0]["parameters"] = {
        "filters": [{"field": "region", "values": ["north", "south"]}],
        "limit": 10,
    }

    case = EvaluationCase.model_validate(payload)
    restored = EvaluationCase.model_validate_json(case.model_dump_json())

    assert restored == case
    assert restored.model_dump() == payload


@pytest.mark.parametrize(
    ("field", "value", "problem_key"),
    [
        ("parameters", {"bad_parameter": object()}, "bad_parameter"),
        ("metadata", {"bad_metadata": object()}, "bad_metadata"),
    ],
)
def test_case_schema_rejects_non_json_values_with_problem_key(
    field: str, value: object, problem_key: str
) -> None:
    payload = valid_case()
    if field == "parameters":
        payload["expected_tool_calls"][0][field] = value
    else:
        payload[field] = value

    with pytest.raises(ValidationError, match=problem_key):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("business_task", "unknown_task"),
        ("primary_capability", "unknown_capability"),
        ("difficulty", "extreme"),
        ("split", "test"),
    ],
)
def test_case_schema_rejects_unknown_enum_values(field: str, value: str) -> None:
    payload = valid_case()
    payload[field] = value

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize("split", ["development", "holdout"])
def test_case_schema_accepts_only_the_frozen_splits(split: str) -> None:
    payload = valid_case()
    payload["split"] = split

    assert EvaluationCase.model_validate(payload).split == split


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "CASE_01"),
        ("dataset_id", "not-a-dataset-id"),
        ("generator_config_hash", "not-a-sha256"),
        ("generator_config_hash", "g" * 64),
        ("generator_config_hash", "A" * 64),
        ("generator_config_hash", 1),
    ],
)
def test_case_schema_rejects_invalid_traceability_identifiers(
    field: str, value: object
) -> None:
    payload = valid_case()
    payload[field] = value

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_case_schema_requires_generator_config_hash() -> None:
    payload = valid_case()
    del payload["generator_config_hash"]

    with pytest.raises(ValidationError) as exc_info:
        EvaluationCase.model_validate(payload)

    assert [error["loc"] for error in exc_info.value.errors()] == [
        ("generator_config_hash",)
    ]


@pytest.mark.parametrize(
    "evidence_id",
    [
        "EV_CASE_001_01",
        "EV_GMV_DIAGNOSIS_01",
        "EV_UNKNOWN_TASK_001",
        "ev_gmv_diagnosis_001",
    ],
)
def test_gold_evidence_rejects_non_task_level_ids(evidence_id: str) -> None:
    payload = valid_case()
    payload["gold_evidence"][0]["evidence_id"] = evidence_id

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_task_evidence_id_is_stable_and_reusable_across_cases() -> None:
    first_payload = valid_case()
    second_payload = valid_case()
    second_payload["case_id"] = "CASE_002"

    first = EvaluationCase.model_validate(first_payload)
    second = EvaluationCase.model_validate(second_payload)

    assert first.gold_evidence[0].evidence_id == "EV_GMV_DIAGNOSIS_001"
    assert second.gold_evidence[0].evidence_id == first.gold_evidence[0].evidence_id


def test_case_schema_rejects_duplicate_evidence_ids() -> None:
    payload = valid_case()
    payload["gold_evidence"].append(deepcopy(payload["gold_evidence"][0]))

    with pytest.raises(ValidationError, match="gold evidence ids must be unique"):
        EvaluationCase.model_validate(payload)


def test_case_schema_requires_gold_metrics_to_be_backed_by_evidence() -> None:
    payload = valid_case()
    payload["gold_metrics"] = {"unsupported_metric": 1.0}
    payload["gold_metric_evidence"] = {
        "unsupported_metric": "EV_GMV_DIAGNOSIS_001"
    }
    payload["numeric_tolerances"] = {"unsupported_metric": 0.1}

    with pytest.raises(ValidationError, match="unsupported_metric"):
        EvaluationCase.model_validate(payload)


def test_gold_metric_uses_its_explicit_evidence_id_not_another_matching_row() -> None:
    payload = valid_case()
    payload["gold_evidence"].append(
        {
            "evidence_id": "EV_GMV_DIAGNOSIS_002",
            "source": "gmv_change.sql",
            "dimensions": {"product_id": "P002"},
            "metrics": {"gmv_change_rate": -0.25},
        }
    )
    payload["gold_metric_evidence"]["gmv_change_rate"] = "EV_GMV_DIAGNOSIS_002"

    with pytest.raises(ValidationError, match="gmv_change_rate.*EV_GMV_DIAGNOSIS_002"):
        EvaluationCase.model_validate(payload)


def test_gold_metrics_must_not_be_empty() -> None:
    payload = valid_case()
    payload["gold_metrics"] = {}
    payload["gold_metric_evidence"] = {}
    payload["numeric_tolerances"] = {}

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize(
    ("mapping", "problem_key"),
    [
        ({}, "gmv_change_rate"),
        (
            {
                "gmv_change_rate": "EV_GMV_DIAGNOSIS_001",
                "unknown_metric": "EV_GMV_DIAGNOSIS_001",
            },
            "unknown_metric",
        ),
        ({"gmv_change_rate": "EV_GMV_DIAGNOSIS_999"}, "EV_GMV_DIAGNOSIS_999"),
    ],
)
def test_gold_metric_evidence_mapping_validates_metric_names_and_evidence_ids(
    mapping: dict[str, str], problem_key: str
) -> None:
    payload = valid_case()
    payload["gold_metric_evidence"] = mapping

    with pytest.raises(ValidationError, match=problem_key):
        EvaluationCase.model_validate(payload)


def test_gold_evidence_rejects_duplicate_dimension_rows() -> None:
    payload = valid_case()
    duplicate = deepcopy(payload["gold_evidence"][0])
    duplicate["evidence_id"] = "EV_GMV_DIAGNOSIS_002"
    payload["gold_evidence"].append(duplicate)

    with pytest.raises(
        ValidationError,
        match="EV_GMV_DIAGNOSIS_001.*EV_GMV_DIAGNOSIS_002.*period",
    ):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize("target", ["gold_metrics", "gold_evidence"])
@pytest.mark.parametrize("value", [True, "1.25", nan, inf, -inf])
def test_gold_numeric_values_reject_bool_numeric_strings_and_non_finite_numbers(
    target: str, value: object
) -> None:
    payload = valid_case()
    if target == "gold_metrics":
        payload["gold_metrics"]["gmv_change_rate"] = value
    else:
        payload["gold_evidence"][0]["metrics"]["gmv_change_rate"] = value

    with pytest.raises(ValidationError, match="gmv_change_rate"):
        EvaluationCase.model_validate(payload)


def test_gold_numeric_values_preserve_int_and_float_types_through_roundtrip() -> None:
    payload = valid_case()
    payload["gold_metrics"] = {"order_count": 12, "gmv_change_rate": -0.12}
    payload["gold_metric_evidence"] = {
        "order_count": "EV_GMV_DIAGNOSIS_001",
        "gmv_change_rate": "EV_GMV_DIAGNOSIS_001",
    }
    payload["gold_evidence"][0]["metrics"] = {
        "order_count": 12,
        "gmv_change_rate": -0.12,
    }

    case = EvaluationCase.model_validate(payload)
    restored = EvaluationCase.model_validate_json(case.model_dump_json())

    assert type(restored.gold_metrics["order_count"]) is int
    assert type(restored.gold_metrics["gmv_change_rate"]) is float


def test_allowed_alternatives_require_non_empty_unique_paths_distinct_from_primary() -> None:
    payload = valid_case()
    primary = deepcopy(payload["expected_tool_calls"])
    alternative = [{"name": "query_sales_summary", "parameters": {"window_days": 30}}]

    for alternatives, problem_key in [
        ([[]], "allowed_alternatives\\[0\\]"),
        ([primary], "allowed_alternatives\\[0\\]"),
        ([alternative, deepcopy(alternative)], "allowed_alternatives\\[1\\]"),
    ]:
        candidate = deepcopy(payload)
        candidate["allowed_alternatives"] = alternatives
        with pytest.raises(ValidationError, match=problem_key):
            EvaluationCase.model_validate(candidate)


@pytest.mark.parametrize(
    "success_criteria",
    [
        [],
        sorted(REQUIRED_SUCCESS_CRITERIA - {"correct_tool"}),
        sorted(REQUIRED_SUCCESS_CRITERIA | {"weighted_score"}),
        ["correct_tool"] * 2
        + sorted(REQUIRED_SUCCESS_CRITERIA - {"correct_tool"}),
    ],
)
def test_case_schema_requires_the_exact_unique_success_gates(
    success_criteria: list[str],
) -> None:
    payload = valid_case()
    payload["success_criteria"] = success_criteria

    with pytest.raises(ValidationError, match="success criteria must contain"):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize("tolerance", ["0.001", True, -0.001, inf, nan])
def test_case_schema_rejects_invalid_numeric_tolerances(tolerance: object) -> None:
    payload = valid_case()
    payload["numeric_tolerances"] = {"gmv_change_rate": tolerance}

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_case_schema_accepts_finite_float_numeric_tolerance() -> None:
    payload = valid_case()
    payload["numeric_tolerances"] = {"gmv_change_rate": 0.001}

    case = EvaluationCase.model_validate(payload)

    assert case.numeric_tolerances["gmv_change_rate"] == 0.001
    assert type(case.numeric_tolerances["gmv_change_rate"]) is float


def test_case_schema_requires_tolerances_for_float_gold_metrics() -> None:
    payload = valid_case()
    payload["numeric_tolerances"] = {}

    with pytest.raises(ValidationError, match="float gold metrics require tolerances"):
        EvaluationCase.model_validate(payload)


def test_case_schema_rejects_tolerance_without_matching_numeric_gold_metric() -> None:
    payload = valid_case()
    payload["numeric_tolerances"] = {"unknown_metric": 0.001}

    with pytest.raises(
        ValidationError, match="tolerances must reference numeric gold metrics"
    ):
        EvaluationCase.model_validate(payload)


@pytest.mark.parametrize(
    "capability_tags",
    [
        ["metric_calculation", "metric_calculation"],
        ["root_cause_analysis"],
    ],
)
def test_case_schema_requires_unique_tags_including_primary_capability(
    capability_tags: list[str],
) -> None:
    payload = valid_case()
    payload["capability_tags"] = capability_tags

    with pytest.raises(
        ValidationError,
        match="capability tags must be unique and include primary capability",
    ):
        EvaluationCase.model_validate(payload)


def test_case_model_forbids_undeclared_fields() -> None:
    payload = valid_case()
    payload["unexpected"] = True

    with pytest.raises(ValidationError) as exc_info:
        EvaluationCase.model_validate(payload)

    assert [error["loc"] for error in exc_info.value.errors()] == [("unexpected",)]


def test_nested_model_forbids_undeclared_fields_without_false_positive() -> None:
    payload = valid_case()
    payload["expected_tool_calls"][0]["unexpected"] = True

    with pytest.raises(ValidationError) as exc_info:
        EvaluationCase.model_validate(payload)

    assert [error["loc"] for error in exc_info.value.errors()] == [
        ("expected_tool_calls", 0, "unexpected")
    ]
