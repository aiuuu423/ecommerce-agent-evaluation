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
        "gold_evidence": [
            {
                "evidence_id": "EV_CASE_001_01",
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


def test_component_models_preserve_tool_parameters_and_evidence_structure() -> None:
    tool_call = ExpectedToolCall.model_validate(
        {"name": "query_sales", "parameters": {"window_days": 30}}
    )
    evidence = GoldEvidence.model_validate(
        {
            "evidence_id": "EV_CASE_001_01",
            "source": "gmv_change.sql",
            "dimensions": {"product_id": "P001"},
            "metrics": {"gmv": 100.0},
        }
    )

    assert tool_call.parameters == {"window_days": 30}
    assert evidence.dimensions == {"product_id": "P001"}
    assert evidence.metrics == {"gmv": 100.0}


def test_case_schema_accepts_a_traceable_case() -> None:
    case = EvaluationCase.model_validate(valid_case())

    assert case.business_task is BusinessTask.GMV_DIAGNOSIS
    assert case.primary_capability is Capability.METRIC_CALCULATION
    assert case.split == "development"
    assert case.gold_evidence[0].evidence_id == "EV_CASE_001_01"
    assert case.model_dump(mode="json")["gold_evidence"][0] == {
        "evidence_id": "EV_CASE_001_01",
        "source": "gmv_change.sql",
        "dimensions": {"period": "current_vs_previous"},
        "metrics": {"gmv_change_rate": -0.12},
    }


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
    ],
)
def test_case_schema_rejects_invalid_traceability_identifiers(
    field: str, value: str
) -> None:
    payload = valid_case()
    payload[field] = value

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_case_schema_rejects_evidence_from_another_case() -> None:
    payload = valid_case()
    payload["gold_evidence"][0]["evidence_id"] = "EV_CASE_002_01"

    with pytest.raises(ValidationError, match="evidence_id must belong to case_id"):
        EvaluationCase.model_validate(payload)


def test_case_schema_rejects_duplicate_evidence_ids() -> None:
    payload = valid_case()
    payload["gold_evidence"].append(deepcopy(payload["gold_evidence"][0]))

    with pytest.raises(ValidationError, match="gold evidence ids must be unique"):
        EvaluationCase.model_validate(payload)


def test_case_schema_requires_gold_metrics_to_be_backed_by_evidence() -> None:
    payload = valid_case()
    payload["gold_metrics"] = {"unsupported_metric": 1.0}
    payload["numeric_tolerances"] = {"unsupported_metric": 0.1}

    with pytest.raises(ValidationError, match="gold metrics must be backed by evidence"):
        EvaluationCase.model_validate(payload)


def test_gold_metric_may_match_any_evidence_row_with_the_same_metric_name() -> None:
    payload = valid_case()
    payload["gold_evidence"].append(
        {
            "evidence_id": "EV_CASE_001_02",
            "source": "gmv_change.sql",
            "dimensions": {"product_id": "P002"},
            "metrics": {"gmv_change_rate": -0.25},
        }
    )

    assert EvaluationCase.model_validate(payload).gold_metrics == {
        "gmv_change_rate": -0.12
    }


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


@pytest.mark.parametrize("tolerance", [-0.001, inf, nan])
def test_case_schema_rejects_invalid_numeric_tolerances(tolerance: float) -> None:
    payload = valid_case()
    payload["numeric_tolerances"] = {"gmv_change_rate": tolerance}

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


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


def test_case_and_nested_models_forbid_undeclared_fields() -> None:
    payload = valid_case()
    payload["unexpected"] = True
    payload["expected_tool_calls"][0]["unexpected"] = True

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)
