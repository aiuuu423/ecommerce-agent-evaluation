import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from difflib import SequenceMatcher
from hashlib import sha256
from math import inf, nan
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.data.case_generator import (
    PRODUCT_LEVEL_GOLD_TASKS,
    _distractor_product,
    _question_stem,
    build_cases,
    write_cases,
)
from app.data.generator import build_snapshot
from app.data.gold import build_gold_bundle
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


def build_test_case_set(
    tmp_path: Path,
) -> tuple[list[EvaluationCase], dict[str, object], dict[str, object]]:
    development_dir = tmp_path / "development-v1"
    public_validation_dir = tmp_path / "public-validation-v1"
    development_manifest = build_snapshot(
        "configs/data/synthetic_v1.yaml", development_dir
    )
    public_validation_manifest = build_snapshot(
        "configs/data/synthetic_public_validation_v1.yaml",
        public_validation_dir,
    )
    return (
        build_cases(development_dir, public_validation_dir),
        development_manifest,
        public_validation_manifest,
    )


def valid_case() -> dict[str, object]:
    return {
        "case_id": "CASE_001",
        "case_version": "1.2",
        "statistical_cluster_id": "gmv_diagnosis:metric_calculation",
        "tool_contract_version": "1.0",
        "tool_contract_sha256": "b" * 64,
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


@pytest.mark.parametrize("split", ["development", "public_validation"])
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


def test_builds_exact_balanced_traceable_case_set(tmp_path: Path) -> None:
    cases, development_manifest, public_validation_manifest = build_test_case_set(
        tmp_path
    )

    assert len(cases) == 100
    assert [case.case_id for case in cases] == [
        f"CASE_{number:03d}" for number in range(1, 101)
    ]
    assert Counter(case.business_task.value for case in cases) == {
        "gmv_diagnosis": 20,
        "product_anomaly": 20,
        "conversion_decline": 20,
        "products_to_watch": 20,
        "next_week_priority": 20,
    }
    assert Counter(case.primary_capability.value for case in cases) == {
        capability.value: 10 for capability in Capability
    }
    assert Counter(case.difficulty for case in cases) == {
        "easy": 30,
        "medium": 40,
        "hard": 30,
    }
    assert Counter(case.split for case in cases) == {
        "development": 70,
        "public_validation": 30,
    }
    assert {
        task: Counter(case.split for case in cases if case.business_task.value == task)
        for task in BusinessTask
    } == {
        task: Counter({"development": 14, "public_validation": 6})
        for task in BusinessTask
    }
    public_validation = [
        case for case in cases if case.split == "public_validation"
    ]
    assert Counter(case.primary_capability for case in public_validation) == {
        capability: 3 for capability in Capability
    }
    assert Counter(case.difficulty for case in public_validation) == {
        "easy": 9,
        "medium": 12,
        "hard": 9,
    }
    expected_manifests = {
        "development": development_manifest,
        "public_validation": public_validation_manifest,
    }
    for case in cases:
        manifest = expected_manifests[case.split]
        assert case.dataset_id == manifest["dataset_id"]
        assert case.dataset_version == manifest["dataset_version"]
        assert case.generator_config_hash == manifest["config_sha256"]
        assert case.statistical_cluster_id == (
            f"{case.business_task.value}:{case.primary_capability.value}"
        )
    assert (
        development_manifest["dataset_id"]
        != public_validation_manifest["dataset_id"]
    )


def test_cases_bind_stable_evidence_metrics_and_capability_semantics(
    tmp_path: Path,
) -> None:
    first, _, _ = build_test_case_set(tmp_path)
    second, _, _ = build_test_case_set(tmp_path)

    assert [case.model_dump(mode="json") for case in first] == [
        case.model_dump(mode="json") for case in second
    ]
    for case in first:
        evidence = {row.evidence_id: row for row in case.gold_evidence}
        assert set(case.gold_metric_evidence) == set(case.gold_metrics)
        for metric, evidence_id in case.gold_metric_evidence.items():
            assert evidence[evidence_id].metrics[metric] == case.gold_metrics[metric]
        assert case.metadata["source_label"] == "Synthetic E-commerce Data"
        assert case.metadata["validation_role"] == case.split

    insufficient = [
        case
        for case in first
        if case.primary_capability is Capability.DATA_INSUFFICIENCY
    ]
    assert len(insufficient) == 10
    assert all(
        {row.dimensions.get("product_id") for row in case.gold_evidence} == {"P005"}
        for case in insufficient
    )
    expected = {
        BusinessTask.GMV_DIAGNOSIS: {
            "metrics": {"gmv_change_rate", "aov_change_rate"},
            "unanswerable": {"traffic_change_rate", "current_cvr", "cvr_change"},
            "tools": ["query_sales", "calculate_metrics"],
            "answer_fragment": "不能用不完整流量可靠计算当前CVR",
        },
        BusinessTask.PRODUCT_ANOMALY: {
            "metrics": {"current_observed_days", "previous_observed_days"},
            "unanswerable": {"traffic_change_rate", "current_cvr", "cvr_change"},
            "tools": ["query_product", "query_traffic"],
            "answer_fragment": "可确认数据缺失异常",
        },
        BusinessTask.CONVERSION_DECLINE: {
            "metrics": {
                "previous_cvr",
                "current_observed_days",
                "previous_observed_days",
            },
            "unanswerable": {"current_cvr", "cvr_change"},
            "tools": ["query_traffic", "query_sales", "calculate_metrics"],
            "answer_fragment": "不能判断P005的转化下降幅度或排名",
        },
        BusinessTask.PRODUCTS_TO_WATCH: {
            "metrics": {
                "gmv_change_rate",
                "refund_rate",
                "current_observed_days",
                "previous_observed_days",
            },
            "unanswerable": {"current_cvr", "cvr_change"},
            "tools": ["query_sales", "query_traffic", "calculate_metrics"],
            "answer_fragment": "应以data_quality_review关注P005",
        },
        BusinessTask.NEXT_WEEK_PRIORITY: {
            "metrics": {
                "evidence_value",
                "current_observed_days",
                "previous_observed_days",
            },
            "unanswerable": {"current_cvr", "cvr_change"},
            "tools": ["query_traffic"],
            "answer_fragment": "下周应优先repair_data_quality",
        },
    }
    for case in insufficient:
        spec = expected[case.business_task]
        assert set(case.gold_metrics) == spec["metrics"]
        assert set(case.metadata["answerable_metrics"]) == spec["metrics"]
        assert set(case.metadata["unanswerable_metrics"]) == spec["unanswerable"]
        assert [call.name for call in case.expected_tool_calls] == spec["tools"]
        assert spec["answer_fragment"] in case.reference_answer
        for call in case.expected_tool_calls:
            if "product_ids" in call.parameters:
                assert call.parameters["product_ids"] == ["P005"]


def test_paraphrase_variants_share_an_explicit_statistical_cluster(
    tmp_path: Path,
) -> None:
    cases, _, _ = build_test_case_set(tmp_path)

    families: dict[tuple[BusinessTask, Capability], list[EvaluationCase]] = {}
    for case in cases:
        families.setdefault(
            (case.business_task, case.primary_capability),
            [],
        ).append(case)

    assert len(families) == 50
    for family_cases in families.values():
        assert {case.metadata["variant"] for case in family_cases} == {1, 2}
        assert len({case.metadata["semantic_family_id"] for case in family_cases}) == 1
        assert len({case.statistical_cluster_id for case in family_cases}) == 1

    stems_by_split: dict[str, set[str]] = {
        "development": set(),
        "public_validation": set(),
    }
    for case in cases:
        stem = _question_stem(
            case.business_task,
            case.split,
            int(case.metadata["variant"]) - 1,
            case.primary_capability is not Capability.DATA_INSUFFICIENCY,
        )
        assert case.user_input.startswith(stem)
        stems_by_split[case.split].add(stem)

    cross_split_similarity = [
        SequenceMatcher(None, development, public_validation).ratio()
        for development in stems_by_split["development"]
        for public_validation in stems_by_split["public_validation"]
    ]
    assert max(cross_split_similarity) < 0.72


def test_tool_calls_have_tool_specific_parameters_and_real_distractors(
    tmp_path: Path,
) -> None:
    cases, _, _ = build_test_case_set(tmp_path)

    contract = yaml.safe_load(
        Path("configs/evaluation/tool_contract_v1.yaml").read_text(encoding="utf-8")
    )
    required_parameters = {
        name: set(spec["required_parameters"])
        for name, spec in contract["tools"].items()
    }
    for case in cases:
        for call in case.expected_tool_calls:
            assert set(call.parameters) == required_parameters[call.name]

    adversarial = [
        case
        for case in cases
        if case.primary_capability is Capability.ADVERSARIAL_DISTRACTOR
    ]
    assert len(adversarial) == 10
    for case in adversarial:
        dataset_dir = tmp_path / (
            "development-v1"
            if case.split == "development"
            else "public-validation-v1"
        )
        products = set(
            __import__("pandas").read_parquet(dataset_dir / "products.parquet")[
                "product_id"
            ]
        )
        gold = build_gold_bundle(dataset_dir)
        product_gold_union = {
            row["product_id"]
            for task in PRODUCT_LEVEL_GOLD_TASKS
            for row in gold["tasks"][task]["evidence"]
        }
        distractor = case.metadata["distractor_product_id"]
        query_product = next(
            call for call in case.expected_tool_calls if call.name == "query_product"
        )
        assert distractor in products
        assert distractor not in product_gold_union
        assert distractor in query_product.parameters["product_ids"]
        assert distractor in case.user_input
        assert case.metadata["distractor_assertion_supported"] is False
        assert f"{distractor}是唯一主因的断言为假" in case.reference_answer
        assert "P999" not in case.user_input


def test_list_cases_are_explicit_top_k_and_every_evidence_metric_is_scorable(
    tmp_path: Path,
) -> None:
    cases, _, _ = build_test_case_set(tmp_path)

    list_tasks = {
        BusinessTask.PRODUCT_ANOMALY,
        BusinessTask.CONVERSION_DECLINE,
        BusinessTask.PRODUCTS_TO_WATCH,
        BusinessTask.NEXT_WEEK_PRIORITY,
    }
    for case in cases:
        if (
            case.business_task in list_tasks
            and case.primary_capability is not Capability.DATA_INSUFFICIENCY
        ):
            assert "Top-3" in case.user_input
            assert case.metadata["top_k"] == 3
            assert len(case.gold_evidence) == 3
        evidence_metrics = {
            metric
            for evidence in case.gold_evidence
            for metric in evidence.metrics
        }
        assert set(case.gold_metrics) == evidence_metrics
        assert set(case.gold_metric_evidence) == evidence_metrics


def test_versioned_tool_contract_fixes_gmv_next_week_and_adversarial_paths(
    tmp_path: Path,
) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    contract = yaml.safe_load(
        Path("configs/evaluation/tool_contract_v1.yaml").read_text(encoding="utf-8")
    )

    for case in cases:
        path = [call.name for call in case.expected_tool_calls]
        assert case.tool_contract_version == "1.0"
        assert len(case.tool_contract_sha256) == 64
        expected = list(contract["paths"][case.business_task.value])
        override = contract["capability_overrides"].get(
            case.primary_capability.value, {}
        )
        expected = list(override.get("paths", {}).get(case.business_task.value, expected))
        expected = list(dict.fromkeys([*override.get("prepend_tools", []), *expected]))
        assert path == expected


def test_generator_rejects_parameters_that_diverge_from_the_yaml_contract(
    tmp_path: Path,
) -> None:
    contract_path = Path("configs/evaluation/tool_contract_v1.yaml")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract["tools"]["query_sales"]["required_parameters"].remove("include_refunds")
    changed_contract = tmp_path / "tool_contract.yaml"
    changed_contract.write_text(
        yaml.safe_dump(contract, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    development_dir = tmp_path / "development-v1"
    public_validation_dir = tmp_path / "public-validation-v1"
    build_snapshot("configs/data/synthetic_v1.yaml", development_dir)
    build_snapshot(
        "configs/data/synthetic_public_validation_v1.yaml",
        public_validation_dir,
    )

    with pytest.raises(
        ValueError,
        match=r"query_sales parameters do not match tool contract.*include_refunds",
    ):
        build_cases(development_dir, public_validation_dir, changed_contract)


def test_difficulty_is_ranked_by_recorded_complexity_with_frozen_distribution(
    tmp_path: Path,
) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    bands = {
        difficulty: [
            case.metadata["complexity_score"]
            for case in cases
            if case.difficulty == difficulty
        ]
        for difficulty in ("easy", "medium", "hard")
    }

    assert max(bands["easy"]) <= min(bands["medium"])
    assert max(bands["medium"]) <= min(bands["hard"])
    assert {difficulty: len(scores) for difficulty, scores in bands.items()} == {
        "easy": 30,
        "medium": 40,
        "hard": 30,
    }


def test_distractor_selection_changes_when_its_false_assertion_becomes_gold(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "v1"
    manifest = build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    gold = build_gold_bundle(dataset_dir)
    products = set(
        __import__("pandas").read_parquet(dataset_dir / "products.parquet")["product_id"]
    )
    task = BusinessTask.PRODUCT_ANOMALY
    first = _distractor_product(gold, products, manifest["dataset_id"], task)

    counterfactual_gold = deepcopy(gold)
    counterfactual_gold["tasks"][BusinessTask.PRODUCT_ANOMALY]["evidence"].append(
        {"product_id": first}
    )
    second = _distractor_product(
        counterfactual_gold,
        products,
        manifest["dataset_id"],
        task,
    )

    assert second != first
    assert second in products
    assert all(
        second
        not in {
            row.get("product_id")
            for row in counterfactual_gold["tasks"][gold_task]["evidence"]
        }
        for gold_task in PRODUCT_LEVEL_GOLD_TASKS
    )


def test_every_gold_metric_is_reachable_from_expected_tool_path(tmp_path: Path) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    sales_metrics = {
        "current_gmv",
        "previous_gmv",
        "current_orders",
        "previous_orders",
        "current_aov",
        "previous_aov",
        "gmv_change_rate",
        "aov_change_rate",
        "current_refund_rate",
        "refund_rate",
    }
    traffic_metrics = {
        "current_visits",
        "previous_visits",
        "traffic_change_rate",
        "current_observed_days",
        "previous_observed_days",
    }
    derived_metrics = {"current_cvr", "previous_cvr", "cvr_change"}

    for case in cases:
        tools = {call.name for call in case.expected_tool_calls}
        metrics = set(case.gold_metrics)
        assert not metrics & sales_metrics or "query_sales" in tools
        assert not metrics & traffic_metrics or "query_traffic" in tools
        if metrics & derived_metrics:
            assert {"query_sales", "query_traffic", "calculate_metrics"} <= tools
        if "evidence_value" in metrics:
            evidence_metric = case.gold_evidence[0].dimensions["evidence_metric"]
            required_tool = (
                "query_traffic"
                if evidence_metric in {"observed_days", "traffic_change_rate"}
                else "query_sales"
            )
            assert required_tool in tools


def test_jsonl_and_manifest_are_reproducible_and_do_not_leak_config(
    tmp_path: Path,
) -> None:
    cases, development_manifest, public_validation_manifest = build_test_case_set(
        tmp_path
    )
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"

    first_manifest = write_cases(cases, first_path)
    second_manifest = write_cases(cases, second_path)

    assert (first_path / "cases.jsonl").read_bytes() == (
        second_path / "cases.jsonl"
    ).read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest["case_count"] == 100
    assert first_manifest["split_counts"] == {
        "development": 70,
        "public_validation": 30,
    }
    assert first_manifest["datasets"]["development"]["dataset_id"] == (
        development_manifest["dataset_id"]
    )
    assert first_manifest["datasets"]["public_validation"]["dataset_id"] == (
        public_validation_manifest["dataset_id"]
    )
    assert first_manifest["split_strategy"]["cross_split_dataset_isolation"] is True
    assert first_manifest["split_strategy"]["statistical_unit"] == (
        "statistical_cluster_id"
    )
    assert first_manifest["jsonl_sha256"]
    assert first_manifest["case_set_id"]
    assert json.loads(
        (first_path / "manifest.json").read_text(encoding="utf-8")
    ) == first_manifest
    serialized = (first_path / "cases.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        "anomaly_id",
        "multiplier",
        "synthetic_v1.yaml",
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
    ):
        assert forbidden not in serialized


def test_committed_case_snapshot_is_frozen_from_both_configs(tmp_path: Path) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    rebuilt_path = tmp_path / "rebuilt"
    rebuilt_manifest = write_cases(cases, rebuilt_path)
    committed_path = Path("data/evaluation_cases/v1")

    assert (rebuilt_path / "cases.jsonl").read_bytes() == (
        committed_path / "cases.jsonl"
    ).read_bytes()
    assert (rebuilt_path / "manifest.json").read_bytes() == (
        committed_path / "manifest.json"
    ).read_bytes()
    assert rebuilt_manifest["case_set_id"] == "ecebfe8b691271fd"
    assert (
        rebuilt_manifest["jsonl_sha256"]
        == "f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03"
    )

    split_difficulty_mapping = "\n".join(
        f"{case.case_id}:{case.split}:{case.difficulty}" for case in cases
    ).encode()
    assert (
        sha256(split_difficulty_mapping).hexdigest()
        == "414f9d77a68db37013278f754f5456242f4ac392c5d8a3453aa00488686e3aa1"
    )


def test_case_snapshot_is_an_atomic_immutable_directory(tmp_path: Path) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    output = tmp_path / "case-snapshot-v1"

    manifest = write_cases(cases, output)
    assert write_cases(cases, output) == manifest
    (output / "cases.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable"):
        write_cases(cases, output)
    assert not list(tmp_path.glob(".case-snapshot-v1.tmp-*"))


def test_concurrent_case_writers_publish_one_complete_snapshot(tmp_path: Path) -> None:
    cases, _, _ = build_test_case_set(tmp_path)
    output = tmp_path / "case-snapshot-v1"
    barrier = threading.Barrier(2)

    def write_after_barrier(_: int) -> dict[str, object]:
        barrier.wait(timeout=10)
        return write_cases(cases, output)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write_after_barrier, range(2)))

    assert results[0] == results[1]
    assert set(path.name for path in output.iterdir()) == {
        "cases.jsonl",
        "manifest.json",
    }
    assert not list(tmp_path.glob(".case-snapshot-v1.tmp-*"))
