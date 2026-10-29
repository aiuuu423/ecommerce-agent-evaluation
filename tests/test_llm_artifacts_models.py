from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.experiments.llm_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    REAL_OUTPUT_FILE_NAMES,
    LatencyStats,
    RealCaseRunRecord,
    RealRunSummary,
    StoredPromptSnapshot,
    TokenStats,
)
from app.tools.schemas import PRODUCT_COLUMNS, PriorToolExecution


def completed_record_payload() -> dict[str, object]:
    return {
        "sequence": 1,
        "case_id": "CASE_001",
        "split": "development",
        "dataset_id": "0123456789abcdef",
        "dataset_version": "v1",
        "agent_version": "v1",
        "repeat_index": 1,
        "user_input": "分析商品异常",
        "status": "completed",
        "final_answer": "商品 P001 的转化率下降。",
        "error_code": None,
        "prior_tool_executions": (
            {
                "call_id": "call_001",
                "tool_name": "query_sales",
                "arguments": {
                    "product_ids": ["P001"],
                    "options": {"include_refunds": True},
                },
                "result": {
                    "result_id": "result_0001",
                    "rows": [{"product_id": "P001", "gmv": 42.5}],
                    "warnings": [],
                },
            },
        ),
        "decision_trace": (
            {
                "sequence": 1,
                "event_type": "tool_result",
                "payload": {
                    "call_id": "call_001",
                    "result": {"row_count": 1, "cached": False},
                    "tags": ["sales", None],
                },
            },
        ),
        "latency_ms": 120,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }


def summary_payload() -> dict[str, object]:
    return {
        "total": 2,
        "completed": 1,
        "failed": 1,
        "stopped": 0,
        "split_counts": {
            "development": 1,
            "public_validation": 1,
        },
        "split_status_counts": {
            "development": {"completed": 1, "failed": 0},
            "public_validation": {"completed": 0, "failed": 1},
        },
        "first_case_id": "CASE_001",
        "last_case_id": "CASE_002",
        "latency": {"p50": 100, "p95": 200, "sample_size": 2},
        "tokens": {
            "available": False,
            "prompt_total": None,
            "completion_total": None,
            "total_total": None,
            "p50": None,
            "p95": None,
            "sample_size": 0,
        },
        "cost": "unavailable",
        "evaluation_status": "pending_not_run",
    }


def test_artifact_v2_constants_are_stable() -> None:
    assert ARTIFACT_SCHEMA_VERSION == "2.0.0"
    assert REAL_OUTPUT_FILE_NAMES == {
        "case_runs.jsonl",
        "summary.json",
        "prompt_snapshot.json",
    }


def test_completed_record_accepts_answer_without_error() -> None:
    record = RealCaseRunRecord(**completed_record_payload())

    assert record.final_answer == "商品 P001 的转化率下降。"
    assert record.total_tokens == 120


def test_completed_record_preserves_json_payload_dicts() -> None:
    payload = completed_record_payload()

    dumped = RealCaseRunRecord(**payload).model_dump(mode="json")

    assert dumped["prior_tool_executions"] == list(
        payload["prior_tool_executions"]
    )
    assert dumped["decision_trace"] == list(payload["decision_trace"])


def test_record_rejects_pydantic_objects_at_json_artifact_boundary() -> None:
    payload = completed_record_payload()
    payload["prior_tool_executions"] = (
        PriorToolExecution(
            call_id="call_001",
            arguments={"product_ids": ["P001"]},
            result={
                "result_id": "result_0001",
                "tool_name": "query_product",
                "dataset_id": "0123456789abcdef",
                "source_label": "Synthetic E-commerce Data",
                "columns": PRODUCT_COLUMNS,
                "rows": [],
                "row_count": 0,
                "warnings": [],
            },
        ),
    )
    payload["decision_trace"] = ()

    with pytest.raises(ValidationError):
        RealCaseRunRecord(**payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("final_answer", None), ("error_code", "provider_timeout")],
)
def test_completed_record_rejects_missing_answer_or_error(
    field_name: str,
    value: object,
) -> None:
    payload = completed_record_payload()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        RealCaseRunRecord(**payload)


def test_failed_record_requires_error_and_no_answer_with_null_usage() -> None:
    payload = completed_record_payload()
    payload.update(
        status="failed",
        final_answer=None,
        error_code="provider_timeout",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    record = RealCaseRunRecord(**payload)

    assert record.error_code == "provider_timeout"
    assert record.final_answer is None
    assert record.prompt_tokens is None


@pytest.mark.parametrize(
    ("final_answer", "error_code"),
    [(None, None), ("partial answer", "provider_timeout")],
)
def test_failed_record_rejects_missing_error_or_answer(
    final_answer: str | None,
    error_code: str | None,
) -> None:
    payload = completed_record_payload()
    payload.update(
        status="failed",
        final_answer=final_answer,
        error_code=error_code,
    )

    with pytest.raises(ValidationError):
        RealCaseRunRecord(**payload)


def test_record_rejects_partial_token_usage() -> None:
    payload = completed_record_payload()
    payload["completion_tokens"] = None

    with pytest.raises(ValidationError, match="token"):
        RealCaseRunRecord(**payload)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_record_rejects_token_total_mismatch_for_usage(
    status: str,
) -> None:
    payload = completed_record_payload()
    if status == "failed":
        payload.update(
            status="failed",
            final_answer=None,
            error_code="provider_timeout",
        )
    payload["total_tokens"] = 121

    with pytest.raises(
        ValidationError,
        match="total_tokens must equal prompt_tokens plus completion_tokens",
    ):
        RealCaseRunRecord(**payload)


def test_record_is_frozen_strict_and_forbids_extra_fields() -> None:
    record = RealCaseRunRecord(**completed_record_payload())

    with pytest.raises(ValidationError):
        record.sequence = 2

    strict_payload = completed_record_payload()
    strict_payload["sequence"] = "1"
    with pytest.raises(ValidationError):
        RealCaseRunRecord(**strict_payload)

    extra_payload = completed_record_payload()
    extra_payload["unexpected"] = True
    with pytest.raises(ValidationError):
        RealCaseRunRecord(**extra_payload)


def test_prompt_snapshot_preserves_complete_prompt_text() -> None:
    prompt_text = "You are an analyst.\n\nKeep every instruction.\n"

    snapshot = StoredPromptSnapshot(
        prompt_name="agent-system",
        prompt_version="v2",
        prompt_text=prompt_text,
        prompt_sha256="a" * 64,
        change_note="Add real-provider instructions.",
        source_file="app/prompts/agent_system.txt",
    )

    dumped = snapshot.model_dump()
    assert snapshot.prompt_text == prompt_text
    assert dumped["prompt_text"] == prompt_text
    assert "text" not in dumped


@pytest.mark.parametrize("prompt_sha256", ["A" * 64, "a" * 63, "g" * 64])
def test_prompt_snapshot_rejects_invalid_sha256(prompt_sha256: str) -> None:
    with pytest.raises(ValidationError):
        StoredPromptSnapshot(
            prompt_name="agent-system",
            prompt_version="v2",
            prompt_text="Complete prompt.",
            prompt_sha256=prompt_sha256,
            change_note="Initial snapshot.",
            source_file="app/prompts/agent_system.txt",
        )


def test_summary_accepts_unavailable_cost_and_tokens() -> None:
    summary = RealRunSummary(**summary_payload())

    assert summary.cost == "unavailable"
    assert summary.tokens == TokenStats(
        available=False,
        prompt_total=None,
        completion_total=None,
        total_total=None,
        p50=None,
        p95=None,
        sample_size=0,
    )
    assert summary.latency == LatencyStats(p50=100, p95=200, sample_size=2)


@pytest.mark.parametrize(
    "updates",
    [
        {"completed": 2},
        {"split_counts": {"development": 2, "public_validation": 1}},
    ],
)
def test_summary_rejects_status_or_split_count_total_mismatch(
    updates: dict[str, object],
) -> None:
    payload = deepcopy(summary_payload())
    payload.update(updates)

    with pytest.raises(ValidationError, match="total"):
        RealRunSummary(**payload)
