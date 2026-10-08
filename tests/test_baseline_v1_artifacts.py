import json
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from app.baselines.v1.config import policy_snapshot
from app.experiments.artifacts import (
    ArtifactWriter,
    CaseRunRecord,
    RunArtifactBundle,
    RunManifest,
    RunSummary,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    verify_published_run,
)

SHA = "a" * 64


def record(sequence: int, case_id: str) -> CaseRunRecord:
    return CaseRunRecord(
        sequence=sequence,
        case_id=case_id,
        split="development",
        dataset_id="0123456789abcdef",
        dataset_version="v1",
        user_input=f"分析 {case_id}",
        status="completed",
        final_answer="完成",
        prior_tool_executions=(),
        decision_trace=(),
        usage=None,
        error_code=None,
    )


def summary() -> RunSummary:
    return RunSummary(
        total=2,
        completed=2,
        failed=0,
        stopped=0,
        split_counts={"development": 2, "public_validation": 0},
        split_status_counts={
            "development": {"completed": 2, "failed": 0},
            "public_validation": {"completed": 0, "failed": 0},
        },
        first_case_id="CASE_001",
        last_case_id="CASE_002",
        published=True,
        evaluation_status="pending_not_run",
    )


def manifest(run_id: str = "baseline-v1__test") -> RunManifest:
    snapshot_sha256 = sha256(canonical_json_bytes(policy_snapshot())).hexdigest()
    return RunManifest(
        artifact_schema_version="1.0",
        run_id=run_id,
        baseline_name="deterministic-baseline-v1",
        baseline_version="1.0.0",
        started_at_utc="2026-10-08T01:00:00Z",
        completed_at_utc="2026-10-08T01:01:00Z",
        git_commit="1" * 40,
        git_dirty=False,
        runtime_versions={
            "python": "3.12",
            "pydantic": "2.11",
            "duckdb": "1.3",
            "pandas": "2.3",
            "pyarrow": "19.0",
        },
        case_set_id="0123456789abcdef",
        case_jsonl_sha256=SHA,
        case_manifest_sha256=SHA,
        datasets=(
            {
                "split": "development",
                "dataset_id": "0123456789abcdef",
                "dataset_version": "v1",
                "manifest_sha256": SHA,
            },
            {
                "split": "public_validation",
                "dataset_id": "fedcba9876543210",
                "dataset_version": "v1",
                "manifest_sha256": SHA,
            },
        ),
        tool_contract_version="1.0",
        tool_contract_sha256=SHA,
        policy_snapshot_sha256=snapshot_sha256,
        policy_source_files=("app/baselines/v1/config.py",),
        policy_source_sha256=SHA,
        output_files=None,
        evaluation_status="pending_not_run",
        usage_status="unavailable",
    )


def bundle(run_id: str = "baseline-v1__test") -> RunArtifactBundle:
    return RunArtifactBundle(
        records=(record(1, "CASE_001"), record(2, "CASE_002")),
        expected_case_ids=("CASE_001", "CASE_002"),
        summary=summary(),
        policy_snapshot=policy_snapshot(),
        manifest=manifest(run_id),
    )


@pytest.mark.parametrize(
    "forbidden",
    ["gold_metrics", "expected_tool_calls", "score", "latency_ms", "unknown"],
)
def test_case_run_record_rejects_forbidden_and_unknown_fields(forbidden: str) -> None:
    payload = record(1, "CASE_001").model_dump(mode="python")
    payload[forbidden] = 1
    with pytest.raises(ValidationError):
        CaseRunRecord.model_validate(payload)


def test_all_artifact_models_are_strict_and_reject_non_finite_numbers() -> None:
    with pytest.raises(ValidationError):
        RunSummary.model_validate({**summary().model_dump(), "accuracy": 1})
    with pytest.raises(ValidationError):
        RunManifest.model_validate({**manifest().model_dump(), "latency": 1})
    with pytest.raises((ValidationError, ValueError)):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises((ValidationError, ValueError)):
        canonical_json_bytes({"value": float("inf")})


def test_manifest_rejects_unknown_runtime_dependency() -> None:
    payload = manifest().model_dump(mode="python")
    payload["runtime_versions"]["numpy"] = "2.0"
    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload, strict=True)


def test_canonical_json_and_jsonl_are_byte_stable() -> None:
    assert canonical_json_bytes({"b": 2, "a": "中文"}) == (
        b'{"a":"\xe4\xb8\xad\xe6\x96\x87","b":2}\n'
    )
    records = (record(1, "CASE_001"), record(2, "CASE_002"))
    first = canonical_jsonl_bytes(records)
    second = canonical_jsonl_bytes(records)
    assert first == second
    assert first.endswith(b"\n")
    assert len(first.splitlines()) == 2


def test_publish_rejects_existing_target_without_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "baseline-v1__test"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ArtifactWriter(tmp_path).publish(bundle())

    assert marker.read_text(encoding="utf-8") == "keep"


def test_publish_cleans_same_root_temporary_directory_on_failure(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path)
    with mock.patch.object(writer, "_write_file", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            writer.publish(bundle())

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.model_copy(
            update={"expected_case_ids": ("CASE_001",)}
        ),
        lambda value: value.model_copy(
            update={"expected_case_ids": ("CASE_002", "CASE_001")}
        ),
    ],
)
def test_publish_rejects_wrong_line_count_or_case_order(
    tmp_path: Path,
    mutate: Callable[[RunArtifactBundle], RunArtifactBundle],
) -> None:
    invalid = mutate(bundle())
    with pytest.raises(ValueError):
        ArtifactWriter(tmp_path).publish(invalid)
    assert list(tmp_path.iterdir()) == []


def test_publish_rereads_schema_and_hashes_then_atomically_replaces(
    tmp_path: Path,
) -> None:
    published = ArtifactWriter(tmp_path).publish(bundle())

    verified = verify_published_run(published)
    assert verified.manifest.run_id == "baseline-v1__test"
    assert [item.case_id for item in verified.records] == ["CASE_001", "CASE_002"]
    assert set(verified.manifest.output_files.root) == {
        "case_runs.jsonl",
        "summary.json",
        "policy_snapshot.json",
    }
    assert "run_manifest.json" not in verified.manifest.output_files.root
    assert sorted(path.name for path in published.iterdir()) == [
        "case_runs.jsonl",
        "policy_snapshot.json",
        "run_manifest.json",
        "summary.json",
    ]
    assert not list(tmp_path.glob(".baseline-v1__test.tmp-*"))


def test_verify_rejects_tampered_schema_hash_or_case_order(tmp_path: Path) -> None:
    published = ArtifactWriter(tmp_path).publish(bundle())
    case_runs = published / "case_runs.jsonl"
    lines = case_runs.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["gold_metrics"] = {"gmv": 1}
    lines[0] = json.dumps(tampered)
    case_runs.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        verify_published_run(published)


def test_bundle_revalidates_constructed_nested_models() -> None:
    invalid_summary = RunSummary.model_construct(**deepcopy(summary().model_dump()))
    object.__setattr__(invalid_summary, "completed", 3)
    with pytest.raises(ValidationError):
        RunArtifactBundle(
            records=(record(1, "CASE_001"), record(2, "CASE_002")),
            expected_case_ids=("CASE_001", "CASE_002"),
            summary=invalid_summary,
            policy_snapshot=policy_snapshot(),
            manifest=manifest(),
        )
