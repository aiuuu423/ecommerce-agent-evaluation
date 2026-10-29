import json
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from app.experiments import llm_artifacts
from app.experiments.artifacts import (
    DatasetArtifactIdentity,
    RuntimeVersions,
    canonical_json_bytes,
)
from app.experiments.llm_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PromptIdentity,
    PublishedRealRun,
    RealArtifactWriter,
    RealCaseRunRecord,
    RealOutputFileHashes,
    RealRunArtifactBundle,
    RealRunManifest,
    RealRunSummary,
    StagedRealRun,
    StoredPromptSnapshot,
    canonical_real_jsonl_bytes,
    verify_published_real_run,
)
from app.prompts.base import compute_prompt_sha256

SHA = "a" * 64
RUN_ID = "real-v1__20261009T000000Z__abcd1234"
PROMPT_TEXT = "系统\n\n完整指令。\n"
DEVELOPMENT_DATASET_ID = "e1e81533c25e03e5"
VALIDATION_DATASET_ID = "0123456789abcdef"


def make_record(
    sequence: int = 1,
    case_id: str = "CASE_001",
    *,
    split: str = "development",
    agent_version: str = "v1",
) -> RealCaseRunRecord:
    return RealCaseRunRecord(
        sequence=sequence,
        case_id=case_id,
        split=split,
        dataset_id=(
            DEVELOPMENT_DATASET_ID
            if split == "development"
            else VALIDATION_DATASET_ID
        ),
        dataset_version="1.0.0",
        agent_version=agent_version,
        repeat_index=1,
        user_input=f"诊断 {case_id} 最近30天 GMV 变化",
        status="completed",
        final_answer="GMV 上升",
        error_code=None,
        prior_tool_executions=(
            {
                "call_id": "call_001",
                "tool_name": "query_sales",
                "arguments": {"product_ids": ["P001"]},
                "result": {"result_id": "result_001", "rows": [{"gmv": 42.5}]},
            },
        ),
        decision_trace=(
            {
                "sequence": 1,
                "event_type": "tool_result",
                "payload": {"result_id": "result_001", "row_count": 1},
            },
        ),
        latency_ms=100 if sequence == 1 else 200,
        prompt_tokens=7 if sequence == 1 else 20,
        completion_tokens=3 if sequence == 1 else 10,
        total_tokens=10 if sequence == 1 else 30,
    )


def prompt() -> StoredPromptSnapshot:
    return StoredPromptSnapshot(
        prompt_name="real-llm-v1",
        prompt_version="1.0.0",
        prompt_text=PROMPT_TEXT,
        prompt_sha256=compute_prompt_sha256(PROMPT_TEXT),
        change_note="initial",
        source_file="app/prompts/v1.py",
    )


def summary() -> RealRunSummary:
    return RealRunSummary(
        total=2,
        completed=2,
        failed=0,
        stopped=0,
        split_counts={"development": 1, "public_validation": 1},
        split_status_counts={
            "development": {"completed": 1, "failed": 0},
            "public_validation": {"completed": 1, "failed": 0},
        },
        first_case_id="CASE_001",
        last_case_id="CASE_002",
        latency={"p50": 100, "p95": 200, "sample_size": 2},
        tokens={
            "available": True,
            "prompt_total": 27,
            "completion_total": 13,
            "total_total": 40,
            "p50": 10,
            "p95": 30,
            "sample_size": 2,
        },
        cost="unavailable",
        evaluation_status="pending_not_run",
    )


def manifest(*, run_id: str = RUN_ID) -> RealRunManifest:
    snapshot = prompt()
    return RealRunManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        run_id=run_id,
        comparison_group_id="formal__20261009T000000Z__abcd1234",
        experiment_mode="formal",
        agent_version="v1",
        started_at_utc="2026-10-09T00:00:00.000000Z",
        completed_at_utc="2026-10-09T00:01:00.000000Z",
        git_commit="0" * 40,
        git_dirty=False,
        runtime_versions=RuntimeVersions(
            python="3.12.0",
            pydantic="2.11.0",
            duckdb="1.3.0",
            pandas="2.3.0",
            pyarrow="19.0.0",
        ),
        case_set_id="1" * 16,
        case_jsonl_sha256=SHA,
        case_manifest_sha256=SHA,
        case_subset_id="all-cases-v1",
        case_subset_sha256=SHA,
        datasets=(
            DatasetArtifactIdentity(
                split="development",
                dataset_id=DEVELOPMENT_DATASET_ID,
                dataset_version="1.0.0",
                manifest_sha256=SHA,
            ),
            DatasetArtifactIdentity(
                split="public_validation",
                dataset_id=VALIDATION_DATASET_ID,
                dataset_version="1.0.0",
                manifest_sha256=SHA,
            ),
        ),
        tool_contract_version="1.0",
        tool_contract_sha256=SHA,
        adapter_name="openai-compatible",
        model="ep-test",
        base_url_host="ark.cn-beijing.volces.com",
        temperature=0.0,
        timeout_seconds=30.0,
        max_steps=8,
        concurrency=1,
        repeats=1,
        prompt={
            "prompt_name": snapshot.prompt_name,
            "prompt_version": snapshot.prompt_version,
            "source_file": snapshot.source_file,
            "prompt_sha256": snapshot.prompt_sha256,
        },
        output_files=None,
        evaluation_status="pending_not_run",
        usage_status="available",
        cost_status="unavailable",
    )


def bundle() -> RealRunArtifactBundle:
    return RealRunArtifactBundle(
        records=(
            make_record(),
            make_record(2, "CASE_002", split="public_validation"),
        ),
        expected_case_ids=("CASE_001", "CASE_002"),
        summary=summary(),
        prompt_snapshot=prompt(),
        manifest=manifest(),
    )


def rewrite_output_hash(path: Path, file_name: str) -> None:
    manifest_path = path / "run_manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["output_files"][file_name] = sha256((path / file_name).read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(payload))


def test_artifact_v2_writer_public_api_is_importable() -> None:
    assert PromptIdentity
    assert PublishedRealRun
    assert RealArtifactWriter
    assert RealOutputFileHashes
    assert RealRunArtifactBundle
    assert RealRunManifest
    assert StagedRealRun


def test_canonical_real_jsonl_roundtrip_preserves_all_record_fields() -> None:
    records = (
        make_record(),
        make_record(2, "CASE_002", split="public_validation"),
    )

    payload = canonical_real_jsonl_bytes(records)
    lines = payload.splitlines()

    assert payload.endswith(b"\n")
    assert len(lines) == 2
    assert [json.loads(line)["case_id"] for line in lines] == [
        "CASE_001",
        "CASE_002",
    ]
    assert set(json.loads(lines[0])) == set(
        records[0].model_dump(mode="json")
    )


def test_record_json_roundtrip_preserves_nonempty_transcript_and_trace() -> None:
    record = make_record()

    restored = RealCaseRunRecord.model_validate_json(
        record.model_dump_json(),
        strict=True,
    )

    assert restored == record
    assert restored.prior_tool_executions == record.prior_tool_executions
    assert restored.decision_trace == record.decision_trace


def test_stage_does_not_publish_until_commit_and_verifier_reads_result(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)

    staged = writer.stage(bundle())

    assert isinstance(staged, StagedRealRun)
    assert staged.temp_dir.is_dir()
    assert not staged.final_dir.exists()
    final = writer.commit(staged)
    published = verify_published_real_run(final)
    assert published.manifest.run_id == RUN_ID
    assert tuple(item.case_id for item in published.records) == (
        "CASE_001",
        "CASE_002",
    )
    assert (tmp_path / ".real-artifact-publish.lock").exists()


def test_publish_rejects_overwrite_without_changing_existing_run(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    final = writer.publish(bundle())
    original_manifest = (final / "run_manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        writer.publish(bundle())

    assert (final / "run_manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


def test_commit_rejects_racing_overwrite_and_cleans_staging(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    staged.final_dir.mkdir()

    with pytest.raises(FileExistsError):
        writer.commit(staged)

    assert not staged.temp_dir.exists()
    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)


def test_commit_rejects_staged_run_from_another_writer(
    tmp_path: Path,
) -> None:
    owner = RealArtifactWriter(tmp_path)
    staged = owner.stage(bundle())

    with pytest.raises(ValueError, match="staged"):
        RealArtifactWriter(tmp_path).commit(staged)

    assert staged.temp_dir.is_dir()
    assert not staged.final_dir.exists()


def test_commit_rejects_replacement_at_verified_staging_path(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    verified_dir = tmp_path / "verified-staging"
    staged.temp_dir.rename(verified_dir)
    staged.temp_dir.mkdir()
    replacement_file = staged.temp_dir / "unverified.txt"
    replacement_file.write_text("unverified", encoding="utf-8")

    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)

    assert replacement_file.read_text(encoding="utf-8") == "unverified"
    assert verified_dir.is_dir()
    assert (verified_dir / "run_manifest.json").is_file()
    assert not staged.final_dir.exists()


@pytest.mark.parametrize("mutation", ["tamper_summary", "add_extra_file"])
def test_commit_reverifies_staged_contents_before_publish(
    tmp_path: Path,
    mutation: str,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    if mutation == "tamper_summary":
        (staged.temp_dir / "summary.json").write_bytes(b"{}\n")
    else:
        (staged.temp_dir / "extra.json").write_bytes(b"{}\n")

    with pytest.raises(ValueError):
        writer.commit(staged)

    assert not staged.temp_dir.exists()
    assert not staged.final_dir.exists()
    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)


def test_commit_rejects_self_consistent_staged_content_rewrite(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    records_path = staged.temp_dir / "case_runs.jsonl"
    payload = [
        json.loads(line) for line in records_path.read_bytes().splitlines()
    ]
    payload[0]["final_answer"] = "另一条合法回答"
    records_path.write_bytes(
        b"".join(canonical_json_bytes(item) for item in payload)
    )
    rewrite_output_hash(staged.temp_dir, "case_runs.jsonl")

    with pytest.raises(ValueError, match="fingerprint"):
        writer.commit(staged)

    assert not staged.temp_dir.exists()
    assert not staged.final_dir.exists()
    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)


def test_commit_reverify_failure_does_not_delete_unknown_replacement(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    verified_dir = tmp_path / "verified-staging"
    replacement_file = staged.temp_dir / "replacement.txt"

    def replace_staging_then_fail(path: Path) -> None:
        assert path == staged.temp_dir
        staged.temp_dir.rename(verified_dir)
        staged.temp_dir.mkdir()
        replacement_file.write_text("unverified", encoding="utf-8")
        raise ValueError("reverify failed")

    with mock.patch(
        "app.experiments.llm_artifacts.verify_published_real_run",
        side_effect=replace_staging_then_fail,
    ):
        with pytest.raises(ValueError, match="reverify"):
            writer.commit(staged)

    assert replacement_file.read_text(encoding="utf-8") == "unverified"
    assert verified_dir.is_dir()
    assert not staged.final_dir.exists()


def test_staging_identity_failure_does_not_consume_capability(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    verified_dir = tmp_path / "verified-staging"
    replacement_dir = tmp_path / "replacement-staging"
    staged.temp_dir.rename(verified_dir)
    staged.temp_dir.mkdir()

    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)

    staged.temp_dir.rename(replacement_dir)
    verified_dir.rename(staged.temp_dir)

    assert writer.commit(staged) == staged.final_dir
    assert replacement_dir.is_dir()
    assert staged.final_dir.is_dir()


def test_commit_rejects_forged_staged_run_without_touching_external_paths(
    tmp_path: Path,
) -> None:
    external_temp = tmp_path / "external-temp"
    external_final = tmp_path / "external-final"
    external_temp.mkdir()
    external_final.mkdir()
    forged = StagedRealRun(
        temp_dir=external_temp,
        final_dir=external_final,
    )

    with pytest.raises(ValueError, match="staged"):
        RealArtifactWriter(tmp_path / "outputs").commit(forged)

    assert external_temp.is_dir()
    assert external_final.is_dir()


def test_commit_rejects_replayed_staged_run(tmp_path: Path) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    final = writer.commit(staged)

    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)

    assert final.is_dir()


def test_open_verified_staging_directory_closes_fd_when_fstat_fails(
    tmp_path: Path,
) -> None:
    identity = llm_artifacts._directory_identity(tmp_path.lstat())

    with (
        mock.patch.object(
            llm_artifacts.os,
            "fstat",
            side_effect=OSError("fstat failed"),
        ),
        mock.patch.object(
            llm_artifacts.os,
            "close",
            wraps=llm_artifacts.os.close,
        ) as close,
    ):
        with pytest.raises(OSError, match="fstat failed"):
            llm_artifacts._open_verified_staging_directory(
                tmp_path,
                identity,
            )

    close.assert_called_once()


def test_commit_rejects_registered_staged_run_with_unsafe_paths(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path / "outputs")
    staged = writer.stage(bundle())
    original_temp = staged.temp_dir
    external_final = tmp_path / "external-final"
    object.__setattr__(staged, "final_dir", external_final)

    with pytest.raises(ValueError, match="staged"):
        writer.commit(staged)

    assert original_temp.is_dir()
    assert not external_final.exists()


def test_rename_directory_exclusive_rejects_existing_empty_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        llm_artifacts._rename_directory_exclusive(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_commit_atomic_rename_rejects_destination_created_after_precheck(
    tmp_path: Path,
) -> None:
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    atomic_rename = llm_artifacts._rename_directory_exclusive

    def create_destination_then_rename(
        source: Path,
        destination: Path,
    ) -> None:
        destination.mkdir()
        (destination / "racer").write_text("wins", encoding="utf-8")
        atomic_rename(source, destination)

    with mock.patch(
        "app.experiments.llm_artifacts._rename_directory_exclusive",
        side_effect=create_destination_then_rename,
    ):
        with pytest.raises(FileExistsError):
            writer.commit(staged)

    assert (staged.final_dir / "racer").read_text(encoding="utf-8") == "wins"
    assert not staged.temp_dir.exists()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("prompt_name", "other-prompt"),
        ("prompt_version", "2.0.0"),
        ("source_file", "app/prompts/other.py"),
        ("prompt_sha256", "b" * 64),
    ],
)
def test_stage_rejects_every_manifest_prompt_identity_mismatch(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
) -> None:
    original = bundle()
    identity = original.manifest.prompt.model_copy(
        update={field_name: bad_value}
    )
    invalid = original.model_copy(
        update={
            "manifest": original.manifest.model_copy(
                update={"prompt": identity}
            )
        }
    )

    with pytest.raises(ValueError, match="prompt"):
        RealArtifactWriter(tmp_path).stage(invalid)

    assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


def test_stage_rejects_prompt_text_changed_without_updating_its_digest(
    tmp_path: Path,
) -> None:
    original = bundle()
    invalid = original.model_copy(
        update={
            "prompt_snapshot": original.prompt_snapshot.model_copy(
                update={"prompt_text": "different"}
            )
        }
    )

    with pytest.raises(ValueError, match="prompt"):
        RealArtifactWriter(tmp_path).stage(invalid)

    assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


def test_verifier_rejects_prompt_text_tampering_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    snapshot_path = final / "prompt_snapshot.json"
    payload = json.loads(snapshot_path.read_bytes())
    payload["prompt_text"] = "tampered"
    snapshot_path.write_bytes(canonical_json_bytes(payload))
    rewrite_output_hash(final, "prompt_snapshot.json")

    with pytest.raises(ValueError, match="prompt"):
        verify_published_real_run(final)


@pytest.mark.parametrize(
    ("file_name", "mutate", "message"),
    [
        (
            "summary.json",
            lambda payload: payload["latency"].update(p50=101),
            "latency",
        ),
        (
            "summary.json",
            lambda payload: payload["tokens"].update(total_total=41),
            "token",
        ),
        (
            "run_manifest.json",
            lambda payload: payload.update(usage_status="unavailable"),
            "usage",
        ),
        (
            "run_manifest.json",
            lambda payload: payload["datasets"].pop(),
            "dataset",
        ),
    ],
)
def test_verifier_rejects_forged_record_derived_metadata(
    tmp_path: Path,
    file_name: str,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    path = final / file_name
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(canonical_json_bytes(payload))
    if file_name != "run_manifest.json":
        rewrite_output_hash(final, file_name)

    with pytest.raises(ValueError, match=message):
        verify_published_real_run(final)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("prompt_name", "other-prompt"),
        ("prompt_version", "2.0.0"),
        ("source_file", "app/prompts/other.py"),
        ("prompt_sha256", "b" * 64),
    ],
)
def test_verifier_rejects_manifest_prompt_identity_mismatch(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    manifest_path = final / "run_manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["prompt"][field_name] = bad_value
    manifest_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="prompt"):
        verify_published_real_run(final)


def test_verifier_rejects_output_file_tampering(tmp_path: Path) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    with (final / "summary.json").open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(ValueError, match="summary.json"):
        verify_published_real_run(final)


def test_verifier_rejects_record_token_mismatch_with_updated_hash(
    tmp_path: Path,
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    records_path = final / "case_runs.jsonl"
    payload = [
        json.loads(line) for line in records_path.read_bytes().splitlines()
    ]
    payload[0]["total_tokens"] = 11
    records_path.write_bytes(
        b"".join(canonical_json_bytes(item) for item in payload)
    )
    rewrite_output_hash(final, "case_runs.jsonl")
    summary_path = final / "summary.json"
    summary_payload = json.loads(summary_path.read_bytes())
    summary_payload["tokens"].update(
        total_total=41,
        p50=11,
    )
    summary_path.write_bytes(canonical_json_bytes(summary_payload))
    rewrite_output_hash(final, "summary.json")

    with pytest.raises(
        ValidationError,
        match="total_tokens must equal prompt_tokens plus completion_tokens",
    ):
        verify_published_real_run(final)


def test_verifier_rejects_noncanonical_json_with_matching_hash(
    tmp_path: Path,
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    summary_path = final / "summary.json"
    payload = json.loads(summary_path.read_bytes())
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rewrite_output_hash(final, "summary.json")

    with pytest.raises(ValueError, match="canonical"):
        verify_published_real_run(final)


def test_verifier_rejects_extra_file(tmp_path: Path) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    (final / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly four"):
        verify_published_real_run(final)


@pytest.mark.parametrize(
    "file_name",
    [
        "case_runs.jsonl",
        "summary.json",
        "prompt_snapshot.json",
        "run_manifest.json",
    ],
)
def test_verifier_rejects_symlink_artifact(
    tmp_path: Path,
    file_name: str,
) -> None:
    final = RealArtifactWriter(tmp_path / "outputs").publish(bundle())
    artifact = final / file_name
    target = tmp_path / f"target-{file_name}"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(target)

    with pytest.raises(ValueError, match="regular non-symlink"):
        verify_published_real_run(final)


def test_verifier_rejects_symlink_directory(tmp_path: Path) -> None:
    final = RealArtifactWriter(tmp_path / "outputs").publish(bundle())
    link = tmp_path / "published-link"
    link.symlink_to(final, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink directory"):
        verify_published_real_run(link)


@pytest.mark.parametrize(
    "contents",
    [
        lambda original: original.rstrip(b"\n"),
        lambda original: original.replace(b"\n", b"\n\n", 1),
    ],
)
def test_verifier_rejects_jsonl_newline_or_empty_line(
    tmp_path: Path,
    contents: Callable[[bytes], bytes],
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    records_path = final / "case_runs.jsonl"
    records_path.write_bytes(contents(records_path.read_bytes()))
    rewrite_output_hash(final, "case_runs.jsonl")

    with pytest.raises(ValueError, match="case_runs.jsonl"):
        verify_published_real_run(final)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload[1].update(sequence=1),
        lambda payload: payload[1].update(case_id="CASE_001"),
    ],
)
def test_verifier_rejects_jsonl_sequence_or_duplicate_id(
    tmp_path: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    final = RealArtifactWriter(tmp_path).publish(bundle())
    records_path = final / "case_runs.jsonl"
    payload = [
        json.loads(line) for line in records_path.read_bytes().splitlines()
    ]
    mutate(payload)
    records_path.write_bytes(
        b"".join(canonical_json_bytes(item) for item in payload)
    )
    rewrite_output_hash(final, "case_runs.jsonl")

    with pytest.raises(ValueError):
        verify_published_real_run(final)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.model_copy(
            update={"expected_case_ids": ("CASE_002", "CASE_001")}
        ),
        lambda value: value.model_copy(
            update={"expected_case_ids": ("CASE_001", "CASE_001")}
        ),
        lambda value: value.model_copy(
            update={
                "records": (
                    make_record(),
                    make_record(1, "CASE_002", split="public_validation"),
                )
            }
        ),
        lambda value: value.model_copy(
            update={"summary": value.summary.model_copy(update={"total": 3})}
        ),
        lambda value: value.model_copy(
            update={"summary": value.summary.model_copy(update={"completed": 1, "failed": 1})}
        ),
        lambda value: value.model_copy(
            update={
                "summary": value.summary.model_copy(
                    update={
                        "split_counts": value.summary.split_counts.model_copy(
                            update={"development": 2, "public_validation": 0}
                        )
                    }
                )
            }
        ),
        lambda value: value.model_copy(
            update={
                "summary": value.summary.model_copy(
                    update={
                        "split_status_counts": value.summary.split_status_counts.model_copy(
                            update={
                                "development": (
                                    value.summary.split_status_counts.development.model_copy(
                                    update={"completed": 0, "failed": 1}
                                    )
                                )
                            }
                        )
                    }
                )
            }
        ),
        lambda value: value.model_copy(
            update={"summary": value.summary.model_copy(update={"first_case_id": "CASE_002"})}
        ),
        lambda value: value.model_copy(
            update={"summary": value.summary.model_copy(update={"last_case_id": "CASE_001"})}
        ),
        lambda value: value.model_copy(
            update={"summary": value.summary.model_copy(update={"stopped": 1, "completed": 1})}
        ),
        lambda value: value.model_copy(
            update={
                "records": (
                    make_record(agent_version="v2"),
                    make_record(
                        2,
                        "CASE_002",
                        split="public_validation",
                        agent_version="v2",
                    ),
                )
            }
        ),
    ],
)
def test_stage_revalidates_bundle_cross_field_invariants(
    tmp_path: Path,
    mutate: Callable[[RealRunArtifactBundle], RealRunArtifactBundle],
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        RealArtifactWriter(tmp_path).stage(mutate(bundle()))

    if tmp_path.exists():
        assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.model_copy(
                update={
                    "summary": value.summary.model_copy(
                        update={
                            "latency": value.summary.latency.model_copy(
                                update={"p50": 101}
                            )
                        }
                    )
                }
            ),
            "latency",
        ),
        (
            lambda value: value.model_copy(
                update={
                    "summary": value.summary.model_copy(
                        update={
                            "tokens": value.summary.tokens.model_copy(
                                update={"total_total": 41}
                            )
                        }
                    )
                }
            ),
            "token",
        ),
        (
            lambda value: value.model_copy(
                update={
                    "manifest": value.manifest.model_copy(
                        update={"usage_status": "unavailable"}
                    )
                }
            ),
            "usage",
        ),
        (
            lambda value: value.model_copy(
                update={
                    "manifest": value.manifest.model_copy(
                        update={"datasets": value.manifest.datasets[:1]}
                    )
                }
            ),
            "dataset",
        ),
        (
            lambda value: value.model_copy(
                update={
                    "manifest": value.manifest.model_copy(
                        update={
                            "datasets": (
                                value.manifest.datasets[0],
                                value.manifest.datasets[0],
                                value.manifest.datasets[1],
                            )
                        }
                    )
                }
            ),
            "dataset",
        ),
    ],
)
def test_stage_rejects_forged_record_derived_metadata(
    tmp_path: Path,
    mutate: Callable[[RealRunArtifactBundle], RealRunArtifactBundle],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealArtifactWriter(tmp_path).stage(mutate(bundle()))

    if tmp_path.exists():
        assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


def test_stage_rejects_published_manifest(tmp_path: Path) -> None:
    original = bundle()
    hashes = RealOutputFileHashes(
        {
            "case_runs.jsonl": SHA,
            "summary.json": SHA,
            "prompt_snapshot.json": SHA,
        }
    )
    invalid = original.model_copy(
        update={
            "manifest": original.manifest.model_copy(
                update={"output_files": hashes}
            )
        }
    )

    with pytest.raises(ValueError, match="output hashes"):
        RealArtifactWriter(tmp_path).stage(invalid)


def test_stage_cleans_temporary_directory_when_readback_fails(
    tmp_path: Path,
) -> None:
    with mock.patch(
        "app.experiments.llm_artifacts.verify_published_real_run",
        side_effect=ValueError("readback failed"),
    ):
        with pytest.raises(ValueError, match="readback"):
            RealArtifactWriter(tmp_path).stage(bundle())

    assert not list(tmp_path.glob(f".{RUN_ID}.tmp-*"))


def test_real_output_hashes_require_exact_names_and_lowercase_sha256() -> None:
    with pytest.raises(ValidationError):
        RealOutputFileHashes({"case_runs.jsonl": SHA})
    with pytest.raises(ValidationError):
        RealOutputFileHashes(
            {
                "case_runs.jsonl": SHA.upper(),
                "summary.json": SHA,
                "prompt_snapshot.json": SHA,
            }
        )


def test_manifest_enforces_fixed_schema_and_positive_execution_counts() -> None:
    payload = deepcopy(manifest().model_dump(mode="python"))
    payload["artifact_schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        RealRunManifest.model_validate(payload, strict=True)

    for field_name in ("max_steps", "concurrency", "repeats"):
        payload = deepcopy(manifest().model_dump(mode="python"))
        payload[field_name] = 0
        with pytest.raises(ValidationError):
            RealRunManifest.model_validate(payload, strict=True)
