import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest import mock

import pytest

import app.experiments.comparisons as comparisons
from app.experiments.artifacts import (
    DatasetArtifactIdentity,
    RuntimeVersions,
    canonical_json_bytes,
)
from app.experiments.comparisons import (
    ComparisonError,
    PublishedComparison,
    _version_order_valid,
    verify_comparison,
    write_comparison,
)
from app.experiments.llm_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PromptIdentity,
    RealArtifactWriter,
    RealCaseRunRecord,
    RealRunArtifactBundle,
    RealRunManifest,
    RealRunSummary,
    StoredPromptSnapshot,
    verify_published_real_run,
    verify_published_real_run_snapshot,
)
from app.experiments.llm_run import PairedRunResult
from app.prompts.base import compute_prompt_sha256

SHA_A = "a" * 64
SHA_B = "b" * 64
GROUP_ID = "pilot__20261029T000000Z__abcd1234"
CASE_IDS = tuple(f"CASE_{index:03d}" for index in range(1, 21))


def _prompt(agent_version: str) -> StoredPromptSnapshot:
    text = f"System prompt for {agent_version}.\n"
    return StoredPromptSnapshot(
        prompt_name=f"real-llm-{agent_version}",
        prompt_version="1.0.0",
        prompt_text=text,
        prompt_sha256=compute_prompt_sha256(text),
        change_note=f"Frozen {agent_version} prompt.",
        source_file=f"app/prompts/{agent_version}.py",
    )


def _records(
    agent_version: str,
    *,
    user_input_suffix: str = "",
) -> tuple[RealCaseRunRecord, ...]:
    return tuple(
        RealCaseRunRecord(
            sequence=index,
            case_id=case_id,
            split="development",
            dataset_id="0123456789abcdef",
            dataset_version="1.0.0",
            agent_version=agent_version,
            repeat_index=1,
            user_input=f"Analyze {case_id}{user_input_suffix}",
            status="completed",
            final_answer=f"Observed answer from {agent_version}",
            error_code=None,
            prior_tool_executions=(),
            decision_trace=(),
            latency_ms=index,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        for index, case_id in enumerate(CASE_IDS, start=1)
    )


def _summary() -> RealRunSummary:
    return RealRunSummary(
        total=20,
        completed=20,
        failed=0,
        stopped=0,
        split_counts={"development": 20, "public_validation": 0},
        split_status_counts={
            "development": {"completed": 20, "failed": 0},
            "public_validation": {"completed": 0, "failed": 0},
        },
        first_case_id=CASE_IDS[0],
        last_case_id=CASE_IDS[-1],
        latency={"p50": 10, "p95": 19, "sample_size": 20},
        tokens={
            "available": False,
            "prompt_total": None,
            "completion_total": None,
            "total_total": None,
            "p50": None,
            "p95": None,
            "sample_size": 0,
        },
        cost="unavailable",
        evaluation_status="pending_not_run",
    )


def _bundle(
    agent_version: str,
    *,
    manifest_updates: dict[str, object] | None = None,
    user_input_suffix: str = "",
) -> RealRunArtifactBundle:
    snapshot = _prompt(agent_version)
    records = _records(agent_version, user_input_suffix=user_input_suffix)
    manifest = RealRunManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        run_id=f"real-{agent_version}-pilot__20261029T000000Z__abcd1234",
        comparison_group_id=GROUP_ID,
        experiment_mode="pilot",
        agent_version=agent_version,
        started_at_utc="2026-10-29T00:00:00.000000Z",
        completed_at_utc="2026-10-29T00:01:00.000000Z",
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
        case_jsonl_sha256=SHA_A,
        case_manifest_sha256=SHA_B,
        case_subset_id="pilot-20-v1",
        case_subset_sha256="c" * 64,
        datasets=(
            DatasetArtifactIdentity(
                split="development",
                dataset_id="0123456789abcdef",
                dataset_version="1.0.0",
                manifest_sha256="d" * 64,
            ),
        ),
        tool_contract_version="1.0.0",
        tool_contract_sha256="e" * 64,
        adapter_name="openai-compatible",
        model="ep-test",
        base_url_host="ark.example.test",
        temperature=0.0,
        timeout_seconds=30.0,
        max_steps=8,
        concurrency=1,
        repeats=1,
        prompt=PromptIdentity(
            prompt_name=snapshot.prompt_name,
            prompt_version=snapshot.prompt_version,
            source_file=snapshot.source_file,
            prompt_sha256=snapshot.prompt_sha256,
        ),
        output_files=None,
        evaluation_status="pending_not_run",
        usage_status="unavailable",
        cost_status="unavailable",
    )
    if manifest_updates:
        manifest = manifest.model_copy(update=manifest_updates)
    return RealRunArtifactBundle(
        records=records,
        expected_case_ids=CASE_IDS,
        summary=_summary(),
        prompt_snapshot=snapshot,
        manifest=manifest,
    )


def _paired(
    tmp_path: Path,
    *,
    v1_bundle: RealRunArtifactBundle | None = None,
    v2_bundle: RealRunArtifactBundle | None = None,
) -> tuple[PairedRunResult, Path]:
    writer = RealArtifactWriter(tmp_path / "runs")
    v1_path = writer.publish(v1_bundle or _bundle("v1"))
    v2_path = writer.publish(v2_bundle or _bundle("v2"))
    v1 = verify_published_real_run(v1_path)
    v2 = verify_published_real_run(v2_path)
    result = PairedRunResult(
        comparison_group_id=GROUP_ID,
        mode="pilot",
        v1=v1,
        v2=v2,
        v1_path=v1_path,
        v2_path=v2_path,
        version_order={case_id: ("v1", "v2") for case_id in CASE_IDS},
        case_ids=CASE_IDS,
    )
    return result, tmp_path / "comparisons"


def _rewrite_comparison(path: Path, mutate: object) -> None:
    file_path = path / "comparison.json"
    payload = json.loads(file_path.read_bytes())
    mutate(payload)
    file_path.write_bytes(canonical_json_bytes(payload))


def test_comparison_publishes_strict_manifest_bound_to_run_files(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)

    path = write_comparison(result, comparisons_root=comparisons_root)
    published = verify_comparison(path)

    assert isinstance(published, PublishedComparison)
    assert published.comparison_group_id == GROUP_ID
    assert published.v1_run_id == result.v1.manifest.run_id
    assert published.v2_run_id == result.v2.manifest.run_id
    assert published.case_ids == CASE_IDS
    assert published.fairness.passed is True
    assert published.fairness.failures == ()
    v1_snapshot = verify_published_real_run_snapshot(result.v1_path)
    v2_snapshot = verify_published_real_run_snapshot(result.v2_path)
    assert published.v1_run_manifest_sha256 == v1_snapshot.run_manifest_sha256
    assert published.v2_run_manifest_sha256 == v2_snapshot.run_manifest_sha256
    assert published.v1_case_runs_sha256 == v1_snapshot.case_runs_sha256
    assert published.v2_case_runs_sha256 == v2_snapshot.case_runs_sha256
    assert (path / "comparison.json").read_bytes().endswith(b"\n")


def test_publish_rejects_self_consistent_run_swap_between_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    replacement_bundle = _bundle("v1").model_copy(
        update={
            "records": tuple(
                record.model_copy(update={"final_answer": "Replacement answer"})
                for record in _bundle("v1").records
            )
        }
    )
    replacement_path = RealArtifactWriter(tmp_path / "replacement").publish(
        replacement_bundle
    )
    original_verifier = verify_published_real_run_snapshot
    calls = 0

    def swap_before_second_v1_snapshot(path: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 3:
            result.v1_path.rename(tmp_path / "original-v1")
            replacement_path.rename(result.v1_path)
        return original_verifier(path)

    monkeypatch.setattr(
        comparisons,
        "verify_published_real_run_snapshot",
        swap_before_second_v1_snapshot,
    )

    with pytest.raises(ComparisonError, match="published run verification changed"):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


@pytest.mark.parametrize(
    ("updates", "expected_check"),
    [
        ({"model": "other-model"}, "model_match"),
        ({"timeout_seconds": 31.0}, "timeout_seconds_match"),
        (
            {
                "runtime_versions": RuntimeVersions(
                    python="3.13.0",
                    pydantic="2.11.0",
                    duckdb="1.3.0",
                    pandas="2.3.0",
                    pyarrow="19.0.0",
                )
            },
            "runtime_versions_match",
        ),
        (
            {
                "datasets": (
                    DatasetArtifactIdentity(
                        split="development",
                        dataset_id="0123456789abcdef",
                        dataset_version="1.0.0",
                        manifest_sha256="f" * 64,
                    ),
                )
            },
            "datasets_match",
        ),
        ({"tool_contract_sha256": "f" * 64}, "tool_contract_match"),
    ],
)
def test_fairness_rejects_must_same_manifest_differences_before_root_creation(
    tmp_path: Path,
    updates: dict[str, object],
    expected_check: str,
) -> None:
    result, comparisons_root = _paired(
        tmp_path,
        v2_bundle=_bundle("v2", manifest_updates=updates),
    )

    with pytest.raises(ComparisonError, match=expected_check):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    [
        ("user_input", "record_pairs_match"),
        ("same_prompt", "prompts_distinct"),
    ],
)
def test_fairness_rejects_record_or_prompt_identity_differences(
    tmp_path: Path,
    mutation: str,
    expected_check: str,
) -> None:
    v2_bundle = _bundle(
        "v2",
        user_input_suffix=" changed" if mutation == "user_input" else "",
    )
    if mutation == "same_prompt":
        v1_prompt = _prompt("v1")
        v2_bundle = v2_bundle.model_copy(
            update={
                "prompt_snapshot": v1_prompt,
                "manifest": v2_bundle.manifest.model_copy(
                    update={
                        "prompt": PromptIdentity(
                            prompt_name=v1_prompt.prompt_name,
                            prompt_version=v1_prompt.prompt_version,
                            source_file=v1_prompt.source_file,
                            prompt_sha256=v1_prompt.prompt_sha256,
                        )
                    }
                ),
            }
        )
    result, comparisons_root = _paired(tmp_path, v2_bundle=v2_bundle)

    with pytest.raises(ComparisonError, match=expected_check):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


def test_fairness_rejects_tampered_version_order_without_creating_root(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    result.version_order[CASE_IDS[0]] = ("v2", "v1")

    with pytest.raises(ComparisonError, match="version_order_valid"):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


@pytest.mark.parametrize(
    ("field_name", "value", "expected_check"),
    [
        ("comparison_group_id", "other-group", "result_group_match"),
        ("mode", "formal", "result_mode_match"),
        ("case_ids", CASE_IDS[:-1], "result_case_ids_match"),
    ],
)
def test_fairness_rejects_result_metadata_that_disagrees_with_disk(
    tmp_path: Path,
    field_name: str,
    value: object,
    expected_check: str,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    result = replace(result, **{field_name: value})

    with pytest.raises(ComparisonError, match=expected_check):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


def test_published_run_tamper_is_rejected_before_fairness(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    (result.v2_path / "case_runs.jsonl").write_bytes(b"{}\n")

    with pytest.raises(ComparisonError, match="published run verification failed"):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


def test_published_run_sequence_tamper_is_rejected_before_fairness(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    records_path = result.v2_path / "case_runs.jsonl"
    records = [json.loads(line) for line in records_path.read_bytes().splitlines()]
    records[0]["sequence"] = 2
    records_path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))
    manifest_path = result.v2_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["output_files"]["case_runs.jsonl"] = sha256(records_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ComparisonError, match="published run verification failed"):
        write_comparison(result, comparisons_root=comparisons_root)

    assert not comparisons_root.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "noncanonical",
        "tamper",
        "fairness_false",
        "extra_file",
        "file_symlink",
        "directory_symlink",
    ],
)
def test_verifier_rejects_invalid_comparison_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    path = write_comparison(result, comparisons_root=comparisons_root)
    verify_path = path
    if mutation == "noncanonical":
        file_path = path / "comparison.json"
        payload = json.loads(file_path.read_bytes())
        file_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif mutation == "tamper":
        _rewrite_comparison(
            path,
            lambda payload: payload.__setitem__(
                "case_ids",
                payload["case_ids"][:-1],
            ),
        )
    elif mutation == "fairness_false":

        def fail_fairness(payload: dict[str, object]) -> None:
            fairness = payload["fairness"]
            assert isinstance(fairness, dict)
            checks = fairness["checks"]
            assert isinstance(checks, dict)
            first_check = next(iter(checks))
            checks[first_check] = False
            fairness["passed"] = False
            fairness["failures"] = [first_check]

        _rewrite_comparison(path, fail_fairness)
    elif mutation == "extra_file":
        (path / "extra.json").write_bytes(b"{}\n")
    elif mutation == "file_symlink":
        file_path = path / "comparison.json"
        real_file = path / "real.json"
        file_path.rename(real_file)
        file_path.symlink_to(real_file.name)
    else:
        verify_path = tmp_path / "comparison-link"
        verify_path.symlink_to(path, target_is_directory=True)

    with pytest.raises(ComparisonError):
        verify_comparison(verify_path)


def test_publish_rejects_overwrite_without_changing_existing(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    path = write_comparison(result, comparisons_root=comparisons_root)
    original = (path / "comparison.json").read_bytes()

    with pytest.raises(ComparisonError, match="already published"):
        write_comparison(result, comparisons_root=comparisons_root)

    assert (path / "comparison.json").read_bytes() == original


def test_atomic_publish_rejects_racing_target_and_preserves_racer(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    atomic_rename = comparisons._rename_directory_exclusive
    target = comparisons_root / GROUP_ID

    def race(source: Path, destination: Path) -> None:
        assert destination == target
        destination.mkdir()
        (destination / "racer").write_text("wins", encoding="utf-8")
        atomic_rename(source, destination)

    with mock.patch(
        "app.experiments.comparisons._rename_directory_exclusive",
        side_effect=race,
    ):
        with pytest.raises(ComparisonError, match="already published"):
            write_comparison(result, comparisons_root=comparisons_root)

    assert (target / "racer").read_text(encoding="utf-8") == "wins"
    assert not list(comparisons_root.glob(f".{GROUP_ID}.tmp-*"))


def test_publish_rejects_staging_content_replaced_before_rename(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    atomic_rename = comparisons._rename_directory_exclusive
    target = comparisons_root / GROUP_ID

    def replace_staging_contents(source: Path, destination: Path) -> None:
        _rewrite_comparison(
            source,
            lambda payload: payload.__setitem__(
                "v1_run_manifest_sha256",
                "0" * 64,
            ),
        )
        verify_comparison(source)
        atomic_rename(source, destination)

    with mock.patch(
        "app.experiments.comparisons._rename_directory_exclusive",
        side_effect=replace_staging_contents,
    ):
        with pytest.raises(ComparisonError):
            write_comparison(result, comparisons_root=comparisons_root)

    assert not target.exists()
    assert not list(comparisons_root.glob(f".{GROUP_ID}.tmp-*"))


def test_manifest_rejects_non_ascii_case_id_digits(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    path = write_comparison(result, comparisons_root=comparisons_root)
    payload = verify_comparison(path).model_dump(mode="python")
    unicode_case_id = "CASE_\u0661\u0662\u0663"
    original_case_id = payload["case_ids"][0]
    payload["case_ids"] = (unicode_case_id, *payload["case_ids"][1:])
    payload["version_order"][unicode_case_id] = payload["version_order"].pop(
        original_case_id
    )

    with pytest.raises(ValueError, match="CASE_NNN"):
        comparisons.ComparisonManifest.model_validate(payload, strict=True)


def test_verifier_rejects_canonical_non_ascii_case_id_digits(
    tmp_path: Path,
) -> None:
    result, comparisons_root = _paired(tmp_path)
    path = write_comparison(result, comparisons_root=comparisons_root)
    file_path = path / "comparison.json"
    payload = json.loads(file_path.read_bytes())
    unicode_case_id = "CASE_\u0661\u0662\u0663"
    original_case_id = payload["case_ids"][0]
    payload["case_ids"][0] = unicode_case_id
    payload["version_order"][unicode_case_id] = payload["version_order"].pop(
        original_case_id
    )
    file_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ComparisonError, match="validation failed"):
        verify_comparison(path)


def test_version_order_validation_uses_case_id_order_for_formal() -> None:
    formal_ids = tuple(f"CASE_{index:03d}" for index in range(1, 101))
    valid = {
        case_id: ("v1", "v2") if index % 2 else ("v2", "v1")
        for index, case_id in enumerate(formal_ids, start=1)
    }
    reverse_inserted = dict(reversed(tuple(valid.items())))

    assert _version_order_valid("formal", reverse_inserted, formal_ids)
    assert not _version_order_valid(
        "formal",
        {**valid, "CASE_101": ("v1", "v2")},
        formal_ids,
    )
    assert not _version_order_valid(
        "formal",
        {case_id: order for case_id, order in valid.items() if case_id != "CASE_050"},
        formal_ids,
    )
    invalid_pair = dict(valid)
    invalid_pair["CASE_001"] = ("v1", "v1")
    assert not _version_order_valid("formal", invalid_pair, formal_ids)
