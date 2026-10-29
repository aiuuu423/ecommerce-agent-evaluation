import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from inspect import signature
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from app.data.generator import build_snapshot
from app.experiments import llm_run
from app.experiments.baseline_v1 import ToolContractIdentity, tool_contract_identity
from app.experiments.cases import (
    CaseBundle,
    DatasetIdentity,
    EvaluationCaseManifest,
    RunnableCase,
)
from app.experiments.llm_artifacts import (
    RealArtifactWriter,
    verify_published_real_run,
)
from app.experiments.llm_run import (
    PairedRunCoordinator,
    PairedRunResult,
    RunLevelError,
    RunSettings,
)
from app.experiments.llm_settings import ProviderConfigError, ProviderSettings
from app.experiments.pilot_cases import PILOT_CASE_IDS, pilot_case_set_sha256

ROOT = Path(__file__).parents[1]
DEVELOPMENT_CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
PUBLIC_VALIDATION_CONFIG = ROOT / "configs/data/synthetic_public_validation_v1.yaml"
TOOL_CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"
TOOL_CONTRACT_SHA256 = sha256(TOOL_CONTRACT.read_bytes()).hexdigest()
SHA = "a" * 64


class ImmediateAnswerTransport:
    def __init__(self, *, include_usage: bool = True) -> None:
        self.include_usage = include_usage
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": deepcopy(headers),
                "payload": deepcopy(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        response: dict[str, Any] = {
            "choices": [{"message": {"content": "离线固定回答"}}]
        }
        if self.include_usage and (len(self.calls) - 1) % 4 < 2:
            response["usage"] = {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        return response


class FailOnceTransport(ImmediateAnswerTransport):
    def __init__(self, fail_call: int) -> None:
        super().__init__()
        self.fail_call = fail_call

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        call_number = len(self.calls) + 1
        if call_number == self.fail_call:
            self.calls.append(
                {
                    "url": url,
                    "headers": deepcopy(headers),
                    "payload": deepcopy(payload),
                    "timeout_seconds": timeout_seconds,
                }
            )
            raise RuntimeError("secret provider failure body")
        return super().post(
            url,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )


@pytest.fixture(scope="module")
def dataset_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("real-llm-run")
    paths = {
        "development": root / "development",
        "public_validation": root / "public-validation",
    }
    build_snapshot(DEVELOPMENT_CONFIG, paths["development"])
    build_snapshot(PUBLIC_VALIDATION_CONFIG, paths["public_validation"])
    return paths


def _dataset_metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _datasets(dataset_dirs: dict[str, Path]) -> dict[str, DatasetIdentity]:
    result = {}
    for split, path in dataset_dirs.items():
        manifest = _dataset_metadata(path)
        result[split] = DatasetIdentity(
            dataset_id=manifest["dataset_id"],
            dataset_version=manifest["dataset_version"],
            generator_config_hash=manifest["config_sha256"],
        )
    return result


def _bundle(
    dataset_dirs: dict[str, Path],
    splits: tuple[str, ...],
) -> CaseBundle:
    datasets = _datasets(dataset_dirs)
    cases = tuple(
        RunnableCase(
            case_id=f"CASE_{index:03d}",
            user_input=f"请离线分析 CASE_{index:03d} 的经营表现",
            split=split,
            dataset_id=datasets[split].dataset_id,
            dataset_version=datasets[split].dataset_version,
        )
        for index, split in enumerate(splits, start=1)
    )
    split_counts = Counter(splits)
    return CaseBundle(
        manifest=EvaluationCaseManifest(
            business_task_counts={"test": len(cases)},
            capability_counts={"test": len(cases)},
            case_count=len(cases),
            case_schema_version="1.0",
            case_set_id="1" * 16,
            datasets=datasets,
            difficulty_counts={"test": len(cases)},
            jsonl_sha256=SHA,
            source_label="Task 9 local tests",
            split_counts={
                "development": split_counts["development"],
                "public_validation": split_counts["public_validation"],
            },
            split_strategy={"kind": "test"},
            tool_contract={
                "version": "1.0",
                "sha256": TOOL_CONTRACT_SHA256,
            },
        ),
        cases=cases,
        manifest_sha256=SHA,
    )


def _pilot_bundle(dataset_dirs: dict[str, Path]) -> CaseBundle:
    public_ids = set(
        case_id
        for case_id in (f"CASE_{index:03d}" for index in range(1, 101))
        if case_id not in PILOT_CASE_IDS
    )
    public_ids = set(sorted(public_ids)[:50])
    return _bundle(
        dataset_dirs,
        tuple(
            "public_validation" if f"CASE_{index:03d}" in public_ids else "development"
            for index in range(1, 101)
        ),
    )


def _formal_bundle(dataset_dirs: dict[str, Path]) -> CaseBundle:
    return _bundle(
        dataset_dirs,
        ("development",) * 50 + ("public_validation",) * 50,
    )


def _mapping(
    bundle: CaseBundle,
    dataset_dirs: dict[str, Path],
) -> dict[str, Path]:
    return {
        bundle.manifest.datasets[split].dataset_id: dataset_dirs[split]
        for split in ("development", "public_validation")
    }


def _provider(*, temperature: float = 0.0) -> ProviderSettings:
    return ProviderSettings(
        api_key="offline-secret",
        model="offline-model",
        base_url="https://offline.example/v1",
        temperature=temperature,
        timeout_seconds=12.0,
    )


def _coordinator(transport: ImmediateAnswerTransport) -> PairedRunCoordinator:
    return PairedRunCoordinator(
        provider=_provider(),
        transport=transport,
        tool_contract_path=TOOL_CONTRACT,
    )


def _payload_version(payload: dict[str, Any]) -> str:
    system_message = payload["messages"][0]
    assert system_message["role"] == "system"
    prompt_text = system_message["content"]
    if prompt_text == llm_run.v1_prompt_snapshot().prompt_text:
        return "v1"
    if prompt_text == llm_run.v2_prompt_snapshot().prompt_text:
        return "v2"
    raise AssertionError("transport call used an unknown system prompt")


def _payload_without_system(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    del normalized["messages"][0]
    return normalized


def test_task9_public_api_is_frozen_and_error_codes_are_exact() -> None:
    settings = RunSettings()
    result_fields = tuple(PairedRunResult.__dataclass_fields__)
    run_signature = signature(PairedRunCoordinator.run)

    assert settings == RunSettings(max_steps=8, concurrency=1, repeats=1)
    assert get_type_hints(PairedRunResult)["mode"] is str
    assert run_signature.parameters["mode"].annotation is str
    assert get_type_hints(PairedRunCoordinator.run)["mode"] is str
    assert result_fields == (
        "comparison_group_id",
        "mode",
        "v1",
        "v2",
        "v1_path",
        "v2_path",
        "version_order",
        "case_ids",
    )
    assert llm_run._SAFE_ERROR_CODES == {
        "catalog_not_found",
        "unknown_tool",
        "invalid_arguments",
        "execution_failed",
        "max_steps_exceeded",
        "adapter_error",
        "adapter_http_error",
        "rate_limited",
        "response_format_error",
        "json_repair_failed",
    }
    with pytest.raises(FrozenInstanceError):
        settings.max_steps = 7  # type: ignore[misc]


def test_pilot_runs_frozen_cases_and_publishes_verified_pair(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    transport = ImmediateAnswerTransport()

    result = _coordinator(transport).run(
        mode="pilot",
        cases=bundle,
        dataset_dirs=_mapping(bundle, dataset_dirs),
        run_output_root=tmp_path,
    )

    assert isinstance(result, PairedRunResult)
    assert result.case_ids == PILOT_CASE_IDS
    assert result.version_order == {
        case_id: ("v1", "v2") for case_id in PILOT_CASE_IDS
    }
    assert result.v1_path.is_dir() and result.v2_path.is_dir()
    assert result.v1.manifest.case_subset_id == "pilot-20-v1"
    assert result.v1.manifest.case_subset_sha256 == pilot_case_set_sha256()
    assert result.v1.manifest.comparison_group_id == result.comparison_group_id
    assert result.v2.manifest.comparison_group_id == result.comparison_group_id
    assert result.comparison_group_id.startswith("pilot__")
    for published, version in ((result.v1, "v1"), (result.v2, "v2")):
        run_id_pattern = rf"real-{version}-pilot__\d{{8}}T\d{{6}}Z__[0-9a-f]{{8}}"
        assert re.fullmatch(run_id_pattern, published.manifest.run_id)
        assert (
            result.v1_path if version == "v1" else result.v2_path
        ).name == published.manifest.run_id
        assert published == verify_published_real_run(
            result.v1_path if version == "v1" else result.v2_path
        )
        assert len(published.records) == 20
        assert tuple(record.case_id for record in published.records) == PILOT_CASE_IDS
        assert tuple(record.sequence for record in published.records) == tuple(
            range(1, 21)
        )
        assert {record.agent_version for record in published.records} == {version}
        assert published.summary.total == published.summary.completed == 20
        assert published.summary.failed == 0
        assert published.summary.tokens.available is True
        assert published.summary.tokens.sample_size == 10
        assert published.summary.tokens.total_total == 100
        assert len(published.manifest.datasets) == 1
        assert published.manifest.datasets[0].split == "development"
        assert all(
            isinstance(event, dict)
            for record in published.records
            for event in record.decision_trace
        )
        assert all(
            isinstance(execution, dict)
            for record in published.records
            for execution in record.prior_tool_executions
        )
    assert len(transport.calls) == 40
    for call_index in range(0, 40, 2):
        first = transport.calls[call_index]["payload"]
        second = transport.calls[call_index + 1]["payload"]
        assert (_payload_version(first), _payload_version(second)) == ("v1", "v2")
        assert _payload_without_system(first) == _payload_without_system(second)


def test_formal_alternates_execution_but_preserves_record_order(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    bundle = _formal_bundle(dataset_dirs)
    transport = ImmediateAnswerTransport(include_usage=False)

    result = _coordinator(transport).run(
        mode="formal",
        cases=bundle,
        dataset_dirs=_mapping(bundle, dataset_dirs),
        run_output_root=tmp_path,
    )

    case_ids = tuple(f"CASE_{index:03d}" for index in range(1, 101))
    assert result.case_ids == case_ids
    assert result.version_order == {
        case_id: (("v1", "v2") if index % 2 else ("v2", "v1"))
        for index, case_id in enumerate(case_ids, start=1)
    }
    assert result.comparison_group_id.startswith("formal__")
    expected_subset_sha256 = (
        "e2dd978689d5674aa6b1ffcdda27d3dc9b5e1b2f8a65487fe50514f3b3d30d11"
    )
    for published, version, path in (
        (result.v1, "v1", result.v1_path),
        (result.v2, "v2", result.v2_path),
    ):
        assert re.fullmatch(
            rf"real-{version}__\d{{8}}T\d{{6}}Z__[0-9a-f]{{8}}",
            published.manifest.run_id,
        )
        assert path.name == published.manifest.run_id
        assert tuple(record.case_id for record in published.records) == case_ids
        assert tuple(record.sequence for record in published.records) == tuple(
            range(1, 101)
        )
        assert published.summary.total == 100
        assert published.summary.tokens.available is False
        assert published.summary.tokens.sample_size == 0
        assert len(published.manifest.datasets) == 2
        assert published.manifest.case_subset_id == "all-100-v1"
        assert published.manifest.case_subset_sha256 == expected_subset_sha256
    assert len(transport.calls) == 200
    for case_index, call_index in enumerate(range(0, 200, 2), start=1):
        first = transport.calls[call_index]["payload"]
        second = transport.calls[call_index + 1]["payload"]
        expected_order = ("v1", "v2") if case_index % 2 else ("v2", "v1")
        assert (_payload_version(first), _payload_version(second)) == expected_order
        assert _payload_without_system(first) == _payload_without_system(second)


def test_pilot_uses_fresh_catalog_registry_and_adapter_per_version_case(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    catalogs: list[object] = []
    registries: list[object] = []
    adapters: list[object] = []
    original_open_dataset = llm_run.open_dataset
    original_build_registry = llm_run.build_default_registry
    original_adapter = llm_run.OpenAICompatibleAdapter

    def tracked_open_dataset(path: Path) -> object:
        catalog = original_open_dataset(path)
        catalogs.append(catalog)
        return catalog

    def tracked_build_registry() -> object:
        registry = original_build_registry()
        registries.append(registry)
        return registry

    def tracked_adapter(**kwargs: object) -> object:
        adapter = original_adapter(**kwargs)  # type: ignore[arg-type]
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(llm_run, "open_dataset", tracked_open_dataset)
    monkeypatch.setattr(llm_run, "build_default_registry", tracked_build_registry)
    monkeypatch.setattr(llm_run, "OpenAICompatibleAdapter", tracked_adapter)

    _coordinator(ImmediateAnswerTransport()).run(
        mode="pilot",
        cases=bundle,
        dataset_dirs=_mapping(bundle, dataset_dirs),
        run_output_root=tmp_path,
    )

    assert len(catalogs) == 42
    assert len({id(catalog) for catalog in catalogs}) == 42
    assert len(registries) == 40
    assert len({id(registry) for registry in registries}) == 40
    assert len(adapters) == 40
    assert len({id(adapter) for adapter in adapters}) == 40


def test_transport_failure_is_isolated_without_retry_or_exception_text(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    transport = FailOnceTransport(fail_call=3)

    result = _coordinator(transport).run(
        mode="pilot",
        cases=bundle,
        dataset_dirs=_mapping(bundle, dataset_dirs),
        run_output_root=tmp_path,
    )

    failed = result.v1.records[1]
    assert failed.case_id == "CASE_005"
    assert failed.status == "failed"
    assert failed.error_code == "adapter_error"
    assert failed.final_answer is None
    assert result.v1.summary.failed == 1
    assert result.v2.summary.failed == 0
    assert len(transport.calls) == 40
    artifact_text = (result.v1_path / "case_runs.jsonl").read_text(encoding="utf-8")
    assert "secret provider failure body" not in artifact_text


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_steps": 7},
        {"concurrency": 2},
        {"repeats": 2},
    ],
)
def test_invalid_run_settings_are_rejected_at_construction(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        RunSettings(**overrides)


def test_invalid_mode_provider_mapping_and_dataset_directory_fail_before_calls(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    mapping = _mapping(bundle, dataset_dirs)

    transport = ImmediateAnswerTransport()
    with pytest.raises(RunLevelError):
        _coordinator(transport).run(
            mode="dry-run",
            cases=bundle,
            dataset_dirs=mapping,
            run_output_root=tmp_path / "mode",
        )
    assert transport.calls == []

    transport = ImmediateAnswerTransport()
    with pytest.raises((ValueError, ProviderConfigError)):
        PairedRunCoordinator(
            provider=_provider(temperature=0.1),
            transport=transport,
            tool_contract_path=TOOL_CONTRACT,
        )
    assert transport.calls == []

    transport = ImmediateAnswerTransport()
    with pytest.raises(RunLevelError):
        _coordinator(transport).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs={},
            run_output_root=tmp_path / "mapping",
        )
    assert transport.calls == []

    transport = ImmediateAnswerTransport()
    swapped = {
        bundle.manifest.datasets["development"].dataset_id: dataset_dirs[
            "public_validation"
        ],
        bundle.manifest.datasets["public_validation"].dataset_id: dataset_dirs[
            "development"
        ],
    }
    with pytest.raises(RunLevelError):
        _coordinator(transport).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs=swapped,
            run_output_root=tmp_path / "identity",
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"api_key": ""}, "missing_api_key"),
        ({"model": " \t "}, "missing_model"),
        ({"base_url": "not-a-url"}, "invalid_base_url"),
        ({"base_url": "https://url-secret@example.test/v1"}, "invalid_base_url"),
        ({"base_url": "https://example.test/v1?secret=1"}, "invalid_base_url"),
        ({"base_url": "https://example.test/v1#secret"}, "invalid_base_url"),
        ({"base_url": "https://example.test:99999/v1"}, "invalid_base_url"),
        ({"temperature": 0.1}, "invalid_temperature"),
        ({"timeout_seconds": -1.0}, "invalid_timeout"),
        ({"timeout_seconds": float("nan")}, "invalid_timeout"),
    ],
)
def test_invalid_provider_settings_fail_at_construction_without_transport_calls(
    overrides: dict[str, object],
    code: str,
) -> None:
    transport = ImmediateAnswerTransport()
    provider = replace(_provider(), **overrides)

    with pytest.raises((ValueError, ProviderConfigError)) as caught:
        PairedRunCoordinator(
            provider=provider,
            transport=transport,
            tool_contract_path=TOOL_CONTRACT,
        )

    assert str(caught.value) == code
    assert transport.calls == []
    assert "offline-secret" not in repr(caught.value)
    assert "url-secret" not in repr(caught.value)


def test_coordinator_stores_normalized_provider_settings() -> None:
    coordinator = PairedRunCoordinator(
        provider=ProviderSettings(
            api_key="  offline-secret  ",
            model="  offline-model  ",
            base_url="  https://offline.example/v1///  ",
            temperature=0.0,
            timeout_seconds=12.0,
        ),
        transport=ImmediateAnswerTransport(),
        tool_contract_path=TOOL_CONTRACT,
    )

    assert coordinator.provider == ProviderSettings(
        api_key="offline-secret",
        model="offline-model",
        base_url="https://offline.example/v1",
        temperature=0.0,
        timeout_seconds=12.0,
    )


@pytest.mark.parametrize("identity_kind", ["dataset", "tool_contract"])
def test_identity_drift_rejects_pair_without_publishing(
    identity_kind: str,
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    transport = ImmediateAnswerTransport()
    output_root = tmp_path / identity_kind
    if identity_kind == "dataset":
        original = llm_run.dataset_source_identity
        call_count = 0

        def drifting_dataset_identity(mapping: dict[str, Path]) -> object:
            nonlocal call_count
            call_count += 1
            identity = original(mapping)
            if call_count == 2:
                return (*identity, ("drift", "b" * 64, "c" * 64))
            return identity

        monkeypatch.setattr(
            llm_run,
            "dataset_source_identity",
            drifting_dataset_identity,
        )
        expected_code = "dataset_identity_drift"
    else:
        initial = tool_contract_identity(TOOL_CONTRACT)
        identities = iter(
            (
                initial,
                ToolContractIdentity(
                    version=initial.version,
                    sha256="b" * 64,
                    required_parameters=initial.required_parameters,
                ),
            )
        )
        monkeypatch.setattr(llm_run, "tool_contract_identity", lambda _path: next(identities))
        expected_code = "tool_contract_identity_drift"

    with pytest.raises(RunLevelError, match=expected_code):
        _coordinator(transport).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs=_mapping(bundle, dataset_dirs),
            run_output_root=output_root,
        )

    assert len(transport.calls) == 40
    assert not output_root.exists() or not any(path.is_dir() for path in output_root.iterdir())


def test_second_commit_failure_rolls_back_first_and_cleans_staging(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    output_root = tmp_path / "publish-failure"
    original_commit = RealArtifactWriter.commit
    commit_count = 0

    def fail_second_commit(
        self: RealArtifactWriter,
        staged: object,
    ) -> Path:
        nonlocal commit_count
        commit_count += 1
        if commit_count == 2:
            original_commit(self, staged)  # type: ignore[arg-type]
            raise OSError("injected second commit failure")
        return original_commit(self, staged)  # type: ignore[arg-type]

    monkeypatch.setattr(RealArtifactWriter, "commit", fail_second_commit)

    with pytest.raises(RunLevelError, match="artifact_publish_error"):
        _coordinator(ImmediateAnswerTransport()).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs=_mapping(bundle, dataset_dirs),
            run_output_root=output_root,
        )

    assert commit_count == 2
    assert output_root.is_dir()
    assert not any(path.is_dir() for path in output_root.iterdir())
    assert not any(".tmp-" in path.name for path in output_root.iterdir())


def test_publish_cleanup_failure_still_attempts_other_side(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    output_root = tmp_path / "cleanup-failure"
    original_commit = RealArtifactWriter.commit
    original_rollback = RealArtifactWriter.rollback_published
    commit_count = 0
    rollback_paths: list[Path] = []

    def fail_after_second_commit(
        self: RealArtifactWriter,
        staged: object,
    ) -> Path:
        nonlocal commit_count
        commit_count += 1
        result = original_commit(self, staged)  # type: ignore[arg-type]
        if commit_count == 2:
            raise OSError("injected publish wrapper failure")
        return result

    def fail_first_rollback(
        self: RealArtifactWriter,
        path: Path,
    ) -> None:
        rollback_paths.append(path)
        if len(rollback_paths) == 1:
            raise OSError("injected cleanup failure")
        original_rollback(self, path)

    monkeypatch.setattr(RealArtifactWriter, "commit", fail_after_second_commit)
    monkeypatch.setattr(
        RealArtifactWriter,
        "rollback_published",
        fail_first_rollback,
    )

    with pytest.raises(RunLevelError, match=r"^artifact_publish_error$") as caught:
        _coordinator(ImmediateAnswerTransport()).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs=_mapping(bundle, dataset_dirs),
            run_output_root=output_root,
        )

    assert caught.value.__cause__ is None
    assert commit_count == 2
    assert len(rollback_paths) == 2
    assert rollback_paths[0] != rollback_paths[1]
    assert not rollback_paths[1].exists()


def test_publish_rollback_does_not_delete_replacement_final(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _pilot_bundle(dataset_dirs)
    output_root = tmp_path / "replacement"
    original_commit = RealArtifactWriter.commit
    commit_count = 0
    replacement_path: Path | None = None
    marker: Path | None = None

    def replace_after_second_commit(
        self: RealArtifactWriter,
        staged: object,
    ) -> Path:
        nonlocal commit_count, replacement_path, marker
        commit_count += 1
        result = original_commit(self, staged)  # type: ignore[arg-type]
        if commit_count == 2:
            replacement_path = result
            result.rename(output_root / "original-published")
            result.mkdir()
            marker = result / "replacement.txt"
            marker.write_text("keep", encoding="utf-8")
            raise OSError("injected replacement failure")
        return result

    monkeypatch.setattr(RealArtifactWriter, "commit", replace_after_second_commit)

    with pytest.raises(RunLevelError, match=r"^artifact_publish_error$"):
        _coordinator(ImmediateAnswerTransport()).run(
            mode="pilot",
            cases=bundle,
            dataset_dirs=_mapping(bundle, dataset_dirs),
            run_output_root=output_root,
        )

    assert replacement_path is not None
    assert marker is not None
    assert replacement_path.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep"
