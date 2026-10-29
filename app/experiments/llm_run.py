import math
import secrets
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from app.agents import AgentRunner, RunRequest, RunResult, TraceEventType
from app.data.database import open_dataset
from app.experiments.artifacts import (
    DatasetArtifactIdentity,
    RuntimeVersions,
    canonical_json_bytes,
)
from app.experiments.baseline_v1 import (
    RunLevelError,
    dataset_source_identity,
    tool_contract_identity,
    validate_registry_contract,
)
from app.experiments.cases import CaseBundle, RunnableCase
from app.experiments.llm_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PromptIdentity,
    PublishedRealRun,
    RealArtifactWriter,
    RealCaseRunRecord,
    RealRunArtifactBundle,
    RealRunManifest,
    RealRunSummary,
    StagedRealRun,
    StoredPromptSnapshot,
    verify_published_real_run,
)
from app.experiments.llm_settings import ProviderSettings, base_url_host
from app.experiments.pilot_cases import pilot_case_set_sha256, select_pilot_cases
from app.llm.openai_compatible import JsonTransport, OpenAICompatibleAdapter
from app.prompts import v1_prompt_snapshot, v2_prompt_snapshot
from app.prompts.base import PromptSnapshot
from app.tools import build_default_registry
from app.tools.schemas import canonical_tool_result_payload

ExperimentMode = Literal["pilot", "formal"]
AgentVersion = Literal["v1", "v2"]
_PROJECT_ROOT = Path(__file__).parents[2]

_SAFE_ERROR_CODES = {
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


@dataclass(frozen=True)
class RunSettings:
    max_steps: int = 8
    concurrency: int = 1
    repeats: int = 1

    def __post_init__(self) -> None:
        values = (self.max_steps, self.concurrency, self.repeats)
        if any(type(value) is not int for value in values) or values != (8, 1, 1):
            raise ValueError("run settings are fixed at max_steps=8, concurrency=1, repeats=1")


@dataclass(frozen=True)
class PairedRunResult:
    comparison_group_id: str
    mode: ExperimentMode
    v1: PublishedRealRun
    v2: PublishedRealRun
    v1_path: Path
    v2_path: Path
    version_order: dict[str, tuple[str, ...]]
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TimedResult:
    result: RunResult
    latency_ms: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _comparison_group_id(mode: ExperimentMode, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}__{timestamp}__{secrets.token_hex(4)}"


def _git_identity() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunLevelError("git_identity_error") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RunLevelError("git_identity_error")
    return commit, bool(status)


def _runtime_versions() -> RuntimeVersions:
    try:
        return RuntimeVersions(
            python=sys.version.split()[0],
            pydantic=version("pydantic"),
            duckdb=version("duckdb"),
            pandas=version("pandas"),
            pyarrow=version("pyarrow"),
        )
    except Exception as exc:
        raise RunLevelError("runtime_identity_error") from exc


def _case_error_code(result: RunResult) -> str:
    for event in reversed(result.decision_trace):
        if event.event_type is TraceEventType.ERROR:
            code = event.payload.get("code")
            if isinstance(code, str) and code in _SAFE_ERROR_CODES:
                return code
            break
    return "case_execution_error"


def _failed_result() -> RunResult:
    return RunResult(
        status="failed",
        final_answer=None,
        prior_tool_executions=[],
        decision_trace=(),
        usage=None,
    )


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _summary(records: tuple[RealCaseRunRecord, ...]) -> RealRunSummary:
    statuses = Counter(record.status for record in records)
    splits = Counter(record.split for record in records)
    split_statuses = Counter((record.split, record.status) for record in records)
    latencies = [record.latency_ms for record in records]
    records_with_usage = [record for record in records if record.total_tokens is not None]
    total_tokens = [
        record.total_tokens
        for record in records_with_usage
        if record.total_tokens is not None
    ]
    usage_available = bool(records_with_usage)
    return RealRunSummary(
        total=len(records),
        completed=statuses["completed"],
        failed=statuses["failed"],
        stopped=0,
        split_counts={
            "development": splits["development"],
            "public_validation": splits["public_validation"],
        },
        split_status_counts={
            split: {
                "completed": split_statuses[(split, "completed")],
                "failed": split_statuses[(split, "failed")],
            }
            for split in ("development", "public_validation")
        },
        first_case_id=records[0].case_id if records else None,
        last_case_id=records[-1].case_id if records else None,
        latency={
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "sample_size": len(records),
        },
        tokens={
            "available": usage_available,
            "prompt_total": (
                sum(record.prompt_tokens or 0 for record in records_with_usage)
                if usage_available
                else None
            ),
            "completion_total": (
                sum(record.completion_tokens or 0 for record in records_with_usage)
                if usage_available
                else None
            ),
            "total_total": sum(total_tokens) if usage_available else None,
            "p50": _percentile(total_tokens, 0.50) if usage_available else None,
            "p95": _percentile(total_tokens, 0.95) if usage_available else None,
            "sample_size": len(records_with_usage),
        },
        cost="unavailable",
        evaluation_status="pending_not_run",
    )


def _stored_prompt(snapshot: PromptSnapshot) -> StoredPromptSnapshot:
    return StoredPromptSnapshot.model_validate(snapshot.model_dump(mode="python"), strict=True)


def _prompt_identity(snapshot: StoredPromptSnapshot) -> PromptIdentity:
    return PromptIdentity(
        prompt_name=snapshot.prompt_name,
        prompt_version=snapshot.prompt_version,
        source_file=snapshot.source_file,
        prompt_sha256=snapshot.prompt_sha256,
    )


def _formal_subset_sha256(case_ids: tuple[str, ...]) -> str:
    return sha256(canonical_json_bytes({"case_ids": case_ids})).hexdigest()


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = path.lstat()
    except OSError:
        return None
    return metadata.st_dev, metadata.st_ino


def _remove_known_directory(path: Path, identity: tuple[int, int], root: Path) -> None:
    try:
        resolved_parent = path.parent.resolve()
        metadata = path.lstat()
    except OSError:
        return
    if (
        resolved_parent == root
        and stat.S_ISDIR(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    ):
        shutil.rmtree(path)


class PairedRunCoordinator:
    def __init__(
        self,
        *,
        provider: ProviderSettings,
        run_settings: RunSettings = RunSettings(),  # noqa: B008
        transport: JsonTransport | None = None,
        tool_contract_path: Path = Path("configs/evaluation/tool_contract_v1.yaml"),
    ) -> None:
        if not isinstance(provider, ProviderSettings):
            raise TypeError("provider must be ProviderSettings")
        if provider.temperature != 0.0:
            raise ValueError("provider temperature must be zero")
        if run_settings != RunSettings():
            raise ValueError("run settings must use the frozen defaults")
        self.provider = provider
        self.run_settings = run_settings
        self.transport = transport
        self.tool_contract_path = tool_contract_path

    def run(
        self,
        *,
        mode: ExperimentMode,
        cases: CaseBundle,
        dataset_dirs: Mapping[str, Path],
        run_output_root: Path,
    ) -> PairedRunResult:
        selected = self._preflight(mode, cases, dataset_dirs)
        selected_ids = {case.dataset_id for case in selected}
        selected_dataset_dirs = {
            dataset_id: dataset_dirs[dataset_id] for dataset_id in selected_ids
        }
        try:
            initial_dataset_identity = dataset_source_identity(selected_dataset_dirs)
        except (OSError, ValueError) as exc:
            raise RunLevelError("dataset_identity_error") from exc
        initial_tool_contract = tool_contract_identity(self.tool_contract_path)
        self._validate_case_contract(
            cases,
            initial_tool_contract.version,
            initial_tool_contract.sha256,
        )
        validate_registry_contract(initial_tool_contract)

        started_at = _utc_now()
        comparison_group_id = _comparison_group_id(mode, started_at)
        snapshots = {
            "v1": _stored_prompt(v1_prompt_snapshot()),
            "v2": _stored_prompt(v2_prompt_snapshot()),
        }
        results: dict[str, dict[str, _TimedResult]] = {"v1": {}, "v2": {}}
        version_order: dict[str, tuple[str, ...]] = {}
        for index, case in enumerate(selected, start=1):
            order: tuple[AgentVersion, AgentVersion]
            if mode == "pilot" or index % 2:
                order = ("v1", "v2")
            else:
                order = ("v2", "v1")
            version_order[case.case_id] = order
            for agent_version in order:
                results[agent_version][case.case_id] = self._execute(
                    case,
                    dataset_dirs[case.dataset_id],
                    snapshots[agent_version].prompt_text,
                )

        try:
            final_dataset_identity = dataset_source_identity(selected_dataset_dirs)
        except (OSError, ValueError) as exc:
            raise RunLevelError("dataset_identity_error") from exc
        if final_dataset_identity != initial_dataset_identity:
            raise RunLevelError("dataset_identity_drift")
        final_tool_contract = tool_contract_identity(self.tool_contract_path)
        if final_tool_contract != initial_tool_contract:
            raise RunLevelError("tool_contract_identity_drift")
        validate_registry_contract(final_tool_contract)

        completed_at = _utc_now()
        git_commit, git_dirty = _git_identity()
        runtime_versions = _runtime_versions()
        manifest_hashes = {
            dataset_id: manifest_hash
            for dataset_id, manifest_hash, _source_hash in initial_dataset_identity
        }
        case_ids = tuple(case.case_id for case in selected)
        if mode == "pilot":
            subset_id = "pilot-20-v1"
            subset_sha256 = pilot_case_set_sha256()
        else:
            subset_id = "all-100-v1"
            subset_sha256 = _formal_subset_sha256(case_ids)

        bundles = {
            agent_version: self._bundle(
                agent_version=agent_version,
                mode=mode,
                comparison_group_id=comparison_group_id,
                selected=selected,
                timed_results=results[agent_version],
                snapshot=snapshots[agent_version],
                cases=cases,
                started_at=started_at,
                completed_at=completed_at,
                git_commit=git_commit,
                git_dirty=git_dirty,
                runtime_versions=runtime_versions,
                manifest_hashes=manifest_hashes,
                subset_id=subset_id,
                subset_sha256=subset_sha256,
                tool_contract_version=initial_tool_contract.version,
                tool_contract_sha256=initial_tool_contract.sha256,
            )
            for agent_version in ("v1", "v2")
        }
        v1_path, v2_path, v1, v2 = self._publish_pair(
            run_output_root,
            bundles["v1"],
            bundles["v2"],
        )
        return PairedRunResult(
            comparison_group_id=comparison_group_id,
            mode=mode,
            v1=v1,
            v2=v2,
            v1_path=v1_path,
            v2_path=v2_path,
            version_order=version_order,
            case_ids=case_ids,
        )

    def _preflight(
        self,
        mode: object,
        cases: CaseBundle,
        dataset_dirs: Mapping[str, Path],
    ) -> tuple[RunnableCase, ...]:
        if mode not in {"pilot", "formal"}:
            raise RunLevelError("invalid_experiment_mode")
        expected_by_split = cases.manifest.datasets
        expected_ids = {
            expected_by_split["development"].dataset_id,
            expected_by_split["public_validation"].dataset_id,
        }
        if len(expected_ids) != 2:
            raise RunLevelError("dataset_split_id_collision")
        if set(dataset_dirs) != expected_ids:
            raise RunLevelError("dataset_mapping_error")
        if len({case.case_id for case in cases.cases}) != len(cases.cases):
            raise RunLevelError("case_identity_error")
        for case in cases.cases:
            expected = expected_by_split[case.split]
            if (
                case.dataset_id != expected.dataset_id
                or case.dataset_version != expected.dataset_version
            ):
                raise RunLevelError("dataset_mapping_error")
        for split in ("development", "public_validation"):
            expected = expected_by_split[split]
            try:
                with open_dataset(dataset_dirs[expected.dataset_id]) as catalog:
                    summary = catalog.verified_summary
                    manifest = catalog.manifest
                    if (
                        summary["dataset_id"] != expected.dataset_id
                        or summary["dataset_version"] != expected.dataset_version
                        or manifest.get("dataset_id") != expected.dataset_id
                        or manifest.get("dataset_version") != expected.dataset_version
                        or manifest.get("config_sha256") != expected.generator_config_hash
                    ):
                        raise RunLevelError("dataset_identity_mismatch")
            except RunLevelError:
                raise
            except Exception as exc:
                raise RunLevelError("dataset_identity_mismatch") from exc
        if mode == "pilot":
            try:
                return select_pilot_cases(cases)
            except Exception as exc:
                raise RunLevelError("pilot_case_selection_error") from exc
        if len(cases.cases) != 100:
            raise RunLevelError("formal_case_count_error")
        return cases.cases

    @staticmethod
    def _validate_case_contract(cases: CaseBundle, version: str, digest: str) -> None:
        if (
            cases.manifest.tool_contract.version != version
            or cases.manifest.tool_contract.sha256 != digest
        ):
            raise RunLevelError("tool_contract_identity_mismatch")

    def _execute(
        self,
        case: RunnableCase,
        dataset_dir: Path,
        system_prompt: str,
    ) -> _TimedResult:
        started = time.perf_counter_ns()
        try:
            with open_dataset(dataset_dir) as catalog:
                adapter = OpenAICompatibleAdapter(
                    base_url=self.provider.base_url,
                    api_key=self.provider.api_key,
                    model=self.provider.model,
                    temperature=self.provider.temperature,
                    timeout_seconds=self.provider.timeout_seconds,
                    system_prompt=system_prompt,
                    transport=self.transport,
                )
                runner = AgentRunner(
                    registry=build_default_registry(),
                    adapter=adapter,
                )
                result = runner.run(
                    RunRequest(
                        user_input=case.user_input,
                        max_steps=self.run_settings.max_steps,
                    ),
                    catalog,
                )
        except Exception:
            result = _failed_result()
        elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
        return _TimedResult(result=result, latency_ms=elapsed_ms)

    def _bundle(
        self,
        *,
        agent_version: AgentVersion,
        mode: ExperimentMode,
        comparison_group_id: str,
        selected: tuple[RunnableCase, ...],
        timed_results: Mapping[str, _TimedResult],
        snapshot: StoredPromptSnapshot,
        cases: CaseBundle,
        started_at: datetime,
        completed_at: datetime,
        git_commit: str,
        git_dirty: bool,
        runtime_versions: RuntimeVersions,
        manifest_hashes: Mapping[str, str],
        subset_id: str,
        subset_sha256: str,
        tool_contract_version: str,
        tool_contract_sha256: str,
    ) -> RealRunArtifactBundle:
        records = tuple(
            self._record(sequence, case, agent_version, timed_results[case.case_id])
            for sequence, case in enumerate(selected, start=1)
        )
        observed_splits = {case.split for case in selected}
        datasets = tuple(
            DatasetArtifactIdentity(
                split=split,
                dataset_id=cases.manifest.datasets[split].dataset_id,
                dataset_version=cases.manifest.datasets[split].dataset_version,
                manifest_sha256=manifest_hashes[cases.manifest.datasets[split].dataset_id],
            )
            for split in ("development", "public_validation")
            if split in observed_splits
        )
        summary = _summary(records)
        manifest = RealRunManifest(
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            run_id=f"{comparison_group_id}__{agent_version}",
            comparison_group_id=comparison_group_id,
            experiment_mode=mode,
            agent_version=agent_version,
            started_at_utc=_utc_text(started_at),
            completed_at_utc=_utc_text(completed_at),
            git_commit=git_commit,
            git_dirty=git_dirty,
            runtime_versions=runtime_versions,
            case_set_id=cases.manifest.case_set_id,
            case_jsonl_sha256=cases.manifest.jsonl_sha256,
            case_manifest_sha256=cases.manifest_sha256,
            case_subset_id=subset_id,
            case_subset_sha256=subset_sha256,
            datasets=datasets,
            tool_contract_version=tool_contract_version,
            tool_contract_sha256=tool_contract_sha256,
            adapter_name="openai-compatible",
            model=self.provider.model,
            base_url_host=base_url_host(self.provider.base_url),
            temperature=self.provider.temperature,
            timeout_seconds=self.provider.timeout_seconds,
            max_steps=self.run_settings.max_steps,
            concurrency=self.run_settings.concurrency,
            repeats=self.run_settings.repeats,
            prompt=_prompt_identity(snapshot),
            output_files=None,
            evaluation_status="pending_not_run",
            usage_status="available" if summary.tokens.available else "unavailable",
            cost_status="unavailable",
        )
        return RealRunArtifactBundle(
            records=records,
            expected_case_ids=tuple(case.case_id for case in selected),
            summary=summary,
            prompt_snapshot=snapshot,
            manifest=manifest,
        )

    @staticmethod
    def _record(
        sequence: int,
        case: RunnableCase,
        agent_version: AgentVersion,
        timed: _TimedResult,
    ) -> RealCaseRunRecord:
        result = timed.result
        usage = result.usage
        return RealCaseRunRecord(
            sequence=sequence,
            case_id=case.case_id,
            split=case.split,
            dataset_id=case.dataset_id,
            dataset_version=case.dataset_version,
            agent_version=agent_version,
            repeat_index=1,
            user_input=case.user_input,
            status=result.status,
            final_answer=result.final_answer if result.status == "completed" else None,
            error_code=None if result.status == "completed" else _case_error_code(result),
            prior_tool_executions=tuple(
                {
                    "call_id": execution.call_id,
                    "arguments": execution.model_dump(mode="json")["arguments"],
                    "result": canonical_tool_result_payload(execution.result),
                }
                for execution in result.prior_tool_executions
            ),
            decision_trace=tuple(
                event.model_dump(mode="json") for event in result.decision_trace
            ),
            latency_ms=timed.latency_ms,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
        )

    @staticmethod
    def _publish_pair(
        output_root: Path,
        v1_bundle: RealRunArtifactBundle,
        v2_bundle: RealRunArtifactBundle,
    ) -> tuple[Path, Path, PublishedRealRun, PublishedRealRun]:
        writer = RealArtifactWriter(output_root)
        staged: list[tuple[StagedRealRun, tuple[int, int]]] = []
        committed: list[tuple[Path, tuple[int, int]]] = []
        verified: list[PublishedRealRun] = []
        root = output_root.resolve()
        try:
            for bundle in (v1_bundle, v2_bundle):
                item = writer.stage(bundle)
                identity = _path_identity(item.temp_dir)
                if identity is None:
                    raise ValueError("staged run identity is unavailable")
                verify_published_real_run(item.temp_dir)
                staged.append((item, identity))
            for item, identity in staged:
                final_path = writer.commit(item)
                final_identity = _path_identity(final_path)
                if final_identity != identity:
                    raise ValueError("committed run identity changed")
                committed.append((final_path, identity))
            verified = [
                verify_published_real_run(final_path)
                for final_path, _identity in committed
            ]
        except BaseException:
            for item, identity in staged:
                _remove_known_directory(item.temp_dir, identity, root)
                _remove_known_directory(item.final_dir, identity, root)
            raise RunLevelError("artifact_publish_error") from None
        return committed[0][0], committed[1][0], verified[0], verified[1]


__all__ = [
    "PairedRunCoordinator",
    "PairedRunResult",
    "RunLevelError",
    "RunSettings",
]
