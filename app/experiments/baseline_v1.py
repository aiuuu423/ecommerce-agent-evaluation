import argparse
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import yaml

from app.agents import AgentRunner, RunRequest, RunResult, TraceEventType
from app.baselines.v1.config import POLICY_NAME, POLICY_VERSION, policy_snapshot
from app.baselines.v1.policy import BaselinePolicyV1
from app.baselines.v1.schemas import PolicyContext
from app.data.database import Catalog, open_dataset
from app.data.manifest import TABLE_NAMES, file_sha256
from app.llm import DeterministicAdapter
from app.tools import build_default_registry
from app.tools.schemas import canonical_tool_result_payload

from .artifacts import (
    ArtifactWriter,
    CaseRunRecord,
    DatasetArtifactIdentity,
    PolicySnapshot,
    RunArtifactBundle,
    RunManifest,
    RunSummary,
    RuntimeVersions,
    canonical_json_bytes,
    verify_published_run,
)
from .cases import CaseBundle, load_runnable_cases

_PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CASE_DIR = Path("data/evaluation_cases/v1")
DEFAULT_DEVELOPMENT_DATASET = Path("data/synthetic/v1")
DEFAULT_PUBLIC_VALIDATION_DATASET = Path("data/synthetic/public-validation-v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/experiment_runs")
DEFAULT_TOOL_CONTRACT = _PROJECT_ROOT / "configs/evaluation/tool_contract_v1.yaml"
_POLICY_SOURCE_FILES = (
    "app/baselines/v1/answers.py",
    "app/baselines/v1/config.py",
    "app/baselines/v1/parsing.py",
    "app/baselines/v1/policy.py",
    "app/baselines/v1/schemas.py",
)
_SAFE_CASE_ERROR_CODES = {
    "adapter_start_error",
    "adapter_error",
    "adapter_protocol_error",
    "duplicate_call_id",
    "unknown_tool",
    "invalid_arguments",
    "invalid_tool_result",
    "tool_execution_error",
    "duplicate_result_id",
    "max_steps_exceeded",
}


class RunLevelError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ToolContractIdentity:
    version: str
    sha256: str
    required_parameters: tuple[tuple[str, tuple[str, ...]], ...]


def _read_regular_file(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError("path is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(path, flags)
    try:
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise OSError("path changed while opening")
        chunks = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def tool_contract_identity(path: Path) -> ToolContractIdentity:
    try:
        contents = _read_regular_file(path)
        payload = yaml.safe_load(contents)
    except (OSError, yaml.YAMLError) as exc:
        raise RunLevelError("tool_contract_identity_error") from exc
    if not isinstance(payload, dict):
        raise RunLevelError("tool_contract_identity_error")
    contract_version = payload.get("contract_version")
    tools = payload.get("tools")
    if not isinstance(contract_version, str) or not contract_version:
        raise RunLevelError("tool_contract_identity_error")
    if not isinstance(tools, dict) or not tools:
        raise RunLevelError("tool_contract_identity_error")
    required_parameters = []
    for tool_name, definition in tools.items():
        if not isinstance(tool_name, str) or not isinstance(definition, dict):
            raise RunLevelError("tool_contract_identity_error")
        required = definition.get("required_parameters")
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) or not item for item in required)
            or len(required) != len(set(required))
        ):
            raise RunLevelError("tool_contract_identity_error")
        required_parameters.append((tool_name, tuple(required)))
    return ToolContractIdentity(
        version=contract_version,
        sha256=sha256(contents).hexdigest(),
        required_parameters=tuple(sorted(required_parameters)),
    )


def validate_registry_contract(contract: ToolContractIdentity) -> None:
    registry = build_default_registry()
    exported = {
        item["function"]["name"]: tuple(
            item["function"]["parameters"].get("required", ())
        )
        for item in registry.openai_tools()
    }
    expected = dict(contract.required_parameters)
    if any(exported.get(name) != required for name, required in expected.items()):
        raise RunLevelError("tool_registry_contract_drift")


def catalog_as_of_date(catalog: Catalog) -> date:
    try:
        result = catalog.execute(
            """
            select
                (select max(order_date) from orders),
                (select max(date) from traffic),
                (select max(date) from marketing)
            """
        )
        row = result.fetchone()
    except Exception as exc:
        raise RunLevelError("dataset_date_query_error") from exc
    if row is None:
        raise RunLevelError("dataset_date_unavailable")
    try:
        values = tuple(row)
    except Exception:
        raise RunLevelError("dataset_date_invalid") from None
    if len(values) != 3 or any(value is None for value in values):
        raise RunLevelError("dataset_date_unavailable")
    if any(type(value) is not date for value in values):
        raise RunLevelError("dataset_date_invalid")
    if len(set(values)) != 1:
        raise RunLevelError("dataset_date_mismatch")
    return values[0]


def policy_source_identity() -> tuple[str, tuple[str, ...]]:
    digest = sha256()
    for relative_path in _POLICY_SOURCE_FILES:
        path = _PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), _POLICY_SOURCE_FILES


def dataset_source_identity(
    dataset_dirs: Mapping[str, Path],
) -> tuple[tuple[str, str, str], ...]:
    identities = []
    artifact_names = (
        "manifest.json",
        "data_quality_report.json",
        *(f"{table_name}.parquet" for table_name in TABLE_NAMES),
    )
    for dataset_id, path in sorted(dataset_dirs.items()):
        digest = sha256()
        for artifact_name in artifact_names:
            digest.update(artifact_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update((path / artifact_name).read_bytes())
            digest.update(b"\0")
        identities.append(
            (
                dataset_id,
                file_sha256(path / "manifest.json"),
                digest.hexdigest(),
            )
        )
    return tuple(identities)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _run_id(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"baseline-v1__{timestamp}__{secrets.token_hex(4)}"


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
            if isinstance(code, str) and code in _SAFE_CASE_ERROR_CODES:
                return code
    return "case_execution_error"


def _summary(records: tuple[CaseRunRecord, ...]) -> RunSummary:
    statuses = Counter(record.status for record in records)
    splits = Counter(record.split for record in records)
    split_statuses = Counter((record.split, record.status) for record in records)
    return RunSummary(
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
        published=True,
        evaluation_status="pending_not_run",
    )


class BaselineBatchRunner:
    def __init__(self, tool_contract_path: Path = DEFAULT_TOOL_CONTRACT) -> None:
        self.tool_contract_path = tool_contract_path

    def run(
        self,
        cases: CaseBundle,
        dataset_dirs: Mapping[str, Path],
        output_root: Path,
    ) -> Path:
        expected_by_split = cases.manifest.datasets
        if (
            expected_by_split["development"].dataset_id
            == expected_by_split["public_validation"].dataset_id
        ):
            raise RunLevelError("dataset_split_id_collision")
        expected_ids = {
            expected_by_split["development"].dataset_id,
            expected_by_split["public_validation"].dataset_id,
        }
        if set(dataset_dirs) != expected_ids:
            raise RunLevelError("dataset_mapping_error")
        for case in cases.cases:
            expected = expected_by_split[case.split]
            if (
                case.dataset_id != expected.dataset_id
                or case.dataset_version != expected.dataset_version
            ):
                raise RunLevelError("dataset_mapping_error")

        initial_tool_contract = tool_contract_identity(self.tool_contract_path)
        if (
            cases.manifest.tool_contract.version != initial_tool_contract.version
            or cases.manifest.tool_contract.sha256 != initial_tool_contract.sha256
        ):
            raise RunLevelError("tool_contract_identity_mismatch")
        validate_registry_contract(initial_tool_contract)

        started_at = _utc_now()
        snapshot = PolicySnapshot.model_validate(policy_snapshot(), strict=True)
        snapshot_sha256 = sha256(canonical_json_bytes(snapshot)).hexdigest()
        try:
            initial_policy_identity = policy_source_identity()
        except OSError as exc:
            raise RunLevelError("policy_identity_error") from exc

        with ExitStack() as stack:
            catalogs: dict[str, Catalog] = {}
            for split in ("development", "public_validation"):
                expected = expected_by_split[split]
                dataset_dir = dataset_dirs[expected.dataset_id]
                try:
                    catalog = stack.enter_context(open_dataset(dataset_dir))
                except Exception as exc:
                    raise RunLevelError("catalog_open_error") from exc
                summary = catalog.verified_summary
                if (
                    summary["dataset_id"] != expected.dataset_id
                    or summary["dataset_version"] != expected.dataset_version
                    or catalog.manifest.get("dataset_id") != expected.dataset_id
                    or catalog.manifest.get("dataset_version")
                    != expected.dataset_version
                    or catalog.manifest.get("config_sha256")
                    != expected.generator_config_hash
                ):
                    raise RunLevelError("dataset_identity_mismatch")
                catalogs[expected.dataset_id] = catalog

            try:
                initial_dataset_identity = dataset_source_identity(dataset_dirs)
            except OSError as exc:
                raise RunLevelError("dataset_identity_error") from exc
            as_of_dates = {
                dataset_id: catalog_as_of_date(catalog)
                for dataset_id, catalog in catalogs.items()
            }

            records: list[CaseRunRecord] = []
            for sequence, case in enumerate(cases.cases, start=1):
                try:
                    policy = BaselinePolicyV1(
                        PolicyContext(
                            dataset_id=case.dataset_id,
                            dataset_version=case.dataset_version,
                            as_of_date=as_of_dates[case.dataset_id],
                        )
                    )
                    adapter = DeterministicAdapter(policy)
                    runner = AgentRunner(
                        registry=build_default_registry(),
                        adapter=adapter,
                    )
                    result = runner.run(
                        RunRequest(user_input=case.user_input),
                        catalogs[case.dataset_id],
                    )
                except Exception:
                    result = RunResult(
                        status="failed",
                        final_answer=None,
                        prior_tool_executions=[],
                        decision_trace=(),
                        usage=None,
                    )
                records.append(
                    CaseRunRecord(
                        sequence=sequence,
                        case_id=case.case_id,
                        split=case.split,
                        dataset_id=case.dataset_id,
                        dataset_version=case.dataset_version,
                        user_input=case.user_input,
                        status=result.status,
                        final_answer=result.final_answer,
                        prior_tool_executions=tuple(
                            {
                                "call_id": execution.call_id,
                                "arguments": execution.model_dump(mode="json")[
                                    "arguments"
                                ],
                                "result": canonical_tool_result_payload(
                                    execution.result
                                ),
                            }
                            for execution in result.prior_tool_executions
                        ),
                        decision_trace=tuple(
                            event.model_dump(mode="json")
                            for event in result.decision_trace
                            if result.status == "completed"
                            or event.event_type is not TraceEventType.ERROR
                        ),
                        usage=None,
                        error_code=(
                            None
                            if result.status == "completed"
                            else _case_error_code(result)
                        ),
                    )
                )

            try:
                final_dataset_identity = dataset_source_identity(dataset_dirs)
            except OSError as exc:
                raise RunLevelError("dataset_identity_error") from exc
            if final_dataset_identity != initial_dataset_identity:
                raise RunLevelError("dataset_identity_drift")
            try:
                final_policy_identity = policy_source_identity()
            except OSError as exc:
                raise RunLevelError("policy_identity_error") from exc
            if (
                final_policy_identity != initial_policy_identity
                or sha256(canonical_json_bytes(policy_snapshot())).hexdigest()
                != snapshot_sha256
            ):
                raise RunLevelError("policy_identity_drift")
            final_tool_contract = tool_contract_identity(self.tool_contract_path)
            if final_tool_contract != initial_tool_contract:
                raise RunLevelError("tool_contract_identity_drift")
            validate_registry_contract(final_tool_contract)

        completed_at = _utc_now()
        git_commit, git_dirty = _git_identity()
        dataset_manifest_hashes = {
            dataset_id: manifest_hash
            for dataset_id, manifest_hash, _ in initial_dataset_identity
        }
        dataset_artifacts = tuple(
            DatasetArtifactIdentity(
                split=split,
                dataset_id=identity.dataset_id,
                dataset_version=identity.dataset_version,
                manifest_sha256=dataset_manifest_hashes[identity.dataset_id],
            )
            for split, identity in (
                ("development", expected_by_split["development"]),
                ("public_validation", expected_by_split["public_validation"]),
            )
        )
        record_tuple = tuple(records)
        manifest = RunManifest(
            artifact_schema_version="1.0",
            run_id=_run_id(started_at),
            baseline_name=POLICY_NAME,
            baseline_version=POLICY_VERSION,
            started_at_utc=_utc_text(started_at),
            completed_at_utc=_utc_text(completed_at),
            git_commit=git_commit,
            git_dirty=git_dirty,
            runtime_versions=_runtime_versions(),
            case_set_id=cases.manifest.case_set_id,
            case_jsonl_sha256=cases.manifest.jsonl_sha256,
            case_manifest_sha256=cases.manifest_sha256,
            datasets=dataset_artifacts,
            tool_contract_version=initial_tool_contract.version,
            tool_contract_sha256=initial_tool_contract.sha256,
            policy_snapshot_sha256=snapshot_sha256,
            policy_source_files=initial_policy_identity[1],
            policy_source_sha256=initial_policy_identity[0],
            output_files=None,
            evaluation_status="pending_not_run",
            usage_status="unavailable",
        )
        bundle = RunArtifactBundle(
            records=record_tuple,
            expected_case_ids=tuple(case.case_id for case in cases.cases),
            summary=_summary(record_tuple),
            policy_snapshot=snapshot,
            manifest=manifest,
        )
        try:
            return ArtifactWriter(output_root).publish(bundle)
        except (OSError, ValueError):
            raise RunLevelError("artifact_publish_error") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Phase 3 baseline"
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DATASET,
    )
    parser.add_argument(
        "--public-validation-dataset",
        type=Path,
        default=DEFAULT_PUBLIC_VALIDATION_DATASET,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _stderr_code(error: RunLevelError) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]*", error.code):
        return error.code
    return "run_level_error"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        try:
            cases = load_runnable_cases(args.case_dir)
        except (OSError, ValueError) as exc:
            raise RunLevelError("case_load_error") from exc
        datasets = {
            cases.manifest.datasets[
                "development"
            ].dataset_id: args.development_dataset,
            cases.manifest.datasets[
                "public_validation"
            ].dataset_id: args.public_validation_dataset,
        }
        published_path = BaselineBatchRunner().run(
            cases,
            datasets,
            args.output_root,
        )
        try:
            published = verify_published_run(published_path)
        except (OSError, ValueError):
            raise RunLevelError("artifact_verify_error") from None
    except RunLevelError as exc:
        print(_stderr_code(exc), file=sys.stderr)
        return 1

    sys.stdout.write(
        canonical_json_bytes(
            {
                "run_id": published.manifest.run_id,
                "path": str(published_path),
                "completed": published.summary.completed,
                "failed": published.summary.failed,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
