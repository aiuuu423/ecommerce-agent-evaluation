import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.experiments.artifacts import canonical_json_bytes
from app.experiments.llm_artifacts import (
    PublishedRealRun,
    _cleanup_verified_staging,
    _directory_identity,
    _rename_directory_exclusive,
    verify_published_real_run,
)
from app.experiments.llm_run import PairedRunResult

COMPARISON_SCHEMA_VERSION = "1.0.0"
_COMPARISON_FILE_NAME = "comparison.json"
_PUBLISH_LOCK_NAME = ".comparison-publish.lock"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

_CHECK_NAMES = (
    "group_ids_match",
    "modes_match",
    "result_group_match",
    "result_mode_match",
    "agent_versions_valid",
    "model_match",
    "adapter_name_match",
    "base_url_host_match",
    "temperature_match",
    "timeout_seconds_match",
    "max_steps_match",
    "concurrency_match",
    "repeats_match",
    "git_commit_match",
    "git_dirty_match",
    "runtime_versions_match",
    "case_set_id_match",
    "case_jsonl_sha256_match",
    "case_manifest_sha256_match",
    "case_subset_id_match",
    "case_subset_sha256_match",
    "datasets_match",
    "tool_contract_match",
    "artifact_schema_match",
    "evaluation_status_match",
    "cost_status_match",
    "record_counts_match",
    "result_case_ids_match",
    "record_case_ids_match",
    "record_pairs_match",
    "record_versions_match",
    "prompts_distinct",
    "version_order_valid",
)


class ComparisonError(RuntimeError):
    pass


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class FairnessReport(_ComparisonModel):
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report(self) -> "FairnessReport":
        if set(self.checks) != set(_CHECK_NAMES):
            raise ValueError("fairness checks do not match the comparison contract")
        expected_failures = tuple(name for name in _CHECK_NAMES if not self.checks[name])
        if self.failures != expected_failures:
            raise ValueError("fairness failures do not match checks")
        if self.passed != (not expected_failures):
            raise ValueError("fairness passed flag does not match checks")
        return self


class ComparisonManifest(_ComparisonModel):
    artifact_schema_version: Literal["1.0.0"]
    comparison_group_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    experiment_mode: Literal["pilot", "formal"]
    v1_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    v2_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    v1_run_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    v2_run_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    v1_case_runs_sha256: str = Field(pattern=_SHA256_PATTERN)
    v2_case_runs_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_subset_id: str = Field(min_length=1)
    case_subset_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_ids: tuple[str, ...] = Field(min_length=1)
    version_order: dict[str, tuple[Literal["v1", "v2"], ...]]
    fairness: FairnessReport

    @model_validator(mode="after")
    def validate_comparison(self) -> "ComparisonManifest":
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case IDs must be unique")
        if any(
            len(case_id) != 8 or not case_id.startswith("CASE_") or not case_id[5:].isdigit()
            for case_id in self.case_ids
        ):
            raise ValueError("case IDs must use CASE_NNN format")
        if set(self.version_order) != set(self.case_ids):
            raise ValueError("version order keys must equal case IDs")
        if not _version_order_valid(
            self.experiment_mode,
            self.version_order,
            self.case_ids,
        ):
            raise ValueError("version order is invalid")
        if not self.fairness.passed:
            raise ValueError("published comparison fairness must pass")
        if self.fairness.failures:
            raise ValueError("published comparison fairness failures must be empty")
        if not all(self.fairness.checks.values()):
            raise ValueError("published comparison fairness checks must all pass")
        return self


PublishedComparison = ComparisonManifest


def _version_order_valid(
    mode: str,
    version_order: Mapping[str, Sequence[str]],
    case_ids: Sequence[str],
) -> bool:
    if mode not in {"pilot", "formal"}:
        return False
    if len(case_ids) != len(set(case_ids)):
        return False
    if set(version_order) != set(case_ids):
        return False
    for index, case_id in enumerate(case_ids, start=1):
        expected = ("v1", "v2") if mode == "pilot" or index % 2 == 1 else ("v2", "v1")
        if tuple(version_order[case_id]) != expected:
            return False
    return True


def _build_fairness(
    result: PairedRunResult,
    v1: PublishedRealRun,
    v2: PublishedRealRun,
) -> FairnessReport:
    m1 = v1.manifest
    m2 = v2.manifest
    result_case_ids = tuple(result.case_ids)
    v1_ids = tuple(record.case_id for record in v1.records)
    v2_ids = tuple(record.case_id for record in v2.records)
    pair_fields = (
        "sequence",
        "case_id",
        "split",
        "dataset_id",
        "dataset_version",
        "repeat_index",
        "user_input",
    )
    checks = {
        "group_ids_match": m1.comparison_group_id == m2.comparison_group_id,
        "modes_match": m1.experiment_mode == m2.experiment_mode,
        "result_group_match": (
            result.comparison_group_id == m1.comparison_group_id == m2.comparison_group_id
        ),
        "result_mode_match": (result.mode == m1.experiment_mode == m2.experiment_mode),
        "agent_versions_valid": (m1.agent_version == "v1" and m2.agent_version == "v2"),
        "model_match": m1.model == m2.model,
        "adapter_name_match": m1.adapter_name == m2.adapter_name,
        "base_url_host_match": m1.base_url_host == m2.base_url_host,
        "temperature_match": m1.temperature == m2.temperature,
        "timeout_seconds_match": m1.timeout_seconds == m2.timeout_seconds,
        "max_steps_match": m1.max_steps == m2.max_steps,
        "concurrency_match": m1.concurrency == m2.concurrency,
        "repeats_match": m1.repeats == m2.repeats,
        "git_commit_match": m1.git_commit == m2.git_commit,
        "git_dirty_match": m1.git_dirty == m2.git_dirty,
        "runtime_versions_match": m1.runtime_versions == m2.runtime_versions,
        "case_set_id_match": m1.case_set_id == m2.case_set_id,
        "case_jsonl_sha256_match": (m1.case_jsonl_sha256 == m2.case_jsonl_sha256),
        "case_manifest_sha256_match": (m1.case_manifest_sha256 == m2.case_manifest_sha256),
        "case_subset_id_match": m1.case_subset_id == m2.case_subset_id,
        "case_subset_sha256_match": (m1.case_subset_sha256 == m2.case_subset_sha256),
        "datasets_match": m1.datasets == m2.datasets,
        "tool_contract_match": (
            m1.tool_contract_version == m2.tool_contract_version
            and m1.tool_contract_sha256 == m2.tool_contract_sha256
        ),
        "artifact_schema_match": (m1.artifact_schema_version == m2.artifact_schema_version),
        "evaluation_status_match": (m1.evaluation_status == m2.evaluation_status),
        "cost_status_match": m1.cost_status == m2.cost_status,
        "record_counts_match": (len(v1.records) == len(v2.records) == len(result_case_ids)),
        "result_case_ids_match": (result_case_ids == v1_ids == v2_ids),
        "record_case_ids_match": (v1_ids == result_case_ids and v2_ids == result_case_ids),
        "record_pairs_match": (
            len(v1.records) == len(v2.records)
            and all(
                all(
                    getattr(record_v1, field_name) == getattr(record_v2, field_name)
                    for field_name in pair_fields
                )
                for record_v1, record_v2 in zip(
                    v1.records,
                    v2.records,
                    strict=True,
                )
            )
        ),
        "record_versions_match": (
            all(record.agent_version == "v1" for record in v1.records)
            and all(record.agent_version == "v2" for record in v2.records)
        ),
        "prompts_distinct": (
            m1.prompt.prompt_sha256 != m2.prompt.prompt_sha256 and m1.prompt != m2.prompt
        ),
        "version_order_valid": _version_order_valid(
            result.mode,
            result.version_order,
            result_case_ids,
        ),
    }
    failures = tuple(name for name in _CHECK_NAMES if not checks[name])
    return FairnessReport(
        passed=not failures,
        checks=checks,
        failures=failures,
    )


def _verified_runs(
    result: PairedRunResult,
) -> tuple[PublishedRealRun, PublishedRealRun]:
    try:
        v1 = verify_published_real_run(result.v1_path)
        v2 = verify_published_real_run(result.v2_path)
    except (OSError, ValueError) as exc:
        raise ComparisonError("published run verification failed") from exc
    return v1, v2


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest(
    result: PairedRunResult,
    v1: PublishedRealRun,
    v2: PublishedRealRun,
    fairness: FairnessReport,
) -> ComparisonManifest:
    return ComparisonManifest(
        artifact_schema_version=COMPARISON_SCHEMA_VERSION,
        comparison_group_id=result.comparison_group_id,
        experiment_mode=result.mode,
        v1_run_id=v1.manifest.run_id,
        v2_run_id=v2.manifest.run_id,
        v1_run_manifest_sha256=_file_sha256(result.v1_path / "run_manifest.json"),
        v2_run_manifest_sha256=_file_sha256(result.v2_path / "run_manifest.json"),
        v1_case_runs_sha256=_file_sha256(result.v1_path / "case_runs.jsonl"),
        v2_case_runs_sha256=_file_sha256(result.v2_path / "case_runs.jsonl"),
        case_subset_id=v1.manifest.case_subset_id,
        case_subset_sha256=v1.manifest.case_subset_sha256,
        case_ids=tuple(result.case_ids),
        version_order={
            case_id: tuple(result.version_order[case_id])
            for case_id in result.case_ids
            if case_id in result.version_order
        },
        fairness=fairness,
    )


def write_comparison(
    result: PairedRunResult,
    *,
    comparisons_root: Path,
) -> Path:
    v1, v2 = _verified_runs(result)
    fairness = _build_fairness(result, v1, v2)
    if not fairness.passed:
        raise ComparisonError(f"fairness checks failed: {', '.join(fairness.failures)}")

    try:
        manifest = _manifest(result, v1, v2, fairness)
        confirmed_v1, confirmed_v2 = _verified_runs(result)
    except ComparisonError:
        raise
    except (OSError, ValueError) as exc:
        raise ComparisonError("comparison manifest validation failed") from exc
    if confirmed_v1 != v1 or confirmed_v2 != v2:
        raise ComparisonError("published run verification changed")

    comparisons_root.mkdir(parents=True, exist_ok=True)
    root = comparisons_root.resolve()
    target = root / manifest.comparison_group_id
    if target.exists() or target.is_symlink():
        raise ComparisonError("comparison already published")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{manifest.comparison_group_id}.tmp-",
            dir=root,
        )
    )
    identity = _directory_identity(staging.lstat())
    try:
        (staging / _COMPARISON_FILE_NAME).write_bytes(canonical_json_bytes(manifest))
        verify_comparison(staging)
        with FileLock(root / _PUBLISH_LOCK_NAME):
            if target.exists() or target.is_symlink():
                raise ComparisonError("comparison already published")
            try:
                _rename_directory_exclusive(staging, target)
            except FileExistsError as exc:
                raise ComparisonError("comparison already published") from exc
    except BaseException:
        _cleanup_verified_staging(staging, identity)
        raise
    return target


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_comparison_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ComparisonError("comparison must be a non-symlink directory") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ComparisonError("comparison must be a non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise ComparisonError("comparison must be a non-symlink directory") from exc
    try:
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ComparisonError("comparison must be a non-symlink directory")
        if set(os.listdir(directory_fd)) != {_COMPARISON_FILE_NAME}:
            raise ComparisonError("comparison must contain exactly comparison.json")
        file_before = os.stat(
            _COMPARISON_FILE_NAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(file_before.st_mode):
            raise ComparisonError("comparison.json must be a regular non-symlink file")
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(
                _COMPARISON_FILE_NAME,
                file_flags,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ComparisonError("comparison.json must be a regular non-symlink file") from exc
        try:
            file_opened = os.fstat(file_fd)
            if not stat.S_ISREG(file_opened.st_mode) or (
                file_opened.st_dev,
                file_opened.st_ino,
            ) != (file_before.st_dev, file_before.st_ino):
                raise ComparisonError("comparison.json must be a regular non-symlink file")
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def verify_comparison(path: Path) -> PublishedComparison:
    contents = _read_comparison_file(path)
    try:
        json.loads(contents, parse_constant=_reject_constant)
        manifest = ComparisonManifest.model_validate_json(
            contents,
            strict=True,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ComparisonError("comparison.json validation failed") from exc
    if canonical_json_bytes(manifest) != contents:
        raise ComparisonError("comparison.json is not canonical JSON")
    return manifest


__all__ = [
    "ComparisonError",
    "ComparisonManifest",
    "FairnessReport",
    "PublishedComparison",
    "verify_comparison",
    "write_comparison",
]
