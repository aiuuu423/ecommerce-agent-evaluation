import ctypes
import errno
import json
import math
import os
import secrets
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from app.experiments.artifacts import (
    DatasetArtifactIdentity,
    RuntimeVersions,
    SplitCounts,
    SplitRunStatusCounts,
    canonical_json_bytes,
)
from app.prompts.base import compute_prompt_sha256
from app.tools.schemas import JsonValue

ARTIFACT_SCHEMA_VERSION = "2.0.0"
REAL_OUTPUT_FILE_NAMES = {
    "case_runs.jsonl",
    "summary.json",
    "prompt_snapshot.json",
}
_REAL_PUBLISH_LOCK_NAME = ".real-artifact-publish.lock"

AgentVersion = Literal["v1", "v2"]
ExperimentMode = Literal["pilot", "formal"]


class LlmArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class RealCaseRunRecord(LlmArtifactModel):
    sequence: int = Field(ge=1)
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    split: Literal["development", "public_validation"]
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    agent_version: AgentVersion
    repeat_index: Literal[1]
    user_input: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    final_answer: str | None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    prior_tool_executions: tuple[dict[str, JsonValue], ...]
    decision_trace: tuple[dict[str, JsonValue], ...]
    latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)

    @model_validator(mode="after")
    def validate_status_and_tokens(self) -> "RealCaseRunRecord":
        if self.status == "completed":
            if self.final_answer is None:
                raise ValueError("completed records require final_answer")
            if self.error_code is not None:
                raise ValueError("completed records cannot contain error_code")
        elif self.final_answer is not None or self.error_code is None:
            raise ValueError("failed records require error_code and no final_answer")

        token_availability = {
            self.prompt_tokens is not None,
            self.completion_tokens is not None,
            self.total_tokens is not None,
        }
        if len(token_availability) != 1:
            raise ValueError("token fields must be all present or all None")
        if (
            self.total_tokens is not None
            and self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens
            != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError(
                "total_tokens must equal prompt_tokens plus completion_tokens"
            )
        return self


class StoredPromptSnapshot(LlmArtifactModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    change_note: str = Field(min_length=1)
    source_file: str = Field(min_length=1)


class LatencyStats(LlmArtifactModel):
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    sample_size: int = Field(ge=0)


class TokenStats(LlmArtifactModel):
    available: bool
    prompt_total: int | None = Field(default=None, ge=0)
    completion_total: int | None = Field(default=None, ge=0)
    total_total: int | None = Field(default=None, ge=0)
    p50: int | None = Field(default=None, ge=0)
    p95: int | None = Field(default=None, ge=0)
    sample_size: int = Field(ge=0)


class RealRunSummary(LlmArtifactModel):
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    stopped: int = Field(ge=0)
    split_counts: SplitCounts
    split_status_counts: SplitRunStatusCounts
    first_case_id: str | None = Field(
        default=None,
        pattern=r"^CASE_[0-9]{3}$",
    )
    last_case_id: str | None = Field(
        default=None,
        pattern=r"^CASE_[0-9]{3}$",
    )
    latency: LatencyStats
    tokens: TokenStats
    cost: Literal["unavailable"]
    evaluation_status: Literal["pending_not_run"]

    @model_validator(mode="after")
    def validate_counts(self) -> "RealRunSummary":
        if self.completed + self.failed + self.stopped != self.total:
            raise ValueError("status counts must equal total")
        if (
            self.split_counts.development
            + self.split_counts.public_validation
            != self.total
        ):
            raise ValueError("split counts must equal total")
        return self


class PromptIdentity(LlmArtifactModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RealOutputFileHashes(RootModel[dict[str, str]]):
    model_config = ConfigDict(
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_file_hashes(self) -> "RealOutputFileHashes":
        if set(self.root) != REAL_OUTPUT_FILE_NAMES:
            raise ValueError(
                "real output hashes must cover exactly the three non-manifest files"
            )
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.root.values()
        ):
            raise ValueError("output hashes must be lowercase SHA-256 values")
        return self


class RealRunManifest(LlmArtifactModel):
    artifact_schema_version: Literal["2.0.0"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    comparison_group_id: str = Field(min_length=1)
    experiment_mode: ExperimentMode
    agent_version: AgentVersion
    started_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
    )
    completed_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
    )
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    runtime_versions: RuntimeVersions
    case_set_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    case_jsonl_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_subset_id: str = Field(min_length=1)
    case_subset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    datasets: tuple[DatasetArtifactIdentity, ...]
    tool_contract_version: str = Field(min_length=1)
    tool_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url_host: str = Field(min_length=1)
    temperature: float
    timeout_seconds: float
    max_steps: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    repeats: int = Field(ge=1)
    prompt: PromptIdentity
    output_files: RealOutputFileHashes | None = None
    evaluation_status: Literal["pending_not_run"]
    usage_status: Literal["available", "unavailable"]
    cost_status: Literal["unavailable"]


def _record_json_payload(record: RealCaseRunRecord) -> dict[str, object]:
    return {
        "sequence": record.sequence,
        "case_id": record.case_id,
        "split": record.split,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "agent_version": record.agent_version,
        "repeat_index": record.repeat_index,
        "user_input": record.user_input,
        "status": record.status,
        "final_answer": record.final_answer,
        "error_code": record.error_code,
        "prior_tool_executions": record.prior_tool_executions,
        "decision_trace": record.decision_trace,
        "latency_ms": record.latency_ms,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
    }


def _record_validation_payload(record: RealCaseRunRecord) -> dict[str, object]:
    return _record_json_payload(record)


def canonical_real_jsonl_bytes(
    records: Sequence[RealCaseRunRecord],
) -> bytes:
    validated = tuple(
        RealCaseRunRecord.model_validate(
            _record_validation_payload(record),
            strict=True,
        )
        for record in records
    )
    return b"".join(
        canonical_json_bytes(_record_json_payload(record))
        for record in validated
    )


def _validate_summary_against_records(
    summary: RealRunSummary,
    records: Sequence[RealCaseRunRecord],
) -> None:
    statuses = Counter(record.status for record in records)
    splits = Counter(record.split for record in records)
    split_statuses = Counter(
        (record.split, record.status) for record in records
    )
    if summary.total != len(records):
        raise ValueError("summary total does not match record count")
    if summary.stopped != 0:
        raise ValueError("published summaries cannot contain stopped cases")
    if (
        summary.completed != statuses["completed"]
        or summary.failed != statuses["failed"]
    ):
        raise ValueError("summary status counts do not match records")
    expected_splits = {
        "development": splits["development"],
        "public_validation": splits["public_validation"],
    }
    if summary.split_counts.model_dump() != expected_splits:
        raise ValueError("summary split counts do not match records")
    expected_split_statuses = {
        split: {
            status: split_statuses[(split, status)]
            for status in ("completed", "failed")
        }
        for split in ("development", "public_validation")
    }
    if summary.split_status_counts.model_dump() != expected_split_statuses:
        raise ValueError("summary split status counts do not match records")
    first_case_id = records[0].case_id if records else None
    last_case_id = records[-1].case_id if records else None
    if (
        summary.first_case_id != first_case_id
        or summary.last_case_id != last_case_id
    ):
        raise ValueError("summary case order boundaries do not match records")


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _validate_record_derived_metadata(
    summary: RealRunSummary,
    manifest: RealRunManifest,
    records: Sequence[RealCaseRunRecord],
) -> None:
    _validate_summary_against_records(summary, records)

    latencies = [record.latency_ms for record in records]
    expected_latency = LatencyStats(
        p50=_percentile(latencies, 0.50),
        p95=_percentile(latencies, 0.95),
        sample_size=len(records),
    )
    if summary.latency != expected_latency:
        raise ValueError("summary latency does not match records")

    records_with_usage = [
        record for record in records if record.total_tokens is not None
    ]
    total_tokens = [
        record.total_tokens
        for record in records_with_usage
        if record.total_tokens is not None
    ]
    usage_available = bool(records_with_usage)
    expected_tokens = TokenStats(
        available=usage_available,
        prompt_total=(
            sum(record.prompt_tokens or 0 for record in records_with_usage)
            if usage_available
            else None
        ),
        completion_total=(
            sum(
                record.completion_tokens or 0
                for record in records_with_usage
            )
            if usage_available
            else None
        ),
        total_total=sum(total_tokens) if usage_available else None,
        p50=_percentile(total_tokens, 0.50) if usage_available else None,
        p95=_percentile(total_tokens, 0.95) if usage_available else None,
        sample_size=len(records_with_usage),
    )
    if summary.tokens != expected_tokens:
        raise ValueError("summary token statistics do not match records")

    expected_usage_status = (
        "available" if summary.tokens.available else "unavailable"
    )
    if manifest.usage_status != expected_usage_status:
        raise ValueError("manifest usage status does not match summary tokens")

    dataset_identities = tuple(
        (dataset.split, dataset.dataset_id, dataset.dataset_version)
        for dataset in manifest.datasets
    )
    if len(dataset_identities) != len(set(dataset_identities)):
        raise ValueError("manifest dataset identities must be unique")
    observed_datasets = {
        (record.split, record.dataset_id, record.dataset_version)
        for record in records
    }
    if set(dataset_identities) != observed_datasets:
        raise ValueError("manifest datasets do not match records")


def _validate_prompt_identity(
    prompt_snapshot: StoredPromptSnapshot,
    prompt_identity: PromptIdentity,
) -> None:
    if (
        compute_prompt_sha256(prompt_snapshot.prompt_text)
        != prompt_snapshot.prompt_sha256
    ):
        raise ValueError("prompt snapshot text SHA-256 does not match snapshot")
    expected = {
        "prompt_name": prompt_snapshot.prompt_name,
        "prompt_version": prompt_snapshot.prompt_version,
        "source_file": prompt_snapshot.source_file,
        "prompt_sha256": prompt_snapshot.prompt_sha256,
    }
    if prompt_identity.model_dump() != expected:
        raise ValueError("prompt snapshot identity does not match manifest")


class RealRunArtifactBundle(LlmArtifactModel):
    records: tuple[RealCaseRunRecord, ...]
    expected_case_ids: tuple[str, ...]
    summary: RealRunSummary
    prompt_snapshot: StoredPromptSnapshot
    manifest: RealRunManifest

    @model_validator(mode="before")
    @classmethod
    def normalize_record_instances(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        records = value.get("records")
        if not isinstance(records, Sequence):
            return value
        normalized = dict(value)
        normalized["records"] = tuple(
            _record_validation_payload(record)
            if isinstance(record, RealCaseRunRecord)
            else record
            for record in records
        )
        return normalized

    @model_validator(mode="after")
    def validate_bundle(self) -> "RealRunArtifactBundle":
        actual_ids = tuple(record.case_id for record in self.records)
        if actual_ids != self.expected_case_ids:
            raise ValueError(
                "case record order does not match expected case order"
            )
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("case IDs must be unique")
        if tuple(record.sequence for record in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise ValueError(
                "case record sequence must be contiguous and one-based"
            )
        _validate_record_derived_metadata(
            self.summary,
            self.manifest,
            self.records,
        )
        if any(
            record.agent_version != self.manifest.agent_version
            for record in self.records
        ):
            raise ValueError(
                "record agent versions do not match manifest agent version"
            )
        _validate_prompt_identity(
            self.prompt_snapshot,
            self.manifest.prompt,
        )
        if self.manifest.output_files is not None:
            raise ValueError(
                "unpublished bundle manifest cannot contain output hashes"
            )
        return self


class PublishedRealRun(LlmArtifactModel):
    records: tuple[RealCaseRunRecord, ...]
    summary: RealRunSummary
    prompt_snapshot: StoredPromptSnapshot
    manifest: RealRunManifest


@dataclass(frozen=True)
class VerifiedRealRunSnapshot:
    run: PublishedRealRun
    run_manifest_sha256: str
    case_runs_sha256: str


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_json(
    contents: bytes,
    file_name: str,
    model: type[BaseModel],
) -> BaseModel:
    try:
        json.loads(contents, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{file_name} is not valid UTF-8 JSON") from exc
    validated = model.model_validate_json(contents, strict=True)
    if canonical_json_bytes(validated) != contents:
        raise ValueError(f"{file_name} is not canonical JSON")
    return validated


def _load_records(contents: bytes) -> tuple[RealCaseRunRecord, ...]:
    if contents and not contents.endswith(b"\n"):
        raise ValueError("case_runs.jsonl must end with a newline")
    records: list[RealCaseRunRecord] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line:
            raise ValueError(f"case_runs.jsonl line {line_number} is empty")
        try:
            json.loads(line, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"case_runs.jsonl line {line_number} is not valid JSON"
            ) from exc
        record = RealCaseRunRecord.model_validate_json(line, strict=True)
        expected_line = canonical_json_bytes(
            _record_json_payload(record)
        ).rstrip(b"\n")
        if expected_line != line:
            raise ValueError(
                f"case_runs.jsonl line {line_number} is not canonical"
            )
        records.append(record)
    if tuple(record.sequence for record in records) != tuple(
        range(1, len(records) + 1)
    ):
        raise ValueError(
            "case record sequence must be contiguous and one-based"
        )
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("case IDs must be unique")
    return tuple(records)


def _read_regular_file_at(directory_fd: int, file_name: str) -> bytes:
    before = os.stat(file_name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(
            f"{file_name} must be a regular non-symlink file"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        file_fd = os.open(file_name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(
            f"{file_name} must be a regular non-symlink file"
        ) from exc
    try:
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError(
                f"{file_name} must be a regular non-symlink file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _read_real_directory_fd(directory_fd: int) -> dict[str, bytes]:
    expected_names = REAL_OUTPUT_FILE_NAMES | {"run_manifest.json"}
    if set(os.listdir(directory_fd)) != expected_names:
        raise ValueError(
            "published run must contain exactly four artifact files"
        )
    return {
        file_name: _read_regular_file_at(directory_fd, file_name)
        for file_name in expected_names
    }


def _read_real_directory(path: Path) -> dict[str, bytes]:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise ValueError(
            "published run must be a non-symlink directory"
        ) from None
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("published run must be a non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            "published run must be a non-symlink directory"
        ) from exc
    try:
        opened = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError(
                "published run must be a non-symlink directory"
            )
        return _read_real_directory_fd(directory_fd)
    finally:
        os.close(directory_fd)


def _verify_real_run_files(
    files: Mapping[str, bytes],
) -> PublishedRealRun:
    manifest = _load_json(
        files["run_manifest.json"],
        "run_manifest.json",
        RealRunManifest,
    )
    summary = _load_json(
        files["summary.json"],
        "summary.json",
        RealRunSummary,
    )
    prompt_snapshot = _load_json(
        files["prompt_snapshot.json"],
        "prompt_snapshot.json",
        StoredPromptSnapshot,
    )
    records = _load_records(files["case_runs.jsonl"])
    assert isinstance(manifest, RealRunManifest)
    assert isinstance(summary, RealRunSummary)
    assert isinstance(prompt_snapshot, StoredPromptSnapshot)
    if manifest.output_files is None:
        raise ValueError("published manifest must contain output hashes")
    for file_name, expected_hash in manifest.output_files.root.items():
        if sha256(files[file_name]).hexdigest() != expected_hash:
            raise ValueError(
                f"{file_name} SHA-256 does not match manifest"
            )
    _validate_prompt_identity(prompt_snapshot, manifest.prompt)
    _validate_record_derived_metadata(summary, manifest, records)
    if any(
        record.agent_version != manifest.agent_version
        for record in records
    ):
        raise ValueError(
            "record agent versions do not match manifest agent version"
        )
    return PublishedRealRun(
        records=tuple(
            _record_validation_payload(record) for record in records
        ),
        summary=summary,
        prompt_snapshot=prompt_snapshot,
        manifest=manifest,
    )


def verify_published_real_run_snapshot(
    path: Path,
) -> VerifiedRealRunSnapshot:
    files = _read_real_directory(path)
    return VerifiedRealRunSnapshot(
        run=_verify_real_run_files(files),
        run_manifest_sha256=sha256(files["run_manifest.json"]).hexdigest(),
        case_runs_sha256=sha256(files["case_runs.jsonl"]).hexdigest(),
    )


def verify_published_real_run(path: Path) -> PublishedRealRun:
    return verify_published_real_run_snapshot(path).run


@dataclass(frozen=True)
class StagedRealRun:
    temp_dir: Path
    final_dir: Path


DirectoryIdentity = tuple[int, int]


@dataclass(frozen=True)
class _StagedCapability:
    staged: StagedRealRun
    temp_dir: Path
    final_dir: Path
    identity: DirectoryIdentity
    file_fingerprints: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PublishedCapability:
    final_dir: Path
    identity: DirectoryIdentity
    file_fingerprints: tuple[tuple[str, str], ...]


def _directory_identity(metadata: os.stat_result) -> DirectoryIdentity:
    return metadata.st_dev, metadata.st_ino


def _verified_directory_identity(
    path: Path,
    expected_identity: DirectoryIdentity,
) -> DirectoryIdentity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(
            "staged directory does not match verified filesystem identity"
        ) from exc
    identity = _directory_identity(metadata)
    if not stat.S_ISDIR(metadata.st_mode) or identity != expected_identity:
        raise ValueError(
            "staged directory does not match verified filesystem identity"
        )
    return identity


def _open_verified_staging_directory(
    path: Path,
    expected_identity: DirectoryIdentity,
) -> int:
    before_identity = _verified_directory_identity(path, expected_identity)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            "staged directory does not match verified filesystem identity"
        ) from exc
    try:
        opened = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(opened) != expected_identity
            or _directory_identity(opened) != before_identity
        ):
            raise ValueError(
                "staged directory does not match verified filesystem identity"
            )
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd


def _raise_rename_error(error_number: int, destination: Path) -> None:
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


def _rename_directory_exclusive_impl(
    source: Path,
    destination: Path,
) -> None:
    if sys.platform == "win32":
        os.rename(source, destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as exc:
            raise RuntimeError("renamex_np is unavailable") from exc
        rename.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x4)
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise RuntimeError("renameat2 is unavailable") from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    else:
        raise RuntimeError(
            f"exclusive directory rename is unsupported on {sys.platform}"
        )
    if result != 0:
        _raise_rename_error(ctypes.get_errno(), destination)


def _rename_directory_exclusive(source: Path, destination: Path) -> None:
    _rename_directory_exclusive_impl(source, destination)


def _hidden_sibling(path: Path, purpose: str) -> Path:
    while True:
        candidate = path.with_name(
            f".{path.name}.{purpose}-{secrets.token_hex(8)}"
        )
        if not candidate.exists() and not candidate.is_symlink():
            return candidate


def _cleanup_verified_staging(
    path: Path,
    expected_identity: DirectoryIdentity,
) -> None:
    try:
        _verified_directory_identity(path, expected_identity)
    except ValueError:
        return

    cleanup_path = _hidden_sibling(path, "cleanup")
    try:
        _rename_directory_exclusive_impl(path, cleanup_path)
    except OSError:
        return
    try:
        _verified_directory_identity(cleanup_path, expected_identity)
    except ValueError:
        try:
            _rename_directory_exclusive_impl(cleanup_path, path)
        except OSError:
            pass
        return
    shutil.rmtree(cleanup_path, ignore_errors=True)


def _move_verified_directory_to_hidden(
    path: Path,
    expected_identity: DirectoryIdentity,
    *,
    purpose: str,
    expected_file_fingerprints: tuple[tuple[str, str], ...] | None = None,
) -> Path:
    _verified_directory_identity(path, expected_identity)
    hidden_path = _hidden_sibling(path, purpose)
    _rename_directory_exclusive_impl(path, hidden_path)
    try:
        _verified_directory_identity(hidden_path, expected_identity)
        if expected_file_fingerprints is not None:
            directory_fd = _open_verified_staging_directory(
                hidden_path,
                expected_identity,
            )
            try:
                hidden_files = _read_real_directory_fd(directory_fd)
            finally:
                os.close(directory_fd)
            hidden_fingerprints = tuple(
                sorted(
                    (file_name, sha256(contents).hexdigest())
                    for file_name, contents in hidden_files.items()
                )
            )
            if hidden_fingerprints != expected_file_fingerprints:
                raise ValueError(
                    "published artifact file fingerprint mismatch"
                )
    except BaseException:
        try:
            _rename_directory_exclusive_impl(hidden_path, path)
        except OSError:
            pass
        raise
    return hidden_path


def _remove_retired_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise OSError(
            f"failed to remove retired directory: {path}"
        ) from exc


def _move_unverified_final_aside(final_dir: Path, temp_dir: Path) -> None:
    try:
        final_dir.lstat()
    except OSError:
        return
    destination = temp_dir
    if destination.exists() or destination.is_symlink():
        destination = _hidden_sibling(temp_dir, "rejected")
    try:
        _rename_directory_exclusive_impl(final_dir, destination)
    except OSError:
        pass


class RealArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self._staged_capabilities: dict[int, _StagedCapability] = {}
        self._published_capabilities: dict[Path, _PublishedCapability] = {}

    def _write_file(self, path: Path, contents: bytes) -> None:
        path.write_bytes(contents)

    def stage(self, bundle: RealRunArtifactBundle) -> StagedRealRun:
        validation_payload = bundle.model_dump(mode="python")
        validation_payload["records"] = tuple(
            _record_validation_payload(record) for record in bundle.records
        )
        validated = RealRunArtifactBundle.model_validate(
            validation_payload,
            strict=True,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        output_root = self.output_root.resolve()
        final_dir = output_root / validated.manifest.run_id
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(
                f"run already exists: {validated.manifest.run_id}"
            )
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{validated.manifest.run_id}.tmp-",
                dir=output_root,
            )
        )
        identity = _directory_identity(temp_dir.lstat())
        try:
            files = {
                "case_runs.jsonl": canonical_real_jsonl_bytes(
                    validated.records
                ),
                "summary.json": canonical_json_bytes(validated.summary),
                "prompt_snapshot.json": canonical_json_bytes(
                    validated.prompt_snapshot
                ),
            }
            for file_name, contents in files.items():
                self._write_file(temp_dir / file_name, contents)
            hashes = {
                file_name: sha256(contents).hexdigest()
                for file_name, contents in files.items()
            }
            published_manifest = validated.manifest.model_copy(
                update={
                    "output_files": RealOutputFileHashes(hashes)
                }
            )
            self._write_file(
                temp_dir / "run_manifest.json",
                canonical_json_bytes(published_manifest),
            )
            verify_published_real_run(temp_dir)
            directory_fd = _open_verified_staging_directory(
                temp_dir,
                identity,
            )
            try:
                staged_files = _read_real_directory_fd(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            _cleanup_verified_staging(temp_dir, identity)
            raise
        file_fingerprints = tuple(
            sorted(
                (file_name, sha256(contents).hexdigest())
                for file_name, contents in staged_files.items()
            )
        )
        staged = StagedRealRun(temp_dir=temp_dir, final_dir=final_dir)
        self._staged_capabilities[id(staged)] = _StagedCapability(
            staged=staged,
            temp_dir=temp_dir,
            final_dir=final_dir,
            identity=identity,
            file_fingerprints=file_fingerprints,
        )
        return staged

    def _validate_staged_capability(
        self,
        staged: StagedRealRun,
    ) -> _StagedCapability:
        capability = self._staged_capabilities.get(id(staged))
        if capability is None or capability.staged is not staged:
            raise ValueError("staged run was not created by this writer")

        output_root = self.output_root.resolve()
        temp_dir = staged.temp_dir
        final_dir = staged.final_dir
        safe_prefix = f".{capability.final_dir.name}.tmp-"
        if (
            temp_dir != capability.temp_dir
            or final_dir != capability.final_dir
            or temp_dir.parent != output_root
            or final_dir.parent != output_root
            or not temp_dir.name.startswith(safe_prefix)
            or temp_dir.name == safe_prefix
        ):
            raise ValueError("staged run paths are not safe for this writer")
        return capability

    def commit(self, staged: StagedRealRun) -> Path:
        capability = self._validate_staged_capability(staged)
        temp_dir = capability.temp_dir
        final_dir = capability.final_dir
        directory_fd = _open_verified_staging_directory(
            temp_dir,
            capability.identity,
        )
        del self._staged_capabilities[id(staged)]
        try:
            with FileLock(self.output_root / _REAL_PUBLISH_LOCK_NAME):
                if final_dir.exists() or final_dir.is_symlink():
                    raise FileExistsError(
                        f"run already exists: {final_dir.name}"
                    )
                staged_files = _read_real_directory_fd(directory_fd)
                file_fingerprints = tuple(
                    sorted(
                        (file_name, sha256(contents).hexdigest())
                        for file_name, contents in staged_files.items()
                    )
                )
                if file_fingerprints != capability.file_fingerprints:
                    raise ValueError(
                        "staged artifact file fingerprint mismatch"
                    )
                verify_published_real_run(temp_dir)
                _verified_directory_identity(
                    temp_dir,
                    capability.identity,
                )
                _rename_directory_exclusive(temp_dir, final_dir)
                try:
                    _verified_directory_identity(
                        final_dir,
                        capability.identity,
                    )
                except ValueError:
                    _move_unverified_final_aside(final_dir, temp_dir)
                    raise
                self._published_capabilities[final_dir] = _PublishedCapability(
                    final_dir=final_dir,
                    identity=capability.identity,
                    file_fingerprints=file_fingerprints,
                )
        except BaseException:
            _cleanup_verified_staging(temp_dir, capability.identity)
            raise
        finally:
            os.close(directory_fd)
        return staged.final_dir

    def discard(self, staged: StagedRealRun) -> None:
        capability = self._validate_staged_capability(staged)
        _verified_directory_identity(
            capability.temp_dir,
            capability.identity,
        )
        cleanup_path = _move_verified_directory_to_hidden(
            capability.temp_dir,
            capability.identity,
            purpose="cleanup",
        )
        del self._staged_capabilities[id(staged)]
        _remove_retired_directory(cleanup_path)

    def rollback_published(self, path: Path) -> None:
        capability = self._published_capabilities.get(path)
        if capability is None or capability.final_dir != path:
            raise ValueError("published run was not created by this writer")
        if (
            path.parent != self.output_root.resolve()
            or path != capability.final_dir
        ):
            raise ValueError("published run path is not safe for this writer")

        try:
            directory_fd = _open_verified_staging_directory(
                path,
                capability.identity,
            )
        except ValueError:
            raise ValueError(
                "published directory does not match verified filesystem identity"
            ) from None
        try:
            published_files = _read_real_directory_fd(directory_fd)
        finally:
            os.close(directory_fd)
        file_fingerprints = tuple(
            sorted(
                (file_name, sha256(contents).hexdigest())
                for file_name, contents in published_files.items()
            )
        )
        if file_fingerprints != capability.file_fingerprints:
            raise ValueError("published artifact file fingerprint mismatch")
        _verified_directory_identity(path, capability.identity)

        rollback_path = _move_verified_directory_to_hidden(
            path,
            capability.identity,
            purpose="rollback",
            expected_file_fingerprints=capability.file_fingerprints,
        )
        del self._published_capabilities[path]
        _remove_retired_directory(rollback_path)

    def publish(self, bundle: RealRunArtifactBundle) -> Path:
        staged = self.stage(bundle)
        return self.commit(staged)
