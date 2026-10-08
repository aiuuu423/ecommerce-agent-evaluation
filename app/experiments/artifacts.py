import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from app.agents.schemas import TraceEvent
from app.tools.schemas import PriorToolExecution

CaseSplit = Literal["development", "public_validation"]
RunStatus = Literal["completed", "failed"]
EvaluationStatus = Literal["pending_not_run"]
UsageStatus = Literal["unavailable"]
Sha256Hex = str
_OUTPUT_FILE_NAMES = {
    "case_runs.jsonl",
    "summary.json",
    "policy_snapshot.json",
}
_TASK_NAMES = {
    "product_anomaly",
    "conversion_decline",
    "next_week_priority",
    "products_to_watch",
    "gmv_diagnosis",
}


class ArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class CaseRunRecord(ArtifactModel):
    sequence: int = Field(ge=1)
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    split: CaseSplit
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    status: RunStatus
    final_answer: str | None
    prior_tool_executions: tuple[PriorToolExecution, ...]
    decision_trace: tuple[TraceEvent, ...]
    usage: None = None
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def validate_status_fields(self) -> "CaseRunRecord":
        if self.status == "completed":
            if self.final_answer is None:
                raise ValueError("completed records require final_answer")
            if self.error_code is not None:
                raise ValueError("completed records cannot contain error_code")
        elif self.final_answer is not None or self.error_code is None:
            raise ValueError("failed records require error_code and no final_answer")
        return self


class SplitStatusCounts(ArtifactModel):
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)


class SplitCounts(ArtifactModel):
    development: int = Field(ge=0)
    public_validation: int = Field(ge=0)


class SplitRunStatusCounts(ArtifactModel):
    development: SplitStatusCounts
    public_validation: SplitStatusCounts


class RunSummary(ArtifactModel):
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    stopped: int = Field(ge=0)
    split_counts: SplitCounts
    split_status_counts: SplitRunStatusCounts
    first_case_id: str | None = Field(default=None, pattern=r"^CASE_[0-9]{3}$")
    last_case_id: str | None = Field(default=None, pattern=r"^CASE_[0-9]{3}$")
    published: bool
    evaluation_status: EvaluationStatus

    @model_validator(mode="after")
    def validate_counts(self) -> "RunSummary":
        if self.completed + self.failed + self.stopped != self.total:
            raise ValueError("status counts must equal total")
        if self.split_counts.development + self.split_counts.public_validation != self.total:
            raise ValueError("split counts must equal total")
        status_total = sum(
            (
                self.split_status_counts.development.completed,
                self.split_status_counts.development.failed,
                self.split_status_counts.public_validation.completed,
                self.split_status_counts.public_validation.failed,
            )
        )
        if status_total != self.total:
            raise ValueError("split status counts must equal total")
        if self.total == 0 and (self.first_case_id is not None or self.last_case_id is not None):
            raise ValueError("empty summaries cannot define case order boundaries")
        if self.total > 0 and (self.first_case_id is None or self.last_case_id is None):
            raise ValueError("non-empty summaries require case order boundaries")
        return self


class PolicySnapshot(ArtifactModel):
    policy_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    routing_priority: list[str]
    routing_proximity_bonus: int = Field(ge=0)
    routing_proximity_max_chars: int = Field(ge=0)
    routing_keywords: dict[str, list[list[str]]]
    default_window_days: int = Field(gt=0)
    default_top_k: int = Field(gt=0)
    tool_paths: dict[str, list[str]]
    metrics: dict[str, list[str]]
    stable_sort_fields: dict[str, list[str]]
    answer_template_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task_mappings(self) -> "PolicySnapshot":
        if set(self.routing_priority) != _TASK_NAMES or len(self.routing_priority) != len(
            _TASK_NAMES
        ):
            raise ValueError("routing_priority must contain every supported task exactly once")
        for field_name in (
            "routing_keywords",
            "tool_paths",
            "metrics",
            "stable_sort_fields",
        ):
            if set(getattr(self, field_name)) != _TASK_NAMES:
                raise ValueError(f"{field_name} must define every supported task")
        return self


class DatasetArtifactIdentity(ArtifactModel):
    split: CaseSplit
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeVersions(ArtifactModel):
    python: str = Field(min_length=1)
    pydantic: str = Field(min_length=1)
    duckdb: str = Field(min_length=1)
    pandas: str = Field(min_length=1)
    pyarrow: str = Field(min_length=1)


class OutputFileHashes(RootModel[dict[str, str]]):
    model_config = ConfigDict(
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_file_hashes(self) -> "OutputFileHashes":
        if set(self.root) != _OUTPUT_FILE_NAMES:
            raise ValueError("output hashes must cover exactly the three non-manifest files")
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.root.values()
        ):
            raise ValueError("output hashes must be lowercase SHA-256 values")
        return self


class RunManifest(ArtifactModel):
    artifact_schema_version: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    baseline_name: str = Field(min_length=1)
    baseline_version: str = Field(min_length=1)
    started_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    completed_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    runtime_versions: RuntimeVersions
    case_set_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    case_jsonl_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    datasets: tuple[DatasetArtifactIdentity, ...] = Field(min_length=1)
    tool_contract_version: str = Field(min_length=1)
    tool_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_source_files: tuple[str, ...] = Field(min_length=1)
    policy_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_files: OutputFileHashes | None = None
    evaluation_status: EvaluationStatus
    usage_status: UsageStatus

    @field_validator("datasets")
    @classmethod
    def validate_dataset_splits(
        cls,
        value: tuple[DatasetArtifactIdentity, ...],
    ) -> tuple[DatasetArtifactIdentity, ...]:
        splits = [item.split for item in value]
        if set(splits) != {"development", "public_validation"} or len(splits) != 2:
            raise ValueError("datasets must define both splits exactly once")
        return value

    @field_validator("policy_source_files")
    @classmethod
    def validate_source_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("policy source files must be unique")
        if any(Path(item).is_absolute() or ".." in Path(item).parts for item in value):
            raise ValueError("policy source files must be safe relative paths")
        return value


class RunArtifactBundle(ArtifactModel):
    records: tuple[CaseRunRecord, ...]
    expected_case_ids: tuple[str, ...]
    summary: RunSummary
    policy_snapshot: PolicySnapshot
    manifest: RunManifest

    @model_validator(mode="after")
    def validate_bundle(self) -> "RunArtifactBundle":
        actual_ids = tuple(record.case_id for record in self.records)
        if actual_ids != self.expected_case_ids:
            raise ValueError("case record order does not match expected case order")
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("case IDs must be unique")
        if tuple(record.sequence for record in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise ValueError("case record sequence must be contiguous and one-based")
        _validate_summary_against_records(self.summary, self.records)
        if self.manifest.output_files is not None:
            raise ValueError("unpublished bundle manifest cannot contain output hashes")
        return self


class PublishedRun(ArtifactModel):
    records: tuple[CaseRunRecord, ...]
    summary: RunSummary
    policy_snapshot: PolicySnapshot
    manifest: RunManifest


def _json_value(value: object, path: str = "$") -> object:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"), path)
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path}: JSON number must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path}: JSON object keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path}: unsupported JSON value {type(value).__name__}")


def canonical_json_bytes(model_or_mapping: object) -> bytes:
    payload = _json_value(model_or_mapping)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(records: Sequence[CaseRunRecord]) -> bytes:
    validated = tuple(
        CaseRunRecord.model_validate(record.model_dump(mode="python"), strict=True)
        for record in records
    )
    return b"".join(canonical_json_bytes(record) for record in validated)


def _validate_summary_against_records(
    summary: RunSummary,
    records: Sequence[CaseRunRecord],
) -> None:
    statuses = Counter(record.status for record in records)
    splits = Counter(record.split for record in records)
    split_statuses = Counter((record.split, record.status) for record in records)
    if summary.total != len(records):
        raise ValueError("summary total does not match record count")
    if summary.stopped != 0:
        raise ValueError("published summaries cannot contain stopped cases")
    if summary.completed != statuses["completed"] or summary.failed != statuses["failed"]:
        raise ValueError("summary status counts do not match records")
    if summary.split_counts.model_dump() != {
        "development": splits["development"],
        "public_validation": splits["public_validation"],
    }:
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
    first = records[0].case_id if records else None
    last = records[-1].case_id if records else None
    if summary.first_case_id != first or summary.last_case_id != last:
        raise ValueError("summary case order boundaries do not match records")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_json(path: Path, model: type[BaseModel]) -> BaseModel:
    contents = path.read_bytes()
    try:
        json.loads(contents, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid UTF-8 JSON") from exc
    validated = model.model_validate_json(contents, strict=True)
    if canonical_json_bytes(validated) != contents:
        raise ValueError(f"{path.name} is not canonical JSON")
    return validated


def _load_records(path: Path) -> tuple[CaseRunRecord, ...]:
    contents = path.read_bytes()
    if contents and not contents.endswith(b"\n"):
        raise ValueError("case_runs.jsonl must end with a newline")
    records: list[CaseRunRecord] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line:
            raise ValueError(f"case_runs.jsonl line {line_number} is empty")
        try:
            json.loads(line, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"case_runs.jsonl line {line_number} is not valid JSON"
            ) from exc
        record = CaseRunRecord.model_validate_json(line, strict=True)
        if canonical_json_bytes(record).rstrip(b"\n") != line:
            raise ValueError(f"case_runs.jsonl line {line_number} is not canonical")
        records.append(record)
    if tuple(record.sequence for record in records) != tuple(range(1, len(records) + 1)):
        raise ValueError("case record sequence must be contiguous and one-based")
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("case IDs must be unique")
    return tuple(records)


def verify_published_run(path: Path) -> PublishedRun:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("published run must be a non-symlink directory")
    expected_names = _OUTPUT_FILE_NAMES | {"run_manifest.json"}
    if {item.name for item in path.iterdir()} != expected_names:
        raise ValueError("published run must contain exactly four artifact files")

    manifest = _load_json(path / "run_manifest.json", RunManifest)
    summary = _load_json(path / "summary.json", RunSummary)
    policy = _load_json(path / "policy_snapshot.json", PolicySnapshot)
    records = _load_records(path / "case_runs.jsonl")
    assert isinstance(manifest, RunManifest)
    assert isinstance(summary, RunSummary)
    assert isinstance(policy, PolicySnapshot)
    if manifest.output_files is None:
        raise ValueError("published manifest must contain output hashes")
    for file_name, expected_hash in manifest.output_files.root.items():
        if sha256((path / file_name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{file_name} SHA-256 does not match manifest")
    if sha256((path / "policy_snapshot.json").read_bytes()).hexdigest() != (
        manifest.policy_snapshot_sha256
    ):
        raise ValueError("policy snapshot SHA-256 does not match identity")
    _validate_summary_against_records(summary, records)
    return PublishedRun(
        records=records,
        summary=summary,
        policy_snapshot=policy,
        manifest=manifest,
    )


class ArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def _write_file(self, path: Path, contents: bytes) -> None:
        path.write_bytes(contents)

    def publish(self, bundle: RunArtifactBundle) -> Path:
        validated = RunArtifactBundle.model_validate(
            bundle.model_dump(mode="python"),
            strict=True,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.output_root / validated.manifest.run_id
        if final_dir.exists():
            raise FileExistsError(f"run already exists: {validated.manifest.run_id}")

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{validated.manifest.run_id}.tmp-",
                dir=self.output_root,
            )
        )
        try:
            files = {
                "case_runs.jsonl": canonical_jsonl_bytes(validated.records),
                "summary.json": canonical_json_bytes(validated.summary),
                "policy_snapshot.json": canonical_json_bytes(validated.policy_snapshot),
            }
            policy_hash = sha256(files["policy_snapshot.json"]).hexdigest()
            if validated.manifest.policy_snapshot_sha256 != policy_hash:
                raise ValueError("policy snapshot SHA-256 does not match manifest")
            for file_name, contents in files.items():
                self._write_file(temporary_dir / file_name, contents)
            hashes = {
                file_name: sha256(contents).hexdigest()
                for file_name, contents in files.items()
            }
            published_manifest = validated.manifest.model_copy(
                update={"output_files": OutputFileHashes(hashes)}
            )
            self._write_file(
                temporary_dir / "run_manifest.json",
                canonical_json_bytes(published_manifest),
            )
            verify_published_run(temporary_dir)
            if final_dir.exists():
                raise FileExistsError(f"run already exists: {validated.manifest.run_id}")
            temporary_dir.replace(final_dir)
            return final_dir
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
