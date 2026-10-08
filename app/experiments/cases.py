import json
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from typing_extensions import TypeAliasType

from app.data.manifest import manifest_id
from app.data.schemas import EvaluationCase, Sha256Hex

CaseSplit = Literal["development", "public_validation"]
FrozenJsonValue = TypeAliasType(
    "FrozenJsonValue",
    None
    | bool
    | int
    | float
    | str
    | list["FrozenJsonValue"]
    | tuple["FrozenJsonValue", ...]
    | dict[str, "FrozenJsonValue"]
    | MappingProxyType,
)
CountMapping = dict[str, int] | MappingProxyType
DatasetMapping = dict[CaseSplit, "DatasetIdentity"] | MappingProxyType
SplitCountMapping = dict[CaseSplit, int] | MappingProxyType
StrategyMapping = dict[str, FrozenJsonValue] | MappingProxyType
_MANIFEST_MAPPING_FIELDS = (
    "business_task_counts",
    "capability_counts",
    "datasets",
    "difficulty_counts",
    "split_counts",
    "split_strategy",
)


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key}")
        result[key] = value
    return result


def _load_json_object(contents: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{source} must contain a JSON object")
    return value


def _read_regular_file(case_dir: Path, file_name: str) -> bytes:
    if case_dir.is_symlink():
        raise ValueError("case directory must not be a symlink")
    try:
        directory_mode = case_dir.lstat().st_mode
    except OSError as exc:
        raise ValueError("case directory is not accessible") from exc
    if not stat.S_ISDIR(directory_mode):
        raise ValueError("case directory must be a directory")

    root = case_dir.resolve(strict=True)
    path = case_dir / file_name
    if path.is_symlink():
        raise ValueError(f"{file_name} must not be a symlink")
    try:
        file_mode = path.lstat().st_mode
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{file_name} is not accessible") from exc
    if not stat.S_ISREG(file_mode):
        raise ValueError(f"{file_name} must be a regular file")
    if resolved_path.parent != root:
        raise ValueError(f"{file_name} escapes the case directory")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{file_name} could not be opened safely") from exc
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError(f"{file_name} must be a regular file")
        return source.read()


class CaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class DatasetIdentity(CaseModel):
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    generator_config_hash: Sha256Hex


class ToolContractIdentity(CaseModel):
    version: str = Field(min_length=1)
    sha256: Sha256Hex


class EvaluationCaseManifest(CaseModel):
    business_task_counts: CountMapping
    capability_counts: CountMapping
    case_count: int = Field(gt=0)
    case_schema_version: str = Field(min_length=1)
    case_set_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    datasets: DatasetMapping
    difficulty_counts: CountMapping
    jsonl_sha256: Sha256Hex
    source_label: str = Field(min_length=1)
    split_counts: SplitCountMapping
    split_strategy: StrategyMapping
    tool_contract: ToolContractIdentity

    @model_validator(mode="after")
    def validate_counts_and_splits(self) -> "EvaluationCaseManifest":
        expected_splits = {"development", "public_validation"}
        if set(self.datasets) != expected_splits:
            raise ValueError("manifest datasets must define both splits")
        if set(self.split_counts) != expected_splits:
            raise ValueError("manifest split_counts must define both splits")
        count_mappings = (
            self.business_task_counts,
            self.capability_counts,
            self.difficulty_counts,
            self.split_counts,
        )
        if any(
            type(count) is not int or count < 0
            for counts in count_mappings
            for count in counts.values()
        ):
            raise ValueError("manifest counts must be non-negative integers")
        for field_name in _MANIFEST_MAPPING_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _deep_freeze(getattr(self, field_name)),
            )
        return self

    @field_serializer(*_MANIFEST_MAPPING_FIELDS, mode="plain")
    def serialize_frozen_mapping(self, value: object, info: Any) -> object:
        if info.mode == "json":
            return _json_compatible(value)
        return value


class RunnableCase(CaseModel):
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    user_input: str = Field(min_length=5)
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    split: CaseSplit


class CaseBundle(CaseModel):
    manifest: EvaluationCaseManifest
    cases: tuple[RunnableCase, ...]
    manifest_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_bundle(self) -> "CaseBundle":
        if len(self.cases) != self.manifest.case_count:
            raise ValueError("case count does not match manifest")
        split_counts = Counter(case.split for case in self.cases)
        if dict(split_counts) != self.manifest.split_counts:
            raise ValueError("split counts do not match manifest")
        return self


def load_runnable_cases(case_dir: Path) -> CaseBundle:
    manifest_contents = _read_regular_file(case_dir, "manifest.json")
    _load_json_object(manifest_contents, source="manifest.json")
    manifest = EvaluationCaseManifest.model_validate_json(
        manifest_contents,
        strict=True,
    )
    contents = _read_regular_file(case_dir, "cases.jsonl")
    if sha256(contents).hexdigest() != manifest.jsonl_sha256:
        raise ValueError("cases.jsonl SHA-256 does not match manifest")
    if b"\r" in contents.replace(b"\r\n", b""):
        raise ValueError("cases.jsonl contains a bare CR line ending")

    runnable_cases: list[RunnableCase] = []
    business_task_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    canonical_lines: list[str] = []
    seen_case_ids: set[str] = set()
    raw_lines = contents.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if not raw_line.strip():
            raise ValueError(f"cases.jsonl line {line_number} is empty")
        _load_json_object(
            raw_line,
            source=f"cases.jsonl line {line_number}",
        )
        case = EvaluationCase.model_validate_json(
            raw_line,
            strict=True,
        )
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        if case.case_version != manifest.case_schema_version:
            raise ValueError(f"{case.case_id}: case version does not match manifest")
        if (
            case.tool_contract_version != manifest.tool_contract.version
            or case.tool_contract_sha256 != manifest.tool_contract.sha256
        ):
            raise ValueError(f"{case.case_id}: tool contract does not match manifest")

        expected_dataset = manifest.datasets[case.split]
        if (
            case.dataset_id != expected_dataset.dataset_id
            or case.dataset_version != expected_dataset.dataset_version
            or case.generator_config_hash != expected_dataset.generator_config_hash
        ):
            raise ValueError(f"{case.case_id}: dataset mapping does not match manifest")

        business_task_counts[case.business_task.value] += 1
        capability_counts[case.primary_capability.value] += 1
        difficulty_counts[case.difficulty] += 1
        canonical_lines.append(
            json.dumps(
                case.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        runnable_cases.append(
            RunnableCase(
                case_id=case.case_id,
                user_input=case.user_input,
                dataset_id=case.dataset_id,
                dataset_version=case.dataset_version,
                split=case.split,
            )
        )

    if len(runnable_cases) != manifest.case_count:
        raise ValueError("case count does not match manifest")
    split_counts = Counter(case.split for case in runnable_cases)
    if dict(split_counts) != manifest.split_counts:
        raise ValueError("split counts do not match manifest")
    if dict(business_task_counts) != manifest.business_task_counts:
        raise ValueError("business task counts do not match manifest")
    if dict(capability_counts) != manifest.capability_counts:
        raise ValueError("capability counts do not match manifest")
    if dict(difficulty_counts) != manifest.difficulty_counts:
        raise ValueError("difficulty counts do not match manifest")
    if manifest_id({"lines": canonical_lines}) != manifest.case_set_id:
        raise ValueError("case set ID does not match manifest")

    return CaseBundle(
        manifest=manifest,
        cases=tuple(runnable_cases),
        manifest_sha256=sha256(manifest_contents).hexdigest(),
    )
