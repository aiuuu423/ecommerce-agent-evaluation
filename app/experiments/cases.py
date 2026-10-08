import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.data.manifest import manifest_id
from app.data.schemas import EvaluationCase, JsonValue, Sha256Hex

CaseSplit = Literal["development", "public_validation"]


class CaseModel(BaseModel):
    model_config = ConfigDict(
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
    business_task_counts: dict[str, int]
    capability_counts: dict[str, int]
    case_count: int = Field(gt=0)
    case_schema_version: str = Field(min_length=1)
    case_set_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    datasets: dict[CaseSplit, DatasetIdentity]
    difficulty_counts: dict[str, int]
    jsonl_sha256: Sha256Hex
    source_label: str = Field(min_length=1)
    split_counts: dict[CaseSplit, int]
    split_strategy: dict[str, JsonValue]
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
        return self


class RunnableCase(CaseModel):
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    user_input: str = Field(min_length=5)
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    split: CaseSplit


class CaseBundle(CaseModel):
    manifest: EvaluationCaseManifest
    cases: tuple[RunnableCase, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> "CaseBundle":
        if len(self.cases) != self.manifest.case_count:
            raise ValueError("case count does not match manifest")
        split_counts = Counter(case.split for case in self.cases)
        if dict(split_counts) != self.manifest.split_counts:
            raise ValueError("split counts do not match manifest")
        return self


def load_runnable_cases(case_dir: Path) -> CaseBundle:
    manifest_path = case_dir / "manifest.json"
    cases_path = case_dir / "cases.jsonl"
    manifest = EvaluationCaseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
        strict=True,
    )
    contents = cases_path.read_bytes()
    if sha256(contents).hexdigest() != manifest.jsonl_sha256:
        raise ValueError("cases.jsonl SHA-256 does not match manifest")

    runnable_cases: list[RunnableCase] = []
    canonical_lines: list[str] = []
    seen_case_ids: set[str] = set()
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"cases.jsonl line {line_number} is empty")
        case = EvaluationCase.model_validate_json(raw_line, strict=True)
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)

        expected_dataset = manifest.datasets[case.split]
        if (
            case.dataset_id != expected_dataset.dataset_id
            or case.dataset_version != expected_dataset.dataset_version
            or case.generator_config_hash != expected_dataset.generator_config_hash
        ):
            raise ValueError(f"{case.case_id}: dataset mapping does not match manifest")

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
    if manifest_id({"lines": canonical_lines}) != manifest.case_set_id:
        raise ValueError("case set ID does not match manifest")

    return CaseBundle(manifest=manifest, cases=tuple(runnable_cases))
