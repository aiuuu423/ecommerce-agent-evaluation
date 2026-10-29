from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.experiments.artifacts import SplitCounts, SplitRunStatusCounts
from app.tools.schemas import JsonValue

ARTIFACT_SCHEMA_VERSION = "2.0.0"
REAL_OUTPUT_FILE_NAMES = {
    "case_runs.jsonl",
    "summary.json",
    "prompt_snapshot.json",
}

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
