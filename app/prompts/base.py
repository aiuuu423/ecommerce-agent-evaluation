from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PromptModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


def compute_prompt_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class PromptSnapshot(PromptModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    change_note: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "prompt_name",
        "prompt_version",
        "prompt_text",
        "change_note",
        "source_file",
    )
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def make_prompt_snapshot(
    *,
    name: str,
    version: str,
    text: str,
    change_note: str,
    source_file: str,
) -> PromptSnapshot:
    return PromptSnapshot(
        prompt_name=name,
        prompt_version=version,
        prompt_text=text,
        change_note=change_note,
        source_file=source_file,
        prompt_sha256=compute_prompt_sha256(text),
    )
