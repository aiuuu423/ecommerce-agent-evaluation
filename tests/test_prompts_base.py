import pytest
from pydantic import ValidationError

from app.prompts.base import (
    PromptSnapshot,
    compute_prompt_sha256,
    make_prompt_snapshot,
)


def test_compute_prompt_sha256_uses_utf8() -> None:
    assert (
        compute_prompt_sha256("你好")
        == "670d9743542cae3ea7ebe36af56bd53648b0a1126162e78d81a32934a711302e"
    )


def test_make_prompt_snapshot_builds_frozen_snapshot() -> None:
    snapshot = make_prompt_snapshot(
        name="real-llm-v1",
        version="1.0.0",
        text="系统提示文本",
        change_note="initial",
        source_file="app/prompts/v1.py",
    )

    assert isinstance(snapshot, PromptSnapshot)
    assert snapshot.prompt_name == "real-llm-v1"
    assert snapshot.prompt_version == "1.0.0"
    assert snapshot.prompt_text == "系统提示文本"
    assert snapshot.change_note == "initial"
    assert snapshot.source_file == "app/prompts/v1.py"
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.prompt_name = "changed"


def test_make_prompt_snapshot_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        make_prompt_snapshot(
            name=" ",
            version="1.0.0",
            text="系统提示文本",
            change_note="initial",
            source_file="app/prompts/v1.py",
        )
