import pytest

from app.prompts.base import compute_prompt_sha256
from app.prompts.v1 import (
    V1_PROMPT_NAME,
    V1_PROMPT_VERSION,
    v1_prompt_snapshot,
)


def test_v1_prompt_snapshot_has_expected_metadata() -> None:
    snapshot = v1_prompt_snapshot()

    assert snapshot.prompt_name == V1_PROMPT_NAME == "real-llm-v1"
    assert snapshot.prompt_version == V1_PROMPT_VERSION
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    assert snapshot.source_file == "app/prompts/v1.py"


@pytest.mark.parametrize(
    "forbidden_token",
    [
        "CASE_",
        "gold",
        "Gold",
        "expected",
        "score",
        "evidence_id",
        "E1",
        "tolerance",
        "容差",
    ],
)
def test_v1_prompt_text_does_not_leak_evaluation_tokens(
    forbidden_token: str,
) -> None:
    assert forbidden_token not in v1_prompt_snapshot().prompt_text
