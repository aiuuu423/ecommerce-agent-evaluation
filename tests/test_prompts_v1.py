import re

from app.prompts import V1_PROMPT_NAME, V1_PROMPT_VERSION, v1_prompt_snapshot
from app.prompts.base import compute_prompt_sha256


def test_v1_prompt_snapshot_has_expected_metadata() -> None:
    snapshot = v1_prompt_snapshot()

    assert snapshot.prompt_name == V1_PROMPT_NAME == "real-llm-v1"
    assert snapshot.prompt_version == V1_PROMPT_VERSION == "1.0.0"
    assert (
        snapshot.prompt_sha256
        == "9e314d9474bfde9aeab30a4c35524e286d9e2e406642a41eb2405310179b6a23"
    )
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    assert snapshot.source_file == "app/prompts/v1.py"


def test_v1_prompt_text_does_not_leak_evaluation_tokens() -> None:
    text = v1_prompt_snapshot().prompt_text
    lowered_text = text.lower()

    for forbidden_token in (
        "case_",
        "gold",
        "expected",
        "score",
        "evidence_id",
        "tolerance",
    ):
        assert forbidden_token not in lowered_text

    assert re.search(r"\bE(?:10|[1-9])\b", text, re.IGNORECASE) is None
    assert "容差" not in text
