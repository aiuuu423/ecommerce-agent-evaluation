import re

from app.prompts import V2_PROMPT_NAME, V2_PROMPT_VERSION, v2_prompt_snapshot
from app.prompts.base import compute_prompt_sha256


def test_v2_prompt_snapshot_has_expected_metadata() -> None:
    snapshot = v2_prompt_snapshot()

    assert snapshot.prompt_name == V2_PROMPT_NAME == "real-llm-v2"
    assert snapshot.prompt_version == V2_PROMPT_VERSION == "1.0.0"
    assert (
        snapshot.prompt_sha256
        == "b6d02ede9e6621704b6cd8bced15a8bdb2201f1db7e9561af73bb8e8f339fd5d"
    )
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    assert snapshot.source_file == "app/prompts/v2.py"


def test_v2_prompt_requires_evidence_bound_structured_answers() -> None:
    text = v2_prompt_snapshot().prompt_text

    for required_text in (
        "result_id",
        "结论",
        "关键证据",
        "建议或限制",
        "数据不足",
    ):
        assert required_text in text


def test_v2_prompt_text_does_not_leak_evaluation_tokens() -> None:
    text = v2_prompt_snapshot().prompt_text
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

    assert "容差" not in text
    assert re.search(r"\bE(?:10|[1-9])\b", text, re.IGNORECASE) is None
