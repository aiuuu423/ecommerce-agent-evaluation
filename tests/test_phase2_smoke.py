import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.agents import AgentRunner, RunResult
from app.experiments.phase2_smoke import main

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_phase2_offline_smoke_needs_no_key_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = tmp_path / "snapshot"
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    exit_code = main(["--config", str(CONFIG), "--work-dir", str(tmp_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "completed"
    assert payload["adapter"] == "deterministic"
    assert payload["tool_names"] == ["query_sales", "calculate_metrics"]
    assert payload["dataset_id"] == "e1e81533c25e03e5"
    assert payload["trace_event_count"] > 0
    assert "gmv_change_rate" in payload["final_answer"]
    assert (snapshot / "manifest.json").is_file()
    before = _file_hashes(snapshot)
    assert main(["--config", str(CONFIG), "--work-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert _file_hashes(snapshot) == before
    assert not (tmp_path / "run_manifest.json").exists()


def test_phase2_offline_smoke_runner_failure_uses_stable_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failed = RunResult(
        status="failed",
        final_answer=None,
        prior_tool_executions=[],
        decision_trace=(),
        usage=None,
    )
    monkeypatch.setattr(AgentRunner, "run", lambda self, request, catalog: failed)

    exit_code = main(["--config", str(CONFIG), "--work-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: offline smoke runner did not complete\n"
    assert (tmp_path / "snapshot" / "manifest.json").is_file()
