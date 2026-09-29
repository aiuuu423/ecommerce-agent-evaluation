import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.data.generator import build_snapshot
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
    build_snapshot(CONFIG, snapshot)
    before = _file_hashes(tmp_path)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    exit_code = main(["--config", str(CONFIG), "--work-dir", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["adapter"] == "deterministic"
    assert payload["tool_names"] == ["query_sales", "calculate_metrics"]
    assert payload["dataset_id"] == "e1e81533c25e03e5"
    assert payload["trace_event_count"] > 0
    assert "gmv_change_rate" in payload["final_answer"]
    assert _file_hashes(tmp_path) == before
    assert not (tmp_path / "run_manifest.json").exists()


def test_phase2_offline_smoke_without_snapshot_tells_user_to_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--config", str(CONFIG), "--work-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "make phase1-data" in captured.err
    assert not (tmp_path / "snapshot").exists()
