import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "data/synthetic/phase1_sha256_baseline.json"
CASES = ROOT / "data/evaluation_cases/v1/cases.jsonl"
MANIFEST = ROOT / "data/evaluation_cases/v1/manifest.json"
CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"

FROZEN_BASELINE_SHA256 = "84140e698e2c006e1f3b71018cfd893119038940a395ae51c69f7613030b309f"
FROZEN_CASES_SHA256 = "f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03"
FROZEN_MANIFEST_SHA256 = "9396ce3a86175874d9846bc5c34490e73cdb9cee612c487900c4e1b6c1eeb489"
FROZEN_CONTRACT_SHA256 = "471e0f09d2a609d3f8799565020cc635f201dd2e95fb33e1a170c256d9e944cb"
FROZEN_CASE_SET_ID = "ecebfe8b691271fd"
FROZEN_DATASET_IDS = {
    "development": "e1e81533c25e03e5",
    "public_validation": "c17d4926cfa7cb26",
}
EXPECTED_LOCK_PATHS = {
    "evaluation_cases/.v1.lock",
    "synthetic/.public-validation-v1.lock",
    "synthetic/.v1.lock",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def clean_phase1_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("phase1-frozen")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.data.phase1",
            "--development-config",
            str(ROOT / "configs/data/synthetic_v1.yaml"),
            "--public-validation-config",
            str(ROOT / "configs/data/synthetic_public_validation_v1.yaml"),
            "--tool-contract",
            str(ROOT / "configs/evaluation/tool_contract_v1.yaml"),
            "--output-root",
            str(output_root),
            "--stage",
            "all",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_root


def test_committed_freeze_files_match_literal_hashes() -> None:
    assert _sha256(BASELINE) == FROZEN_BASELINE_SHA256
    assert _sha256(CASES) == FROZEN_CASES_SHA256
    assert _sha256(MANIFEST) == FROZEN_MANIFEST_SHA256
    assert _sha256(CONTRACT) == FROZEN_CONTRACT_SHA256


def test_phase1_clean_build_matches_frozen_baseline(
    clean_phase1_build: Path,
) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["artifact_count"] == 16
    assert baseline["dataset_ids"] == FROZEN_DATASET_IDS
    assert baseline["case_set_id"] == FROZEN_CASE_SET_ID
    expected_paths = {artifact["path"] for artifact in baseline["artifacts"]}
    actual_file_paths = {
        path.relative_to(clean_phase1_build).as_posix()
        for path in clean_phase1_build.rglob("*")
        if path.is_file()
    }
    actual_lock_paths = {
        path for path in actual_file_paths if Path(path).suffix == ".lock"
    }
    actual_asset_paths = actual_file_paths - actual_lock_paths
    assert len(expected_paths) == 16
    assert actual_lock_paths == EXPECTED_LOCK_PATHS
    assert actual_asset_paths == expected_paths
    for artifact in baseline["artifacts"]:
        path = clean_phase1_build / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]


def test_phase1_contract_cases_and_manifest_remain_bound() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert contract["contract_version"] == "1.0"
    assert manifest["case_set_id"] == FROZEN_CASE_SET_ID
    assert manifest["jsonl_sha256"] == FROZEN_CASES_SHA256
    assert manifest["tool_contract"] == {
        "version": "1.0",
        "sha256": FROZEN_CONTRACT_SHA256,
    }
    assert {
        split: item["dataset_id"] for split, item in manifest["datasets"].items()
    } == FROZEN_DATASET_IDS
    for line in CASES.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        assert case["tool_contract_version"] == "1.0"
        assert case["tool_contract_sha256"] == FROZEN_CONTRACT_SHA256
        assert case["dataset_id"] == FROZEN_DATASET_IDS[case["split"]]
