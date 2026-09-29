import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _versioned_config(source: Path, destination: Path, version: str) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["dataset_version"] = version
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_all_make_phase1_targets_delegate_to_the_single_phase1_entrypoint() -> None:
    for target, stage in (
        ("phase1-data", "data"),
        ("phase1-cases", "cases"),
        ("phase1", "all"),
    ):
        completed = subprocess.run(
            ["make", "--dry-run", f"PYTHON={sys.executable}", target],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert "-m app.data.phase1" in completed.stdout
        assert f"--stage {stage}" in completed.stdout
        assert "-m app.data.generator" not in completed.stdout
        assert "-m app.data.case_generator" not in completed.stdout


def test_non_v1_configs_drive_all_phase1_output_directories(tmp_path: Path) -> None:
    development_config = tmp_path / "development.yaml"
    public_validation_config = tmp_path / "public_validation.yaml"
    output_root = tmp_path / "artifacts"
    _versioned_config(
        ROOT / "configs/data/synthetic_v1.yaml",
        development_config,
        "development-v2",
    )
    _versioned_config(
        ROOT / "configs/data/synthetic_public_validation_v1.yaml",
        public_validation_config,
        "public-validation-v2",
    )

    completed = subprocess.run(
        [
            "make",
            f"PYTHON={sys.executable}",
            f"DEVELOPMENT_CONFIG={development_config}",
            f"PUBLIC_VALIDATION_CONFIG={public_validation_config}",
            f"PHASE1_OUTPUT_ROOT={output_root}",
            "phase1",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    development_manifest = json.loads(
        (output_root / "synthetic/development-v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    public_validation_manifest = json.loads(
        (output_root / "synthetic/public-validation-v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    case_manifest = json.loads(
        (output_root / "evaluation_cases/development-v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert development_manifest["dataset_version"] == "development-v2"
    assert (
        public_validation_manifest["dataset_version"]
        == "public-validation-v2"
    )
    assert case_manifest["datasets"]["development"]["dataset_version"] == (
        "development-v2"
    )
    assert case_manifest["datasets"]["public_validation"]["dataset_version"] == (
        "public-validation-v2"
    )
    assert not (output_root / "synthetic/v1").exists()
    assert not (output_root / "evaluation_cases/v1").exists()
