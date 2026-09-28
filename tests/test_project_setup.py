import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).parents[1]


def dependency_names(dependencies: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(dependency).name) for dependency in dependencies}


def test_project_metadata_declares_python_311_and_phase_1_dependencies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]

    assert project["requires-python"] == ">=3.11"
    assert dependency_names(project["dependencies"]) == {
        "duckdb",
        "filelock",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
    }


def test_lock_file_exactly_pins_declared_dependencies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = dependency_names(
        config["project"]["dependencies"] + config["project"]["optional-dependencies"]["dev"]
    )
    locked_requirements = [
        Requirement(line)
        for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith((" ", "#"))
    ]

    assert all(
        len(requirement.specifier) == 1
        and next(iter(requirement.specifier)).operator == "=="
        for requirement in locked_requirements
    )
    assert declared <= {
        canonicalize_name(requirement.name) for requirement in locked_requirements
    }


def test_makefile_python_entry_is_overridable_and_fails_clearly() -> None:
    success = subprocess.run(
        ["make", f"PYTHON={sys.executable}", "check-python"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert success.returncode == 0, success.stderr

    failure = subprocess.run(
        ["make", "PYTHON=missing-python-command", "check-python"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failure.returncode != 0
    assert "Python 3.11+ executable" in failure.stdout


def test_generated_outputs_are_ignored_but_keep_files_are_trackable() -> None:
    def is_ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", path],
            cwd=ROOT,
            check=False,
        )
        return result.returncode == 0

    for path in (
        ".env",
        ".env.local",
        ".env.production",
        "data/synthetic/example.parquet",
        "data/evaluation_cases/example.jsonl",
        "data/evaluation_cases/example.manifest.json",
        "data/results/example.json",
    ):
        assert is_ignored(path), f"{path} should be ignored"

    for path in (
        ".env.example",
        "data/synthetic/.gitkeep",
        "data/results/.gitkeep",
    ):
        assert not is_ignored(path), f"{path} should remain trackable"
