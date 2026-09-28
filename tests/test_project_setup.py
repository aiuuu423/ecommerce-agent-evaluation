import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_project_metadata_declares_python_311_and_phase_1_dependencies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]

    assert project["requires-python"] == ">=3.11"
    dependencies = " ".join(project["dependencies"]).lower()
    for package in ("pandas", "numpy", "pydantic", "duckdb", "pyarrow", "pyyaml"):
        assert package in dependencies


def test_generated_outputs_are_ignored_but_keep_files_are_trackable() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/synthetic/*" in ignore
    assert "!data/synthetic/.gitkeep" in ignore
    assert "data/evaluation_cases/*.jsonl" in ignore
