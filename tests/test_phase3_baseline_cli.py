import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.experiments.baseline_v1 import RunLevelError, main

ROOT = Path(__file__).parents[1]


def _case_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            datasets={
                "development": SimpleNamespace(dataset_id="a" * 16),
                "public_validation": SimpleNamespace(dataset_id="b" * 16),
            }
        )
    )


def test_main_uses_plan_defaults_and_emits_only_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _case_bundle()
    published = tmp_path / "outputs" / "baseline-v1__test"
    loaded_paths: list[Path] = []

    def load_cases(path: Path) -> SimpleNamespace:
        loaded_paths.append(path)
        return bundle

    monkeypatch.setattr(
        "app.experiments.baseline_v1.load_runnable_cases",
        load_cases,
    )
    run_calls: list[tuple[object, object, Path]] = []

    def run(
        self: object,
        cases: object,
        dataset_dirs: object,
        output_root: Path,
    ) -> Path:
        run_calls.append((cases, dataset_dirs, output_root))
        return published

    monkeypatch.setattr("app.experiments.baseline_v1.BaselineBatchRunner.run", run)
    monkeypatch.setattr(
        "app.experiments.baseline_v1.verify_published_run",
        lambda path: SimpleNamespace(
            manifest=SimpleNamespace(run_id="baseline-v1__test"),
            summary=SimpleNamespace(completed=3, failed=1),
        ),
    )

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == (
        '{"completed":3,"failed":1,"path":"'
        f'{published}","run_id":"baseline-v1__test"' + "}\n"
    )
    assert json.loads(captured.out) == {
        "run_id": "baseline-v1__test",
        "path": str(published),
        "completed": 3,
        "failed": 1,
    }
    assert run_calls == [
        (
            bundle,
            {
                "a" * 16: Path("data/synthetic/v1"),
                "b" * 16: Path("data/synthetic/public-validation-v1"),
            },
            Path("outputs/experiment_runs"),
        )
    ]
    assert loaded_paths == [Path("data/evaluation_cases/v1")]


def test_run_level_error_emits_only_safe_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "app.experiments.baseline_v1.load_runnable_cases",
        lambda path: _case_bundle(),
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.BaselineBatchRunner.run",
        lambda self, cases, dataset_dirs, output_root: (_ for _ in ()).throw(
            RunLevelError("catalog_open_error")
        ),
    )

    exit_code = main(
        [
            "--case-dir",
            str(tmp_path / "cases"),
            "--development-dataset",
            str(tmp_path / "development"),
            "--public-validation-dataset",
            str(tmp_path / "public validation"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "catalog_open_error\n"
    assert not (tmp_path / "outputs").exists()


def test_missing_input_returns_safe_error_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "outputs"

    exit_code = main(
        [
            "--case-dir",
            str(tmp_path / "missing-cases"),
            "--development-dataset",
            str(tmp_path / "missing-development"),
            "--public-validation-dataset",
            str(tmp_path / "missing-public-validation"),
            "--output-root",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "case_load_error\n"
    assert not output_root.exists()


@pytest.mark.parametrize(
    "verify_error",
    [
        PermissionError("/private/published/run"),
        ValueError("invalid artifact at /private/published/run"),
    ],
)
def test_verify_errors_emit_only_safe_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verify_error: Exception,
) -> None:
    published_path = tmp_path / "private" / "published-run"
    monkeypatch.setattr(
        "app.experiments.baseline_v1.load_runnable_cases",
        lambda path: _case_bundle(),
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.BaselineBatchRunner.run",
        lambda self, cases, dataset_dirs, output_root: published_path,
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.verify_published_run",
        lambda path: (_ for _ in ()).throw(verify_error),
    )

    exit_code = main(["--output-root", str(tmp_path / "outputs")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "artifact_verify_error\n"
    assert "Traceback" not in captured.err
    assert str(tmp_path) not in captured.err


def test_verify_programming_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.experiments.baseline_v1.load_runnable_cases",
        lambda path: _case_bundle(),
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.BaselineBatchRunner.run",
        lambda self, cases, dataset_dirs, output_root: tmp_path / "published-run",
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.verify_published_run",
        lambda path: (_ for _ in ()).throw(RuntimeError("programming error")),
    )

    with pytest.raises(RuntimeError, match="programming error"):
        main([])


def test_make_target_quotes_python_path_with_spaces(tmp_path: Path) -> None:
    python_dir = tmp_path / "Python Runtime" / "bin"
    python_dir.mkdir(parents=True)
    python_path = python_dir / "python"
    python_path.symlink_to(sys.executable)
    python_with_spaces = str(python_path)
    check = subprocess.run(
        ["make", f"PYTHON={python_with_spaces}", "check-python"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            "make",
            "--dry-run",
            f"PYTHON={python_with_spaces}",
            "phase3-baseline",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert check.returncode == 0, check.stderr
    assert result.returncode == 0, result.stderr
    assert f'"{python_with_spaces}" -m app.experiments.baseline_v1' in result.stdout


def test_module_entrypoint_propagates_main_exit_code(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.experiments.baseline_v1",
            "--case-dir",
            str(tmp_path / "missing"),
            "--development-dataset",
            str(tmp_path / "missing-development"),
            "--public-validation-dataset",
            str(tmp_path / "missing-public-validation"),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "case_load_error\n"
    assert not (tmp_path / "outputs").exists()
