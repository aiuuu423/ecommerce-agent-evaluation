import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import app.experiments.llm_cli as llm_cli
from app.experiments.baseline_v1 import RunLevelError
from app.experiments.comparisons import ComparisonError
from app.experiments.llm_cli import (
    PILOT_FAILURE_RATE_LIMIT,
    _dataset_dirs,
    formal_main,
    pilot_gate,
    pilot_main,
)

ROOT = Path(__file__).parents[1]
PROVIDER_ENV = (
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _records(failed: int, total: int = 20) -> tuple[SimpleNamespace, ...]:
    return tuple(
        [SimpleNamespace(status="failed")] * failed
        + [SimpleNamespace(status="completed")] * (total - failed)
    )


def _cases() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            datasets={
                "development": SimpleNamespace(dataset_id="development-id"),
                "public_validation": SimpleNamespace(dataset_id="validation-id"),
            }
        )
    )


def _result(v1_failed: int = 0, v2_failed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        comparison_group_id="pilot__group",
        v1=SimpleNamespace(
            manifest=SimpleNamespace(run_id="v1-run"),
            records=_records(v1_failed),
        ),
        v2=SimpleNamespace(
            manifest=SimpleNamespace(run_id="v2-run"),
            records=_records(v2_failed),
        ),
    )


def _install_success_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: SimpleNamespace,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    cases = _cases()
    monkeypatch.setenv("LLM_API_KEY", "provider-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_cli, "load_runnable_cases", lambda path: cases)

    class FakeCoordinator:
        def __init__(self, **kwargs: object) -> None:
            calls["coordinator_kwargs"] = kwargs

        def run(self, **kwargs: object) -> SimpleNamespace:
            calls["run_kwargs"] = kwargs
            return result

    def fake_write_comparison(
        received: object,
        *,
        comparisons_root: Path,
    ) -> Path:
        calls["comparison"] = (received, comparisons_root)
        return Path("mock-comparisons/pilot__group")

    monkeypatch.setattr(llm_cli, "PairedRunCoordinator", FakeCoordinator)
    monkeypatch.setattr(llm_cli, "write_comparison", fake_write_comparison)
    return calls


def test_missing_provider_config_is_stable_for_both_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for main in (pilot_main, formal_main):
        assert main([]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "provider_config_error: missing_api_key\n"


@pytest.mark.parametrize(
    ("failed", "expected_ok", "expected_rate"),
    [(0, True, 0.0), (2, True, 0.10), (3, False, 0.15)],
)
def test_pilot_gate_boundary(
    failed: int,
    expected_ok: bool,
    expected_rate: float,
) -> None:
    ok, rate = pilot_gate(_records(failed))

    assert ok is expected_ok
    assert rate == pytest.approx(expected_rate)


def test_pilot_gate_rejects_empty_record_set() -> None:
    assert pilot_gate([]) == (False, 1.0)
    assert PILOT_FAILURE_RATE_LIMIT == 0.10


@pytest.mark.parametrize(
    ("v1_failed", "v2_failed", "expected_exit"),
    [(2, 2, 0), (3, 0, 2), (0, 3, 2)],
)
def test_pilot_writes_comparison_before_reporting_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    v1_failed: int,
    v2_failed: int,
    expected_exit: int,
) -> None:
    result = _result(v1_failed, v2_failed)
    calls = _install_success_mocks(monkeypatch, result=result)

    assert pilot_main([]) == expected_exit

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "comparison_group_id": "pilot__group",
        "comparison_path": "mock-comparisons/pilot__group",
        "mode": "pilot",
        "pilot_gate": {
            "failure_rate_limit": 0.10,
            "passed": expected_exit == 0,
            "v1_failure_rate": v1_failed / 20,
            "v2_failure_rate": v2_failed / 20,
        },
        "v1_run_id": "v1-run",
        "v2_run_id": "v2-run",
    }
    assert calls["comparison"][0] is result
    assert calls["run_kwargs"]["mode"] == "pilot"


def test_formal_succeeds_without_pilot_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result(20, 20)
    calls = _install_success_mocks(monkeypatch, result=result)

    assert formal_main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "formal"
    assert "pilot_gate" not in payload
    assert calls["run_kwargs"]["mode"] == "formal"


def test_dataset_dirs_uses_manifest_ids_and_cli_paths() -> None:
    args = SimpleNamespace(
        development_dataset=Path("cli-development"),
        public_validation_dataset=Path("cli-validation"),
    )

    assert _dataset_dirs(_cases(), args) == {
        "development-id": Path("cli-development"),
        "validation-id": Path("cli-validation"),
    }


class _ValidatedModel(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        _ValidatedModel.model_validate({"value": "secret-value"})
    return caught.value


@pytest.mark.parametrize(
    "error",
    [ValueError("secret-value"), OSError("/secret/path"), _validation_error()],
)
def test_case_load_errors_are_stable_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "provider-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def fail_load(path: Path) -> object:
        raise error

    monkeypatch.setattr(llm_cli, "load_runnable_cases", fail_load)

    assert pilot_main(["--case-dir", "/secret/path"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "case_config_error\n"
    assert "secret" not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_run_level_error_is_stable_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "provider-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_cli, "load_runnable_cases", lambda path: _cases())

    class FailingCoordinator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise RunLevelError("dataset_identity_error")

    monkeypatch.setattr(llm_cli, "PairedRunCoordinator", FailingCoordinator)

    assert formal_main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "dataset_identity_error\n"
    assert "secret" not in captured.out + captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("error", [ValueError("secret-value"), _validation_error()])
def test_unexpected_run_validation_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "provider-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_cli, "load_runnable_cases", lambda path: _cases())

    class FailingCoordinator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise error

    monkeypatch.setattr(llm_cli, "PairedRunCoordinator", FailingCoordinator)

    with pytest.raises(type(error), match="secret-value"):
        formal_main([])


def test_comparison_error_is_stable_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_success_mocks(monkeypatch, result=_result())

    def fail_comparison(result: object, *, comparisons_root: Path) -> Path:
        raise ComparisonError("secret comparison path /secret/path")

    monkeypatch.setattr(llm_cli, "write_comparison", fail_comparison)

    assert pilot_main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "comparison_error\n"
    assert "secret" not in captured.out + captured.err


def test_unexpected_comparison_value_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_success_mocks(monkeypatch, result=_result())

    def fail_comparison(result: object, *, comparisons_root: Path) -> Path:
        raise ValueError("secret-value")

    monkeypatch.setattr(llm_cli, "write_comparison", fail_comparison)

    with pytest.raises(ValueError, match="secret-value"):
        formal_main([])


def test_module_dispatch_calls_the_selected_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        llm_cli,
        "pilot_main",
        lambda argv: calls.append(("pilot", list(argv))) or 11,
    )
    monkeypatch.setattr(
        llm_cli,
        "formal_main",
        lambda argv: calls.append(("formal", list(argv))) or 12,
    )

    assert llm_cli._module_main(["pilot", "--case-dir", "pilot-cases"]) == 11
    assert llm_cli._module_main(["formal", "--case-dir", "formal-cases"]) == 12
    assert calls == [
        ("pilot", ["--case-dir", "pilot-cases"]),
        ("formal", ["--case-dir", "formal-cases"]),
    ]


def test_module_help_is_offline_for_both_commands() -> None:
    for command in ("pilot", "formal"):
        completed = subprocess.run(
            [sys.executable, "-m", "app.experiments.llm_cli", command, "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout
        assert "--case-dir" in completed.stdout
        assert completed.stderr == ""


def test_project_wiring_is_exact() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["scripts"] == {
        "build-phase1": "app.data.phase1:main",
        "phase2-smoke": "app.experiments.phase2_smoke:main",
        "phase3-baseline": "app.experiments.baseline_v1:main",
        "real-llm-pilot": "app.experiments.llm_cli:pilot_main",
        "real-llm-formal": "app.experiments.llm_cli:formal_main",
    }

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "outputs/real_runs/" in ignored
    assert "outputs/paired_comparisons/" in ignored

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for variable in (
        "REAL_RUN_CASE_DIR",
        "REAL_RUN_DEVELOPMENT_DATASET",
        "REAL_RUN_PUBLIC_VALIDATION_DATASET",
        "REAL_RUN_OUTPUT_ROOT",
        "REAL_RUN_COMPARISONS_ROOT",
    ):
        assert f"{variable} ?=" in makefile
    phony_line = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert "real-llm-pilot" in phony_line
    assert "real-llm-formal" in phony_line
    assert "real-llm-pilot: check-python" in makefile
    assert "real-llm-formal: check-python" in makefile
    assert "llm_cli pilot" in makefile
    assert "llm_cli formal" in makefile
    assert "pilot technical gate failed; do not run formal" in makefile
