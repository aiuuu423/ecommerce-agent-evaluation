import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .artifacts import canonical_json_bytes
from .baseline_v1 import RunLevelError
from .cases import load_runnable_cases
from .comparisons import ComparisonError, write_comparison
from .llm_run import PairedRunCoordinator, RunSettings
from .llm_settings import ProviderConfigError, load_provider_settings

DEFAULT_CASE_DIR = Path("data/evaluation_cases/v1")
DEFAULT_DEVELOPMENT_DATASET = Path("data/synthetic/v1")
DEFAULT_PUBLIC_VALIDATION_DATASET = Path("data/synthetic/public-validation-v1")
DEFAULT_RUN_OUTPUT_ROOT = Path("outputs/real_runs")
DEFAULT_COMPARISONS_ROOT = Path("outputs/paired_comparisons")
PILOT_FAILURE_RATE_LIMIT = 0.10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-llm V1/V2 paired experiment"
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DATASET,
    )
    parser.add_argument(
        "--public-validation-dataset",
        type=Path,
        default=DEFAULT_PUBLIC_VALIDATION_DATASET,
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=DEFAULT_RUN_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--comparisons-root",
        type=Path,
        default=DEFAULT_COMPARISONS_ROOT,
    )
    return parser


def pilot_gate(records: Sequence[object]) -> tuple[bool, float]:
    total = len(records)
    if total == 0:
        return False, 1.0
    failed = sum(1 for record in records if record.status != "completed")
    rate = failed / total
    return rate <= PILOT_FAILURE_RATE_LIMIT, rate


def _dataset_dirs(cases: object, args: argparse.Namespace) -> dict[str, Path]:
    return {
        cases.manifest.datasets["development"].dataset_id: args.development_dataset,
        cases.manifest.datasets["public_validation"].dataset_id: (
            args.public_validation_dataset
        ),
    }


def _run(mode: str, argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    try:
        provider = load_provider_settings(os.environ)
    except ProviderConfigError as exc:
        print(f"provider_config_error: {exc.code}", file=sys.stderr)
        return 1

    try:
        cases = load_runnable_cases(args.case_dir)
    except (OSError, ValueError, ValidationError):
        print("case_config_error", file=sys.stderr)
        return 1

    try:
        coordinator = PairedRunCoordinator(
            provider=provider,
            run_settings=RunSettings(),
        )
        result = coordinator.run(
            mode=mode,
            cases=cases,
            dataset_dirs=_dataset_dirs(cases, args),
            run_output_root=args.run_output_root,
        )
    except ProviderConfigError as exc:
        print(f"provider_config_error: {exc.code}", file=sys.stderr)
        return 1
    except RunLevelError as exc:
        print(exc.code, file=sys.stderr)
        return 1

    try:
        comparison_path = write_comparison(
            result,
            comparisons_root=args.comparisons_root,
        )
    except ComparisonError:
        print("comparison_error", file=sys.stderr)
        return 1

    payload = {
        "comparison_group_id": result.comparison_group_id,
        "mode": mode,
        "v1_run_id": result.v1.manifest.run_id,
        "v2_run_id": result.v2.manifest.run_id,
        "comparison_path": str(comparison_path),
    }
    if mode == "pilot":
        v1_ok, v1_rate = pilot_gate(result.v1.records)
        v2_ok, v2_rate = pilot_gate(result.v2.records)
        passed = v1_ok and v2_ok
        payload["pilot_gate"] = {
            "failure_rate_limit": PILOT_FAILURE_RATE_LIMIT,
            "v1_failure_rate": v1_rate,
            "v2_failure_rate": v2_rate,
            "passed": passed,
        }
        sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
        return 0 if passed else 2

    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
    return 0


def pilot_main(argv: Sequence[str] | None = None) -> int:
    return _run("pilot", argv)


def formal_main(argv: Sequence[str] | None = None) -> int:
    return _run("formal", argv)


def _module_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a real-llm paired experiment",
        add_help=False,
    )
    parser.add_argument("command", choices=("pilot", "formal"))
    args, command_argv = parser.parse_known_args(argv)
    if args.command == "pilot":
        return pilot_main(command_argv)
    return formal_main(command_argv)


if __name__ == "__main__":
    raise SystemExit(_module_main())
