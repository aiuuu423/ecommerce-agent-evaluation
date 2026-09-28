import argparse
from pathlib import Path

from app.data.case_generator import DEFAULT_TOOL_CONTRACT, build_cases, write_cases
from app.data.config import load_data_config
from app.data.generator import build_snapshot

DEFAULT_DEVELOPMENT_CONFIG = Path("configs/data/synthetic_v1.yaml")
DEFAULT_HOLDOUT_CONFIG = Path("configs/data/synthetic_holdout_v1.yaml")
DEFAULT_OUTPUT_ROOT = Path("data")


def build_phase1(
    development_config: Path | str = DEFAULT_DEVELOPMENT_CONFIG,
    holdout_config: Path | str = DEFAULT_HOLDOUT_CONFIG,
    tool_contract: Path | str = DEFAULT_TOOL_CONTRACT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    root = Path(output_root)
    development_version = load_data_config(development_config).dataset_version
    holdout_version = load_data_config(holdout_config).dataset_version
    development_dir = root / "synthetic" / development_version
    holdout_dir = root / "synthetic" / holdout_version

    development_manifest = build_snapshot(development_config, development_dir)
    holdout_manifest = build_snapshot(holdout_config, holdout_dir)
    case_manifest = write_cases(
        build_cases(development_dir, holdout_dir, tool_contract),
        root / "evaluation_cases" / "v1",
    )
    return {
        "development": development_manifest,
        "holdout": holdout_manifest,
        "cases": case_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete Phase 1 data and evaluation case artifacts"
    )
    parser.add_argument(
        "--development-config",
        type=Path,
        default=DEFAULT_DEVELOPMENT_CONFIG,
    )
    parser.add_argument(
        "--holdout-config",
        type=Path,
        default=DEFAULT_HOLDOUT_CONFIG,
    )
    parser.add_argument(
        "--tool-contract",
        type=Path,
        default=DEFAULT_TOOL_CONTRACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    manifests = build_phase1(
        args.development_config,
        args.holdout_config,
        args.tool_contract,
        args.output_root,
    )
    print(
        "Built Phase 1 artifacts: "
        f"development={manifests['development']['dataset_id']}, "
        f"holdout={manifests['holdout']['dataset_id']}, "
        f"cases={manifests['cases']['case_set_id']}"
    )


if __name__ == "__main__":
    main()
