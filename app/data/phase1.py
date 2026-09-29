import argparse
from pathlib import Path

from app.data.case_generator import DEFAULT_TOOL_CONTRACT, build_cases, write_cases
from app.data.config import load_data_config
from app.data.generator import build_snapshot

DEFAULT_DEVELOPMENT_CONFIG = Path("configs/data/synthetic_v1.yaml")
DEFAULT_PUBLIC_VALIDATION_CONFIG = Path(
    "configs/data/synthetic_public_validation_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path("data")
PHASE1_STAGES = ("data", "cases", "all")


def phase1_paths(
    development_config: Path | str = DEFAULT_DEVELOPMENT_CONFIG,
    public_validation_config: Path | str = DEFAULT_PUBLIC_VALIDATION_CONFIG,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    root = Path(output_root)
    development_version = load_data_config(development_config).dataset_version
    public_validation_version = load_data_config(
        public_validation_config
    ).dataset_version
    return {
        "development": root / "synthetic" / development_version,
        "public_validation": root / "synthetic" / public_validation_version,
        "cases": root / "evaluation_cases" / development_version,
    }


def build_phase1(
    development_config: Path | str = DEFAULT_DEVELOPMENT_CONFIG,
    public_validation_config: Path | str = DEFAULT_PUBLIC_VALIDATION_CONFIG,
    tool_contract: Path | str = DEFAULT_TOOL_CONTRACT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    stage: str = "all",
) -> dict[str, object]:
    if stage not in PHASE1_STAGES:
        raise ValueError(f"unknown Phase 1 stage: {stage}")

    paths = phase1_paths(development_config, public_validation_config, output_root)
    manifests: dict[str, object] = {}
    if stage in {"data", "all"}:
        manifests["development"] = build_snapshot(
            development_config, paths["development"]
        )
        manifests["public_validation"] = build_snapshot(
            public_validation_config,
            paths["public_validation"],
        )
    if stage in {"cases", "all"}:
        manifests["cases"] = write_cases(
            build_cases(
                paths["development"],
                paths["public_validation"],
                tool_contract,
            ),
            paths["cases"],
        )
    return manifests


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
        "--public-validation-config",
        type=Path,
        default=DEFAULT_PUBLIC_VALIDATION_CONFIG,
    )
    parser.add_argument(
        "--tool-contract",
        type=Path,
        default=DEFAULT_TOOL_CONTRACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=PHASE1_STAGES, default="all")
    args = parser.parse_args()

    manifests = build_phase1(
        args.development_config,
        args.public_validation_config,
        args.tool_contract,
        args.output_root,
        args.stage,
    )
    identifiers = []
    if "development" in manifests:
        identifiers.append(f"development={manifests['development']['dataset_id']}")
        identifiers.append(
            "public_validation="
            f"{manifests['public_validation']['dataset_id']}"
        )
    if "cases" in manifests:
        identifiers.append(f"cases={manifests['cases']['case_set_id']}")
    print(f"Built Phase 1 {args.stage} artifacts: {', '.join(identifiers)}")


if __name__ == "__main__":
    main()
