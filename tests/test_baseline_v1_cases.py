import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from app.data.manifest import manifest_id
from app.experiments.cases import RunnableCase, load_runnable_cases

ROOT = Path(__file__).parents[1]
FROZEN_CASE_DIR = ROOT / "data/evaluation_cases/v1"
FROZEN_CASE_SET_ID = "ecebfe8b691271fd"
FROZEN_CASES_SHA256 = (
    "f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03"
)


def _canonical_line(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_case_dir(
    destination: Path,
    cases: list[dict[str, object]],
    *,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    lines = [_canonical_line(case) for case in cases]
    contents = ("\n".join(lines) + "\n").encode()
    source_manifest = json.loads(
        (FROZEN_CASE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    source_manifest["case_count"] = len(cases)
    source_manifest["case_set_id"] = manifest_id({"lines": lines})
    source_manifest["jsonl_sha256"] = hashlib.sha256(contents).hexdigest()
    source_manifest["split_counts"] = {
        split: sum(case["split"] == split for case in cases)
        for split in ("development", "public_validation")
    }
    source_manifest["business_task_counts"] = {
        task: sum(case["business_task"] == task for case in cases)
        for task in {case["business_task"] for case in cases}
    }
    source_manifest["capability_counts"] = {
        capability: sum(case["primary_capability"] == capability for case in cases)
        for capability in {case["primary_capability"] for case in cases}
    }
    source_manifest["difficulty_counts"] = {
        difficulty: sum(case["difficulty"] == difficulty for case in cases)
        for difficulty in {case["difficulty"] for case in cases}
    }
    if mutate_manifest is not None:
        mutate_manifest(source_manifest)
    destination.mkdir()
    (destination / "cases.jsonl").write_bytes(contents)
    (destination / "manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


@pytest.fixture
def two_source_cases() -> list[dict[str, object]]:
    cases = [
        json.loads(line)
        for line in (FROZEN_CASE_DIR / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return [
        next(case for case in cases if case["split"] == "public_validation"),
        next(case for case in cases if case["split"] == "development"),
    ]


def test_load_runnable_cases_validates_identity_and_projects_only_safe_fields(
    tmp_path: Path,
    two_source_cases: list[dict[str, object]],
) -> None:
    case_dir = _write_case_dir(tmp_path / "cases", two_source_cases)

    bundle = load_runnable_cases(case_dir)

    assert [case.case_id for case in bundle.cases] == [
        case["case_id"] for case in two_source_cases
    ]
    assert bundle.manifest.case_count == 2
    assert bundle.manifest.split_counts == {
        "development": 1,
        "public_validation": 1,
    }
    allowed_fields = {
        "case_id",
        "user_input",
        "dataset_id",
        "dataset_version",
        "split",
    }
    assert all(set(case.model_dump()) == allowed_fields for case in bundle.cases)
    assert "gold_metrics" in two_source_cases[0]
    assert "expected_tool_calls" in two_source_cases[0]
    assert "gold" not in json.dumps(
        [case.model_dump(mode="json") for case in bundle.cases]
    ).lower()
    with pytest.raises(ValidationError, match="frozen"):
        bundle.cases[0].user_input = "tampered"


def test_case_bundle_deeply_freezes_manifest_mappings_and_serializes_canonically(
    tmp_path: Path,
    two_source_cases: list[dict[str, object]],
) -> None:
    def add_nested_strategy(manifest: dict[str, object]) -> None:
        split_strategy = manifest["split_strategy"]
        assert isinstance(split_strategy, dict)
        split_strategy["nested"] = {"levels": [{"enabled": True}]}

    case_dir = _write_case_dir(
        tmp_path / "cases",
        two_source_cases,
        mutate_manifest=add_nested_strategy,
    )
    source_manifest = json.loads(
        (case_dir / "manifest.json").read_text(encoding="utf-8")
    )

    bundle = load_runnable_cases(case_dir)

    manifest = bundle.manifest
    frozen_mappings = (
        manifest.business_task_counts,
        manifest.capability_counts,
        manifest.datasets,
        manifest.difficulty_counts,
        manifest.split_counts,
        manifest.split_strategy,
        manifest.split_strategy["nested"],
        manifest.split_strategy["nested"]["levels"][0],
    )
    assert all(isinstance(mapping, MappingProxyType) for mapping in frozen_mappings)
    assert isinstance(manifest.split_strategy["nested"]["levels"], tuple)
    for mapping in frozen_mappings:
        with pytest.raises(TypeError):
            mapping["tampered"] = True
    with pytest.raises(TypeError):
        manifest.split_strategy["nested"]["levels"][0] = {"enabled": False}

    serialized = manifest.model_dump(mode="json")
    assert serialized == source_manifest
    canonical = json.dumps(
        bundle.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert json.loads(canonical)["manifest"] == source_manifest


def test_runnable_case_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunnableCase.model_validate(
            {
                "case_id": "CASE_001",
                "user_input": "分析最近30天GMV变化",
                "dataset_id": "e1e81533c25e03e5",
                "dataset_version": "v1",
                "split": "development",
                "gold_metrics": {},
            }
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda cases, manifest: manifest.__setitem__("jsonl_sha256", "0" * 64),
            "SHA-256",
        ),
        (
            lambda cases, manifest: manifest.__setitem__("case_count", 3),
            "case count",
        ),
        (
            lambda cases, manifest: manifest.__setitem__(
                "split_counts", {"development": 2, "public_validation": 0}
            ),
            "split counts",
        ),
        (
            lambda cases, manifest: manifest.__setitem__("case_set_id", "0" * 16),
            "case set ID",
        ),
    ],
)
def test_load_runnable_cases_rejects_manifest_identity_mismatches(
    tmp_path: Path,
    two_source_cases: list[dict[str, object]],
    change: Callable[[list[dict[str, object]], dict[str, object]], None],
    error: str,
) -> None:
    case_dir = _write_case_dir(
        tmp_path / "cases",
        two_source_cases,
        mutate_manifest=lambda manifest: change(two_source_cases, manifest),
    )

    with pytest.raises(ValueError, match=error):
        load_runnable_cases(case_dir)


def test_load_runnable_cases_rejects_duplicate_case_ids(
    tmp_path: Path,
    two_source_cases: list[dict[str, object]],
) -> None:
    two_source_cases[1]["case_id"] = two_source_cases[0]["case_id"]
    case_dir = _write_case_dir(tmp_path / "cases", two_source_cases)

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_runnable_cases(case_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset_id", "0" * 16),
        ("dataset_version", "wrong-version"),
        ("generator_config_hash", "0" * 64),
    ],
)
def test_load_runnable_cases_rejects_dataset_mapping_mismatch(
    tmp_path: Path,
    two_source_cases: list[dict[str, object]],
    field: str,
    replacement: str,
) -> None:
    two_source_cases[0][field] = replacement
    case_dir = _write_case_dir(tmp_path / "cases", two_source_cases)

    with pytest.raises(ValueError, match="dataset mapping"):
        load_runnable_cases(case_dir)


def test_frozen_case_snapshot_regression() -> None:
    bundle = load_runnable_cases(FROZEN_CASE_DIR)

    assert bundle.manifest.case_set_id == FROZEN_CASE_SET_ID
    assert bundle.manifest.jsonl_sha256 == FROZEN_CASES_SHA256
    assert bundle.manifest.case_count == len(bundle.cases) == 100
    assert bundle.manifest.split_counts == {
        "development": 70,
        "public_validation": 30,
    }
    assert [case.case_id for case in bundle.cases] == [
        f"CASE_{index:03d}" for index in range(1, 101)
    ]
