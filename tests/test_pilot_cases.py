from collections import Counter

import pytest

from app.experiments.cases import (
    CaseBundle,
    CaseSplit,
    DatasetIdentity,
    EvaluationCaseManifest,
    RunnableCase,
)
from app.experiments.pilot_cases import (
    PILOT_CASE_IDS,
    PILOT_CASE_SET_VERSION,
    PilotCaseSetError,
    pilot_case_set_sha256,
    select_pilot_cases,
)

EXPECTED_PILOT_CASE_IDS = (
    "CASE_003",
    "CASE_005",
    "CASE_013",
    "CASE_017",
    "CASE_022",
    "CASE_027",
    "CASE_029",
    "CASE_032",
    "CASE_041",
    "CASE_053",
    "CASE_055",
    "CASE_059",
    "CASE_063",
    "CASE_065",
    "CASE_076",
    "CASE_079",
    "CASE_087",
    "CASE_089",
    "CASE_091",
    "CASE_097",
)
EXPECTED_PILOT_CASE_SET_SHA256 = (
    "ff324e509b5e9212b2fd3b6b80a9993515f310fbed24adc6aeeff0cf7b875616"
)
SHA256 = "a" * 64


def _case_bundle(
    case_ids: tuple[str, ...],
    *,
    public_validation_ids: frozenset[str] = frozenset(),
    case_splits: tuple[CaseSplit, ...] | None = None,
) -> CaseBundle:
    datasets = {
        "development": DatasetIdentity(
            dataset_id="1" * 16,
            dataset_version="development-v1",
            generator_config_hash=SHA256,
        ),
        "public_validation": DatasetIdentity(
            dataset_id="2" * 16,
            dataset_version="public-validation-v1",
            generator_config_hash=SHA256,
        ),
    }
    splits = case_splits or tuple(
        "public_validation" if case_id in public_validation_ids else "development"
        for case_id in case_ids
    )
    cases = tuple(
        RunnableCase(
            case_id=case_id,
            user_input=f"Run evaluation for {case_id}",
            dataset_id=datasets[split].dataset_id,
            dataset_version=datasets[split].dataset_version,
            split=split,
        )
        for case_id, split in zip(case_ids, splits, strict=True)
    )
    if "public_validation" not in splits:
        cases += (
            RunnableCase(
                case_id="CASE_001",
                user_input="Run non-pilot public validation case",
                dataset_id=datasets["public_validation"].dataset_id,
                dataset_version=datasets["public_validation"].dataset_version,
                split="public_validation",
            ),
        )
    split_counts = Counter(case.split for case in cases)
    return CaseBundle(
        manifest=EvaluationCaseManifest(
            business_task_counts={"test": len(cases)},
            capability_counts={"test": len(cases)},
            case_count=len(cases),
            case_schema_version="1.0",
            case_set_id="3" * 16,
            datasets=datasets,
            difficulty_counts={"test": len(cases)},
            jsonl_sha256=SHA256,
            source_label="pilot case tests",
            split_counts=dict(split_counts),
            split_strategy={"kind": "test"},
            tool_contract={"version": "1.0", "sha256": SHA256},
        ),
        cases=cases,
        manifest_sha256=SHA256,
    )


def test_pilot_case_set_constants_are_frozen() -> None:
    assert PILOT_CASE_SET_VERSION == "1.0.0"
    assert PILOT_CASE_IDS == EXPECTED_PILOT_CASE_IDS
    assert len(PILOT_CASE_IDS) == 20
    assert len(set(PILOT_CASE_IDS)) == 20


def test_pilot_case_set_sha256_matches_frozen_literal() -> None:
    assert pilot_case_set_sha256() == EXPECTED_PILOT_CASE_SET_SHA256


def test_select_pilot_cases_uses_frozen_order_for_shuffled_bundle() -> None:
    bundle = _case_bundle(tuple(reversed(EXPECTED_PILOT_CASE_IDS)))

    selected = select_pilot_cases(bundle)

    assert isinstance(selected, tuple)
    assert all(type(case) is RunnableCase for case in selected)
    assert tuple(case.case_id for case in selected) == EXPECTED_PILOT_CASE_IDS


def test_select_pilot_cases_rejects_missing_id() -> None:
    bundle = _case_bundle(EXPECTED_PILOT_CASE_IDS[:-1])

    with pytest.raises(
        PilotCaseSetError,
        match=r"^pilot case set is incomplete$",
    ):
        select_pilot_cases(bundle)


def test_select_pilot_cases_rejects_non_development_case() -> None:
    bundle = _case_bundle(
        EXPECTED_PILOT_CASE_IDS,
        public_validation_ids=frozenset({"CASE_041"}),
    )

    with pytest.raises(
        PilotCaseSetError,
        match=r"^pilot cases must all be development cases$",
    ):
        select_pilot_cases(bundle)


@pytest.mark.parametrize(
    "duplicate_splits",
    [
        ("development", "public_validation"),
        ("public_validation", "development"),
    ],
)
def test_select_pilot_cases_rejects_duplicate_id_regardless_of_order(
    duplicate_splits: tuple[CaseSplit, CaseSplit],
) -> None:
    other_case_ids = tuple(
        case_id for case_id in EXPECTED_PILOT_CASE_IDS if case_id != "CASE_041"
    )
    bundle = _case_bundle(
        other_case_ids + ("CASE_041", "CASE_041"),
        case_splits=("development",) * len(other_case_ids) + duplicate_splits,
    )

    with pytest.raises(
        PilotCaseSetError,
        match=r"^pilot case IDs must be unique$",
    ):
        select_pilot_cases(bundle)
