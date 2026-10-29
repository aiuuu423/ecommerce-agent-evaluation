import hashlib
import json

from app.experiments.cases import CaseBundle, RunnableCase

PILOT_CASE_SET_VERSION = "1.0.0"
PILOT_CASE_IDS = (
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


class PilotCaseSetError(RuntimeError):
    pass


def pilot_case_set_sha256() -> str:
    payload = {
        "version": PILOT_CASE_SET_VERSION,
        "case_ids": list(PILOT_CASE_IDS),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_pilot_cases(bundle: CaseBundle) -> tuple[RunnableCase, ...]:
    cases_by_id = {case.case_id: case for case in bundle.cases}
    if any(case_id not in cases_by_id for case_id in PILOT_CASE_IDS):
        raise PilotCaseSetError("pilot case set is incomplete")

    selected = tuple(cases_by_id[case_id] for case_id in PILOT_CASE_IDS)
    if any(case.split != "development" for case in selected):
        raise PilotCaseSetError("pilot cases must all be development cases")
    return selected
