import json
from contextlib import contextmanager
from datetime import date
from hashlib import sha256
from pathlib import Path
from unittest import mock

import pytest

from app.agents import RunResult, TraceEvent, TraceEventType
from app.data.database import open_dataset
from app.data.generator import build_snapshot
from app.experiments.artifacts import ArtifactWriter, verify_published_run
from app.experiments.baseline_v1 import (
    BaselineBatchRunner,
    RunLevelError,
    catalog_as_of_date,
)
from app.experiments.cases import (
    CaseBundle,
    DatasetIdentity,
    EvaluationCaseManifest,
    RunnableCase,
)

ROOT = Path(__file__).parents[1]
DEVELOPMENT_CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
PUBLIC_VALIDATION_CONFIG = (
    ROOT / "configs/data/synthetic_public_validation_v1.yaml"
)
SHA = "a" * 64


@pytest.fixture(scope="module")
def dataset_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("baseline-batch")
    paths = {
        "development": root / "development",
        "public_validation": root / "public-validation",
    }
    build_snapshot(DEVELOPMENT_CONFIG, paths["development"])
    build_snapshot(PUBLIC_VALIDATION_CONFIG, paths["public_validation"])
    return paths


def _dataset_metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _case_bundle(dataset_dirs: dict[str, Path]) -> CaseBundle:
    development = _dataset_metadata(dataset_dirs["development"])
    public_validation = _dataset_metadata(dataset_dirs["public_validation"])
    datasets = {
        "development": DatasetIdentity(
            dataset_id=development["dataset_id"],
            dataset_version=development["dataset_version"],
            generator_config_hash=development["config_sha256"],
        ),
        "public_validation": DatasetIdentity(
            dataset_id=public_validation["dataset_id"],
            dataset_version=public_validation["dataset_version"],
            generator_config_hash=public_validation["config_sha256"],
        ),
    }
    cases = (
        RunnableCase(
            case_id="CASE_001",
            user_input="诊断最近30天 GMV 变化",
            split="development",
            dataset_id=datasets["development"].dataset_id,
            dataset_version=datasets["development"].dataset_version,
        ),
        RunnableCase(
            case_id="CASE_002",
            user_input="列出最近30天需要关注的商品",
            split="public_validation",
            dataset_id=datasets["public_validation"].dataset_id,
            dataset_version=datasets["public_validation"].dataset_version,
        ),
        RunnableCase(
            case_id="CASE_003",
            user_input="分析最近30天 P003 的转化下降",
            split="development",
            dataset_id=datasets["development"].dataset_id,
            dataset_version=datasets["development"].dataset_version,
        ),
        RunnableCase(
            case_id="CASE_004",
            user_input="给出最近30天下周经营优先级",
            split="public_validation",
            dataset_id=datasets["public_validation"].dataset_id,
            dataset_version=datasets["public_validation"].dataset_version,
        ),
    )
    return CaseBundle(
        manifest=EvaluationCaseManifest(
            business_task_counts={"gmv_change": 1, "products_to_watch": 1,
                                  "conversion_decline": 1, "next_week_priorities": 1},
            capability_counts={"diagnosis": 4},
            case_count=4,
            case_schema_version="1.0",
            case_set_id="1" * 16,
            datasets=datasets,
            difficulty_counts={"medium": 4},
            jsonl_sha256=SHA,
            source_label="test cases",
            split_counts={"development": 2, "public_validation": 2},
            split_strategy={"kind": "test"},
            tool_contract={"version": "1.0", "sha256": SHA},
        ),
        cases=cases,
        manifest_sha256=SHA,
    )


def _dataset_mapping(
    bundle: CaseBundle,
    dataset_dirs: dict[str, Path],
) -> dict[str, Path]:
    return {
        bundle.manifest.datasets["development"].dataset_id: dataset_dirs["development"],
        bundle.manifest.datasets["public_validation"].dataset_id: (
            dataset_dirs["public_validation"]
        ),
    }


def test_batch_routes_two_catalogs_in_order_with_fresh_runtime_per_case(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    mapping = _dataset_mapping(bundle, dataset_dirs)
    policies: list[object] = []
    adapters: list[object] = []
    runners: list[object] = []
    routed_dataset_ids: list[str] = []

    class TrackingPolicy:
        def __init__(self, context: object) -> None:
            self.context = context
            policies.append(self)

    class TrackingAdapter:
        def __init__(self, policy: object) -> None:
            self.policy = policy
            adapters.append(self)

    class TrackingRunner:
        def __init__(self, registry: object, adapter: object) -> None:
            self.registry = registry
            self.adapter = adapter
            runners.append(self)

        def run(self, request: object, catalog: object) -> RunResult:
            routed_dataset_ids.append(catalog.verified_summary["dataset_id"])
            return RunResult(
                status="completed",
                final_answer=f"完成：{request.user_input}",
                prior_tool_executions=[],
                decision_trace=(),
                usage=None,
            )

    monkeypatch.setattr(
        "app.experiments.baseline_v1.BaselinePolicyV1", TrackingPolicy
    )
    monkeypatch.setattr(
        "app.experiments.baseline_v1.DeterministicAdapter", TrackingAdapter
    )
    monkeypatch.setattr("app.experiments.baseline_v1.AgentRunner", TrackingRunner)

    published = BaselineBatchRunner().run(bundle, mapping, tmp_path)
    verified = verify_published_run(published)

    assert [record.case_id for record in verified.records] == [
        case.case_id for case in bundle.cases
    ]
    assert routed_dataset_ids == [case.dataset_id for case in bundle.cases]
    assert len({id(item) for item in policies}) == len(bundle.cases)
    assert len({id(item) for item in adapters}) == len(bundle.cases)
    assert len({id(item) for item in runners}) == len(bundle.cases)
    assert all(record.usage is None for record in verified.records)
    assert verified.summary.split_counts.model_dump() == {
        "development": 2,
        "public_validation": 2,
    }
    assert verified.summary.evaluation_status == "pending_not_run"
    assert verified.manifest.usage_status == "unavailable"
    assert verified.manifest.started_at_utc.endswith("Z")
    assert verified.manifest.completed_at_utc.endswith("Z")
    assert len(verified.manifest.git_commit) == 40
    assert verified.manifest.runtime_versions.python
    assert verified.manifest.policy_snapshot_sha256
    assert verified.manifest.case_manifest_sha256 == bundle.manifest_sha256
    assert [item.dataset_id for item in verified.manifest.datasets] == [
        bundle.manifest.datasets["development"].dataset_id,
        bundle.manifest.datasets["public_validation"].dataset_id,
    ]
    assert [item.manifest_sha256 for item in verified.manifest.datasets] == [
        sha256(
            (dataset_dirs["development"] / "manifest.json").read_bytes()
        ).hexdigest(),
        sha256(
            (dataset_dirs["public_validation"] / "manifest.json").read_bytes()
        ).hexdigest(),
    ]


def test_failed_case_is_recorded_with_safe_code_and_next_case_runs(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    calls = 0

    def run(self: object, request: object, catalog: object) -> RunResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return RunResult(
                status="failed",
                final_answer=None,
                prior_tool_executions=[],
                decision_trace=(
                    TraceEvent(
                        sequence=1,
                        event_type=TraceEventType.ERROR,
                        payload={
                            "code": "tool_execution_error",
                            "summary": "sensitive detail must not be copied",
                        },
                    ),
                ),
                usage=None,
            )
        return RunResult(
            status="completed",
            final_answer="完成",
            prior_tool_executions=[],
            decision_trace=(),
            usage=None,
        )

    monkeypatch.setattr("app.experiments.baseline_v1.AgentRunner.run", run)

    published = BaselineBatchRunner().run(
        bundle,
        _dataset_mapping(bundle, dataset_dirs),
        tmp_path,
    )
    verified = verify_published_run(published)

    assert calls == 4
    assert verified.records[0].status == "failed"
    assert verified.records[0].error_code == "tool_execution_error"
    assert verified.records[0].final_answer is None
    assert verified.records[1].status == "completed"
    artifact_text = (published / "case_runs.jsonl").read_text(encoding="utf-8")
    assert "sensitive detail" not in artifact_text


def test_catalog_as_of_date_uses_fact_table_maxima_and_requires_equality(
    dataset_dirs: dict[str, Path],
) -> None:
    with open_dataset(dataset_dirs["development"]) as catalog:
        assert catalog_as_of_date(catalog) == date(2026, 4, 30)
        catalog.execute(
            "select max(order_date), date '2026-04-29', date '2026-04-30' "
            "from orders"
        )
        original_fetchone = catalog.fetchone
        with mock.patch.object(
            catalog,
            "fetchone",
            return_value=original_fetchone()[:1]
            + (date(2026, 4, 29), date(2026, 4, 30)),
        ):
            with pytest.raises(RunLevelError) as exc_info:
                catalog_as_of_date(catalog)
    assert exc_info.value.code == "dataset_date_mismatch"


class _AsOfCatalog:
    def __init__(
        self,
        row: object = (date(2026, 4, 30),) * 3,
        *,
        execute_error: Exception | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.execute_error = execute_error
        self.fetch_error = fetch_error

    def execute(self, query: str) -> "_AsOfCatalog":
        if self.execute_error is not None:
            raise self.execute_error
        return self

    def fetchone(self) -> object:
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.row


class _ExplodingRow:
    def __len__(self) -> int:
        return 3

    def __iter__(self):
        raise RuntimeError("secret row iteration failure")


@pytest.mark.parametrize(
    ("catalog", "expected_code"),
    [
        (_AsOfCatalog(execute_error=RuntimeError("secret execute failure")),
         "dataset_date_query_error"),
        (_AsOfCatalog(fetch_error=RuntimeError("secret fetch failure")),
         "dataset_date_query_error"),
        (_AsOfCatalog(None), "dataset_date_unavailable"),
        (_AsOfCatalog(()), "dataset_date_unavailable"),
        (_AsOfCatalog(object()), "dataset_date_invalid"),
        (_AsOfCatalog(_ExplodingRow()), "dataset_date_invalid"),
        (_AsOfCatalog(("2026-04-30",) * 3), "dataset_date_invalid"),
        (
            _AsOfCatalog(
                (date(2026, 4, 30), date(2026, 4, 29), date(2026, 4, 30))
            ),
            "dataset_date_mismatch",
        ),
    ],
)
def test_catalog_as_of_date_normalizes_all_invalid_results(
    catalog: _AsOfCatalog,
    expected_code: str,
) -> None:
    with pytest.raises(RunLevelError) as exc_info:
        catalog_as_of_date(catalog)  # type: ignore[arg-type]

    assert exc_info.value.code == expected_code
    assert str(exc_info.value) == expected_code


@pytest.mark.parametrize("row", [object(), _ExplodingRow()])
def test_malformed_as_of_rows_never_publish(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    row: object,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    monkeypatch.setattr(
        "app.experiments.baseline_v1.catalog_as_of_date",
        lambda catalog: catalog_as_of_date(_AsOfCatalog(row)),
    )

    with mock.patch.object(ArtifactWriter, "publish") as publish:
        with pytest.raises(RunLevelError) as exc_info:
            BaselineBatchRunner().run(
                bundle,
                _dataset_mapping(bundle, dataset_dirs),
                tmp_path / "outputs",
            )

    assert exc_info.value.code == "dataset_date_invalid"
    assert str(exc_info.value) == "dataset_date_invalid"
    assert exc_info.value.__cause__ is None
    publish.assert_not_called()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("split", "field", "replacement"),
    [
        ("development", "dataset_id", "0" * 16),
        ("development", "dataset_version", "wrong"),
        ("development", "config_sha256", "0" * 64),
        ("public_validation", "dataset_id", "0" * 16),
        ("public_validation", "dataset_version", "wrong"),
        ("public_validation", "config_sha256", "0" * 64),
    ],
)
def test_batch_rejects_each_split_dataset_manifest_mismatch_without_publishing(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    field: str,
    replacement: str,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    mapping = _dataset_mapping(bundle, dataset_dirs)
    target_path = dataset_dirs[split]

    @contextmanager
    def open_with_tampered_manifest(path: Path):
        with open_dataset(path) as catalog:
            if path == target_path:
                payload = dict(catalog.manifest)
                payload[field] = replacement
                object.__setattr__(catalog, "_Catalog__manifest", payload)
            yield catalog

    monkeypatch.setattr(
        "app.experiments.baseline_v1.open_dataset",
        open_with_tampered_manifest,
    )
    with mock.patch.object(ArtifactWriter, "publish") as publish:
        with pytest.raises(RunLevelError) as exc_info:
            BaselineBatchRunner().run(bundle, mapping, tmp_path / "outputs")

    assert exc_info.value.code == "dataset_identity_mismatch"
    publish.assert_not_called()
    assert not (tmp_path / "outputs").exists()


def test_batch_explicitly_rejects_same_dataset_id_for_both_splits(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    original = _case_bundle(dataset_dirs)
    manifest_payload = original.manifest.model_dump(mode="json")
    development = manifest_payload["datasets"]["development"]
    manifest_payload["datasets"]["public_validation"] = development
    cases_payload = [case.model_dump(mode="json") for case in original.cases]
    for case in cases_payload:
        if case["split"] == "public_validation":
            case["dataset_id"] = development["dataset_id"]
            case["dataset_version"] = development["dataset_version"]
    bundle = CaseBundle(
        manifest=EvaluationCaseManifest.model_validate(
            manifest_payload,
            strict=True,
        ),
        cases=tuple(
            RunnableCase.model_validate(case, strict=True) for case in cases_payload
        ),
        manifest_sha256=original.manifest_sha256,
    )
    mapping = {development["dataset_id"]: dataset_dirs["development"]}

    with mock.patch.object(ArtifactWriter, "publish") as publish:
        with pytest.raises(RunLevelError) as exc_info:
            BaselineBatchRunner().run(bundle, mapping, tmp_path / "outputs")

    assert exc_info.value.code == "dataset_split_id_collision"
    publish.assert_not_called()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "error_code",
    [
        "dataset_date_query_error",
        "dataset_date_unavailable",
        "dataset_date_invalid",
        "dataset_date_mismatch",
    ],
)
def test_as_of_run_level_errors_never_publish(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    monkeypatch.setattr(
        "app.experiments.baseline_v1.catalog_as_of_date",
        mock.Mock(side_effect=RunLevelError(error_code)),
    )

    with mock.patch.object(ArtifactWriter, "publish") as publish:
        with pytest.raises(RunLevelError) as exc_info:
            BaselineBatchRunner().run(
                bundle,
                _dataset_mapping(bundle, dataset_dirs),
                tmp_path / "outputs",
            )

    assert exc_info.value.code == error_code
    publish.assert_not_called()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("mapping", "dataset_mapping_error"),
        ("catalog", "catalog_open_error"),
        ("catalog_identity", "dataset_identity_mismatch"),
        ("dataset_drift", "dataset_identity_drift"),
        ("policy_drift", "policy_identity_drift"),
    ],
)
def test_run_level_errors_never_publish(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_code: str,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    mapping = _dataset_mapping(bundle, dataset_dirs)
    if failure == "mapping":
        mapping.pop(next(iter(mapping)))
    elif failure == "catalog":
        mapping[next(iter(mapping))] = tmp_path / "missing"
    elif failure == "catalog_identity":
        development_id = bundle.manifest.datasets["development"].dataset_id
        public_validation_id = (
            bundle.manifest.datasets["public_validation"].dataset_id
        )
        mapping[development_id], mapping[public_validation_id] = (
            mapping[public_validation_id],
            mapping[development_id],
        )
    elif failure == "dataset_drift":
        identities = iter(
            [
                (
                    ("a" * 16, "1" * 64, "a" * 64),
                    ("b" * 16, "2" * 64, "b" * 64),
                ),
                (
                    ("a" * 16, "1" * 64, "a" * 64),
                    ("b" * 16, "2" * 64, "c" * 64),
                ),
            ]
        )
        monkeypatch.setattr(
            "app.experiments.baseline_v1.dataset_source_identity",
            lambda paths: next(identities),
        )
    else:
        identities = iter([("a" * 64, ("a.py",)), ("b" * 64, ("a.py",))])
        monkeypatch.setattr(
            "app.experiments.baseline_v1.policy_source_identity",
            lambda: next(identities),
        )

    with mock.patch.object(ArtifactWriter, "publish") as publish:
        with pytest.raises(RunLevelError) as exc_info:
            BaselineBatchRunner().run(bundle, mapping, tmp_path / "outputs")

    assert exc_info.value.code == expected_code
    publish.assert_not_called()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "publish_error",
    [
        PermissionError("/private/output/path"),
        ValueError("invalid artifact at /private/output/path"),
    ],
)
def test_expected_artifact_publish_errors_are_normalized_without_cause(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    publish_error: Exception,
) -> None:
    bundle = _case_bundle(dataset_dirs)
    monkeypatch.setattr(
        "app.experiments.baseline_v1.AgentRunner.run",
        lambda self, request, catalog: RunResult(
            status="completed",
            final_answer="完成",
            prior_tool_executions=[],
            decision_trace=(),
            usage=None,
        ),
    )
    monkeypatch.setattr(
        ArtifactWriter,
        "publish",
        mock.Mock(side_effect=publish_error),
    )

    with pytest.raises(RunLevelError) as exc_info:
        BaselineBatchRunner().run(
            bundle,
            _dataset_mapping(bundle, dataset_dirs),
            tmp_path / "outputs",
        )

    assert exc_info.value.code == "artifact_publish_error"
    assert str(exc_info.value) == "artifact_publish_error"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_artifact_publish_programming_errors_propagate(
    tmp_path: Path,
    dataset_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.experiments.baseline_v1.AgentRunner.run",
        lambda self, request, catalog: RunResult(
            status="completed",
            final_answer="完成",
            prior_tool_executions=[],
            decision_trace=(),
            usage=None,
        ),
    )
    monkeypatch.setattr(
        ArtifactWriter,
        "publish",
        mock.Mock(side_effect=RuntimeError("programming error")),
    )
    bundle = _case_bundle(dataset_dirs)

    with pytest.raises(RuntimeError, match="programming error"):
        BaselineBatchRunner().run(
            bundle,
            _dataset_mapping(bundle, dataset_dirs),
            tmp_path / "outputs",
        )
