import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from app.data.phase1 import build_phase1

ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_data_exploration.ipynb"


def load_notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


@pytest.fixture
def isolated_notebook_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "notebooks").mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = \"notebook-execution-test\"\n",
        encoding="utf-8",
    )
    shutil.copytree(ROOT / "app", project_root / "app")

    build_phase1(
        development_config=ROOT / "configs/data/synthetic_v1.yaml",
        public_validation_config=(
            ROOT / "configs/data/synthetic_public_validation_v1.yaml"
        ),
        output_root=project_root / "data",
        stage="data",
    )
    return project_root


def test_data_exploration_notebook_has_required_sections() -> None:
    notebook = load_notebook()
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    for section in (
        "Synthetic Data 声明",
        "数据范围",
        "质量检查",
        "指标分布",
        "异常场景",
        "双数据集说明",
        "局限性",
    ):
        assert section in markdown

    assert "Synthetic E-commerce Data" in markdown
    assert "Development" in markdown
    assert "Public Validation" in markdown


def test_notebook_is_read_only_and_contains_no_execution_state() -> None:
    notebook = load_notebook()
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert all(cell.get("execution_count") is None for cell in notebook.cells)
    assert all(not cell.get("outputs", []) for cell in notebook.cells)
    for write_operation in (
        ".write_",
        ".to_csv(",
        ".to_json(",
        ".to_parquet(",
        "build_snapshot(",
    ):
        assert write_operation not in code


def test_notebook_uses_verified_catalog_without_public_validation_query_access() -> None:
    notebook = load_notebook()
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    content = "\n".join(cell.source for cell in notebook.cells)

    assert "Path.cwd()" in code
    assert "pyproject.toml" in code
    assert "PHASE1_DATA_ROOT" not in code
    assert 'PROJECT_ROOT / "data/synthetic/v1"' in code
    assert 'PROJECT_ROOT / "data/synthetic/public-validation-v1"' in code
    assert "open_dataset" in code
    assert ".fetch_df()" in code
    assert ".verified_summary" in code
    assert ".manifest" not in code
    assert "make phase1-data" in content
    assert "pd.read_parquet" not in code
    assert "read_text(" not in code
    assert 'catalogs["Public Validation"].execute' not in code
    assert 'catalogs = {"Development": open_dataset(SNAPSHOTS["Development"])}' in code
    assert (
        'with open_dataset(SNAPSHOTS["Public Validation"]) '
        "as public_validation_catalog:"
    ) in code
    assert "del public_validation_catalog" in code

    lowered = content.lower()
    for forbidden in (
        ".yaml",
        "seed",
        "multiplier",
        "config_sha256",
        "generator_source_sha256",
        "product_id",
    ):
        assert forbidden not in lowered


def test_notebook_covers_business_metrics_without_agent_effect_numbers() -> None:
    notebook = load_notebook()
    content = "\n".join(cell.source for cell in notebook.cells)

    for metric in ("GMV", "Orders", "AOV", "CTR", "CVR", "Refund Rate", "ROAS"):
        assert metric in content
    assert "Pending / Not Run" in content
    assert "优化后提升" not in content
    assert "准确率提升" not in content
    assert not re.search(
        r"(?:Agent|Baseline|Optimized|V1|V2).{0,30}\d+(?:\.\d+)?%",
        content,
        flags=re.IGNORECASE,
    )


@pytest.mark.parametrize("execution_subdirectory", [Path(), Path("notebooks")])
def test_notebook_executes_from_supported_working_directories(
    execution_subdirectory: Path,
    isolated_notebook_project: Path,
) -> None:
    project_root = isolated_notebook_project
    execution_cwd = project_root / execution_subdirectory
    notebook = load_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
expected_summary_columns = {
    "dataset",
    "source_label",
    "dataset_version",
    "dataset_id",
    "start_date",
    "end_date",
    "rows_products",
    "rows_customers",
    "rows_traffic",
    "rows_marketing",
    "rows_orders",
    "quality_status",
}
assert set(dataset_summary.columns) == expected_summary_columns
assert set(dataset_summary["dataset"]) == {"Development", "Public Validation"}
assert dataset_summary["quality_status"].eq("pass (verified)").all()

expected_metric_columns = {
    "date",
    "observed_rows",
    "total_rows",
    "coverage",
    "GMV",
    "Orders",
    "AOV",
    "CTR",
    "CVR",
    "Refund Rate",
    "ROAS",
}
assert set(daily_metrics.columns) == expected_metric_columns
assert daily_metrics["coverage"].between(0, 1).all()
missing_days = daily_metrics.loc[daily_metrics["coverage"].lt(1)]
complete_days = daily_metrics.loc[daily_metrics["coverage"].eq(1)]
assert not missing_days.empty
assert missing_days[["CTR", "CVR"]].isna().all().all()
assert complete_days[["CTR", "CVR"]].notna().all().all()
assert coverage_gaps[["CTR", "CVR"]].isna().all().all()

assert set(catalogs) == {"Development"}
development_count = catalogs["Development"].execute(
    "select count(*) from orders"
).fetchone()[0]
assert development_count > 0

public_validation_dataset_id = dataset_summary.loc[
    dataset_summary["dataset"].eq("Public Validation"), "dataset_id"
].item()
queryable_public_validation_handles = []
for variable_name, value in list(globals().items()):
    if not hasattr(value, "verified_summary") or not hasattr(value, "execute"):
        continue
    if value.verified_summary["dataset_id"] != public_validation_dataset_id:
        continue
    try:
        value.execute("select 1").fetchone()
    except ValueError:
        continue
    queryable_public_validation_handles.append(variable_name)
assert queryable_public_validation_handles == []
"""
        )
    )

    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
    ).execute(cwd=execution_cwd)

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", [])
    )
    assert all(cell.get("execution_count") is None for cell in load_notebook().cells)
    assert all(not cell.get("outputs", []) for cell in load_notebook().cells)
