import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_data_exploration.ipynb"


def load_notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


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
    assert "Holdout" in markdown


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


def test_notebook_uses_both_snapshots_with_robust_project_discovery() -> None:
    notebook = load_notebook()
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    content = "\n".join(cell.source for cell in notebook.cells)

    assert "Path.cwd()" in code
    assert "pyproject.toml" in code
    assert "data/synthetic/v1" in code
    assert "data/synthetic/holdout-v1" in code
    assert "make phase1-data" in content


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
