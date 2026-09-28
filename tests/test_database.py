import json
import shutil
from pathlib import Path

import duckdb
import pytest

from app.data import database
from app.data.database import open_dataset
from app.data.generator import build_snapshot
from app.data.manifest import file_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
TABLES = {"products", "customers", "traffic", "marketing", "orders"}


@pytest.fixture(scope="session")
def snapshot_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return_path = tmp_path_factory.mktemp("database") / "template"
    build_snapshot(CONFIG, return_path)
    return return_path


@pytest.fixture
def dataset_dir(tmp_path: Path, snapshot_template: Path) -> Path:
    return shutil.copytree(snapshot_template, tmp_path / "v1")


def _manifest(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(dataset_dir: Path, manifest: dict) -> None:
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_quality(dataset_dir: Path, quality: dict) -> None:
    quality_path = dataset_dir / "data_quality_report.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    manifest = _manifest(dataset_dir)
    manifest["data_quality_report"]["sha256"] = file_sha256(quality_path)
    _write_manifest(dataset_dir, manifest)


def test_open_dataset_registers_exactly_five_read_only_views(
    dataset_dir: Path,
) -> None:
    manifest = _manifest(dataset_dir)

    connection = open_dataset(dataset_dir)
    catalog = sorted(
        connection.execute(
            """
            select table_name, table_type
            from information_schema.tables
            where table_schema = 'main'
            """
        ).fetchall()
    )

    assert catalog == sorted((name, "VIEW") for name in TABLES)
    for name in TABLES:
        count = connection.execute(f'select count(*) from "{name}"').fetchone()
        assert count == (manifest["tables"][name]["rows"],)
        with pytest.raises(duckdb.Error):
            connection.execute(f'insert into "{name}" select * from "{name}" limit 1')


@pytest.mark.parametrize("table_name", sorted(TABLES))
def test_open_dataset_rejects_each_tampered_parquet(
    dataset_dir: Path,
    table_name: str,
) -> None:
    with (dataset_dir / f"{table_name}.parquet").open("ab") as target:
        target.write(b"tampered")

    with pytest.raises(ValueError, match=rf"table hash mismatch: {table_name}"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_missing_table_file(dataset_dir: Path) -> None:
    (dataset_dir / "orders.parquet").unlink()

    with pytest.raises(ValueError, match="missing table file: orders"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_missing_manifest_table(dataset_dir: Path) -> None:
    manifest = _manifest(dataset_dir)
    del manifest["tables"]["orders"]
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="exactly the five required tables"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_extra_manifest_table(dataset_dir: Path) -> None:
    manifest = _manifest(dataset_dir)
    manifest["tables"]["bonus"] = manifest["tables"]["products"]
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="exactly the five required tables"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_extra_parquet_file(dataset_dir: Path) -> None:
    shutil.copyfile(
        dataset_dir / "products.parquet",
        dataset_dir / "bonus.parquet",
    )

    with pytest.raises(ValueError, match="unexpected parquet files"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_unsafe_table_identifier(dataset_dir: Path) -> None:
    manifest = _manifest(dataset_dir)
    manifest["tables"]["products; drop view orders"] = manifest["tables"].pop(
        "products"
    )
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="unsafe table identifier"):
        open_dataset(dataset_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.pop("dataset_id"), "manifest is invalid"),
        (
            lambda manifest: manifest.__setitem__("dataset_id", "not-a-digest"),
            "manifest is invalid",
        ),
        (
            lambda manifest: manifest.__setitem__("dataset_id", "0" * 16),
            "manifest dataset identity mismatch",
        ),
        (
            lambda manifest: manifest["tables"]["products"].__setitem__(
                "rows", "40"
            ),
            "table metadata is invalid: products",
        ),
        (
            lambda manifest: manifest["tables"]["products"].__setitem__(
                "file", "../products.parquet"
            ),
            "table metadata is invalid: products",
        ),
    ],
)
def test_open_dataset_rejects_invalid_manifest(
    dataset_dir: Path,
    mutation,
    message: str,
) -> None:
    manifest = _manifest(dataset_dir)
    mutation(manifest)
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match=message):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_malformed_manifest_json(dataset_dir: Path) -> None:
    (dataset_dir / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is invalid"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_missing_quality_report(dataset_dir: Path) -> None:
    (dataset_dir / "data_quality_report.json").unlink()

    with pytest.raises(ValueError, match="missing quality report"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_tampered_quality_report(dataset_dir: Path) -> None:
    quality_path = dataset_dir / "data_quality_report.json"
    quality_path.write_text('{"status":"pass"}', encoding="utf-8")

    with pytest.raises(ValueError, match="quality report hash mismatch"):
        open_dataset(dataset_dir)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda quality: quality.__setitem__("status", "fail"),
        lambda quality: quality["row_counts"].__setitem__("products", 41),
        lambda quality: quality.__setitem__("source_label", "Untrusted Data"),
        lambda quality: quality["failed_checks"].append("tampered"),
    ],
)
def test_open_dataset_rejects_invalid_quality_report(
    dataset_dir: Path,
    mutation,
) -> None:
    quality = json.loads(
        (dataset_dir / "data_quality_report.json").read_text(encoding="utf-8")
    )
    mutation(quality)
    _write_quality(dataset_dir, quality)

    with pytest.raises(ValueError, match="quality report is invalid"):
        open_dataset(dataset_dir)


def test_validation_finishes_before_duckdb_connection(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = json.loads(
        (dataset_dir / "data_quality_report.json").read_text(encoding="utf-8")
    )
    quality["status"] = "fail"
    _write_quality(dataset_dir, quality)

    def unexpected_connect(*args, **kwargs):
        raise AssertionError("DuckDB must not open before validation completes")

    monkeypatch.setattr(database.duckdb, "connect", unexpected_connect)

    with pytest.raises(ValueError, match="quality report is invalid"):
        open_dataset(dataset_dir)
