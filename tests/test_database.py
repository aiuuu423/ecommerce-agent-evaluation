import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from app.data import database
from app.data.database import open_dataset
from app.data.generator import build_snapshot
from app.data.manifest import (
    file_sha256,
    manifest_id,
)

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
        with pytest.raises(PermissionError, match="read-only"):
            connection.execute(f'insert into "{name}" select * from "{name}" limit 1')
    connection.close()


def _refresh_dataset_id(manifest: dict) -> None:
    manifest["dataset_id"] = manifest_id(
        {
            name: {
                "rows": metadata["rows"],
                "logical_sha256": metadata["logical_sha256"],
            }
            for name, metadata in manifest["tables"].items()
        }
    )


def test_open_dataset_recomputes_rows_and_logical_hashes_from_parquet(
    dataset_dir: Path,
) -> None:
    manifest = _manifest(dataset_dir)
    manifest["tables"]["products"]["rows"] += 1
    _refresh_dataset_id(manifest)
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="table row count mismatch: products"):
        open_dataset(dataset_dir)

    manifest = _manifest(dataset_dir)
    manifest["tables"]["products"]["rows"] -= 1
    manifest["tables"]["products"]["logical_sha256"] = "0" * 64
    _refresh_dataset_id(manifest)
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="table logical hash mismatch: products"):
        open_dataset(dataset_dir)


def test_open_dataset_recomputes_dataset_id_from_parquet(dataset_dir: Path) -> None:
    manifest = _manifest(dataset_dir)
    manifest["dataset_id"] = "0" * 16
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="dataset identity mismatch"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_valid_parquet_replacement(dataset_dir: Path) -> None:
    manifest = _manifest(dataset_dir)
    path = dataset_dir / "products.parquet"
    products = pd.read_parquet(path)
    products.loc[0, "product_name"] = "replacement"
    products.to_parquet(path, index=False)
    manifest["tables"]["products"]["sha256"] = file_sha256(path)
    _write_manifest(dataset_dir, manifest)

    with pytest.raises(ValueError, match="table logical hash mismatch: products"):
        open_dataset(dataset_dir)


def test_open_dataset_materializes_verified_files_in_memory(dataset_dir: Path) -> None:
    catalog = open_dataset(dataset_dir)
    expected = catalog.execute("select count(*) from products").fetchone()

    (dataset_dir / "products.parquet").unlink()

    assert catalog.execute("select count(*) from products").fetchone() == expected
    catalog.close()


@pytest.mark.parametrize(
    "statement",
    [
        "create table injected(value integer)",
        "delete from products",
        "copy products to '/tmp/products.csv'",
        "attach ':memory:' as other",
    ],
)
def test_catalog_rejects_non_select_statements(
    dataset_dir: Path,
    statement: str,
) -> None:
    with open_dataset(dataset_dir) as catalog:
        with pytest.raises(PermissionError, match="read-only"):
            catalog.execute(statement)


def test_catalog_rejects_external_scans(dataset_dir: Path, tmp_path: Path) -> None:
    external = tmp_path / "external.csv"
    external.write_text("value\n1\n", encoding="utf-8")

    with open_dataset(dataset_dir) as catalog:
        with pytest.raises(duckdb.PermissionException):
            catalog.execute(f"select * from read_csv('{external}')")


def test_catalog_exposes_read_only_description_for_gold_style_query(
    dataset_dir: Path,
) -> None:
    with open_dataset(dataset_dir) as catalog:
        relation = catalog.execute(
            """
            with product_totals as (
                select product_id, count(*) as order_count
                from orders
                group by product_id
            )
            select product_id, order_count
            from product_totals
            order by product_id
            limit 1
            """
        )

        assert [column[0] for column in relation.description] == [
            "product_id",
            "order_count",
        ]
        assert relation.fetchall()[0][0] == "P001"
        with pytest.raises(AttributeError):
            relation.description = ()
        with pytest.raises(PermissionError, match="read-only"):
            relation.execute("delete from orders")


def test_catalog_context_manager_closes_connection(dataset_dir: Path) -> None:
    with open_dataset(dataset_dir) as catalog:
        assert catalog.execute("select 1").fetchone() == (1,)

    with pytest.raises(ValueError, match="closed"):
        catalog.execute("select 1")


@pytest.mark.parametrize(
    "artifact",
    [
        "manifest.json",
        "data_quality_report.json",
        "products.parquet",
    ],
)
def test_open_dataset_rejects_artifact_symlinks(
    dataset_dir: Path,
    artifact: str,
) -> None:
    path = dataset_dir / artifact
    target = dataset_dir.parent / f"real-{artifact}"
    path.rename(target)
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        open_dataset(dataset_dir)


def test_open_dataset_rejects_dataset_directory_symlink(dataset_dir: Path) -> None:
    link = dataset_dir.parent / "linked"
    link.symlink_to(dataset_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        open_dataset(link)


def test_open_dataset_closes_connection_when_registration_fails(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = duckdb.connect(database=":memory:")
    closed = False

    class FailingConnection:
        def register(self, *args, **kwargs):
            raise duckdb.Error("registration failed")

        def close(self):
            nonlocal closed
            closed = True
            connection.close()

    monkeypatch.setattr(database.duckdb, "connect", lambda *args, **kwargs: FailingConnection())

    with pytest.raises(duckdb.Error, match="registration failed"):
        open_dataset(dataset_dir)

    assert closed


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
