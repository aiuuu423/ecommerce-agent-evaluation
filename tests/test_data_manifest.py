import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow
import pytest
import yaml

from app.data import generator
from app.data.config import load_data_config
from app.data.generator import build_snapshot
from app.data.manifest import contained_path, dataset_id_for_tables, file_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
TABLES = {"products", "customers", "traffic", "marketing", "orders"}


@pytest.fixture
def small_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload.update(
        {
            "dataset_version": "test-v1",
            "days": 60,
            "product_count": 7,
            "customer_count": 20,
            "anomalies": [],
        }
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _config_variant(config_path: Path, *, seed_delta: int) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["seed"] += seed_delta
    variant = config_path.with_name(f"config-seed-{payload['seed']}.yaml")
    variant.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return variant


def test_snapshot_writes_parquet_quality_report_and_stable_manifest(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"

    first = build_snapshot(small_config, output)
    second = build_snapshot(small_config, output)

    assert first == second
    assert len(first["dataset_id"]) == 16
    assert first["config_sha256"] == sha256(small_config.read_bytes()).hexdigest()
    assert first["source_label"] == "Synthetic E-commerce Data"
    assert first["writer"] == {
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "parquet_engine": "pyarrow",
        "compression": "snappy",
        "index": False,
    }
    assert first["data_quality_report"] == {
        "file": "data_quality_report.json",
        "sha256": file_sha256(output / "data_quality_report.json"),
    }
    assert set(first["tables"]) == TABLES
    for metadata in first["tables"].values():
        table_path = output / metadata["file"]
        assert table_path.is_file()
        assert metadata["rows"] > 0
        assert len(metadata["logical_sha256"]) == 64
        assert metadata["sha256"] == file_sha256(table_path)

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == first
    quality = json.loads(
        (output / "data_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality["status"] == "pass"
    assert quality["row_counts"] == {
        name: metadata["rows"] for name, metadata in first["tables"].items()
    }


def test_snapshot_reads_and_hashes_config_in_one_operation(
    tmp_path: Path, small_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = generator.load_data_config_with_sha256
    calls = 0

    def recording_load(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(generator, "load_data_config_with_sha256", recording_load)

    build_snapshot(small_config, tmp_path / "v1")

    assert calls == 1


@pytest.mark.parametrize("relative", ["../escape.parquet", "/tmp/escape.parquet"])
def test_output_artifacts_must_be_contained(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="contained"):
        contained_path(tmp_path / "snapshot", relative)


def test_failed_snapshot_leaves_no_partial_output(
    tmp_path: Path, small_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v1"
    original = pd.DataFrame.to_parquet
    writes = 0

    def fail_after_first_write(frame, path, *args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated write failure")
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_first_write)

    with pytest.raises(OSError, match="simulated"):
        build_snapshot(small_config, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".v1.tmp-*"))


def test_existing_snapshot_with_same_identity_returns_without_rewriting(
    tmp_path: Path, small_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v1"
    expected = build_snapshot(small_config, output)

    def fail_write(*args, **kwargs):
        raise AssertionError("an immutable snapshot must not be rewritten")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    assert build_snapshot(small_config, output) == expected


def test_existing_snapshot_rejects_tampered_quality_report_hash(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"
    build_snapshot(small_config, output)
    report_path = output / "data_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_label"] = "tampered"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="quality report hash mismatch"):
        build_snapshot(small_config, output)


def test_existing_snapshot_rejects_non_passing_quality_report(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"
    build_snapshot(small_config, output)
    report_path = output / "data_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_quality_report"]["sha256"] = file_sha256(report_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="quality report status is not pass"):
        build_snapshot(small_config, output)


def test_existing_snapshot_with_different_identity_is_rejected(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"
    expected = build_snapshot(small_config, output)
    other_config = _config_variant(small_config, seed_delta=1)

    with pytest.raises(ValueError, match="immutable"):
        build_snapshot(other_config, output)

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == expected
    assert not list(tmp_path.glob(".v1.backup-*"))


def test_dataset_id_is_sensitive_to_normalized_logical_table_content(
    small_config: Path,
) -> None:
    tables = generator.generate_dataset(load_data_config(small_config))
    changed = {name: frame.copy(deep=True) for name, frame in tables.items()}
    changed["products"].loc[0, "product_name"] += " changed"

    assert dataset_id_for_tables(tables) != dataset_id_for_tables(changed)


def test_dataset_id_can_be_recomputed_from_parquet_independently_of_physical_sha(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"
    manifest = build_snapshot(small_config, output)
    tables = {
        name: pd.read_parquet(output / metadata["file"])
        for name, metadata in manifest["tables"].items()
    }

    assert dataset_id_for_tables(tables) == manifest["dataset_id"]

    products_path = output / manifest["tables"]["products"]["file"]
    original_file_sha = file_sha256(products_path)
    tables["products"].to_parquet(
        products_path,
        index=False,
        engine="pyarrow",
        compression="gzip",
    )

    assert file_sha256(products_path) != original_file_sha
    assert dataset_id_for_tables(
        {
            name: pd.read_parquet(output / metadata["file"])
            for name, metadata in manifest["tables"].items()
        }
    ) == manifest["dataset_id"]


def test_concurrent_builds_publish_one_complete_snapshot(
    tmp_path: Path, small_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v1"
    barrier = threading.Barrier(2)
    original_generate = generator.generate_dataset

    def synchronized_generate(config):
        tables = original_generate(config)
        barrier.wait(timeout=10)
        return tables

    monkeypatch.setattr(generator, "generate_dataset", synchronized_generate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: build_snapshot(small_config, output), range(2))
        )

    assert results[0] == results[1]
    assert set(path.name for path in output.iterdir()) == {
        "products.parquet",
        "customers.parquet",
        "traffic.parquet",
        "marketing.parquet",
        "orders.parquet",
        "manifest.json",
        "data_quality_report.json",
    }
    assert not list(tmp_path.glob(".v1.tmp-*"))


def test_readers_only_observe_absent_or_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".v1.tmp-test"
    output = tmp_path / "v1"
    staging.mkdir()
    expected_names = {"manifest.json", "products.parquet"}
    for name in expected_names:
        (staging / name).write_text(name, encoding="utf-8")

    observations: list[set[str] | None] = []
    stop = threading.Event()

    def read_repeatedly() -> None:
        while not stop.is_set():
            try:
                observations.append({path.name for path in output.iterdir()})
            except FileNotFoundError:
                observations.append(None)

    original_rename = os.rename

    def delayed_rename(source, target):
        time.sleep(0.02)
        original_rename(source, target)
        time.sleep(0.02)

    monkeypatch.setattr(generator.os, "rename", delayed_rename)
    reader = threading.Thread(target=read_repeatedly)
    reader.start()
    try:
        generator._publish_snapshot(staging, output)
    finally:
        stop.set()
        reader.join(timeout=5)

    assert None in observations
    assert expected_names in observations
    assert all(
        observation is None or observation == expected_names
        for observation in observations
    )


def test_cli_build_is_idempotent_and_rejects_different_identity(
    tmp_path: Path, small_config: Path
) -> None:
    output = tmp_path / "v1"
    command = [
        sys.executable,
        "-m",
        "app.data.generator",
        "--config",
        str(small_config),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    different = subprocess.run(
        [
            *command[:4],
            str(_config_variant(small_config, seed_delta=1)),
            *command[5:],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert first.returncode == 0
    assert second.returncode == 0
    assert manifest["dataset_id"] in first.stdout
    assert manifest["dataset_id"] in second.stdout
    assert different.returncode != 0
    assert "immutable" in different.stderr
