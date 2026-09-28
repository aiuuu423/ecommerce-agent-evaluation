import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from app.data import generator
from app.data.generator import build_snapshot
from app.data.manifest import contained_path, file_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
TABLES = {"products", "customers", "traffic", "marketing", "orders"}


def test_snapshot_writes_parquet_quality_report_and_stable_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v1"

    first = build_snapshot(CONFIG, output)
    second = build_snapshot(CONFIG, output)

    assert first == second
    assert len(first["dataset_id"]) == 16
    assert first["config_sha256"] == sha256(CONFIG.read_bytes()).hexdigest()
    assert first["source_label"] == "Synthetic E-commerce Data"
    assert set(first["tables"]) == TABLES
    for metadata in first["tables"].values():
        table_path = output / metadata["file"]
        assert table_path.is_file()
        assert metadata["rows"] > 0
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = generator.load_data_config_with_sha256
    calls = 0

    def recording_load(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(generator, "load_data_config_with_sha256", recording_load)

    build_snapshot(CONFIG, tmp_path / "v1")

    assert calls == 1


@pytest.mark.parametrize("relative", ["../escape.parquet", "/tmp/escape.parquet"])
def test_output_artifacts_must_be_contained(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="contained"):
        contained_path(tmp_path / "snapshot", relative)


def test_failed_snapshot_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        build_snapshot(CONFIG, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".v1.tmp-*"))


def test_failed_rebuild_preserves_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v1"
    expected = build_snapshot(CONFIG, output)

    def fail_write(*args, **kwargs):
        raise OSError("simulated rebuild failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="simulated"):
        build_snapshot(CONFIG, output)

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == expected
