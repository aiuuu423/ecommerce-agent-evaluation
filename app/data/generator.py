import argparse
import fcntl
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pyarrow

from app.data.config import SyntheticDataConfig, load_data_config_with_sha256
from app.data.manifest import (
    TABLE_NAMES,
    contained_path,
    dataset_id_for_tables,
    file_sha256,
    json_sha256,
    logical_table_sha256,
    validate_manifest,
    write_json,
)
from app.data.schemas import (
    CustomerRow,
    MarketingRow,
    OrderRow,
    ProductRow,
    TrafficRow,
)
from app.data.validation import validate_dataset

CENT = Decimal("0.01")
PRICE_MIN = 30
PRICE_MAX_EXCLUSIVE = 500
MARGIN_BASIS_POINTS_MIN = 2500
MARGIN_BASIS_POINTS_MAX_EXCLUSIVE = 6001
BASIS_POINTS_PER_UNIT = Decimal(10_000)
LAUNCH_AGE_DAYS_MIN = 30
LAUNCH_AGE_DAYS_MAX_EXCLUSIVE = 720
NEW_CUSTOMER_PROBABILITY = 0.28
MAX_CLICK_THROUGH_RATE = 0.22
MAX_CONVERSION_RATE = 0.25
MAX_REFUND_PROBABILITY = 0.65
MAX_ORDER_SEQUENCE = 999_999
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"
DimensionValue = TypeVar("DimensionValue")


def _rng(config: SyntheticDataConfig, stream_name: str) -> np.random.Generator:
    digest = sha256(stream_name.encode("utf-8")).digest()
    stream_entropy = [
        int.from_bytes(digest[offset : offset + 4], "big")
        for offset in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence([config.seed, *stream_entropy]))


def _shuffled_balanced_values(
    values: Sequence[DimensionValue],
    count: int,
    rng: np.random.Generator,
) -> list[DimensionValue]:
    balanced = [values[index % len(values)] for index in range(count)]
    rng.shuffle(balanced)
    return balanced


def _launch_date(start_date: date, launch_age_days: int) -> date:
    return start_date - timedelta(days=launch_age_days)


def generate_products(config: SyntheticDataConfig) -> pd.DataFrame:
    prices = _rng(config, "products.price").integers(
        PRICE_MIN,
        PRICE_MAX_EXCLUSIVE,
        size=config.product_count,
    )
    margin_basis_points = _rng(config, "products.margin").integers(
        MARGIN_BASIS_POINTS_MIN,
        MARGIN_BASIS_POINTS_MAX_EXCLUSIVE,
        size=config.product_count,
    )
    launch_ages = _rng(config, "products.launch_date").integers(
        LAUNCH_AGE_DAYS_MIN,
        LAUNCH_AGE_DAYS_MAX_EXCLUSIVE,
        size=config.product_count,
    )
    categories = _shuffled_balanced_values(
        config.categories,
        config.product_count,
        _rng(config, "products.category"),
    )
    rows: list[dict[str, object]] = []

    for index, (price_value, margin_basis_points_value, launch_age, category) in enumerate(
        zip(prices, margin_basis_points, launch_ages, categories, strict=True),
        start=1,
    ):
        price = Decimal(int(price_value)).quantize(CENT)
        margin = Decimal(int(margin_basis_points_value)) / BASIS_POINTS_PER_UNIT
        cost = (price * (Decimal(1) - margin)).quantize(
            CENT, rounding=ROUND_HALF_UP
        )
        row = ProductRow(
            product_id=f"P{index:03d}",
            product_name=f"模拟商品 {index:03d}",
            category=category,
            price=price,
            cost=cost,
            launch_date=_launch_date(config.start_date, int(launch_age)),
        )
        rows.append(row.model_dump())

    return pd.DataFrame(rows)


def generate_customers(config: SyntheticDataConfig) -> pd.DataFrame:
    new_customer_draws = _rng(config, "customers.is_new_customer").random(
        config.customer_count
    )
    regions = _shuffled_balanced_values(
        config.regions,
        config.customer_count,
        _rng(config, "customers.region"),
    )
    channels = _shuffled_balanced_values(
        config.channels,
        config.customer_count,
        _rng(config, "customers.channel"),
    )
    rows: list[dict[str, object]] = []

    for index, (new_customer_draw, region, channel) in enumerate(
        zip(new_customer_draws, regions, channels, strict=True),
        start=1,
    ):
        row = CustomerRow(
            customer_id=f"C{index:04d}",
            is_new_customer=bool(new_customer_draw < NEW_CUSTOMER_PROBABILITY),
            region=region,
            channel=channel,
        )
        rows.append(row.model_dump())

    return pd.DataFrame(rows)


def _anomaly_effects(
    config: SyntheticDataConfig,
    product_id: str,
    day_index: int,
) -> Mapping[str, float | bool]:
    effects: dict[str, float | bool] = {
        "traffic_multiplier": 1.0,
        "conversion_multiplier": 1.0,
        "refund_multiplier": 1.0,
        "missing_traffic": False,
    }
    for anomaly in config.anomalies:
        if (
            anomaly.product_id != product_id
            or day_index < anomaly.start_day
            or day_index > anomaly.end_day
        ):
            continue
        if anomaly.kind in {"traffic_drop", "extreme_traffic_spike"}:
            effects["traffic_multiplier"] *= anomaly.multiplier
        elif anomaly.kind in {"sales_drop", "conversion_drop"}:
            effects["conversion_multiplier"] *= anomaly.multiplier
        elif anomaly.kind == "high_refund":
            effects["refund_multiplier"] *= anomaly.multiplier
        elif anomaly.kind == "missing_traffic":
            effects["missing_traffic"] = True
        elif anomaly.kind == "multi_factor_drop":
            effects["traffic_multiplier"] *= anomaly.multiplier
            effects["conversion_multiplier"] *= anomaly.multiplier
    return effects


def _fact_rng(
    config: SyntheticDataConfig,
    field_name: str,
    current_date: date,
    product_id: str,
    order_sequence: int | None = None,
) -> np.random.Generator:
    coordinates = f"{field_name}|{current_date.isoformat()}|{product_id}"
    if order_sequence is not None:
        coordinates = f"{coordinates}|{order_sequence}"
    return _rng(config, coordinates)


def _order_id(current_date: date, product_id: str, order_sequence: int) -> str:
    if not 1 <= order_sequence <= MAX_ORDER_SEQUENCE:
        raise ValueError(
            f"order sequence must be between 1 and {MAX_ORDER_SEQUENCE}"
        )
    return f"O{current_date:%Y%m%d}-{product_id}-{order_sequence:06d}"


def _generate_traffic_values(
    config: SyntheticDataConfig,
    current_date: date,
    product_id: str,
    day_index: int,
    effects: Mapping[str, float | bool],
) -> tuple[int, int, int]:
    product_number = int(product_id[1:])
    weekly_factor = 1.12 if current_date.weekday() >= 5 else 1.0
    trend_factor = 1.0 + day_index * 0.0008
    base_impressions = 850 + product_number * 12
    impressions = int(
        _fact_rng(config, "traffic.impressions", current_date, product_id).poisson(
            base_impressions
            * weekly_factor
            * trend_factor
            * float(effects["traffic_multiplier"])
        )
    )
    click_through_rate = min(
        MAX_CLICK_THROUGH_RATE,
        0.07 + (product_number % 5) * 0.01,
    )
    clicks = int(
        _fact_rng(config, "traffic.clicks", current_date, product_id).binomial(
            impressions, click_through_rate
        )
    )
    visits = clicks + int(
        _fact_rng(config, "traffic.visits", current_date, product_id).poisson(
            max(4, clicks * 0.12)
        )
    )
    return impressions, clicks, visits


def _generate_traffic_row(
    current_date: date,
    product_id: str,
    values: tuple[int, int, int],
    *,
    is_missing: bool,
) -> dict[str, object]:
    impressions, clicks, visits = values
    row = TrafficRow(
        date=current_date,
        product_id=product_id,
        impressions=None if is_missing else impressions,
        clicks=None if is_missing else clicks,
        visits=None if is_missing else visits,
        is_missing=is_missing,
    )
    return row.model_dump()


def _generate_marketing_row(
    current_date: date,
    product_id: str,
    day_index: int,
    impressions: int,
) -> dict[str, object]:
    product_number = int(product_id[1:])
    spend_rate = Decimal("0.012") + Decimal(day_index % 7) * Decimal("0.0005")
    row = MarketingRow(
        date=current_date,
        product_id=product_id,
        campaign_id=f"M{product_number:03d}",
        spend=(Decimal(impressions) * spend_rate).quantize(
            CENT, rounding=ROUND_HALF_UP
        ),
    )
    return row.model_dump()


def _generate_order_rows(
    config: SyntheticDataConfig,
    current_date: date,
    product_id: str,
    unit_price: Decimal,
    visits: int,
    effects: Mapping[str, float | bool],
) -> list[dict[str, object]]:
    product_number = int(product_id[1:])
    conversion_rate = min(
        MAX_CONVERSION_RATE,
        (0.035 + (product_number % 4) * 0.008)
        * float(effects["conversion_multiplier"]),
    )
    order_count = int(
        _fact_rng(config, "orders.count", current_date, product_id).binomial(
            visits, conversion_rate
        )
    )
    if order_count > MAX_ORDER_SEQUENCE:
        raise ValueError("generated order count exceeds order ID capacity")

    refund_probability = min(
        MAX_REFUND_PROBABILITY,
        (0.035 + (product_number % 3) * 0.01)
        * float(effects["refund_multiplier"]),
    )
    rows: list[dict[str, object]] = []
    for order_sequence in range(1, order_count + 1):
        customer_number = int(
            _fact_rng(
                config,
                "orders.customer",
                current_date,
                product_id,
                order_sequence,
            ).integers(1, config.customer_count + 1)
        )
        quantity = int(
            _fact_rng(
                config,
                "orders.quantity",
                current_date,
                product_id,
                order_sequence,
            ).choice([1, 1, 1, 2, 2, 3])
        )
        is_refund = bool(
            _fact_rng(
                config,
                "orders.refund",
                current_date,
                product_id,
                order_sequence,
            ).random()
            < refund_probability
        )
        row = OrderRow(
            order_id=_order_id(current_date, product_id, order_sequence),
            product_id=product_id,
            customer_id=f"C{customer_number:04d}",
            order_date=current_date,
            quantity=quantity,
            unit_price=unit_price,
            revenue=(unit_price * quantity).quantize(CENT, rounding=ROUND_HALF_UP),
            is_refund=is_refund,
            status="refunded" if is_refund else "paid",
        )
        rows.append(row.model_dump())
    return rows


def _generate_product_day(
    config: SyntheticDataConfig,
    current_date: date,
    day_index: int,
    product_id: str,
    unit_price: Decimal,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    effects = _anomaly_effects(config, product_id, day_index)
    traffic_values = _generate_traffic_values(
        config, current_date, product_id, day_index, effects
    )
    traffic_row = _generate_traffic_row(
        current_date,
        product_id,
        traffic_values,
        is_missing=bool(effects["missing_traffic"]),
    )
    marketing_row = _generate_marketing_row(
        current_date, product_id, day_index, traffic_values[0]
    )
    order_rows = _generate_order_rows(
        config,
        current_date,
        product_id,
        unit_price,
        traffic_values[2],
        effects,
    )
    return traffic_row, marketing_row, order_rows


def generate_dataset(config: SyntheticDataConfig) -> dict[str, pd.DataFrame]:
    products = generate_products(config)
    customers = generate_customers(config)
    traffic_rows: list[dict[str, object]] = []
    marketing_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []

    for day_index in range(config.days):
        current_date = config.start_date + timedelta(days=day_index)
        for product in products.itertuples(index=False):
            traffic_row, marketing_row, daily_orders = _generate_product_day(
                config,
                current_date,
                day_index,
                product.product_id,
                product.price,
            )
            traffic_rows.append(traffic_row)
            marketing_rows.append(marketing_row)
            order_rows.extend(daily_orders)

    traffic = pd.DataFrame(traffic_rows)
    nullable_metrics = ["impressions", "clicks", "visits"]
    traffic[nullable_metrics] = traffic[nullable_metrics].astype(object).where(
        traffic[nullable_metrics].notna(),
        None,
    )
    return {
        "products": products,
        "customers": customers,
        "traffic": traffic,
        "marketing": pd.DataFrame(marketing_rows),
        "orders": pd.DataFrame(order_rows),
    }


def _publish_snapshot(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"snapshot output already exists: {output_dir}")
    os.rename(staging_dir, output_dir)


@contextmanager
def _snapshot_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _logical_table_manifest(
    tables: dict[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "file": f"{name}.parquet",
            "rows": len(tables[name]),
            "logical_sha256": logical_table_sha256(tables[name]),
        }
        for name in TABLE_NAMES
    }


def _load_existing_snapshot(
    output: Path,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"snapshot output is not an immutable directory: {output}")

    manifest_path = contained_path(output, "manifest.json")
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        existing_dataset_id = manifest["dataset_id"]
        table_manifest = manifest["tables"]
        quality_metadata = manifest["data_quality_report"]
        quality_path = contained_path(output, quality_metadata["file"])
        expected_quality_sha = quality_metadata["sha256"]
        existing_metadata = {
            key: manifest[key]
            for key in (
                "dataset_id",
                "config_sha256",
                "dataset_version",
                "schema_version",
                "seed",
                "source_label",
                "writer",
            )
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing snapshot manifest is invalid: {output}") from exc

    if not isinstance(table_manifest, dict) or set(table_manifest) != set(TABLE_NAMES):
        raise ValueError(f"existing snapshot manifest is invalid: {output}")
    expected_metadata = {
        key: expected_identity[key]
        for key in (
            "dataset_id",
            "config_sha256",
            "dataset_version",
            "schema_version",
            "seed",
            "source_label",
            "writer",
        )
    }
    if existing_metadata != expected_metadata:
        raise ValueError(
            "snapshot version directory is immutable and contains a different "
            f"dataset identity: {output}"
        )
    expected_tables = expected_identity["tables"]
    if not isinstance(expected_tables, dict):
        raise TypeError("expected snapshot tables must be a mapping")
    for name in TABLE_NAMES:
        try:
            metadata = table_manifest[name]
            table_path = contained_path(output, metadata["file"])
            expected_file_sha = metadata["sha256"]
            existing_logical_metadata = {
                "file": metadata["file"],
                "rows": metadata["rows"],
                "logical_sha256": metadata["logical_sha256"],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"existing snapshot manifest is invalid: {output}") from exc
        if not table_path.is_file() or file_sha256(table_path) != expected_file_sha:
            raise ValueError(f"existing snapshot physical file hash mismatch: {table_path.name}")
        if existing_logical_metadata != expected_tables[name]:
            raise ValueError(
                "snapshot version directory is immutable and contains a different "
                f"dataset identity: {output}"
            )

    if not quality_path.is_file() or file_sha256(quality_path) != expected_quality_sha:
        raise ValueError(f"existing snapshot quality report hash mismatch: {quality_path.name}")
    try:
        quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing snapshot quality report is invalid: {output}") from exc
    if not isinstance(quality_report, dict) or quality_report.get("status") != "pass":
        raise ValueError(f"existing snapshot quality report status is not pass: {output}")
    if quality_metadata != expected_identity["data_quality_report"]:
        raise ValueError(
            "snapshot version directory is immutable and contains a different "
            f"dataset identity: {output}"
        )

    existing_tables = {
        name: pd.read_parquet(contained_path(output, table_manifest[name]["file"]))
        for name in TABLE_NAMES
    }
    if dataset_id_for_tables(existing_tables) != existing_dataset_id:
        raise ValueError(f"existing snapshot logical identity mismatch: {output}")
    return manifest


def build_snapshot(
    config_path: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, object]:
    config, config_digest = load_data_config_with_sha256(config_path)
    tables = generate_dataset(config)
    quality = validate_dataset(tables, config)
    if quality["status"] != "pass":
        raise ValueError(f"dataset quality failed: {quality['failed_checks']}")

    raw_output = (
        Path(output_dir)
        if output_dir is not None
        else Path("data/synthetic") / config.dataset_version
    )
    try:
        raw_output.lstat()
    except FileNotFoundError:
        pass
    else:
        if raw_output.is_symlink():
            raise ValueError(f"snapshot output must not be a symbolic link: {raw_output}")
    output = raw_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset_id_for_tables(tables)
    logical_tables = _logical_table_manifest(tables)
    writer = {
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "parquet_engine": PARQUET_ENGINE,
        "compression": PARQUET_COMPRESSION,
        "index": False,
    }
    quality_metadata = {
        "file": "data_quality_report.json",
        "sha256": json_sha256(quality),
    }
    expected_identity: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_version": config.dataset_version,
        "schema_version": config.schema_version,
        "source_label": config.source_label,
        "seed": config.seed,
        "config_sha256": config_digest,
        "writer": writer,
        "tables": logical_tables,
        "data_quality_report": quality_metadata,
    }

    with _snapshot_lock(output):
        if output.exists() or output.is_symlink():
            return _load_existing_snapshot(output, expected_identity)

        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        ).resolve()
        try:
            table_manifest: dict[str, dict[str, object]] = {}
            for name in TABLE_NAMES:
                table_path = contained_path(staging, f"{name}.parquet")
                tables[name].to_parquet(
                    table_path,
                    index=False,
                    engine=PARQUET_ENGINE,
                    compression=PARQUET_COMPRESSION,
                )
                table_manifest[name] = {
                    **logical_tables[name],
                    "sha256": file_sha256(table_path),
                }

            quality_path = contained_path(staging, "data_quality_report.json")
            write_json(quality_path, quality)
            manifest: dict[str, object] = {
                "dataset_id": dataset_id,
                "dataset_version": config.dataset_version,
                "schema_version": config.schema_version,
                "source_label": config.source_label,
                "seed": config.seed,
                "config_sha256": config_digest,
                "writer": writer,
                "tables": table_manifest,
                "data_quality_report": quality_metadata,
            }
            validate_manifest(manifest)
            write_json(contained_path(staging, "manifest.json"), manifest)
            _publish_snapshot(staging, output)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 data snapshot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = build_snapshot(args.config, args.output)
    print(f"Built {manifest['source_label']} snapshot {manifest['dataset_id']}")


if __name__ == "__main__":
    main()
