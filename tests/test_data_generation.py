from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from app.data import generator
from app.data.config import load_data_config
from app.data.generator import (
    _launch_date,
    generate_customers,
    generate_products,
)
from app.data.schemas import CustomerRow, ProductRow

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


def test_dimension_generation_is_deterministic() -> None:
    config = load_data_config(CONFIG)

    assert_frame_equal(generate_products(config), generate_products(config))
    assert_frame_equal(generate_customers(config), generate_customers(config))


def test_different_seeds_change_random_dimension_fields() -> None:
    config = load_data_config(CONFIG)
    other_config = config.model_copy(update={"seed": config.seed + 1})

    products = generate_products(config)
    other_products = generate_products(other_config)
    customers = generate_customers(config)
    other_customers = generate_customers(other_config)

    for column in ("category", "price", "cost", "launch_date"):
        assert not products[column].equals(other_products[column])
    for column in ("is_new_customer", "region", "channel"):
        assert not customers[column].equals(other_customers[column])


def test_dimension_generation_is_independent_of_call_order() -> None:
    config = load_data_config(CONFIG)

    products_first = generate_products(config)
    customers_second = generate_customers(config)
    customers_first = generate_customers(config)
    products_second = generate_products(config)

    assert_frame_equal(products_first, products_second)
    assert_frame_equal(customers_first, customers_second)


@pytest.mark.parametrize(
    ("stream_name", "generated_table", "affected_columns"),
    [
        ("products.price", "products", {"price", "cost"}),
        ("products.margin", "products", {"cost"}),
        ("products.launch_date", "products", {"launch_date"}),
        ("products.category", "products", {"category"}),
        ("customers.is_new_customer", "customers", {"is_new_customer"}),
        ("customers.region", "customers", {"region"}),
        ("customers.channel", "customers", {"channel"}),
    ],
)
def test_random_fields_use_isolated_substreams(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    generated_table: str,
    affected_columns: set[str],
) -> None:
    config = load_data_config(CONFIG)
    generate = generate_products if generated_table == "products" else generate_customers
    baseline = generate(config)
    original_rng = generator._rng

    def rng_with_one_changed_stream(current_config, current_stream_name):
        if current_stream_name == stream_name:
            changed_config = config.model_copy(update={"seed": config.seed + 1})
            return original_rng(changed_config, current_stream_name)
        return original_rng(current_config, current_stream_name)

    monkeypatch.setattr(generator, "_rng", rng_with_one_changed_stream)
    changed = generate(config)
    unaffected_columns = [
        column for column in baseline.columns if column not in affected_columns
    ]

    assert_frame_equal(baseline[unaffected_columns], changed[unaffected_columns])
    assert any(
        not baseline[column].equals(changed[column]) for column in affected_columns
    )


def test_dimension_ids_are_sequential_unique_and_counts_match() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    assert len(products) == config.product_count
    assert len(customers) == config.customer_count
    assert products["product_id"].tolist() == [
        f"P{number:03d}" for number in range(1, config.product_count + 1)
    ]
    assert customers["customer_id"].tolist() == [
        f"C{number:04d}" for number in range(1, config.customer_count + 1)
    ]
    assert products["product_id"].is_unique
    assert customers["customer_id"].is_unique


def test_dimensions_cover_configured_values_without_id_periodicity() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    assert set(products["category"]) == set(config.categories)
    assert set(customers["region"]) == set(config.regions)
    assert set(customers["channel"]) == set(config.channels)
    assert products["category"].tolist() != (
        config.categories * config.product_count
    )[: config.product_count]
    assert customers["region"].tolist() != (
        config.regions * config.customer_count
    )[: config.customer_count]
    assert customers["channel"].tolist() != (
        config.channels * config.customer_count
    )[: config.customer_count]


def test_v1_dimensions_are_balanced() -> None:
    config = load_data_config(CONFIG)
    category_counts = generate_products(config)["category"].value_counts()
    customers = generate_customers(config)
    region_counts = customers["region"].value_counts()
    channel_counts = customers["channel"].value_counts()

    assert category_counts.max() - category_counts.min() <= 1
    assert region_counts.max() - region_counts.min() <= 1
    assert channel_counts.max() - channel_counts.min() <= 1


def _has_fixed_period(values: list[str], period: int) -> bool:
    return all(value == values[index % period] for index, value in enumerate(values))


def test_balanced_dimensions_are_not_fixed_cycles_across_seeds() -> None:
    config = load_data_config(CONFIG)

    for seed in range(10):
        seeded_config = config.model_copy(update={"seed": seed})
        products = generate_products(seeded_config)
        customers = generate_customers(seeded_config)

        assert not _has_fixed_period(
            products["category"].tolist(), len(config.categories)
        )
        assert not _has_fixed_period(customers["region"].tolist(), len(config.regions))
        assert not _has_fixed_period(
            customers["channel"].tolist(), len(config.channels)
        )


def test_product_launch_dates_have_expected_range_and_diversity() -> None:
    config = load_data_config(CONFIG)
    launch_dates = generate_products(config)["launch_date"]

    assert launch_dates.min() >= config.start_date - timedelta(days=719)
    assert launch_dates.max() <= config.start_date - timedelta(days=30)
    assert launch_dates.nunique() > 1


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [
        (30, date(2025, 12, 2)),
        (719, date(2024, 1, 13)),
    ],
)
def test_launch_date_pure_function_includes_age_boundaries(
    age_days: int, expected: date
) -> None:
    assert _launch_date(date(2026, 1, 1), age_days) == expected


def test_generated_dimensions_reuse_schema_and_config_contracts() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    validated_products = [
        ProductRow.model_validate(row) for row in products.to_dict(orient="records")
    ]
    for row in customers.to_dict(orient="records"):
        CustomerRow.model_validate(row)

    assert all(product.cost < product.price for product in validated_products)
    assert all(
        product.price.as_tuple().exponent >= -2
        and product.cost.as_tuple().exponent >= -2
        for product in validated_products
    )
    assert set(products["category"]) <= set(config.categories)
    assert set(customers["region"]) <= set(config.regions)
    assert set(customers["channel"]) <= set(config.channels)
    assert all(
        isinstance(value, Decimal) for value in products[["price", "cost"]].stack()
    )
