from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from pandas.testing import assert_frame_equal

from app.data.config import load_data_config
from app.data.generator import generate_customers, generate_products
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


def test_v1_product_categories_are_balanced() -> None:
    config = load_data_config(CONFIG)
    category_counts = generate_products(config)["category"].value_counts()

    assert category_counts.max() - category_counts.min() <= 1
    assert category_counts.to_dict() == {
        category: config.product_count // len(config.categories)
        for category in config.categories
    }


def test_product_launch_dates_have_expected_range_and_diversity() -> None:
    config = load_data_config(CONFIG)
    launch_dates = generate_products(config)["launch_date"]

    assert launch_dates.min() >= config.start_date - timedelta(days=719)
    assert launch_dates.max() <= config.start_date - timedelta(days=30)
    assert launch_dates.nunique() > 1


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
