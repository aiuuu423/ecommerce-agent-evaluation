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
