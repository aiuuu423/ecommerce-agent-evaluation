from collections.abc import Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from typing import TypeVar

import numpy as np
import pandas as pd

from app.data.config import SyntheticDataConfig
from app.data.schemas import CustomerRow, ProductRow

CENT = Decimal("0.01")
PRICE_MIN = 30
PRICE_MAX_EXCLUSIVE = 500
MARGIN_BASIS_POINTS_MIN = 2500
MARGIN_BASIS_POINTS_MAX_EXCLUSIVE = 6001
BASIS_POINTS_PER_UNIT = Decimal(10_000)
LAUNCH_AGE_DAYS_MIN = 30
LAUNCH_AGE_DAYS_MAX_EXCLUSIVE = 720
NEW_CUSTOMER_PROBABILITY = 0.28
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
