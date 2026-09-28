from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from app.data.config import SyntheticDataConfig
from app.data.schemas import CustomerRow, ProductRow

CENT = Decimal("0.01")


def _rng(config: SyntheticDataConfig, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([config.seed, stream]))


def generate_products(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = _rng(config, 1)
    prices = rng.integers(30, 500, size=config.product_count)
    margin_basis_points = rng.integers(2500, 6001, size=config.product_count)
    rows: list[dict[str, object]] = []

    for index, (price_value, margin_basis_points_value) in enumerate(
        zip(prices, margin_basis_points, strict=True),
        start=1,
    ):
        price = Decimal(int(price_value)).quantize(CENT)
        margin = Decimal(int(margin_basis_points_value)) / Decimal(10_000)
        cost = (price * (Decimal(1) - margin)).quantize(
            CENT, rounding=ROUND_HALF_UP
        )
        row = ProductRow(
            product_id=f"P{index:03d}",
            product_name=f"模拟商品 {index:03d}",
            category=config.categories[(index - 1) % len(config.categories)],
            price=price,
            cost=cost,
            launch_date=config.start_date
            - timedelta(days=int(rng.integers(30, 720))),
        )
        rows.append(row.model_dump())

    return pd.DataFrame(rows)


def generate_customers(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = _rng(config, 2)
    rows: list[dict[str, object]] = []

    for index in range(1, config.customer_count + 1):
        row = CustomerRow(
            customer_id=f"C{index:04d}",
            is_new_customer=bool(rng.random() < 0.28),
            region=config.regions[(index - 1) % len(config.regions)],
            channel=config.channels[int(rng.integers(0, len(config.channels)))],
        )
        rows.append(row.model_dump())

    return pd.DataFrame(rows)
