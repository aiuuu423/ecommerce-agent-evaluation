from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from typing import TypeVar

import numpy as np
import pandas as pd

from app.data.config import SyntheticDataConfig
from app.data.schemas import (
    CustomerRow,
    MarketingRow,
    OrderRow,
    ProductRow,
    TrafficRow,
)

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


def generate_dataset(config: SyntheticDataConfig) -> dict[str, pd.DataFrame]:
    products = generate_products(config)
    customers = generate_customers(config)
    impressions_rng = _rng(config, "traffic.impressions")
    clicks_rng = _rng(config, "traffic.clicks")
    visits_rng = _rng(config, "traffic.visits")
    order_count_rng = _rng(config, "orders.count")
    customer_rng = _rng(config, "orders.customer")
    quantity_rng = _rng(config, "orders.quantity")
    refund_rng = _rng(config, "orders.refund")
    traffic_rows: list[dict[str, object]] = []
    marketing_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    order_number = 1

    for day_index in range(config.days):
        current_date = config.start_date + timedelta(days=day_index)
        weekly_factor = 1.12 if current_date.weekday() >= 5 else 1.0
        trend_factor = 1.0 + day_index * 0.0008

        for product in products.itertuples(index=False):
            product_number = int(product.product_id[1:])
            effects = _anomaly_effects(config, product.product_id, day_index)
            base_impressions = 850 + product_number * 12
            impressions = int(
                impressions_rng.poisson(
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
            clicks = int(clicks_rng.binomial(impressions, click_through_rate))
            visits = clicks + int(visits_rng.poisson(max(4, clicks * 0.12)))
            conversion_rate = min(
                MAX_CONVERSION_RATE,
                (0.035 + (product_number % 4) * 0.008)
                * float(effects["conversion_multiplier"]),
            )

            if effects["missing_traffic"]:
                traffic_row = TrafficRow(
                    date=current_date,
                    product_id=product.product_id,
                    impressions=None,
                    clicks=None,
                    visits=None,
                    is_missing=True,
                )
            else:
                traffic_row = TrafficRow(
                    date=current_date,
                    product_id=product.product_id,
                    impressions=impressions,
                    clicks=clicks,
                    visits=visits,
                    is_missing=False,
                )
            traffic_rows.append(traffic_row.model_dump())

            spend_rate = Decimal("0.012") + Decimal(day_index % 7) * Decimal(
                "0.0005"
            )
            marketing_row = MarketingRow(
                date=current_date,
                product_id=product.product_id,
                campaign_id=f"M{product_number:03d}",
                spend=(Decimal(impressions) * spend_rate).quantize(
                    CENT, rounding=ROUND_HALF_UP
                ),
            )
            marketing_rows.append(marketing_row.model_dump())

            order_count = int(order_count_rng.binomial(visits, conversion_rate))
            refund_probability = min(
                MAX_REFUND_PROBABILITY,
                (0.035 + (product_number % 3) * 0.01)
                * float(effects["refund_multiplier"]),
            )
            for _ in range(order_count):
                quantity = int(quantity_rng.choice([1, 1, 1, 2, 2, 3]))
                is_refund = bool(refund_rng.random() < refund_probability)
                order_row = OrderRow(
                    order_id=f"O{order_number:06d}",
                    product_id=product.product_id,
                    customer_id=(
                        f"C{int(customer_rng.integers(1, config.customer_count + 1)):04d}"
                    ),
                    order_date=current_date,
                    quantity=quantity,
                    unit_price=product.price,
                    revenue=(product.price * quantity).quantize(
                        CENT, rounding=ROUND_HALF_UP
                    ),
                    is_refund=is_refund,
                    status="refunded" if is_refund else "paid",
                )
                order_rows.append(order_row.model_dump())
                order_number += 1

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
