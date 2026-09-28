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
MAX_ORDER_SEQUENCE = 999_999
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
