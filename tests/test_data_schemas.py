from datetime import date

import pytest
from pydantic import ValidationError

from app.data.schemas import CustomerRow, MarketingRow, OrderRow, ProductRow, TrafficRow


def test_product_rejects_cost_equal_to_or_above_price() -> None:
    for cost in (50, 60):
        with pytest.raises(ValidationError):
            ProductRow(
                product_id="P001",
                product_name="商品 001",
                category="home",
                price=50,
                cost=cost,
                launch_date=date(2026, 1, 1),
            )


def test_customer_rejects_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        CustomerRow(
            customer_id="C001",
            is_new_customer=True,
            region="north",
            channel="organic",
        )


def test_traffic_rejects_clicks_above_impressions() -> None:
    with pytest.raises(ValidationError):
        TrafficRow(
            date=date(2026, 1, 1),
            product_id="P001",
            impressions=100,
            clicks=120,
            visits=120,
            is_missing=False,
        )


def test_traffic_requires_null_metrics_exactly_when_marked_missing() -> None:
    missing_metrics = TrafficRow(
        date=date(2026, 1, 1),
        product_id="P001",
        impressions=None,
        clicks=None,
        visits=None,
        is_missing=True,
    )
    assert missing_metrics.is_missing is True

    with pytest.raises(ValidationError):
        TrafficRow(
            date=date(2026, 1, 1),
            product_id="P001",
            impressions=100,
            clicks=None,
            visits=None,
            is_missing=True,
        )

    with pytest.raises(ValidationError):
        TrafficRow(
            date=date(2026, 1, 1),
            product_id="P001",
            impressions=None,
            clicks=None,
            visits=None,
            is_missing=False,
        )


def test_marketing_rejects_negative_spend() -> None:
    with pytest.raises(ValidationError):
        MarketingRow(
            date=date(2026, 1, 1),
            product_id="P001",
            campaign_id="M001",
            spend=-1,
        )


def test_order_revenue_must_equal_quantity_times_unit_price() -> None:
    with pytest.raises(ValidationError):
        OrderRow(
            order_id="O000001",
            product_id="P001",
            customer_id="C0001",
            order_date=date(2026, 1, 1),
            quantity=2,
            unit_price=50,
            revenue=80,
            is_refund=False,
            status="paid",
        )


def test_order_refund_flag_and_status_must_agree() -> None:
    with pytest.raises(ValidationError):
        OrderRow(
            order_id="O000001",
            product_id="P001",
            customer_id="C0001",
            order_date=date(2026, 1, 1),
            quantity=2,
            unit_price=50,
            revenue=100,
            is_refund=True,
            status="paid",
        )


def test_rows_reject_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        ProductRow(
            product_id="P001",
            product_name="商品 001",
            category="home",
            price=50,
            cost=30,
            launch_date=date(2026, 1, 1),
            unexpected="value",
        )
