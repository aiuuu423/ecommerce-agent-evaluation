from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from app.data.schemas import CustomerRow, MarketingRow, OrderRow, ProductRow, TrafficRow


def product_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "product_id": "P001",
        "product_name": "商品 001",
        "category": "home",
        "price": "50.00",
        "cost": "30.00",
        "launch_date": date(2026, 1, 1),
    }
    data.update(overrides)
    return data


def customer_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "customer_id": "C0001",
        "is_new_customer": True,
        "region": "north",
        "channel": "organic",
    }
    data.update(overrides)
    return data


def traffic_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "date": date(2026, 1, 1),
        "product_id": "P001",
        "impressions": 100,
        "clicks": 20,
        "visits": 20,
        "is_missing": False,
    }
    data.update(overrides)
    return data


def marketing_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "date": date(2026, 1, 1),
        "product_id": "P001",
        "campaign_id": "M001",
        "spend": "10.00",
    }
    data.update(overrides)
    return data


def order_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "order_id": "O20260101-P001-000001",
        "product_id": "P001",
        "customer_id": "C0001",
        "order_date": date(2026, 1, 1),
        "quantity": 2,
        "unit_price": "50.00",
        "revenue": "100.00",
        "is_refund": False,
        "status": "paid",
    }
    data.update(overrides)
    return data


def assert_model_error(
    model: type[BaseModel],
    data: dict[str, object],
    *,
    location: tuple[str, ...],
    error_type: str,
    message: str | None = None,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(data)

    errors = exc_info.value.errors()
    assert len(errors) == 1, errors
    error = errors[0]
    assert error["loc"] == location
    assert error["type"] == error_type
    if message is not None:
        assert error["msg"] == message


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (ProductRow, product_data()),
        (CustomerRow, customer_data()),
        (TrafficRow, traffic_data()),
        (MarketingRow, marketing_data()),
        (OrderRow, order_data()),
    ],
)
def test_each_row_model_accepts_a_valid_sample(
    model: type[BaseModel], data: dict[str, object]
) -> None:
    assert model.model_validate(data)


@pytest.mark.parametrize("cost", ["50.00", "60.00"])
def test_product_rejects_cost_equal_to_or_above_price(cost: str) -> None:
    assert_model_error(
        ProductRow,
        product_data(cost=cost),
        location=(),
        error_type="value_error",
        message="Value error, cost must be lower than price",
    )


@pytest.mark.parametrize(
    ("model", "data", "location", "pattern"),
    [
        (ProductRow, product_data(product_id="P１２３"), ("product_id",), r"^P[0-9]{3}$"),
        (CustomerRow, customer_data(customer_id="C１２３４"), ("customer_id",), r"^C[0-9]{4}$"),
        (TrafficRow, traffic_data(product_id="invalid"), ("product_id",), r"^P[0-9]{3}$"),
        (MarketingRow, marketing_data(product_id="invalid"), ("product_id",), r"^P[0-9]{3}$"),
        (MarketingRow, marketing_data(campaign_id="M１２３"), ("campaign_id",), r"^M[0-9]{3}$"),
        (
            OrderRow,
            order_data(order_id="O２０２６０１０１-P001-000001"),
            ("order_id",),
            r"^O[0-9]{8}-P[0-9]{3}-[0-9]{6}$",
        ),
        (OrderRow, order_data(product_id="invalid"), ("product_id",), r"^P[0-9]{3}$"),
        (OrderRow, order_data(customer_id="invalid"), ("customer_id",), r"^C[0-9]{4}$"),
    ],
)
def test_identifier_fields_require_their_ascii_format(
    model: type[BaseModel],
    data: dict[str, object],
    location: tuple[str, ...],
    pattern: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(data)

    errors = exc_info.value.errors()
    assert len(errors) == 1, errors
    error = errors[0]
    assert error["loc"] == location
    assert error["type"] == "string_pattern_mismatch"
    assert error["ctx"] == {"pattern": pattern}


def test_row_models_only_validate_foreign_key_format() -> None:
    row = OrderRow.model_validate(
        order_data(product_id="P999", customer_id="C9999")
    )
    assert row.product_id == "P999"
    assert row.customer_id == "C9999"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"impressions": 100, "clicks": 120, "visits": 120},
            "Value error, clicks cannot exceed impressions",
        ),
        (
            {"impressions": 100, "clicks": 80, "visits": 79},
            "Value error, visits cannot be lower than clicks",
        ),
    ],
)
def test_traffic_rejects_invalid_funnel(
    overrides: dict[str, object], message: str
) -> None:
    assert_model_error(
        TrafficRow,
        traffic_data(**overrides),
        location=(),
        error_type="value_error",
        message=message,
    )


def test_traffic_accepts_equal_funnel_boundaries() -> None:
    row = TrafficRow.model_validate(
        traffic_data(impressions=20, clicks=20, visits=20)
    )
    assert row.impressions == row.clicks == row.visits == 20


def test_traffic_requires_null_metrics_exactly_when_marked_missing() -> None:
    missing_metrics = TrafficRow.model_validate(
        traffic_data(
            impressions=None,
            clicks=None,
            visits=None,
            is_missing=True,
        )
    )
    assert missing_metrics.is_missing is True

    assert_model_error(
        TrafficRow,
        traffic_data(clicks=None, visits=None, is_missing=True),
        location=(),
        error_type="value_error",
        message="Value error, missing traffic rows must use null metrics",
    )
    assert_model_error(
        TrafficRow,
        traffic_data(impressions=None, clicks=None, visits=None),
        location=(),
        error_type="value_error",
        message="Value error, non-missing traffic rows require all metrics",
    )


@pytest.mark.parametrize(
    ("model", "make_data", "field"),
    [
        (ProductRow, lambda value: product_data(price=value), "price"),
        (ProductRow, lambda value: product_data(cost=value), "cost"),
        (MarketingRow, lambda value: marketing_data(spend=value), "spend"),
        (OrderRow, lambda value: order_data(unit_price=value), "unit_price"),
        (OrderRow, lambda value: order_data(revenue=value), "revenue"),
    ],
)
def test_money_fields_reject_more_than_two_decimal_places(
    model: type[BaseModel],
    make_data: Callable[[str], dict[str, object]],
    field: str,
) -> None:
    assert_model_error(
        model,
        make_data("10.001"),
        location=(field,),
        error_type="decimal_max_places",
    )


def test_money_fields_are_normalized_as_decimal_values() -> None:
    product = ProductRow.model_validate(product_data(price="50.10", cost="30.05"))
    marketing = MarketingRow.model_validate(marketing_data(spend="0.00"))
    order = OrderRow.model_validate(
        order_data(quantity=2, unit_price="50.10", revenue="100.20")
    )

    assert product.price == Decimal("50.10")
    assert product.cost == Decimal("30.05")
    assert marketing.spend == Decimal("0.00")
    assert order.unit_price == Decimal("50.10")
    assert order.revenue == Decimal("100.20")


@pytest.mark.parametrize(
    ("model", "data", "field", "error_type"),
    [
        (ProductRow, product_data(price="0.00"), "price", "greater_than"),
        (ProductRow, product_data(cost="0.00"), "cost", "greater_than"),
        (MarketingRow, marketing_data(spend="-0.01"), "spend", "greater_than_equal"),
        (OrderRow, order_data(unit_price="0.00"), "unit_price", "greater_than"),
        (OrderRow, order_data(revenue="0.00"), "revenue", "greater_than"),
    ],
)
def test_money_fields_enforce_business_boundaries(
    model: type[BaseModel],
    data: dict[str, object],
    field: str,
    error_type: str,
) -> None:
    assert_model_error(
        model,
        data,
        location=(field,),
        error_type=error_type,
    )


@pytest.mark.parametrize("quantity", [1, 20])
def test_order_accepts_quantity_boundaries(quantity: int) -> None:
    row = OrderRow.model_validate(
        order_data(quantity=quantity, revenue=str(Decimal("50.00") * quantity))
    )
    assert row.quantity == quantity


@pytest.mark.parametrize(
    ("quantity", "error_type"),
    [(0, "greater_than_equal"), (21, "less_than_equal")],
)
def test_order_rejects_quantity_outside_boundaries(
    quantity: int, error_type: str
) -> None:
    assert_model_error(
        OrderRow,
        order_data(quantity=quantity),
        location=("quantity",),
        error_type=error_type,
    )


@pytest.mark.parametrize(
    ("is_refund", "status"),
    [(False, "paid"), (True, "refunded")],
)
def test_order_accepts_consistent_refund_combinations(
    is_refund: bool, status: str
) -> None:
    row = OrderRow.model_validate(order_data(is_refund=is_refund, status=status))
    assert row.is_refund is is_refund
    assert row.status == status


@pytest.mark.parametrize(
    ("is_refund", "status"),
    [(True, "paid"), (False, "refunded")],
)
def test_order_rejects_inconsistent_refund_combinations(
    is_refund: bool, status: str
) -> None:
    assert_model_error(
        OrderRow,
        order_data(is_refund=is_refund, status=status),
        location=(),
        error_type="value_error",
        message="Value error, refund flag and status must agree",
    )


def test_order_revenue_must_equal_quantity_times_unit_price() -> None:
    assert_model_error(
        OrderRow,
        order_data(revenue="80.00"),
        location=(),
        error_type="value_error",
        message="Value error, revenue must equal unit_price * quantity",
    )


def test_rows_reject_undeclared_fields() -> None:
    assert_model_error(
        ProductRow,
        product_data(unexpected="value"),
        location=("unexpected",),
        error_type="extra_forbidden",
    )
