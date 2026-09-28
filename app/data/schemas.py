from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductRow(StrictModel):
    product_id: str = Field(pattern=r"^P\d{3}$")
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: Decimal = Field(gt=0)
    cost: Decimal = Field(gt=0)
    launch_date: date

    @model_validator(mode="after")
    def validate_margin(self) -> "ProductRow":
        if self.cost >= self.price:
            raise ValueError("cost must be lower than price")
        return self


class CustomerRow(StrictModel):
    customer_id: str = Field(pattern=r"^C\d{4}$")
    is_new_customer: bool
    region: str = Field(min_length=1)
    channel: str = Field(min_length=1)


class TrafficRow(StrictModel):
    date: date
    product_id: str
    impressions: int | None = Field(default=None, ge=0)
    clicks: int | None = Field(default=None, ge=0)
    visits: int | None = Field(default=None, ge=0)
    is_missing: bool

    @model_validator(mode="after")
    def validate_funnel(self) -> "TrafficRow":
        values = (self.impressions, self.clicks, self.visits)
        if self.is_missing:
            if any(value is not None for value in values):
                raise ValueError("missing traffic rows must use null metrics")
            return self
        if any(value is None for value in values):
            raise ValueError("non-missing traffic rows require all metrics")
        if not self.impressions >= self.clicks:
            raise ValueError("clicks cannot exceed impressions")
        if self.visits < self.clicks:
            raise ValueError("visits cannot be lower than clicks")
        return self


class MarketingRow(StrictModel):
    date: date
    product_id: str
    campaign_id: str = Field(pattern=r"^M\d{3}$")
    spend: Decimal = Field(ge=0)


class OrderRow(StrictModel):
    order_id: str = Field(pattern=r"^O\d{6}$")
    product_id: str
    customer_id: str
    order_date: date
    quantity: int = Field(ge=1, le=20)
    unit_price: Decimal = Field(gt=0)
    revenue: Decimal = Field(gt=0)
    is_refund: bool
    status: Literal["paid", "refunded"]

    @model_validator(mode="after")
    def validate_order(self) -> "OrderRow":
        if self.revenue != self.unit_price * self.quantity:
            raise ValueError("revenue must equal unit_price * quantity")
        if self.is_refund != (self.status == "refunded"):
            raise ValueError("refund flag and status must agree")
        return self
