from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Money = Annotated[Decimal, Field(decimal_places=2)]
ProductId = Annotated[str, Field(pattern=r"^P[0-9]{3}$")]
CustomerId = Annotated[str, Field(pattern=r"^C[0-9]{4}$")]
OrderId = Annotated[str, Field(pattern=r"^O[0-9]{6}$")]
CampaignId = Annotated[str, Field(pattern=r"^M[0-9]{3}$")]


class DatasetRowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductRow(DatasetRowModel):
    product_id: ProductId
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: Money = Field(gt=0)
    cost: Money = Field(gt=0)
    launch_date: date

    @model_validator(mode="after")
    def validate_margin(self) -> "ProductRow":
        if self.cost >= self.price:
            raise ValueError("cost must be lower than price")
        return self


class CustomerRow(DatasetRowModel):
    customer_id: CustomerId
    is_new_customer: bool
    region: str = Field(min_length=1)
    channel: str = Field(min_length=1)


class TrafficRow(DatasetRowModel):
    date: date
    product_id: ProductId
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


class MarketingRow(DatasetRowModel):
    date: date
    product_id: ProductId
    campaign_id: CampaignId
    spend: Money = Field(ge=0)


class OrderRow(DatasetRowModel):
    order_id: OrderId
    product_id: ProductId
    customer_id: CustomerId
    order_date: date
    quantity: int = Field(ge=1, le=20)
    unit_price: Money = Field(gt=0)
    revenue: Money = Field(gt=0)
    is_refund: bool
    status: Literal["paid", "refunded"]

    @model_validator(mode="after")
    def validate_order(self) -> "OrderRow":
        if self.revenue != self.unit_price * self.quantity:
            raise ValueError("revenue must equal unit_price * quantity")
        if self.is_refund != (self.status == "refunded"):
            raise ValueError("refund flag and status must agree")
        return self
