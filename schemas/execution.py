"""Schemas for order lifecycle and portfolio state."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order(BaseModel):
    order_id: str
    strategy_id: str
    asset: str
    side: str
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    commission: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class PortfolioState(BaseModel):
    timestamp: datetime
    cash: float
    total_value: float
    positions: dict[str, dict]
    daily_pnl: float
    total_pnl: float

