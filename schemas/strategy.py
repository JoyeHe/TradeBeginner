"""Strategy schemas shared by generation, risk, execution, and backtesting."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNCERTAIN = "uncertain"


class PositionAction(str, Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"


class Position(BaseModel):
    asset: str = Field(..., description="Ticker symbol, e.g. AAPL")
    action: PositionAction
    size_pct: float = Field(..., ge=0, le=100)
    entry_price_target: Optional[float] = None
    stop_loss_pct: float = Field(..., ge=0, le=100)
    take_profit_pct: Optional[float] = Field(None, ge=0)
    time_horizon_days: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0, le=1)


class RiskMetrics(BaseModel):
    portfolio_var_95: Optional[float] = None
    max_drawdown_estimate: Optional[float] = None
    correlation_to_existing: Optional[float] = None
    total_exposure_pct: float
    sector_concentrations: Optional[dict[str, float]] = None


class Strategy(BaseModel):
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    market_regime: MarketRegime
    positions: list[Position]
    rationale: str
    data_sources_used: list[str] = Field(default_factory=list)
    risk_metrics: RiskMetrics
    analysis_id: Optional[str] = None
    library_entry_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class UserModifiedStrategy(BaseModel):
    original_strategy: Strategy
    modified_strategy: Strategy
    user_notes: Optional[str] = None
    modification_timestamp: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False

